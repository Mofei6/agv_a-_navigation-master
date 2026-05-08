import csv
import os
import random
from typing import Dict, List, Tuple

from cpp_evaluator import Chromosome, CPPEvaluator, EvaluationResult


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GAAStarCPOptimizer:
    """Upper-level GA optimizer for TAS, with lower-level A*-CP feedback.

    This version keeps explicit task -> AGV assignment in the chromosome:

        Chromosome.task_order: global task priority/order
        Chromosome.agv_assign: task_id -> agv_id

    Important modeling choices in this version:
    1. Fixed AGV set: self.agv_ids comes from planner.active_agvs.
    2. The chromosome explicitly stores AGV IDs. CPP evaluation must execute
       the assigned AGV task sequences, not dynamically choose AGVs.
    3. release_time is ignored by the GA task allocation logic. Initial
       sequences are based on window_start / window_end only.
    4. The GA is strengthened with load-balanced initialization, AGV assignment
       mutation, AGV-load swap mutation, and load-balance repair so that idle
       AGVs are less likely while other AGVs are overloaded.
    """

    def __init__(
        self,
        planner,
        population_size=32,
        generations=35,
        crossover_rate=0.85,
        mutation_rate=0.30,
        elite_size=3,
        tournament_k=3,
        seed=42,
    ):
        self.planner = planner
        self.population_size = int(os.environ.get("GA_POPULATION", population_size))
        self.generations = int(os.environ.get("GA_GENERATIONS", generations))
        self.crossover_rate = float(os.environ.get("GA_CROSSOVER_RATE", crossover_rate))
        self.mutation_rate = float(os.environ.get("GA_MUTATION_RATE", mutation_rate))
        self.elite_size = int(os.environ.get("GA_ELITE_SIZE", elite_size))
        self.tournament_k = int(os.environ.get("GA_TOURNAMENT_K", tournament_k))

        self.rng = random.Random(seed)
        self.evaluator = CPPEvaluator(planner)

        self.task_ids = [t["task_id"] for t in planner.tasks]
        self.task_by_id = {t["task_id"]: t for t in planner.tasks}
        self.agv_ids = list(planner.active_agvs)

        if not self.agv_ids:
            raise ValueError("planner.active_agvs is empty. Fixed active AGVs must be configured first.")

        self.cache: Dict[Tuple, EvaluationResult] = {}
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    # Chromosome creation and repair
    # ------------------------------------------------------------------

    def _window_key(self, tid: str):
        """Task sorting key. Do not use release_time for task assignment."""
        task = self.task_by_id[tid]
        return (
            int(task["window_start"]),
            int(task["window_end"]),
            task["task_id"],
        )

    def _due_key(self, tid: str):
        """Earliest-due-date key. Do not use release_time."""
        task = self.task_by_id[tid]
        return (
            int(task["window_end"]),
            int(task["window_start"]),
            task["task_id"],
        )

    def _cache_key(self, chrom: Chromosome):
        # Assignment is part of the chromosome and must be part of the cache key.
        return (
            tuple(chrom.task_order),
            tuple((tid, chrom.agv_assign.get(tid, "")) for tid in chrom.task_order),
        )

    def _copy_chrom(self, chrom: Chromosome) -> Chromosome:
        return Chromosome(list(chrom.task_order), dict(chrom.agv_assign))

    def _normalize_chromosome(self, chrom: Chromosome) -> Chromosome:
        """Ensure the chromosome has every task exactly once and legal AGV IDs."""
        seen = set()
        clean_order = []

        for tid in chrom.task_order:
            if tid in self.task_by_id and tid not in seen:
                clean_order.append(tid)
                seen.add(tid)

        missing = [tid for tid in self.task_ids if tid not in seen]
        # Put missing tasks by time-window order, not release_time.
        missing.sort(key=self._window_key)
        clean_order.extend(missing)

        clean_assign = {}
        for tid in self.task_ids:
            aid = chrom.agv_assign.get(tid)
            if aid not in self.agv_ids:
                aid = self.rng.choice(self.agv_ids)
            clean_assign[tid] = aid

        chrom.task_order = clean_order
        chrom.agv_assign = clean_assign
        return chrom

    def _round_robin_assign(self, order: List[str]) -> Dict[str, str]:
        """Balanced deterministic assignment by current order."""
        return {tid: self.agv_ids[i % len(self.agv_ids)] for i, tid in enumerate(order)}

    def _balanced_random_assign(self, order: List[str]) -> Dict[str, str]:
        """Randomized assignment, but keep AGV task counts balanced."""
        assign = {}
        loads = {aid: 0 for aid in self.agv_ids}

        for tid in order:
            min_load = min(loads.values())
            # Choose among the least-loaded AGVs. This avoids initial idle AGVs.
            candidates = [aid for aid in self.agv_ids if loads[aid] == min_load]
            aid = self.rng.choice(candidates)
            assign[tid] = aid
            loads[aid] += 1

        return assign

    def _balanced_window_assign(self, order: List[str]) -> Dict[str, str]:
        """Balanced assignment for tasks sorted by time-window start.

        Since all AGVs share the same virtual home in this project, distance from
        home usually does not distinguish AGVs. Therefore the main useful prior
        is load balance. This function is deterministic and stable.
        """
        assign = {}
        loads = {aid: 0 for aid in self.agv_ids}

        for tid in order:
            aid = min(self.agv_ids, key=lambda a: (loads[a], int(a)))
            assign[tid] = aid
            loads[aid] += 1

        return assign

    def _random_chromosome(self) -> Chromosome:
        order = self.task_ids[:]
        self.rng.shuffle(order)

        # Prefer balanced random assignment over fully random assignment.
        if self.rng.random() < 0.80:
            assign = self._balanced_random_assign(order)
        else:
            assign = {tid: self.rng.choice(self.agv_ids) for tid in order}

        chrom = Chromosome(order, assign)
        self._repair_assignment_by_load(chrom, max_moves=3)
        return self._normalize_chromosome(chrom)

    def _initial_population(self) -> List[Chromosome]:
        pop: List[Chromosome] = []

        # Seed 1: time-window order + round-robin assignment.
        order_window = sorted(self.task_ids, key=self._window_key)
        pop.append(Chromosome(order_window[:], self._round_robin_assign(order_window)))

        # Seed 2: time-window order + deterministic balanced assignment.
        pop.append(Chromosome(order_window[:], self._balanced_window_assign(order_window)))

        # Seed 3: time-window order + balanced random assignment.
        pop.append(Chromosome(order_window[:], self._balanced_random_assign(order_window)))

        # Seed 4: earliest due date + round-robin assignment.
        order_edd = sorted(self.task_ids, key=self._due_key)
        pop.append(Chromosome(order_edd[:], self._round_robin_assign(order_edd)))

        # Seed 5: earliest due date + balanced random assignment.
        pop.append(Chromosome(order_edd[:], self._balanced_random_assign(order_edd)))

        # Random diversified individuals.
        while len(pop) < self.population_size:
            pop.append(self._random_chromosome())

        pop = [self._normalize_chromosome(chrom) for chrom in pop[: self.population_size]]
        return pop

    def _loads(self, chrom: Chromosome) -> Dict[str, int]:
        loads = {aid: 0 for aid in self.agv_ids}
        for tid in chrom.task_order:
            aid = chrom.agv_assign.get(tid)
            if aid in loads:
                loads[aid] += 1
        return loads

    def _repair_assignment_by_load(self, chrom: Chromosome, max_moves: int = 2):
        """Repair obviously unbalanced assignment while keeping explicit AGV IDs.

        This is not dynamic AGV selection at runtime. It rewrites genes in the
        chromosome before evaluation, so final task assignment still contains
        explicit AGV IDs.
        """
        if len(self.agv_ids) <= 1:
            return

        chrom = self._normalize_chromosome(chrom)
        loads = self._loads(chrom)

        moves = 0
        while moves < max_moves:
            heavy = max(self.agv_ids, key=lambda aid: (loads[aid], -int(aid)))
            light = min(self.agv_ids, key=lambda aid: (loads[aid], int(aid)))

            if loads[heavy] - loads[light] < 2:
                break

            heavy_tasks = [tid for tid in chrom.task_order if chrom.agv_assign[tid] == heavy]
            if not heavy_tasks:
                break

            # Move one task from the most loaded AGV to the least loaded AGV.
            # Prefer a later task to reduce disruption of early time-window tasks.
            cutoff = max(1, len(heavy_tasks) // 2)
            candidates = heavy_tasks[cutoff:] or heavy_tasks
            tid = self.rng.choice(candidates)

            chrom.agv_assign[tid] = light
            loads[heavy] -= 1
            loads[light] += 1
            moves += 1

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------

    def _tournament(self, scored):
        sample = self.rng.sample(scored, min(self.tournament_k, len(scored)))
        return self._copy_chrom(min(sample, key=lambda x: x[0])[1])

    def _order_crossover(self, p1: List[str], p2: List[str]) -> Tuple[List[str], List[str]]:
        n = len(p1)
        if n < 2:
            return p1[:], p2[:]

        a, b = sorted(self.rng.sample(range(n), 2))

        def ox(x, y):
            child = [None] * n
            child[a:b + 1] = x[a:b + 1]
            fill = [v for v in y if v not in child]
            ptr = 0
            for idx in list(range(b + 1, n)) + list(range(0, b + 1)):
                if child[idx] is None:
                    child[idx] = fill[ptr]
                    ptr += 1
            return child

        return ox(p1, p2), ox(p2, p1)

    def _crossover(self, a: Chromosome, b: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if self.rng.random() > self.crossover_rate:
            return self._copy_chrom(a), self._copy_chrom(b)

        c1_order, c2_order = self._order_crossover(a.task_order, b.task_order)

        # Uniform crossover for explicit task -> AGV assignment.
        c1_assign = {}
        c2_assign = {}
        for tid in self.task_ids:
            a_aid = a.agv_assign.get(tid, self.rng.choice(self.agv_ids))
            b_aid = b.agv_assign.get(tid, self.rng.choice(self.agv_ids))

            if self.rng.random() < 0.5:
                c1_assign[tid] = a_aid
                c2_assign[tid] = b_aid
            else:
                c1_assign[tid] = b_aid
                c2_assign[tid] = a_aid

        c1 = Chromosome(c1_order, c1_assign)
        c2 = Chromosome(c2_order, c2_assign)

        self._normalize_chromosome(c1)
        self._normalize_chromosome(c2)

        # Light repair after crossover to avoid children with very idle AGVs.
        self._repair_assignment_by_load(c1, max_moves=2)
        self._repair_assignment_by_load(c2, max_moves=2)

        return c1, c2

    def _mutate(self, chrom: Chromosome):
        chrom = self._normalize_chromosome(chrom)

        if self.rng.random() > self.mutation_rate:
            return

        n = len(chrom.task_order)
        if n == 0:
            return

        # Include both sequence mutations and explicit AGV assignment mutations.
        op = self.rng.choice([
            "swap_order",
            "invert_order",
            "relocate_order",
            "change_agv",
            "swap_agv_load",
            "rebalance_agv",
        ])

        if op == "swap_order" and n >= 2:
            i, j = self.rng.sample(range(n), 2)
            chrom.task_order[i], chrom.task_order[j] = chrom.task_order[j], chrom.task_order[i]

        elif op == "invert_order" and n >= 2:
            i, j = sorted(self.rng.sample(range(n), 2))
            chrom.task_order[i:j + 1] = list(reversed(chrom.task_order[i:j + 1]))

        elif op == "relocate_order" and n >= 2:
            i, j = self.rng.sample(range(n), 2)
            tid = chrom.task_order.pop(i)
            chrom.task_order.insert(j, tid)

        elif op == "change_agv":
            tid = self.rng.choice(chrom.task_order)
            old = chrom.agv_assign.get(tid)
            choices = [aid for aid in self.agv_ids if aid != old]
            if choices:
                chrom.agv_assign[tid] = self.rng.choice(choices)

        elif op == "swap_agv_load" and n >= 2:
            t1, t2 = self.rng.sample(chrom.task_order, 2)
            chrom.agv_assign[t1], chrom.agv_assign[t2] = (
                chrom.agv_assign.get(t2, self.rng.choice(self.agv_ids)),
                chrom.agv_assign.get(t1, self.rng.choice(self.agv_ids)),
            )

        elif op == "rebalance_agv":
            self._repair_assignment_by_load(chrom, max_moves=2)

        # Extra small probability: mutate another AGV assignment.
        # This helps escape local optima where one AGV is overloaded.
        if self.rng.random() < 0.35 and n >= 1:
            tid = self.rng.choice(chrom.task_order)
            old = chrom.agv_assign.get(tid)
            choices = [aid for aid in self.agv_ids if aid != old]
            if choices:
                chrom.agv_assign[tid] = self.rng.choice(choices)

        # Always normalize and lightly repair after mutation.
        self._normalize_chromosome(chrom)
        if self.rng.random() < 0.75:
            self._repair_assignment_by_load(chrom, max_moves=2)

    # ------------------------------------------------------------------
    # Evaluation and output
    # ------------------------------------------------------------------

    def _evaluate(self, chrom: Chromosome) -> EvaluationResult:
        self._normalize_chromosome(chrom)
        key = self._cache_key(chrom)
        ev = self.cache.get(key)
        if ev is None:
            ev = self.evaluator.evaluate(chrom, detailed=False)
            self.cache[key] = ev
        return ev

    def _write_history(self):
        with open("ga_history.csv", "w", newline="", encoding="utf-8-sig") as f:
            fields = [
                "generation",
                "best_score",
                "generation_best_score",
                "total_tardiness",
                "max_tardiness",
                "late_task_count",
                "makespan",
                "total_distance",
                "total_wait",
                "workload_imbalance",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history)

    def _write_best_chromosome(self, chrom: Chromosome, ev: EvaluationResult):
        # This file is the chromosome itself: every task gene has an explicit AGV ID.
        with open("best_chromosome.csv", "w", newline="", encoding="utf-8-sig") as f:
            fields = ["gene_index", "task_id", "assigned_agv"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, tid in enumerate(chrom.task_order):
                writer.writerow({
                    "gene_index": i,
                    "task_id": tid,
                    "assigned_agv": chrom.agv_assign[tid],
                })

        # This file is the decoded AGV sequence that CPP actually evaluated.
        with open("best_agv_sequences.csv", "w", newline="", encoding="utf-8-sig") as f:
            fields = ["agv_id", "sequence_index", "task_id"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for aid, seq in ev.agv_task_sequences.items():
                for i, tid in enumerate(seq):
                    writer.writerow({
                        "agv_id": aid,
                        "sequence_index": i,
                        "task_id": tid,
                    })

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[Chromosome, EvaluationResult]:
        population = self._initial_population()

        global_best_chrom = None
        global_best_eval = None
        global_best_score = float("inf")

        for gen in range(self.generations):
            scored = []
            for chrom in population:
                ev = self._evaluate(chrom)
                scored.append((ev.score, chrom, ev))

            scored.sort(key=lambda x: x[0])
            gen_best_score, gen_best_chrom, gen_best_eval = scored[0]

            if gen_best_score < global_best_score:
                global_best_score = gen_best_score
                global_best_chrom = self._copy_chrom(gen_best_chrom)
                global_best_eval = gen_best_eval

            self.history.append({
                "generation": gen,
                "best_score": round(global_best_score, 4),
                "generation_best_score": round(gen_best_score, 4),
                "total_tardiness": global_best_eval.total_tardiness,
                "max_tardiness": global_best_eval.max_tardiness,
                "late_task_count": global_best_eval.late_task_count,
                "makespan": global_best_eval.makespan,
                "total_distance": global_best_eval.total_distance,
                "total_wait": global_best_eval.total_wait,
                "workload_imbalance": global_best_eval.workload_imbalance,
            })

            print(
                f"GA gen {gen:03d} | "
                f"best={global_best_score:.2f} | "
                f"gen_best={gen_best_score:.2f} | "
                f"late={global_best_eval.late_task_count} | "
                f"tard={global_best_eval.total_tardiness} | "
                f"max_tard={global_best_eval.max_tardiness} | "
                f"makespan={global_best_eval.makespan} | "
                f"dist={global_best_eval.total_distance} | "
                f"wait={global_best_eval.total_wait} | "
                f"imbalance={global_best_eval.workload_imbalance}"
            )

            # Elitism.
            next_pop = [self._copy_chrom(item[1]) for item in scored[: self.elite_size]]

            while len(next_pop) < self.population_size:
                p1 = self._tournament(scored)
                p2 = self._tournament(scored)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)

                next_pop.append(c1)
                if len(next_pop) < self.population_size:
                    next_pop.append(c2)

            population = next_pop

        # Re-evaluate best in detailed mode to obtain final tracks and outputs.
        final_eval = self.evaluator.evaluate(global_best_chrom, detailed=True)
        self._write_history()
        self._write_best_chromosome(global_best_chrom, final_eval)

        return global_best_chrom, final_eval
