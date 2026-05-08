
import csv
import os
import random
from typing import Dict, List, Tuple

from cpp_evaluator import Chromosome, CPPEvaluator, EvaluationResult


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GAAStarCPOptimizer:
    """Upper-level GA optimizer for TAS, with lower-level A*-CP feedback.

    The original project chose an AGV greedily for each task and committed it
    immediately. This optimizer evaluates complete chromosomes, so a late
    congestion penalty can influence the whole task allocation and sequencing.
    """

    def __init__(
        self,
        planner,
        population_size=32,
        generations=35,
        crossover_rate=0.85,
        mutation_rate=0.25,
        elite_size=3,
        tournament_k=3,
        seed=42,
    ):
        self.planner = planner
        self.population_size = int(os.environ.get("GA_POPULATION", population_size))
        self.generations = int(os.environ.get("GA_GENERATIONS", generations))
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_k = tournament_k
        self.rng = random.Random(seed)
        self.evaluator = CPPEvaluator(planner)

        self.task_ids = [t["task_id"] for t in planner.tasks]
        self.task_by_id = {t["task_id"]: t for t in planner.tasks}
        self.agv_ids = list(planner.active_agvs)

        self.cache: Dict[Tuple, EvaluationResult] = {}
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    # Chromosome creation and repair
    # ------------------------------------------------------------------

    def _cache_key(self, chrom: Chromosome):
        return (
            tuple(chrom.task_order),
            tuple((tid, chrom.agv_assign[tid]) for tid in chrom.task_order),
        )

    def _copy_chrom(self, chrom: Chromosome) -> Chromosome:
        return Chromosome(list(chrom.task_order), dict(chrom.agv_assign))

    def _round_robin_assign(self, order: List[str]) -> Dict[str, str]:
        return {tid: self.agv_ids[i % len(self.agv_ids)] for i, tid in enumerate(order)}

    def _balanced_deadline_assign(self, order: List[str]) -> Dict[str, str]:
        assign = {}
        load = {aid: 0 for aid in self.agv_ids}
        for tid in order:
            task = self.task_by_id[tid]
            dp = self.planner.drop_points[task["destination_id"]]
            station_pos = dp["pos"]
            best_aid = min(
                self.agv_ids,
                key=lambda aid: (
                    load[aid],
                    manhattan((50, 26), station_pos),
                    int(aid),
                ),
            )
            assign[tid] = best_aid
            load[best_aid] += 1
        return assign

    def _random_chromosome(self) -> Chromosome:
        order = self.task_ids[:]
        self.rng.shuffle(order)
        assign = {tid: self.rng.choice(self.agv_ids) for tid in order}
        return Chromosome(order, assign)

    def _initial_population(self) -> List[Chromosome]:
        pop: List[Chromosome] = []

        # Seed 1: release/window order from input.
        order_release = sorted(
            self.task_ids,
            key=lambda tid: (
                self.task_by_id[tid]["release_time"],
                self.task_by_id[tid]["window_start"],
                self.task_by_id[tid]["task_id"],
            ),
        )
        pop.append(Chromosome(order_release, self._round_robin_assign(order_release)))

        # Seed 2: earliest due date.
        order_edd = sorted(
            self.task_ids,
            key=lambda tid: (
                self.task_by_id[tid]["window_end"],
                self.task_by_id[tid]["window_start"],
                self.task_by_id[tid]["release_time"],
            ),
        )
        pop.append(Chromosome(order_edd, self._round_robin_assign(order_edd)))

        # Seed 3: EDD with balanced assignment.
        pop.append(Chromosome(order_edd[:], self._balanced_deadline_assign(order_edd)))

        # Seed 4: release order with random assignment.
        assign = {tid: self.rng.choice(self.agv_ids) for tid in order_release}
        pop.append(Chromosome(order_release[:], assign))

        # Random diversified individuals.
        while len(pop) < self.population_size:
            pop.append(self._random_chromosome())

        return pop[: self.population_size]

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

        c1_assign = {}
        c2_assign = {}
        for tid in self.task_ids:
            if self.rng.random() < 0.5:
                c1_assign[tid] = a.agv_assign[tid]
                c2_assign[tid] = b.agv_assign[tid]
            else:
                c1_assign[tid] = b.agv_assign[tid]
                c2_assign[tid] = a.agv_assign[tid]

        return Chromosome(c1_order, c1_assign), Chromosome(c2_order, c2_assign)

    def _mutate(self, chrom: Chromosome):
        if self.rng.random() > self.mutation_rate:
            return

        n = len(chrom.task_order)

        # Sequence mutation: swap, inversion, or relocate.
        op = self.rng.choice(["swap", "invert", "relocate", "assign"])
        if op == "swap" and n >= 2:
            i, j = self.rng.sample(range(n), 2)
            chrom.task_order[i], chrom.task_order[j] = chrom.task_order[j], chrom.task_order[i]

        elif op == "invert" and n >= 2:
            i, j = sorted(self.rng.sample(range(n), 2))
            chrom.task_order[i:j + 1] = reversed(chrom.task_order[i:j + 1])

        elif op == "relocate" and n >= 2:
            i, j = self.rng.sample(range(n), 2)
            tid = chrom.task_order.pop(i)
            chrom.task_order.insert(j, tid)

        else:
            # Allocation mutation: move a task to a different AGV.
            tid = self.rng.choice(self.task_ids)
            old = chrom.agv_assign[tid]
            choices = [aid for aid in self.agv_ids if aid != old]
            if choices:
                chrom.agv_assign[tid] = self.rng.choice(choices)

        # Small extra chance to change one assignment even after sequence mutation.
        if self.rng.random() < 0.30:
            tid = self.rng.choice(self.task_ids)
            chrom.agv_assign[tid] = self.rng.choice(self.agv_ids)

    # ------------------------------------------------------------------
    # Evaluation and output
    # ------------------------------------------------------------------

    def _evaluate(self, chrom: Chromosome) -> EvaluationResult:
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
                f"tardiness={global_best_eval.total_tardiness} | "
                f"makespan={global_best_eval.makespan} | "
                f"dist={global_best_eval.total_distance}"
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
