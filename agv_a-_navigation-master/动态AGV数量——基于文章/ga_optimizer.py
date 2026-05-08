#!/usr/bin/env python3
"""
MASP-BC 遗传算法优化器。

染色体编码与论文一致：
1) sequence: 任务排列，表示全局优先顺序；
2) assignment: 每个任务分配给哪台 AGV。

解码时按 sequence 扫描，把任务加入对应 AGV 队列；每台 AGV 执行自己的队列。
如果剩余电量不足以执行下一任务，则先插入一次“充满电”操作，充电时长为：
    tau * (battery_capacity - remaining_battery)
最后一个任务后不强制充电，因为论文假设 AGV 可夜间充电，末尾充电不计入当天 makespan。

示例：
python ga_optimizer.py --input agv_tasks.json --out best_schedule.json --history ga_history.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class SimpleRNG:
    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 2**32

    def randrange(self, *args: int) -> int:
        if len(args) == 1:
            start, stop = 0, args[0]
        elif len(args) == 2:
            start, stop = args
        else:
            raise TypeError('randrange expected 1 or 2 args')
        return start + int(self.random() * (stop - start))

    def shuffle(self, seq: List[int]) -> None:
        for i in range(len(seq) - 1, 0, -1):
            j = self.randrange(i + 1)
            seq[i], seq[j] = seq[j], seq[i]

    def sample(self, population, k: int):
        pool = list(population)
        out = []
        for _ in range(k):
            idx = self.randrange(len(pool))
            out.append(pool.pop(idx))
        return out



@dataclass
class Chromosome:
    sequence: List[int]
    assignment: List[int]  # assignment[job_id] = agv_id
    fitness: Optional[float] = None
    makespan: Optional[float] = None
    fleet_size: Optional[int] = None
    schedule: Optional[Dict] = None


def load_instance(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["jobs", "max_agvs", "battery_capacity", "tau"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys in instance JSON: {missing}")
    return data


def theta_value(instance: Dict) -> float:
    """论文中的尺度平衡参数 theta = sum_j(d_j + tau e_j) / |M|。"""
    total_single_agv_upper_bound = sum(j["duration"] + instance["tau"] * j["energy"] for j in instance["jobs"])
    return total_single_agv_upper_bound / instance["max_agvs"]


def decode(chrom: Chromosome, instance: Dict) -> Tuple[float, float, int, Dict]:
    jobs_by_id = {j["id"]: j for j in instance["jobs"]}
    max_agvs = int(instance["max_agvs"])
    capacity = float(instance["battery_capacity"])
    tau = float(instance["tau"])
    theta = theta_value(instance)

    per_agv_jobs: List[List[int]] = [[] for _ in range(max_agvs)]
    for job_id in chrom.sequence:
        agv_id = chrom.assignment[job_id]
        if agv_id < 0 or agv_id >= max_agvs:
            raise ValueError(f"Invalid AGV id {agv_id} for job {job_id}")
        per_agv_jobs[agv_id].append(job_id)

    events: List[Dict] = []
    makespan = 0.0
    used_agvs = 0
    infeas_penalty = 0.0

    for agv_id, queue in enumerate(per_agv_jobs):
        if not queue:
            continue
        used_agvs += 1
        t = 0.0
        remaining = capacity
        charge_count = 0
        for job_id in queue:
            job = jobs_by_id[job_id]
            energy = float(job["energy"])
            duration = float(job["duration"])
            if energy > capacity:
                infeas_penalty += 1e6 + (energy - capacity) * 1e4
                continue

            if remaining + 1e-9 < energy:
                charged_amount = capacity - remaining
                charge_duration = tau * charged_amount
                events.append(
                    {
                        "type": "charge",
                        "agv": agv_id,
                        "start": round(t, 6),
                        "end": round(t + charge_duration, 6),
                        "duration": round(charge_duration, 6),
                        "charged_amount": round(charged_amount, 6),
                        "battery_before": round(remaining, 6),
                        "battery_after": round(capacity, 6),
                    }
                )
                t += charge_duration
                remaining = capacity
                charge_count += 1

            start = t
            end = t + duration
            events.append(
                {
                    "type": "job",
                    "agv": agv_id,
                    "job_id": job_id,
                    "station_id": job["station_id"],
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "duration": round(duration, 6),
                    "energy": round(energy, 6),
                    "battery_before": round(remaining, 6),
                    "battery_after": round(remaining - energy, 6),
                }
            )
            t = end
            remaining -= energy
        makespan = max(makespan, t)

    # 论文目标函数：min 2/3*Cmax + 1/3*theta*AGV数量。
    fitness = (2.0 / 3.0) * makespan + (1.0 / 3.0) * theta * used_agvs + infeas_penalty
    schedule = {
        "objective": round(fitness, 6),
        "makespan": round(makespan, 6),
        "fleet_size": used_agvs,
        "theta": round(theta, 6),
        "battery_capacity": capacity,
        "tau": tau,
        "sequence": chrom.sequence,
        "assignment": chrom.assignment,
        "events": sorted(events, key=lambda e: (e["agv"], e["start"], e["type"])),
    }
    return fitness, makespan, used_agvs, schedule


def random_chromosome(job_ids: List[int], max_agvs: int, rng: SimpleRNG) -> Chromosome:
    seq = job_ids[:]
    rng.shuffle(seq)
    assignment = [rng.randrange(max_agvs) for _ in job_ids]
    return Chromosome(seq, assignment)


def evaluate(chrom: Chromosome, instance: Dict) -> Chromosome:
    fitness, makespan, fleet_size, schedule = decode(chrom, instance)
    chrom.fitness = fitness
    chrom.makespan = makespan
    chrom.fleet_size = fleet_size
    chrom.schedule = schedule
    return chrom


def roulette_select(population: List[Chromosome], rng: SimpleRNG) -> Chromosome:
    # 最小化问题：适应度越小，被选中概率越大。
    eps = 1e-9
    weights = [1.0 / (c.fitness + eps) for c in population]  # type: ignore[arg-type]
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for chrom, w in zip(population, weights):
        acc += w
        if acc >= pick:
            return chrom
    return population[-1]


def pmx(parent1: List[int], parent2: List[int], rng: SimpleRNG) -> Tuple[List[int], List[int]]:
    """Partially Matched Crossover for permutations."""
    n = len(parent1)
    if n < 2:
        return parent1[:], parent2[:]
    a, b = sorted(rng.sample(range(n), 2))

    def make_child(p1: List[int], p2: List[int]) -> List[int]:
        child = [None] * n  # type: ignore[list-item]
        child[a:b + 1] = p1[a:b + 1]
        for i in range(a, b + 1):
            gene = p2[i]
            if gene in child:
                continue
            pos = i
            while True:
                mapped = p1[pos]
                pos = p2.index(mapped)
                if child[pos] is None:
                    child[pos] = gene
                    break
        for i in range(n):
            if child[i] is None:
                child[i] = p2[i]
        return child  # type: ignore[return-value]

    return make_child(parent1, parent2), make_child(parent2, parent1)


def one_point_crossover(a1: List[int], a2: List[int], rng: SimpleRNG) -> Tuple[List[int], List[int]]:
    n = len(a1)
    if n < 2:
        return a1[:], a2[:]
    cut = rng.randrange(1, n)
    return a1[:cut] + a2[cut:], a2[:cut] + a1[cut:]


def mutate(chrom: Chromosome, max_agvs: int, mutation_prob: float, rng: SimpleRNG) -> None:
    if rng.random() < mutation_prob and len(chrom.sequence) >= 2:
        i, j = rng.sample(range(len(chrom.sequence)), 2)
        chrom.sequence[i], chrom.sequence[j] = chrom.sequence[j], chrom.sequence[i]
    if rng.random() < mutation_prob:
        job_id = rng.randrange(len(chrom.assignment))
        chrom.assignment[job_id] = rng.randrange(max_agvs)


def run_ga(
    instance: Dict,
    population_size: int = 100,
    mutation_prob: float = 0.05,
    max_generations: int = 200,
    stall_generations: int = 200,
    elite_size: int = 2,
    seed: int = 123,
) -> Tuple[Chromosome, List[Dict]]:
    rng = SimpleRNG(seed)
    job_ids = [int(j["id"]) for j in instance["jobs"]]
    max_agvs = int(instance["max_agvs"])
    if sorted(job_ids) != list(range(len(job_ids))):
        raise ValueError("Job ids must be contiguous integers from 0 to n-1.")

    population = [evaluate(random_chromosome(job_ids, max_agvs, rng), instance) for _ in range(population_size)]
    best = deepcopy(min(population, key=lambda c: c.fitness))  # type: ignore[arg-type]
    history: List[Dict] = []
    no_improve = 0

    for gen in range(max_generations):
        population.sort(key=lambda c: c.fitness)  # type: ignore[arg-type]
        current_best = population[0]
        avg_fit = sum(c.fitness for c in population if c.fitness is not None) / len(population)
        history.append(
            {
                "generation": gen,
                "best_objective": current_best.fitness,
                "best_makespan": current_best.makespan,
                "best_fleet_size": current_best.fleet_size,
                "avg_objective": avg_fit,
            }
        )

        if current_best.fitness is not None and current_best.fitness + 1e-9 < best.fitness:  # type: ignore[operator]
            best = deepcopy(current_best)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= stall_generations:
            break

        new_pop: List[Chromosome] = [deepcopy(c) for c in population[:elite_size]]
        while len(new_pop) < population_size:
            p1 = roulette_select(population, rng)
            p2 = roulette_select(population, rng)
            cseq1, cseq2 = pmx(p1.sequence, p2.sequence, rng)
            cass1, cass2 = one_point_crossover(p1.assignment, p2.assignment, rng)
            child1 = Chromosome(cseq1, cass1)
            child2 = Chromosome(cseq2, cass2)
            mutate(child1, max_agvs, mutation_prob, rng)
            mutate(child2, max_agvs, mutation_prob, rng)
            new_pop.append(evaluate(child1, instance))
            if len(new_pop) < population_size:
                new_pop.append(evaluate(child2, instance))
        population = new_pop

    return best, history


def write_history(path: Path, history: List[Dict]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize MASP-BC AGV scheduling using a genetic algorithm.")
    parser.add_argument("--input", type=Path, default=Path("agv_tasks.json"), help="input instance JSON")
    parser.add_argument("--out", type=Path, default=Path("best_schedule.json"), help="output best schedule JSON")
    parser.add_argument("--history", type=Path, default=Path("ga_history.csv"), help="output GA history CSV")
    parser.add_argument("--population", type=int, default=100, help="population size")
    parser.add_argument("--mutation", type=float, default=0.05, help="mutation probability")
    parser.add_argument("--generations", type=int, default=200, help="maximum generations")
    parser.add_argument("--stall", type=int, default=200, help="stop after this many non-improving generations")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    args = parser.parse_args()

    instance = load_instance(args.input)
    best, history = run_ga(
        instance,
        population_size=args.population,
        mutation_prob=args.mutation,
        max_generations=args.generations,
        stall_generations=args.stall,
        seed=args.seed,
    )

    result = {
        "instance_name": instance.get("name", "unknown"),
        "summary": {
            "objective": round(best.fitness, 6),
            "makespan": round(best.makespan, 6),
            "fleet_size": best.fleet_size,
            "theta": best.schedule["theta"],
            "generations_recorded": len(history),
        },
        "schedule": best.schedule,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_history(args.history, history)
    print("[OK] best solution")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"[OK] wrote {args.out} and {args.history}")


if __name__ == "__main__":
    main()
