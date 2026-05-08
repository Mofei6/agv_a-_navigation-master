#!/usr/bin/env python3
"""
无限电量 AGV 时间窗调度优化器。

核心功能：
- 保留“求最优 AGV 数量”：自动遍历 1..max_agvs，输出每个车队规模的最好解，并选择综合目标最优的 AGV 数量；
- 去掉充电和电池约束；
- 每个任务有 release_time、window_start、window_end；
- AGV 从固定取料区取货后送至工作站；
- 使用时空 A* + reservation table 避免多 AGV 顶点冲突和对向边冲突；
- GA 双字符串编码：任务全局顺序 + 任务到 AGV 的分配。

示例：
python ga_optimizer.py --input agv_tasks_tw.json --out best_schedule_tw.json --history ga_history_tw.csv --population 80 --generations 120
"""
from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

Cell = Tuple[int, int]
TimedCell = Tuple[int, int, int]


@dataclass
class TaskPlan:
    task_id: str
    agv_id: int
    pickup_id: str
    destination_id: str
    start_time: int
    pickup_arrival: int
    pickup_departure: int
    arrival_time: int
    delivery_start: int
    finish_time: int
    window_start: int
    window_end: int
    tardiness: int
    weighted_tardiness: float
    route: List[TimedCell]


class ReservationTable:
    def __init__(self) -> None:
        self.vertex: Dict[Tuple[int, int, int], int] = {}
        self.edge: Dict[Tuple[int, int, int, int, int], int] = {}

    def copy(self) -> "ReservationTable":
        other = ReservationTable()
        other.vertex = dict(self.vertex)
        other.edge = dict(self.edge)
        return other

    def is_vertex_free(self, cell: Cell, t: int, agv_id: int) -> bool:
        owner = self.vertex.get((cell[0], cell[1], t))
        return owner is None or owner == agv_id

    def is_move_free(self, a: Cell, b: Cell, depart_t: int, agv_id: int) -> bool:
        if not self.is_vertex_free(b, depart_t + 1, agv_id):
            return False
        owner_same = self.edge.get((a[0], a[1], b[0], b[1], depart_t))
        if owner_same is not None and owner_same != agv_id:
            return False
        owner_reverse = self.edge.get((b[0], b[1], a[0], a[1], depart_t))
        if owner_reverse is not None and owner_reverse != agv_id:
            return False
        return True

    def reserve_path(self, path: List[TimedCell], agv_id: int) -> None:
        if not path:
            return
        for x, y, t in path:
            self.vertex[(x, y, t)] = agv_id
        for a, b in zip(path, path[1:]):
            ax, ay, at = a
            bx, by, bt = b
            if bt == at + 1:
                self.edge[(ax, ay, bx, by, at)] = agv_id

    def reserve_wait(self, cell: Cell, start_t: int, end_t: int, agv_id: int) -> bool:
        for t in range(start_t, end_t + 1):
            if not self.is_vertex_free(cell, t, agv_id):
                return False
        for t in range(start_t, end_t + 1):
            self.vertex[(cell[0], cell[1], t)] = agv_id
        return True


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct(came: Dict[Tuple[int, int, int], Tuple[int, int, int]], state: Tuple[int, int, int]) -> List[TimedCell]:
    out = [state]
    while state in came:
        state = came[state]
        out.append(state)
    out.reverse()
    return out


def astar_time_path(
    start: Cell,
    goal: Cell,
    start_t: int,
    grid_w: int,
    grid_h: int,
    reservations: ReservationTable,
    agv_id: int,
    obstacles: Set[Cell],
    allow_diagonal: bool,
    max_wait: int = 500,
) -> Optional[List[TimedCell]]:
    if start in obstacles or goal in obstacles:
        return None
    if allow_diagonal:
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0)]
        heuristic = lambda c: max(abs(c[0] - goal[0]), abs(c[1] - goal[1]))
    else:
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        heuristic = lambda c: manhattan(c, goal)

    start_state = (start[0], start[1], start_t)
    limit_t = start_t + heuristic(start) + max_wait + grid_w * grid_h
    pq = [(heuristic(start), 0, start_state)]
    came: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    best_g = {start_state: 0}

    while pq:
        _, g, state = heapq.heappop(pq)
        x, y, t = state
        if (x, y) == goal:
            return reconstruct(came, state)
        if t >= limit_t:
            continue
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            nt = t + 1
            if nx < 0 or nx >= grid_w or ny < 0 or ny >= grid_h:
                continue
            if (nx, ny) in obstacles:
                continue
            if dx != 0 and dy != 0 and not allow_diagonal:
                continue
            if not reservations.is_move_free((x, y), (nx, ny), t, agv_id):
                continue
            ns = (nx, ny, nt)
            ng = g + 1
            if ng < best_g.get(ns, 10**12):
                best_g[ns] = ng
                came[ns] = state
                heapq.heappush(pq, (ng + heuristic((nx, ny)), ng, ns))
    return None


def merge_routes(a: List[TimedCell], b: List[TimedCell]) -> List[TimedCell]:
    if not a:
        return b[:]
    if not b:
        return a[:]
    if a[-1] == b[0]:
        return a + b[1:]
    return a + b


def add_wait_to_route(route: List[TimedCell], cell: Cell, start_t: int, end_t: int) -> List[TimedCell]:
    if end_t <= start_t:
        return route
    extra = [(cell[0], cell[1], t) for t in range(start_t + 1, end_t + 1)]
    return merge_routes(route, extra)


def point_cell(p: Dict) -> Cell:
    return int(p["x"]), int(p["y"])


def decode_solution(
    instance: Dict,
    sequence: List[int],
    assignment: List[int],
    fleet_size: int,
    max_path_wait: int,
) -> Dict:
    tasks = instance["tasks"]
    layout = instance["layout"]
    grid = layout["grid"]
    grid_w, grid_h = int(grid["width"]), int(grid["height"])
    allow_diagonal = bool(instance.get("params", {}).get("allow_diagonal", False))
    load_time = int(instance.get("params", {}).get("load_time", 2))
    unload_time = int(instance.get("params", {}).get("unload_time", 2))
    supply = layout["supply"]
    pickup_points = layout.get("pickup_points") or [supply]
    by_id = {p["id"]: p for p in layout["points"]}
    obstacles = {(int(o["x"]), int(o["y"])) for o in layout.get("obstacles", [])}
    reservations = ReservationTable()
    agv_state = [{"time": 0, "cell": point_cell(supply), "route": [], "tasks": []} for _ in range(fleet_size)]
    failures = 0
    task_plans: List[TaskPlan] = []

    for task_idx in sequence:
        task = tasks[task_idx]
        agv_id = assignment[task_idx] % fleet_size
        st = agv_state[agv_id]
        current_cell = st["cell"]
        current_time = int(st["time"])
        dest = by_id[task["destination_id"]]
        dest_cell = point_cell(dest)
        pickup_order = sorted(pickup_points, key=lambda p: manhattan(current_cell, point_cell(p)) + manhattan(point_cell(p), dest_cell))
        best_candidate = None

        for pickup in pickup_order[:2]:
            pick_cell = point_cell(pickup)
            trial_res = reservations.copy()
            travel_lb = manhattan(current_cell, pick_cell)
            release = int(task.get("release_time", 0))
            departure = max(current_time, release - travel_lb - 1)
            departure = max(current_time, departure)
            for retry in range(4):
                dep = departure + retry * 3
                path1 = astar_time_path(current_cell, pick_cell, dep, grid_w, grid_h, trial_res, agv_id, obstacles, allow_diagonal, max_path_wait)
                if path1 is None:
                    continue
                local_res = trial_res.copy()
                local_res.reserve_path(path1, agv_id)
                pick_arrival = path1[-1][2]
                load_start = max(pick_arrival, release)
                if not local_res.reserve_wait(pick_cell, pick_arrival, load_start + load_time, agv_id):
                    continue
                pickup_departure = load_start + load_time
                route = add_wait_to_route(path1, pick_cell, pick_arrival, pickup_departure)
                path2 = astar_time_path(pick_cell, dest_cell, pickup_departure, grid_w, grid_h, local_res, agv_id, obstacles, allow_diagonal, max_path_wait)
                if path2 is None:
                    continue
                local_res.reserve_path(path2, agv_id)
                route = merge_routes(route, path2)
                arrival = path2[-1][2]
                delivery_start = max(arrival, int(task.get("window_start", 0)))
                finish = delivery_start + unload_time
                if not local_res.reserve_wait(dest_cell, arrival, finish, agv_id):
                    continue
                route = add_wait_to_route(route, dest_cell, arrival, finish)
                tardiness = max(0, delivery_start - int(task.get("window_end", 0)))
                score = (tardiness * float(task.get("priority_weight", 1.0)), tardiness, finish, manhattan(pick_cell, dest_cell))
                candidate = (score, pickup, local_res, route, dep, pick_arrival, pickup_departure, arrival, delivery_start, finish, tardiness)
                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = candidate
                break

        if best_candidate is None:
            failures += 1
            st["time"] = current_time + 1
            continue

        _, pickup, reservations, route, departure, pick_arrival, pickup_departure, arrival, delivery_start, finish, tardiness = best_candidate
        weighted_tardiness = tardiness * float(task.get("priority_weight", 1.0))
        plan = TaskPlan(
            task_id=task["id"],
            agv_id=agv_id,
            pickup_id=pickup["id"],
            destination_id=task["destination_id"],
            start_time=departure,
            pickup_arrival=pick_arrival,
            pickup_departure=pickup_departure,
            arrival_time=arrival,
            delivery_start=delivery_start,
            finish_time=finish,
            window_start=int(task.get("window_start", 0)),
            window_end=int(task.get("window_end", 0)),
            tardiness=tardiness,
            weighted_tardiness=weighted_tardiness,
            route=route,
        )
        task_plans.append(plan)
        st["time"] = finish
        st["cell"] = point_cell(by_id[task["destination_id"]])
        st["route"] = merge_routes(st["route"], route)
        st["tasks"].append(task["id"])

    total_tardiness = sum(p.tardiness for p in task_plans)
    weighted_tardiness = sum(p.weighted_tardiness for p in task_plans)
    max_tardiness = max([p.tardiness for p in task_plans], default=0)
    late_count = sum(1 for p in task_plans if p.tardiness > 0)
    makespan = max([int(s["time"]) for s in agv_state], default=0)
    planned_count = len(task_plans)
    used_agvs = sum(1 for s in agv_state if s["tasks"])
    return {
        "plans": task_plans,
        "agv_state": agv_state,
        "metrics": {
            "fleet_size": fleet_size,
            "used_agvs": used_agvs,
            "planned_tasks": planned_count,
            "unplanned_tasks": len(tasks) - planned_count + failures,
            "total_tardiness": total_tardiness,
            "weighted_tardiness": weighted_tardiness,
            "max_tardiness": max_tardiness,
            "late_count": late_count,
            "makespan": makespan,
            "path_failures": failures,
        },
    }


def objective(metrics: Dict, fleet_weight: float, tardiness_weight: float, late_weight: float, makespan_weight: float, failure_weight: float) -> float:
    return (
        failure_weight * metrics["unplanned_tasks"]
        + tardiness_weight * metrics["weighted_tardiness"]
        + late_weight * metrics["late_count"]
        + makespan_weight * metrics["makespan"]
        + fleet_weight * metrics["fleet_size"]
    )


def pmx(parent1: List[int], parent2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    n = len(parent1)
    if n < 2:
        return parent1[:], parent2[:]
    a, b = sorted(rng.sample(range(n), 2))
    def make_child(p1, p2):
        child = [None] * n
        child[a:b+1] = p1[a:b+1]
        for i in range(a, b + 1):
            if p2[i] not in child:
                pos = i
                while True:
                    mapped = p1[pos]
                    pos = p2.index(mapped)
                    if child[pos] is None:
                        child[pos] = p2[i]
                        break
        for i in range(n):
            if child[i] is None:
                child[i] = p2[i]
        return child
    return make_child(parent1, parent2), make_child(parent2, parent1)


def crossover_assign(a1: List[int], a2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    n = len(a1)
    if n < 2:
        return a1[:], a2[:]
    cut = rng.randrange(1, n)
    return a1[:cut] + a2[cut:], a2[:cut] + a1[cut:]


def mutate_sequence(seq: List[int], rng: random.Random) -> None:
    if len(seq) >= 2:
        i, j = rng.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]


def mutate_assignment(assign: List[int], fleet_size: int, rng: random.Random) -> None:
    if assign:
        i = rng.randrange(len(assign))
        assign[i] = rng.randrange(fleet_size)


def roulette(population: List[Dict], rng: random.Random) -> Dict:
    inv = [1.0 / (1e-9 + ind["fitness"]) for ind in population]
    total = sum(inv)
    r = rng.random() * total
    acc = 0.0
    for ind, w in zip(population, inv):
        acc += w
        if acc >= r:
            return ind
    return population[-1]


def run_ga_for_fleet(instance: Dict, fleet_size: int, args) -> Tuple[Dict, List[Dict]]:
    rng = random.Random(args.seed + fleet_size * 1009)
    n = len(instance["tasks"])
    base_seq = list(range(n))
    population = []

    for _ in range(args.population):
        seq = base_seq[:]
        rng.shuffle(seq)
        if rng.random() < 0.5:
            seq.sort(key=lambda i: (instance["tasks"][i].get("window_end", 0), instance["tasks"][i].get("release_time", 0)))
            for _ in range(max(1, n // 8)):
                mutate_sequence(seq, rng)
        assign = [rng.randrange(fleet_size) for _ in range(n)]
        decoded = decode_solution(instance, seq, assign, fleet_size, args.max_path_wait)
        fit = objective(decoded["metrics"], args.fleet_weight, args.tardiness_weight, args.late_weight, args.makespan_weight, args.failure_weight)
        population.append({"sequence": seq, "assignment": assign, "fitness": fit, "decoded": decoded})

    population.sort(key=lambda x: x["fitness"])
    best = copy.deepcopy(population[0])
    history = []
    stall = 0

    for gen in range(args.generations):
        new_pop = [copy.deepcopy(population[0]), copy.deepcopy(population[1])] if len(population) >= 2 else [copy.deepcopy(population[0])]
        while len(new_pop) < args.population:
            p1 = roulette(population, rng)
            p2 = roulette(population, rng)
            cseq1, cseq2 = pmx(p1["sequence"], p2["sequence"], rng)
            cass1, cass2 = crossover_assign(p1["assignment"], p2["assignment"], rng)
            for cseq, cass in [(cseq1, cass1), (cseq2, cass2)]:
                if rng.random() < args.mutation:
                    mutate_sequence(cseq, rng)
                if rng.random() < args.mutation:
                    mutate_assignment(cass, fleet_size, rng)
                decoded = decode_solution(instance, cseq, cass, fleet_size, args.max_path_wait)
                fit = objective(decoded["metrics"], args.fleet_weight, args.tardiness_weight, args.late_weight, args.makespan_weight, args.failure_weight)
                new_pop.append({"sequence": cseq, "assignment": cass, "fitness": fit, "decoded": decoded})
                if len(new_pop) >= args.population:
                    break
        population = sorted(new_pop, key=lambda x: x["fitness"])
        gen_best = population[0]
        metrics = gen_best["decoded"]["metrics"]
        history.append({
            "fleet_size": fleet_size,
            "generation": gen,
            "fitness": gen_best["fitness"],
            **metrics,
        })
        if gen_best["fitness"] + 1e-9 < best["fitness"]:
            best = copy.deepcopy(gen_best)
            stall = 0
        else:
            stall += 1
        if stall >= args.stall:
            break
    return best, history


def serialise_solution(instance: Dict, best: Dict, fleet_summaries: List[Dict], args) -> Dict:
    decoded = best["decoded"]
    plans = []
    task_by_id = {t["id"]: t for t in instance["tasks"]}
    for p in decoded["plans"]:
        t = task_by_id[p.task_id]
        plans.append({
            "task_id": p.task_id,
            "agv_id": p.agv_id,
            "pickup_id": p.pickup_id,
            "destination_id": p.destination_id,
            "destination_label": t.get("destination_label", ""),
            "material": t.get("material", ""),
            "release_time": t.get("release_time", 0),
            "window_start": p.window_start,
            "window_end": p.window_end,
            "start_time": p.start_time,
            "pickup_arrival": p.pickup_arrival,
            "pickup_departure": p.pickup_departure,
            "arrival_time": p.arrival_time,
            "delivery_start": p.delivery_start,
            "finish_time": p.finish_time,
            "tardiness": p.tardiness,
            "weighted_tardiness": p.weighted_tardiness,
            "route": [{"x": x, "y": y, "t": tt} for x, y, tt in p.route],
        })
    agvs = []
    for i, st in enumerate(decoded["agv_state"]):
        agvs.append({
            "agv_id": i,
            "tasks": st["tasks"],
            "finish_time": st["time"],
            "route": [{"x": x, "y": y, "t": tt} for x, y, tt in st["route"]],
        })
    return {
        "input_name": instance.get("name", ""),
        "model": "infinite_battery_time_windows_conflict_free_paths",
        "objective_weights": {
            "fleet_weight": args.fleet_weight,
            "tardiness_weight": args.tardiness_weight,
            "late_weight": args.late_weight,
            "makespan_weight": args.makespan_weight,
            "failure_weight": args.failure_weight,
        },
        "summary": {"objective": best["fitness"], **decoded["metrics"]},
        "fleet_summaries": fleet_summaries,
        "chromosome": {
            "sequence_task_indices": best["sequence"],
            "assignment_by_task_index": best["assignment"],
        },
        "layout": instance["layout"],
        "tasks": instance["tasks"],
        "plans": plans,
        "agvs": agvs,
    }


def choose_best(results: List[Dict]) -> Dict:
    return min(results, key=lambda r: (r["best"]["fitness"], r["best"]["decoded"]["metrics"]["weighted_tardiness"], r["fleet_size"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="GA optimizer for AGV time-window scheduling without charging.")
    parser.add_argument("--input", type=Path, default=Path("agv_tasks_tw.json"))
    parser.add_argument("--out", type=Path, default=Path("best_schedule_tw.json"))
    parser.add_argument("--history", type=Path, default=Path("ga_history_tw.csv"))
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--mutation", type=float, default=0.08)
    parser.add_argument("--stall", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-agvs", type=int, default=1)
    parser.add_argument("--max-agvs", type=int, default=None)
    parser.add_argument("--fleet-weight", type=float, default=30.0)
    parser.add_argument("--tardiness-weight", type=float, default=1000.0)
    parser.add_argument("--late-weight", type=float, default=150.0)
    parser.add_argument("--makespan-weight", type=float, default=1.0)
    parser.add_argument("--failure-weight", type=float, default=1000000.0)
    parser.add_argument("--max-path-wait", type=int, default=30)
    args = parser.parse_args()

    instance = json.loads(args.input.read_text(encoding="utf-8"))
    max_agvs = args.max_agvs or int(instance.get("max_agvs", 1))
    min_agvs = max(1, args.min_agvs)
    results = []
    all_history = []

    for k in range(min_agvs, max_agvs + 1):
        best, hist = run_ga_for_fleet(instance, k, args)
        results.append({"fleet_size": k, "best": best})
        all_history.extend(hist)
        m = best["decoded"]["metrics"]
        print(f"[fleet={k}] objective={best['fitness']:.3f} tardiness={m['total_tardiness']} late={m['late_count']} makespan={m['makespan']} unplanned={m['unplanned_tasks']}")

    selected = choose_best(results)
    fleet_summaries = []
    for r in results:
        b = r["best"]
        fleet_summaries.append({"fleet_size": r["fleet_size"], "objective": b["fitness"], **b["decoded"]["metrics"]})

    solution = serialise_solution(instance, selected["best"], fleet_summaries, args)
    args.out.write_text(json.dumps(solution, ensure_ascii=False, indent=2), encoding="utf-8")

    if all_history:
        with args.history.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_history[0].keys()))
            writer.writeheader()
            writer.writerows(all_history)

    s = solution["summary"]
    print(f"[OK] wrote {args.out}")
    print(f"best fleet_size={s['fleet_size']} used_agvs={s['used_agvs']} objective={s['objective']:.3f} total_tardiness={s['total_tardiness']} late_count={s['late_count']} makespan={s['makespan']}")


if __name__ == "__main__":
    main()
