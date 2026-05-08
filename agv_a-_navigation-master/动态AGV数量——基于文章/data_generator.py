#!/usr/bin/env python3
"""
调度任务数据生成器：为 MASP-BC（带电池约束的多目标 AGV 调度）生成可复现实例。

输出 JSON 包含：
- 仓库/充电点 depot 坐标
- 工作站坐标 stations
- 任务 jobs：每个任务有 roundtrip duration 和 energy 消耗
- AGV 最大数量、电池容量、单位充电时间 tau

示例：
python data_generator.py --jobs 80 --agvs 6 --seed 42 --out agv_tasks.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple


class SimpleRNG:
    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 2**32

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()

    def randrange(self, n: int) -> int:
        return int(self.random() * n)

    def choice(self, seq):
        return seq[self.randrange(len(seq))]



@dataclass
class Station:
    id: int
    x: float
    y: float


@dataclass
class Job:
    id: int
    station_id: int
    duration: float  # minutes, including roundtrip + load/unload
    energy: float    # percentage points of battery capacity
    weight: float
    distance: float


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def generate_instance(
    n_jobs: int = 100,
    n_stations: int = 20,
    max_agvs: int = 20,
    seed: int = 42,
    grid_size: int = 100,
    speed: float = 50.0,  # distance units per minute
    load_unload_min: Tuple[float, float] = (1.0, 4.0),
    battery_capacity: float = 100.0,
    full_charge_minutes: float = 60.0,
) -> Dict:
    """生成一组 AGV 任务。

    论文中的能耗以电池容量百分比表示，完整充电 60 分钟，故 tau=60/100=0.6。
    这里构造的每个任务是“仓库 -> 工作站 -> 仓库”的往返任务。
    """
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    if n_stations < 1:
        raise ValueError("n_stations must be positive")
    if max_agvs < 1:
        raise ValueError("max_agvs must be positive")

    rng = SimpleRNG(seed)
    depot = {"x": 0.0, "y": 0.0}

    stations: List[Station] = []
    for sid in range(n_stations):
        # 避免工作站过于靠近仓库，使动画更易观察。
        x = rng.uniform(10, grid_size)
        y = rng.uniform(10, grid_size)
        stations.append(Station(id=sid, x=round(x, 2), y=round(y, 2)))

    jobs: List[Job] = []
    for jid in range(n_jobs):
        station = rng.choice(stations)
        one_way = euclidean((depot["x"], depot["y"]), (station.x, station.y))
        travel_time = 2.0 * one_way / speed
        load_unload = rng.uniform(*load_unload_min)
        weight = rng.uniform(5, 80)

        duration = travel_time + load_unload
        # 能耗模型：基础行驶能耗 + 负载相关能耗 + 装卸动作能耗。单位：电量百分比。
        energy = 0.55 * travel_time + 0.0045 * weight * one_way + 0.25 * load_unload
        # 确保单个任务可由满电 AGV 完成，同时保留电池约束压力。
        energy = min(max(energy, 1.0), battery_capacity * 0.85)

        jobs.append(
            Job(
                id=jid,
                station_id=station.id,
                duration=round(duration, 3),
                energy=round(energy, 3),
                weight=round(weight, 2),
                distance=round(one_way, 3),
            )
        )

    instance = {
        "name": f"masp_bc_{n_jobs}_jobs_seed_{seed}",
        "seed": seed,
        "depot": depot,
        "stations": [asdict(s) for s in stations],
        "jobs": [asdict(j) for j in jobs],
        "max_agvs": max_agvs,
        "battery_capacity": battery_capacity,
        "full_charge_minutes": full_charge_minutes,
        "tau": full_charge_minutes / battery_capacity,
        "notes": "Each job is one roundtrip depot-station-depot. Energy is battery percentage points.",
    }
    return instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MASP-BC AGV scheduling data.")
    parser.add_argument("--jobs", type=int, default=30, help="number of transfer jobs")
    parser.add_argument("--stations", type=int, default=20, help="number of workstations")
    parser.add_argument("--agvs", type=int, default=20, help="maximum available AGVs")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--grid", type=int, default=100, help="coordinate grid size")
    parser.add_argument("--speed", type=float, default=50.0, help="AGV speed in distance units per minute")
    parser.add_argument("--out", type=Path, default=Path("agv_tasks.json"), help="output JSON path")
    args = parser.parse_args()

    data = generate_instance(
        n_jobs=args.jobs,
        n_stations=args.stations,
        max_agvs=args.agvs,
        seed=args.seed,
        grid_size=args.grid,
        speed=args.speed,
    )
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.out} with {len(data['jobs'])} jobs and max_agvs={data['max_agvs']}")


if __name__ == "__main__":
    main()
