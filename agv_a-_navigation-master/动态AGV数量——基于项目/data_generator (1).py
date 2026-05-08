#!/usr/bin/env python3
"""
AGV 调度数据生成/导入器（无限电量 + 时间窗 + 多 AGV 协同路径）。

支持两种方式：
1) 从用户给的两个 CSV 导入：一个布局表，一个任务时间窗表；
2) 没有 CSV 时生成一份可复现实例。

输出 JSON 供 ga_optimizer.py 使用。

示例：
python data_generator.py --layout ce91cfa6-7059-4fc9-bfc5-55e09cc8ad86.csv --tasks 9917c501-70d7-4f3e-88f6-17b3b69e6c08.csv --agvs 6 --out agv_tasks_tw.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from random import Random
from typing import Dict, List, Tuple, Optional


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def to_int(value: str, default: int = 0) -> int:
    return int(round(to_float(value, default)))


def infer_grid(points: List[Dict], margin: int = 3) -> Dict[str, int]:
    xs = [int(p["x"]) for p in points]
    ys = [int(p["y"]) for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    shift_x = margin - min_x if min_x < margin else 0
    shift_y = margin - min_y if min_y < margin else 0
    for p in points:
        p["x"] = int(p["x"]) + shift_x
        p["y"] = int(p["y"]) + shift_y
    return {
        "width": max_x + shift_x + margin + 1,
        "height": max_y + shift_y + margin + 1,
        "shift_x": shift_x,
        "shift_y": shift_y,
    }


def load_from_tables(
    layout_csv: Path,
    tasks_csv: Path,
    max_agvs: int,
    move_time: int,
    load_time: int,
    unload_time: int,
    allow_diagonal: bool,
) -> Dict:
    layout_rows = read_csv(layout_csv)
    task_rows = read_csv(tasks_csv)

    points: List[Dict] = []
    for r in layout_rows:
        p = {
            "type": r.get("type", "").strip(),
            "id": r.get("name", "").strip(),
            "label": r.get("label", "").strip(),
            "x": to_int(r.get("x", "0")),
            "y": to_int(r.get("y", "0")),
            "pitch": r.get("pitch", "").strip(),
        }
        if not p["id"]:
            p["id"] = f"P-{len(points)+1}"
        points.append(p)

    if not points:
        raise ValueError("布局 CSV 中没有点位")

    grid = infer_grid(points)
    by_id = {p["id"]: p for p in points}
    supplies = [p for p in points if p["type"].lower() == "supply"]
    pickup_points = [p for p in points if p["type"].lower() in {"pickup_slot", "pickup", "pick"}]
    drop_points = [p for p in points if p["type"].lower() in {"drop_point", "station", "workstation", "drop"}]

    if not supplies and pickup_points:
        supplies = [pickup_points[0]]
    if not pickup_points and supplies:
        pickup_points = supplies
    if not supplies:
        supplies = [points[0]]
    if not pickup_points:
        pickup_points = [points[0]]
    if not drop_points:
        raise ValueError("布局 CSV 中没有工作站/投递点，请检查 type 是否为 drop_point")

    tasks: List[Dict] = []
    for i, r in enumerate(task_rows):
        dest_id = r.get("destination_id", "").strip()
        if dest_id not in by_id:
            raise ValueError(f"任务 {r.get('task_id', i)} 的 destination_id={dest_id} 不在布局表中")
        priority = r.get("priority", "Normal").strip() or "Normal"
        weight = {"low": 0.8, "normal": 1.0, "high": 1.5, "urgent": 2.0}.get(priority.lower(), 1.0)
        tasks.append({
            "id": r.get("task_id", f"T{i+1:04d}").strip() or f"T{i+1:04d}",
            "destination_id": dest_id,
            "destination_label": r.get("destination_label", by_id[dest_id].get("label", dest_id)).strip(),
            "material": r.get("material", "").strip(),
            "release_time": to_int(r.get("release_time", "0")),
            "window_start": to_int(r.get("window_start", "0")),
            "window_end": to_int(r.get("window_end", "0")),
            "priority": priority,
            "priority_weight": weight,
        })

    instance = {
        "name": f"agv_tw_{len(tasks)}_tasks",
        "model": "infinite_battery_time_windows_conflict_free_paths",
        "layout": {
            "grid": grid,
            "points": points,
            "supply": supplies[0],
            "pickup_points": pickup_points,
            "drop_points": drop_points,
            "obstacles": [],
        },
        "tasks": tasks,
        "max_agvs": max_agvs,
        "params": {
            "move_time": move_time,
            "load_time": load_time,
            "unload_time": unload_time,
            "allow_diagonal": allow_diagonal,
            "battery": "infinite",
            "conflict_policy": "space_time_vertex_and_edge_reservation",
        },
        "notes": "AGV 从 supply/当前位置出发，到固定取料区 pickup_points 取货，再送到 destination_id；无充电约束。",
    }
    return instance


def generate_synthetic(
    n_tasks: int,
    n_stations: int,
    max_agvs: int,
    seed: int,
    grid_width: int,
    grid_height: int,
    move_time: int,
    load_time: int,
    unload_time: int,
    allow_diagonal: bool,
) -> Dict:
    rng = Random(seed)
    points = [
        {"type": "supply", "id": "SUPPLY-01", "label": "SUPPLY", "x": grid_width // 2, "y": grid_height // 2, "pitch": ""},
        {"type": "pickup_slot", "id": "PICK-1", "label": "P1", "x": grid_width // 2 - 1, "y": grid_height // 2, "pitch": ""},
        {"type": "pickup_slot", "id": "PICK-2", "label": "P2", "x": grid_width // 2 + 1, "y": grid_height // 2, "pitch": ""},
    ]
    for i in range(n_stations):
        points.append({
            "type": "drop_point",
            "id": f"DP-{i+1:03d}",
            "label": f"S{i+1}",
            "x": rng.randint(1, grid_width - 2),
            "y": rng.randint(1, grid_height - 2),
            "pitch": "",
        })
    drop_points = [p for p in points if p["type"] == "drop_point"]
    tasks = []
    for i in range(n_tasks):
        dp = rng.choice(drop_points)
        rel = i * rng.randint(2, 6)
        start = rel + rng.randint(10, 30)
        end = start + rng.randint(35, 80)
        priority = rng.choice(["Normal", "Normal", "Normal", "High"])
        tasks.append({
            "id": f"T{i+1:04d}",
            "destination_id": dp["id"],
            "destination_label": dp["label"],
            "material": f"M{rng.randint(1, 6)}",
            "release_time": rel,
            "window_start": start,
            "window_end": end,
            "priority": priority,
            "priority_weight": 1.5 if priority == "High" else 1.0,
        })
    return {
        "name": f"synthetic_agv_tw_{n_tasks}_seed_{seed}",
        "model": "infinite_battery_time_windows_conflict_free_paths",
        "layout": {
            "grid": {"width": grid_width, "height": grid_height, "shift_x": 0, "shift_y": 0},
            "points": points,
            "supply": points[0],
            "pickup_points": [points[1], points[2]],
            "drop_points": drop_points,
            "obstacles": [],
        },
        "tasks": tasks,
        "max_agvs": max_agvs,
        "params": {
            "move_time": move_time,
            "load_time": load_time,
            "unload_time": unload_time,
            "allow_diagonal": allow_diagonal,
            "battery": "infinite",
            "conflict_policy": "space_time_vertex_and_edge_reservation",
        },
        "notes": "Synthetic instance with infinite battery and time windows.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/import AGV time-window scheduling data without charging.")
    parser.add_argument("--layout", type=Path, default=None, help="layout CSV: type,name,label,x,y,pitch")
    parser.add_argument("--tasks", type=Path, default=None, help="task CSV: task_id,destination_id,release_time,window_start,window_end,...")
    parser.add_argument("--agvs", type=int, default=10, help="maximum available AGVs")
    parser.add_argument("--move-time", type=int, default=1, help="minutes per grid move")
    parser.add_argument("--load-time", type=int, default=2, help="minutes spent at pickup")
    parser.add_argument("--unload-time", type=int, default=2, help="minutes spent at workstation")
    parser.add_argument("--allow-diagonal", action="store_true", help="allow diagonal grid moves")
    parser.add_argument("--jobs", type=int, default=30, help="synthetic task count if CSVs are not given")
    parser.add_argument("--stations", type=int, default=20, help="synthetic station count")
    parser.add_argument("--seed", type=int, default=42, help="synthetic random seed")
    parser.add_argument("--grid-width", type=int, default=24)
    parser.add_argument("--grid-height", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("agv_tasks_tw.json"), help="output JSON path")
    args = parser.parse_args()

    if args.layout and args.tasks:
        data = load_from_tables(args.layout, args.tasks, args.agvs, args.move_time, args.load_time, args.unload_time, args.allow_diagonal)
    else:
        data = generate_synthetic(args.jobs, args.stations, args.agvs, args.seed, args.grid_width, args.grid_height, args.move_time, args.load_time, args.unload_time, args.allow_diagonal)

    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.out} | tasks={len(data['tasks'])} | max_agvs={data['max_agvs']} | battery=infinite")


if __name__ == "__main__":
    main()
