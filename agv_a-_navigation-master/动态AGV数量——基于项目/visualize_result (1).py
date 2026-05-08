#!/usr/bin/env python3
"""
AGV 时间窗调度结果可视化。

输出：
- Gantt 图：任务、时间窗、到达/延迟；
- GA 收敛图：不同 AGV 数量的目标值变化；
- AGV 移动动画 GIF：展示无冲突路径协同。

示例：
python visualize_result.py --input agv_tasks_tw.json --schedule best_schedule_tw.json --history ga_history_tw.csv --gif agv_movement_tw.gif --gantt agv_gantt_tw.png --convergence ga_convergence_tw.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def route_position(route: List[Dict], t: int) -> Tuple[int, int]:
    if not route:
        return (0, 0)
    best = route[0]
    for p in route:
        if p["t"] <= t:
            best = p
        else:
            break
    return int(best["x"]), int(best["y"])


def plot_layout(ax, layout: Dict) -> None:
    points = layout["points"]
    for p in points:
        x, y = int(p["x"]), int(p["y"])
        typ = p.get("type", "")
        if typ == "supply":
            ax.scatter([x], [y], marker="s", s=130)
            ax.text(x + 0.15, y + 0.15, p.get("label", p["id"]), fontsize=8)
        elif typ in {"pickup_slot", "pickup"}:
            ax.scatter([x], [y], marker="^", s=110)
            ax.text(x + 0.15, y + 0.15, p.get("label", p["id"]), fontsize=8)
        elif typ in {"drop_point", "station", "workstation"}:
            ax.scatter([x], [y], marker="o", s=45)
            ax.text(x + 0.10, y + 0.10, p.get("label", p["id"]), fontsize=6)


def create_animation(schedule: Dict, gif_path: Path, fps: int, max_frames: int) -> None:
    layout = schedule["layout"]
    grid = layout["grid"]
    agvs = schedule["agvs"]
    makespan = int(schedule["summary"]["makespan"])
    step = max(1, makespan // max_frames) if makespan > max_frames else 1
    frames = list(range(0, makespan + 1, step))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1, int(grid["width"]) + 1)
    ax.set_ylim(-1, int(grid["height"]) + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4)
    plot_layout(ax, layout)
    scatters = []
    labels = []
    for agv in agvs:
        sc = ax.scatter([], [], s=120)
        txt = ax.text(0, 0, "", fontsize=9, weight="bold")
        scatters.append(sc)
        labels.append(txt)
    title = ax.set_title("")

    def init():
        return scatters + labels + [title]

    def update(t):
        occupied = {}
        for i, agv in enumerate(agvs):
            x, y = route_position(agv.get("route", []), t)
            occupied.setdefault((x, y), []).append(agv["agv_id"])
            scatters[i].set_offsets([[x, y]])
            labels[i].set_position((x + 0.15, y + 0.15))
            labels[i].set_text(f"A{agv['agv_id']}")
        conflicts = {cell: ids for cell, ids in occupied.items() if len(ids) > 1}
        suffix = "" if not conflicts else f" | conflict? {conflicts}"
        title.set_text(f"AGV movement, t={t}{suffix}")
        return scatters + labels + [title]

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=1000 / max(1, fps))
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def create_gantt(schedule: Dict, gantt_path: Path) -> None:
    plans = sorted(schedule["plans"], key=lambda p: (p["agv_id"], p["start_time"]))
    if not plans:
        return
    fig_h = max(5, min(18, 0.25 * len(plans) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    yticks, ylabels = [], []
    for row, p in enumerate(plans):
        y = row
        yticks.append(y)
        ylabels.append(f"A{p['agv_id']} {p['task_id']}->{p['destination_label']}")
        ax.barh(y, p["finish_time"] - p["start_time"], left=p["start_time"], height=0.5)
        ax.plot([p["window_start"], p["window_end"]], [y, y], linewidth=4)
        ax.scatter([p["delivery_start"]], [y], marker="x", s=45)
        if p["tardiness"] > 0:
            ax.text(p["delivery_start"] + 1, y, f"late {p['tardiness']}", va="center", fontsize=7)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("time")
    ax.set_title("AGV task schedule with time windows")
    ax.grid(True, axis="x", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(gantt_path, dpi=160)
    plt.close(fig)


def create_convergence(history_path: Path, convergence_path: Path) -> None:
    if not history_path.exists():
        return
    with history_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    by_fleet: Dict[str, List[Tuple[int, float]]] = {}
    for r in rows:
        by_fleet.setdefault(r["fleet_size"], []).append((int(r["generation"]), float(r["fitness"])))
    fig, ax = plt.subplots(figsize=(10, 5))
    for fleet, vals in sorted(by_fleet.items(), key=lambda kv: int(kv[0])):
        vals.sort()
        ax.plot([v[0] for v in vals], [v[1] for v in vals], label=f"{fleet} AGV")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.set_title("GA convergence by fleet size")
    ax.grid(True, linewidth=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(convergence_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize AGV time-window scheduling result.")
    parser.add_argument("--input", type=Path, default=Path("agv_tasks_tw.json"))
    parser.add_argument("--schedule", type=Path, default=Path("best_schedule_tw.json"))
    parser.add_argument("--history", type=Path, default=Path("ga_history_tw.csv"))
    parser.add_argument("--gif", type=Path, default=Path("agv_movement_tw.gif"))
    parser.add_argument("--gantt", type=Path, default=Path("agv_gantt_tw.png"))
    parser.add_argument("--convergence", type=Path, default=Path("ga_convergence_tw.png"))
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=60)
    args = parser.parse_args()

    schedule = load_json(args.schedule)
    create_gantt(schedule, args.gantt)
    create_convergence(args.history, args.convergence)
    create_animation(schedule, args.gif, args.fps, args.max_frames)
    print(f"[OK] wrote {args.gantt}, {args.convergence}, {args.gif}")


if __name__ == "__main__":
    main()
