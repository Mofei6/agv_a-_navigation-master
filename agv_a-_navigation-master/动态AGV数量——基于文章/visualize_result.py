#!/usr/bin/env python3
"""
MASP-BC 结果可视化：
1) 生成 AGV 移动动画 GIF；
2) 生成 AGV 甘特图 PNG；
3) 生成 GA 收敛曲线 PNG（如果提供 history CSV）。

示例：
python visualize_result.py --input agv_tasks.json --schedule best_schedule.json --history ga_history.csv --gif agv_movement.gif
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


def station_lookup(instance: Dict) -> Dict[int, Tuple[float, float]]:
    return {int(s["id"]): (float(s["x"]), float(s["y"])) for s in instance["stations"]}


def position_at_event(event: Dict, t: float, depot: Tuple[float, float], stations: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
    """返回某 AGV 在某事件中的位置。

    job 事件是往返：前 10% 装货在仓库，10%-50% 去工作站，50%-60% 卸货，60%-100% 回仓库。
    charge 事件始终在仓库。
    """
    if event["type"] == "charge":
        return depot
    station = stations[int(event["station_id"])]
    start, end = float(event["start"]), float(event["end"])
    if end <= start:
        return depot
    p = min(max((t - start) / (end - start), 0.0), 1.0)
    if p < 0.10:
        return depot
    if p < 0.50:
        q = (p - 0.10) / 0.40
        return (depot[0] + q * (station[0] - depot[0]), depot[1] + q * (station[1] - depot[1]))
    if p < 0.60:
        return station
    q = (p - 0.60) / 0.40
    return (station[0] + q * (depot[0] - station[0]), station[1] + q * (depot[1] - station[1]))


def agv_position(events: List[Dict], agv_id: int, t: float, depot: Tuple[float, float], stations: Dict[int, Tuple[float, float]]) -> Tuple[float, float, str]:
    agv_events = [e for e in events if int(e["agv"]) == agv_id]
    for e in agv_events:
        if float(e["start"]) <= t <= float(e["end"]):
            pos = position_at_event(e, t, depot, stations)
            label = "charging" if e["type"] == "charge" else f"job {e['job_id']}"
            return pos[0], pos[1], label
    # 空闲时停在仓库。
    return depot[0], depot[1], "idle"


def make_animation(instance: Dict, result: Dict, out_gif: Path, fps: int = 8, frames: int = 240) -> None:
    depot = (float(instance["depot"]["x"]), float(instance["depot"]["y"]))
    stations = station_lookup(instance)
    events = result["schedule"]["events"]
    makespan = float(result["summary"]["makespan"])
    fleet_size = int(result["summary"]["fleet_size"])
    used_agvs = sorted({int(e["agv"]) for e in events})
    if not used_agvs:
        raise ValueError("No AGV events found in schedule.")

    fig, ax = plt.subplots(figsize=(8, 7))
    xs = [x for x, _ in stations.values()] + [depot[0]]
    ys = [y for _, y in stations.values()] + [depot[1]]
    pad = 10
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("AGV movement animation")

    ax.scatter([depot[0]], [depot[1]], marker="s", s=160, label="Depot / charger")
    ax.scatter([p[0] for p in stations.values()], [p[1] for p in stations.values()], marker="o", s=40, label="Stations")
    for sid, (x, y) in stations.items():
        ax.text(x, y, str(sid), fontsize=7, ha="left", va="bottom")
    ax.legend(loc="upper right")

    scat = ax.scatter([], [], s=120)
    labels = [ax.text(0, 0, "", fontsize=9, ha="center", va="bottom") for _ in used_agvs]
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def update(frame: int):
        t = makespan * frame / max(frames - 1, 1)
        positions = []
        for idx, agv_id in enumerate(used_agvs):
            x, y, state = agv_position(events, agv_id, t, depot, stations)
            positions.append((x, y))
            labels[idx].set_position((x, y + 2.0))
            labels[idx].set_text(f"AGV{agv_id}: {state}")
        scat.set_offsets(positions)
        time_text.set_text(f"t = {t:.1f} / {makespan:.1f} min | used AGVs = {fleet_size}")
        return [scat, time_text, *labels]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_gantt(result: Dict, out_png: Path) -> None:
    events = result["schedule"]["events"]
    used_agvs = sorted({int(e["agv"]) for e in events})
    fig, ax = plt.subplots(figsize=(11, max(4, len(used_agvs) * 0.7)))
    yticks, ylabels = [], []
    for row, agv_id in enumerate(used_agvs):
        yticks.append(row)
        ylabels.append(f"AGV {agv_id}")
        for e in [ev for ev in events if int(ev["agv"]) == agv_id]:
            start = float(e["start"])
            duration = float(e["end"]) - start
            ax.barh(row, duration, left=start, height=0.45)
            text = "C" if e["type"] == "charge" else f"J{e['job_id']}"
            ax.text(start + duration / 2, row, text, ha="center", va="center", fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (minutes)")
    ax.set_title("AGV schedule Gantt chart: jobs and charging operations")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def make_convergence(history_csv: Path, out_png: Path) -> None:
    if not history_csv.exists():
        return
    gens, bests, avgs = [], [], []
    with history_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gens.append(int(row["generation"]))
            bests.append(float(row["best_objective"]))
            avgs.append(float(row["avg_objective"]))
    if not gens:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gens, bests, label="best objective")
    ax.plot(gens, avgs, label="average objective")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Objective")
    ax.set_title("GA convergence")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MASP-BC AGV schedule and movement.")
    parser.add_argument("--input", type=Path, default=Path("agv_tasks.json"), help="input instance JSON")
    parser.add_argument("--schedule", type=Path, default=Path("best_schedule.json"), help="optimized schedule JSON")
    parser.add_argument("--history", type=Path, default=Path("ga_history.csv"), help="GA history CSV")
    parser.add_argument("--gif", type=Path, default=Path("agv_movement.gif"), help="output movement GIF")
    parser.add_argument("--gantt", type=Path, default=Path("agv_gantt.png"), help="output Gantt chart PNG")
    parser.add_argument("--convergence", type=Path, default=Path("ga_convergence.png"), help="output convergence PNG")
    parser.add_argument("--frames", type=int, default=240, help="animation frames")
    parser.add_argument("--fps", type=int, default=8, help="animation frames per second")
    args = parser.parse_args()

    instance = load_json(args.input)
    result = load_json(args.schedule)
    make_animation(instance, result, args.gif, fps=args.fps, frames=args.frames)
    make_gantt(result, args.gantt)
    make_convergence(args.history, args.convergence)
    print(f"[OK] wrote {args.gif}, {args.gantt}, {args.convergence}")


if __name__ == "__main__":
    main()
