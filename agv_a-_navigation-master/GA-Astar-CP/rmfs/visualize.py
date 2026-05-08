import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .config import AGVS, TASKS, Params
from .layout import WarehouseLayout
from .collision import EvalResult


def draw_base(ax, layout: WarehouseLayout):
    ax.set_xlim(0.5, layout.width + 0.5)
    ax.set_ylim(0.5, layout.height + 0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(1, layout.width + 1))
    ax.set_yticks(range(1, layout.height + 1))
    ax.grid(True, linewidth=0.3, alpha=0.45)
    for x, y in layout.pods:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightgray', edgecolor='white', linewidth=0.25))
    st = layout.params.station
    ax.add_patch(Rectangle((st[0]-0.5, st[1]-0.5), 1, 1, facecolor='lightblue', edgecolor='black'))
    ax.text(st[0], st[1], 'Station', ha='center', va='center', fontsize=7)

import math
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


def plot_chromosome_encoding(best_chromosome, ev: EvalResult, out: Path):
    """
    可视化最终最优染色体 + 解码后的 AGV 任务序列
    """
    agv_ids = sorted(ev.agv_task_sequences.keys())
    k = len(agv_ids)
    colors = plt.cm.tab10.colors

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.3])

    # ---------------------------
    # 上半部分：原始染色体编码
    # ---------------------------
    ax1 = fig.add_subplot(gs[0])

    n = len(best_chromosome)
    for i, task_id in enumerate(best_chromosome):
        agv_id = agv_ids[i % k]
        color = colors[(agv_id - 1) % len(colors)]
        rect = Rectangle((i, 0), 1, 1, facecolor=color, alpha=0.35, edgecolor='black')
        ax1.add_patch(rect)

        ax1.text(i + 0.5, 0.62, f'{task_id}', ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(i + 0.5, 0.22, f'A{agv_id}', ha='center', va='center', fontsize=8)

    ax1.set_xlim(0, n)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([i + 0.5 for i in range(n)])
    ax1.set_xticklabels([str(i) for i in range(n)], rotation=90, fontsize=8)
    ax1.set_yticks([])
    ax1.set_title('Best chromosome encoding (gene value = task ID, color/label = decoded AGV)')
    ax1.set_xlabel('Gene index')

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=colors[(aid - 1) % len(colors)], alpha=0.35, edgecolor='black',
                  label=f'AGV {aid}')
        for aid in agv_ids
    ]
    ax1.legend(handles=legend_handles, loc='upper right', ncol=min(len(agv_ids), 5))

    # ---------------------------
    # 下半部分：解码后的 AGV 任务序列
    # ---------------------------
    ax2 = fig.add_subplot(gs[1])

    max_len = max(len(seq) for seq in ev.agv_task_sequences.values()) if ev.agv_task_sequences else 0
    ax2.set_xlim(0, max_len + 1.5)
    ax2.set_ylim(0.5, len(agv_ids) + 0.5)
    ax2.set_yticks(range(1, len(agv_ids) + 1))
    ax2.set_yticklabels([f'AGV {aid}' for aid in agv_ids])
    ax2.set_xticks(range(1, max_len + 1))
    ax2.set_xlabel('Sequence position')
    ax2.set_title('Decoded TAS result: task sequence assigned to each AGV')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)

    for row_idx, aid in enumerate(agv_ids, start=1):
        seq = ev.agv_task_sequences[aid]
        color = colors[(aid - 1) % len(colors)]
        for col_idx, tid in enumerate(seq, start=1):
            rect = Rectangle((col_idx - 0.45, row_idx - 0.3), 0.9, 0.6,
                             facecolor=color, alpha=0.45, edgecolor='black')
            ax2.add_patch(rect)
            ax2.text(col_idx, row_idx, str(tid), ha='center', va='center', fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_all_agv_paths(layout: WarehouseLayout, ev: EvalResult, out: Path):
    """
    将所有 AGV 的所有任务路径叠加在一张静态图上
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_base(ax, layout)

    colors = plt.cm.tab10.colors
    task_by_id = {t.task_id: t for t in TASKS}

    # 画任务点
    for tid, task in task_by_id.items():
        ax.scatter([task.loc[0]], [task.loc[1]], s=40, c='white', edgecolors='black', zorder=4)
        ax.text(task.loc[0], task.loc[1], str(tid), fontsize=7, ha='center', va='center', zorder=5)

    # 画 AGV 起点
    for agv in AGVS:
        color = colors[(agv.agv_id - 1) % len(colors)]
        ax.scatter([agv.start[0]], [agv.start[1]], marker='s', s=120, c=[color], edgecolors='black', zorder=6)
        ax.text(agv.start[0], agv.start[1], f'A{agv.agv_id}', fontsize=8, ha='center', va='center', zorder=7)

    # 画路径
    handles = []
    for aid in sorted(ev.segments.keys()):
        color = colors[(aid - 1) % len(colors)]
        segs = sorted(ev.segments[aid], key=lambda s: s.start_time)

        for seg in segs:
            xs = [p[0] for p in seg.path]
            ys = [p[1] for p in seg.path]
            alpha = 0.9 if seg.loaded else 0.5
            style = '-' if seg.loaded else '--'
            ax.plot(xs, ys, style, color=color, linewidth=2, alpha=alpha)

        handles.append(Line2D([0], [0], color=color, lw=2, label=f'AGV {aid}'))

    ax.legend(handles=handles, loc='upper left')
    ax.set_title('All AGV paths of the best solution')
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def build_agv_timeline(ev: EvalResult, agvs):
    """
    将每个 AGV 的分段路径转成逐秒位置序列:
    timeline[agv_id][t] = (x, y)
    """
    T = ev.makespan_agv
    timeline = {}

    agv_start = {a.agv_id: a.start for a in agvs}

    for agv in agvs:
        aid = agv.agv_id
        pos_seq = [None] * (T + 1)
        pos_seq[0] = agv.start

        segs = sorted(ev.segments.get(aid, []), key=lambda s: s.start_time)
        for seg in segs:
            for k, p in enumerate(seg.path):
                t = seg.start_time + k
                if 0 <= t <= T:
                    pos_seq[t] = p

        # 前向填充：如果某一时刻没有显式路径点，则停留在上一位置
        last = agv.start
        for t in range(T + 1):
            if pos_seq[t] is None:
                pos_seq[t] = last
            else:
                last = pos_seq[t]

        timeline[aid] = pos_seq

    return timeline


def get_agv_status_at_time(ev: EvalResult, agv_id: int, t: int):
    """
    返回某一时刻 AGV 所处的 segment 信息
    """
    segs = sorted(ev.segments.get(agv_id, []), key=lambda s: s.start_time)
    for seg in segs:
        if seg.start_time <= t <= seg.end_time:
            return {
                'task_id': seg.task_id,
                'label': seg.label,
                'loaded': seg.loaded
            }
    return {
        'task_id': None,
        'label': 'idle',
        'loaded': False
    }


def animate_agv_paths_gif(layout: WarehouseLayout, ev: EvalResult, agvs, out: Path,
                          fps: int = 4, tail: int = 8):
    """
    生成所有 AGV 路径执行的 GIF 动图
    """
    timeline = build_agv_timeline(ev, agvs)
    T = ev.makespan_agv
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_base(ax, layout)

    # 任务点
    for t in TASKS:
        ax.scatter([t.loc[0]], [t.loc[1]], s=30, c='white', edgecolors='black', zorder=2)
        ax.text(t.loc[0], t.loc[1], str(t.task_id), fontsize=6, ha='center', va='center', zorder=3)

    # 画工作区入口/出口文本
    pin = layout.params.workspace_entrance
    pout = layout.params.workspace_exit
    ax.text(pin[0], pin[1] + 0.6, 'IN', fontsize=8, ha='center')
    ax.text(pout[0], pout[1] + 0.6, 'OUT', fontsize=8, ha='center')

    # 每个 AGV 对应一个点、一个尾迹、一个标签
    point_artists = {}
    trail_artists = {}
    text_artists = {}

    for idx, agv in enumerate(agvs):
        color = colors[idx % len(colors)]

        trail_line, = ax.plot([], [], '-', lw=2, color=color, alpha=0.55)
        point, = ax.plot([], [], 'o', ms=10, color=color, markeredgecolor='black')
        text = ax.text(agv.start[0] + 0.25, agv.start[1] + 0.25,
                       f'A{agv.agv_id}', fontsize=8, color=color)

        trail_artists[agv.agv_id] = trail_line
        point_artists[agv.agv_id] = point
        text_artists[agv.agv_id] = text

    time_text = ax.text(0.02, 1.02, '', transform=ax.transAxes, fontsize=11)
    info_text = ax.text(1.02, 0.98, '', transform=ax.transAxes,
                        fontsize=8, va='top', family='monospace')

    ax.set_title('Animated execution of all AGV paths (best solution)')

    def update(frame):
        time_text.set_text(f't = {frame}s')

        info_lines = []
        for idx, agv in enumerate(agvs):
            aid = agv.agv_id
            pos = timeline[aid][frame]
            x, y = pos

            # 当前点
            point_artists[aid].set_data([x], [y])
            text_artists[aid].set_position((x + 0.25, y + 0.25))

            # 尾迹
            hist = timeline[aid][max(0, frame - tail): frame + 1]
            xs = [p[0] for p in hist]
            ys = [p[1] for p in hist]
            trail_artists[aid].set_data(xs, ys)

            # 状态说明
            status = get_agv_status_at_time(ev, aid, frame)
            task_show = '-' if status['task_id'] is None else status['task_id']
            info_lines.append(
                f'A{aid}: pos=({x:>2},{y:>2})  task={task_show:>2}  {status["label"]}'
            )

        info_text.set_text('\n'.join(info_lines))

        artists = [time_text, info_text]
        artists.extend(point_artists.values())
        artists.extend(trail_artists.values())
        artists.extend(text_artists.values())
        return artists

    anim = FuncAnimation(fig, update, frames=T + 1, interval=1000 / fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)

def plot_history(history, out: Path):
    gens = [h['generation'] for h in history]
    best_times = [h['best_picker_time'] for h in history]
    gen_times = [h['generation_best_picker_time'] for h in history]
    costs = [h['best_cost'] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(gens, gen_times, alpha=0.35, label='Generation best picker time')
    ax1.plot(gens, best_times, linewidth=2, label='Best-so-far picker time')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Picker completion time (s)')
    ax2 = ax1.twinx()
    ax2.plot(gens, costs, linestyle='--', label='Best-so-far AGV cost')
    ax2.set_ylabel('Total AGV cost (CNY)')
    ax1.set_title('Iteration curve of completion time and AGV cost')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_tas(layout: WarehouseLayout, ev: EvalResult, out: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_base(ax, layout)
    task_by_id = {t.task_id: t for t in TASKS}
    for agv in AGVS:
        x, y = agv.start
        ax.scatter([x], [y], marker='s', s=120, edgecolors='black')
        ax.text(x, y, f'No.{agv.agv_id}', ha='center', va='center', fontsize=7)
    for agv_id, seq in ev.agv_task_sequences.items():
        for idx, tid in enumerate(seq, start=1):
            x, y = task_by_id[tid].loc
            ax.scatter([x], [y], s=130, edgecolors='black')
            ax.text(x, y, f'{agv_id}-{idx}', ha='center', va='center', fontsize=7)
    ax.set_title('Grid-based TAS result: AGV-task allocation and sequence')
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_one_path(layout: WarehouseLayout, ev: EvalResult, out: Path, preferred_agv=3, preferred_task=16):
    segs = ev.segments.get(preferred_agv, [])
    selected = [s for s in segs if s.task_id == preferred_task]
    if not selected and segs:
        selected = segs[:4]
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_base(ax, layout)
    for agv in AGVS:
        ax.scatter([agv.start[0]], [agv.start[1]], marker='s', s=80, edgecolors='black')
        ax.text(agv.start[0], agv.start[1], f'No.{agv.agv_id}', ha='center', va='center', fontsize=6)
    markers = ['o', 's', '^', 'D']
    for i, seg in enumerate(selected):
        xs = [p[0] for p in seg.path]
        ys = [p[1] for p in seg.path]
        ax.plot(xs, ys, linewidth=2, marker=markers[i % len(markers)], markersize=3, label=seg.label)
    if selected:
        p0 = selected[0].path[0]
        ptask = selected[0].path[-1]
        pin = layout.params.workspace_entrance
        pout = layout.params.workspace_exit
        for label, p in [('S1 AGV location', p0), ('S2 task location', ptask), ('S3 entrance', pin), ('S4 exit', pout)]:
            ax.annotate(label, xy=p, xytext=(p[0]+0.4, p[1]+0.4), fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.7))
    ax.legend(loc='upper left', fontsize=7)
    ax.set_title('Path planning result for one AGV task')
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_results(ev: EvalResult, chromosome, history, out: Path):
    data = {
        'picker_completion_time_s': ev.makespan_picker,
        'agv_completion_time_s': ev.makespan_agv,
        'total_agv_cost_cny': round(ev.total_cost, 6),
        'waiting_time_s': ev.waiting_time,
        'best_task_permutation': chromosome,
        'agv_task_sequences': {str(k): v for k, v in ev.agv_task_sequences.items()},
        'history_last': history[-10:],
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
