import csv
import os
import random
from dataclasses import dataclass

# ============================================================
# 基础参数
# ============================================================

GRID_W = 52
GRID_H = 52
RANDOM_SEED = 42

STATION_W = 3
STATION_H = 3

MAX_CANDIDATE_AGVS = 10

# 共享虚拟等待区：放在工位布局右侧
SHARED_HOME_X = 50
SHARED_HOME_Y = 26

# 物料区：放在工位布局右侧
SUPPLY_POS = (47, 26)

# 取料点：AGV 实际进入取料点取料；物料区本体不可进入
PICKUP_POINTS = [
    ("PICK-1", "P1", 46, 25),
    ("PICK-2", "P2", 46, 27),
]

TASK_ROUNDS_PER_STATION = int(os.environ.get("TASK_ROUNDS_PER_STATION", "1"))

# ============================================================
# 任务时间窗生成参数
# ============================================================
# 原始代码里每个工位的 current_start 都在 20..50 内随机初始化，
# 当 TASK_ROUNDS_PER_STATION=1 时，所有任务的时间窗开始时间会高度集中。
# 这里改成按全局任务序号生成稀疏时间窗。
#
# 可选模式：
# - wave:  分批生成，例如每 5 个任务为一波，波与波之间间隔较大
# - linear: 所有任务按固定节拍线性展开
TIME_WINDOW_MODE = os.environ.get("TIME_WINDOW_MODE", "wave").strip().lower()

TIME_WINDOW_BASE_START = int(os.environ.get("TIME_WINDOW_BASE_START", "120"))

# linear 模式参数：相邻任务窗口开始时间的平均间隔
TIME_WINDOW_INTERVAL = int(os.environ.get("TIME_WINDOW_INTERVAL", "35"))

# wave 模式参数：每波任务数、波间隔、波内任务间隔
TASKS_PER_WAVE = int(os.environ.get("TASKS_PER_WAVE", "5"))
WAVE_INTERVAL = int(os.environ.get("WAVE_INTERVAL", "180"))
TASK_INTERVAL_IN_WAVE = int(os.environ.get("TASK_INTERVAL_IN_WAVE", "30"))

# 随机扰动，避免所有窗口完全等间隔
TIME_WINDOW_JITTER = int(os.environ.get("TIME_WINDOW_JITTER", "10"))

# 时间窗长度和 release lead
WINDOW_LEN_MIN = int(os.environ.get("WINDOW_LEN_MIN", "180"))
WINDOW_LEN_MAX = int(os.environ.get("WINDOW_LEN_MAX", "260"))
RELEASE_LEAD_MIN = int(os.environ.get("RELEASE_LEAD_MIN", "80"))
RELEASE_LEAD_MAX = int(os.environ.get("RELEASE_LEAD_MAX", "150"))

MATERIALS = ["M1", "M2", "M3", "M4", "M5"]


@dataclass
class AGVHome:
    name: str
    x: int
    y: int
    pitch: int = 180


@dataclass
class DropPoint:
    name: str
    label: str
    x: int  # 工位 3x3 区域左下角 x
    y: int  # 工位 3x3 区域左下角 y


def build_drop_points():
    """
    每个工位占用 3x3 网格。
    x, y 表示工位 3x3 区域的左下角。

    对任意工位 (x, y)：
    - 工位本体占用 x..x+2, y..y+2
    - 卸货点为 (x-1, y) 和 (x-1, y+2)
    - AGV 只能到这两个卸货点之一才能卸货
    """
    raw = [
        # 顶部 209 区：209F 209E 209D 209C 209B 209A
        ("209F", 14, 46),
        ("209E", 18, 46),
        ("209D", 22, 46),
        ("209C", 26, 46),
        ("209B", 30, 46),
        ("209A", 34, 46),

        # 上右 200 区
        ("200A", 26, 38),
        ("200B", 31, 38),
        ("200C", 36, 38),

        # 中上排
        ("130A", 4, 30),
        ("130B", 8, 30),
        ("140A", 12, 30),
        ("140B", 16, 30),
        ("140C", 20, 30),
        ("150", 24, 30),
        ("200A", 30, 30),
        ("200B", 35, 30),
        ("200C", 40, 30),

        # 中下排
        ("130A", 4, 22),
        ("130B", 8, 22),
        ("140A", 12, 22),
        ("140B", 16, 22),
        ("140C", 20, 22),
        ("150", 24, 22),
        ("200A", 30, 22),
        ("200B", 35, 22),
        ("200C", 40, 22),

        # 底部排
        ("130A", 4, 14),
        ("130B", 8, 14),
        ("140A", 12, 14),
        ("140B", 16, 14),
        ("140C", 20, 14),
        ("150", 24, 14),
        ("200A", 30, 14),
        ("200B", 35, 14),
        ("200C", 40, 14),
    ]

    drop_points = []
    used = {}

    for label, x, y in raw:
        if x <= 1:
            raise ValueError(f"工位 {label} 的 x={x} 太小，左侧卸货点会越界")

        used[label] = used.get(label, 0) + 1
        name = f"DP-{label}-{used[label]:02d}"
        drop_points.append(DropPoint(name, label, x, y))

    return drop_points


def build_agvs():
    """
    生成候选 AGV 池。

    所有 AGV 使用同一个共享虚拟等待区位置。
    """
    return [
        AGVHome(str(i + 1), SHARED_HOME_X, SHARED_HOME_Y, 180)
        for i in range(MAX_CANDIDATE_AGVS)
    ]


def station_cells(x, y):
    return {
        (x + dx, y + dy)
        for dx in range(STATION_W)
        for dy in range(STATION_H)
    }


def station_unload_points(x, y):
    """
    工位卸货点：
    - 工位左下角下方一格
    - 工位左上角上方一格

    例如工位左下角为 (3, 4)，工位左上角为 (3, 6)，
    则卸货点为 (3, 3) 和 (3, 7)。
    """
    return [
        (x, y - 1),
        (x, y + STATION_H),
    ]


def validate_layout(drop_points):
    occupied = {}
    blocked = {SUPPLY_POS}

    for _, _, x, y in PICKUP_POINTS:
        if not (1 <= x <= GRID_W and 1 <= y <= GRID_H):
            raise ValueError(f"取料点越界: {(x, y)}")

    for dp in drop_points:
        for cell in station_cells(dp.x, dp.y):
            if not (1 <= cell[0] <= GRID_W and 1 <= cell[1] <= GRID_H):
                raise ValueError(f"工位 {dp.name} 占用格越界: {cell}")

            if cell in occupied:
                raise ValueError(f"工位重叠: {dp.name} 与 {occupied[cell]} 在 {cell}")

            occupied[cell] = dp.name
            blocked.add(cell)

        for p in station_unload_points(dp.x, dp.y):
            if not (1 <= p[0] <= GRID_W and 1 <= p[1] <= GRID_H):
                raise ValueError(f"工位 {dp.name} 卸货点越界: {p}")

            if p in blocked:
                raise ValueError(f"工位 {dp.name} 卸货点被占用: {p}")

    for name, label, x, y in PICKUP_POINTS:
        if (x, y) in blocked:
            raise ValueError(f"取料点 {name} 与障碍重叠: {(x, y)}")

    if SUPPLY_POS in occupied:
        raise ValueError("物料区与工位重叠")

    if (SHARED_HOME_X, SHARED_HOME_Y) in blocked:
        raise ValueError("AGV 等候区与障碍重叠")


def write_positions(drop_points, agvs, filename="agv_position.csv"):
    fields = ["type", "name", "label", "x", "y", "pitch"]

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        writer.writerow({
            "type": "supply",
            "name": "SUPPLY-01",
            "label": "SUPPLY",
            "x": SUPPLY_POS[0],
            "y": SUPPLY_POS[1],
            "pitch": "",
        })

        for name, label, x, y in PICKUP_POINTS:
            writer.writerow({
                "type": "pickup_slot",
                "name": name,
                "label": label,
                "x": x,
                "y": y,
                "pitch": "",
            })

        for dp in drop_points:
            writer.writerow({
                "type": "drop_point",
                "name": dp.name,
                "label": dp.label,
                "x": dp.x,
                "y": dp.y,
                "pitch": "",
            })

        for agv in agvs:
            writer.writerow({
                "type": "agv",
                "name": agv.name,
                "label": f"AGV{agv.name}",
                "x": agv.x,
                "y": agv.y,
                "pitch": agv.pitch,
            })


def generate_time_window(task_index):
    """
    根据全局任务序号生成更稀疏的 release_time/window_start/window_end。

    task_index 从 0 开始。默认 wave 模式会形成类似：
    第 1 波：5 个任务，window_start 分布在 120,150,180,210,240 附近；
    第 2 波：5 个任务，window_start 分布在 300,330,360,390,420 附近；
    以此类推。
    """
    if TIME_WINDOW_MODE == "linear":
        base_start = TIME_WINDOW_BASE_START + task_index * TIME_WINDOW_INTERVAL
    elif TIME_WINDOW_MODE == "wave":
        if TASKS_PER_WAVE <= 0:
            raise ValueError("TASKS_PER_WAVE 必须大于 0")
        wave_id = task_index // TASKS_PER_WAVE
        index_in_wave = task_index % TASKS_PER_WAVE
        base_start = (
            TIME_WINDOW_BASE_START
            + wave_id * WAVE_INTERVAL
            + index_in_wave * TASK_INTERVAL_IN_WAVE
        )
    else:
        raise ValueError(f"未知 TIME_WINDOW_MODE: {TIME_WINDOW_MODE}")

    window_start = base_start + random.randint(-TIME_WINDOW_JITTER, TIME_WINDOW_JITTER)
    window_start = max(0, window_start)

    if WINDOW_LEN_MIN > WINDOW_LEN_MAX:
        raise ValueError("WINDOW_LEN_MIN 不能大于 WINDOW_LEN_MAX")
    if RELEASE_LEAD_MIN > RELEASE_LEAD_MAX:
        raise ValueError("RELEASE_LEAD_MIN 不能大于 RELEASE_LEAD_MAX")

    window_len = random.randint(WINDOW_LEN_MIN, WINDOW_LEN_MAX)
    window_end = window_start + window_len

    lead = random.randint(RELEASE_LEAD_MIN, RELEASE_LEAD_MAX)
    release_time = max(0, window_start - lead)

    return release_time, window_start, window_end


def write_tasks(drop_points, filename="agv_task.csv"):
    fields = [
        "task_id",
        "destination_id",
        "destination_label",
        "material",
        "release_time",
        "window_start",
        "window_end",
        "priority",
    ]

    tasks = []
    task_no = 1

    for round_id in range(TASK_ROUNDS_PER_STATION):
        for dp in drop_points:
            task_index = task_no - 1
            release_time, window_start, window_end = generate_time_window(task_index)
            material = random.choice(MATERIALS)

            tasks.append({
                "task_id": f"T{task_no:04d}",
                "destination_id": dp.name,
                "destination_label": dp.label,
                "material": material,
                "release_time": release_time,
                "window_start": window_start,
                "window_end": window_end,
                "priority": "Normal",
            })

            task_no += 1

    tasks.sort(key=lambda r: (r["release_time"], r["window_start"], r["task_id"]))

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tasks)


def main():
    random.seed(RANDOM_SEED)

    drop_points = build_drop_points()
    agvs = build_agvs()

    validate_layout(drop_points)

    write_positions(drop_points, agvs)
    write_tasks(drop_points)

    print("数据生成完成")
    print(f"地图尺寸: {GRID_W} x {GRID_H}")
    print(f"工位尺寸: {STATION_W} x {STATION_H}")
    print(f"候选 AGV 数量: {MAX_CANDIDATE_AGVS}")
    print(f"工位数量: {len(drop_points)}")
    print(f"任务数量: {len(drop_points) * TASK_ROUNDS_PER_STATION}")
    print(f"时间窗模式: {TIME_WINDOW_MODE}")
    if TIME_WINDOW_MODE == "wave":
        print(f"时间窗参数: base={TIME_WINDOW_BASE_START}, tasks_per_wave={TASKS_PER_WAVE}, wave_interval={WAVE_INTERVAL}, in_wave_interval={TASK_INTERVAL_IN_WAVE}, jitter={TIME_WINDOW_JITTER}")
    else:
        print(f"时间窗参数: base={TIME_WINDOW_BASE_START}, interval={TIME_WINDOW_INTERVAL}, jitter={TIME_WINDOW_JITTER}")
    print(f"时间窗长度: {WINDOW_LEN_MIN}..{WINDOW_LEN_MAX}")
    print(f"release lead: {RELEASE_LEAD_MIN}..{RELEASE_LEAD_MAX}")
    print(f"物料区: {SUPPLY_POS}")
    print(f"取料点: {[p[2:] for p in PICKUP_POINTS]}")
    print(f"AGV 等候区: {(SHARED_HOME_X, SHARED_HOME_Y)}")
    print("输出文件: agv_position.csv, agv_task.csv")


if __name__ == "__main__":
    main()