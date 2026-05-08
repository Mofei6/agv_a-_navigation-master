import csv
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

TASK_ROUNDS_PER_STATION = 1
TASK_GAP_MIN = 28
TASK_GAP_MAX = 68
WINDOW_LEN_MIN = 45
WINDOW_LEN_MAX = 105
RELEASE_LEAD_MIN = 20
RELEASE_LEAD_MAX = 60

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

    for dp in drop_points:
        current_start = random.randint(20, 200)

        for _ in range(TASK_ROUNDS_PER_STATION):
            window_len = random.randint(WINDOW_LEN_MIN, WINDOW_LEN_MAX)
            lead = random.randint(RELEASE_LEAD_MIN, RELEASE_LEAD_MAX)
            release_time = max(0, current_start - 20)
            window_end = current_start + window_len
            material = random.choice(MATERIALS)

            tasks.append({
                "task_id": f"T{task_no:04d}",
                "destination_id": dp.name,
                "destination_label": dp.label,
                "material": material,
                "release_time": release_time,
                "window_start": current_start,
                "window_end": window_end,
                "priority": "Normal",
            })

            task_no += 1
            current_start = window_end + random.randint(TASK_GAP_MIN, TASK_GAP_MAX)

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
    print(f"物料区: {SUPPLY_POS}")
    print(f"取料点: {[p[2:] for p in PICKUP_POINTS]}")
    print(f"AGV 等候区: {(SHARED_HOME_X, SHARED_HOME_Y)}")
    print("输出文件: agv_position.csv, agv_task.csv")


if __name__ == "__main__":
    main()