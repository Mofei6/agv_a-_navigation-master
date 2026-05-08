import csv
import random
from dataclasses import dataclass

GRID_W = 32
GRID_H = 18
RANDOM_SEED = 42

# 最大候选 AGV 数量。
# 注意：这不是最终启用数量。
# navigation.py 会根据任务、延迟、距离和 AGV 成本自动决定实际启用几台。
MAX_CANDIDATE_AGVS = 20

# 所有候选 AGV 共享同一个虚拟等待区出口。
# 等待区内部允许 AGV 重叠；真实道路上只从这个出口进入。
# AGV 等候区放在工位布局右侧
SHARED_HOME_X = 31
SHARED_HOME_Y = 10

# 物料区放在工位布局右侧
SUPPLY_POS = (29, 10)

# 取料点放在物料区左侧上下两个服务位
# AGV 实际进入取料点，不进入 SUPPLY_POS 本体
PICKUP_POINTS = [
    ("PICK-1", "P1", 28, 9),
    ("PICK-2", "P2", 28, 11),
]
TASK_ROUNDS_PER_STATION = 1
TASK_GAP_MIN = 28
TASK_GAP_MAX = 48
WINDOW_LEN_MIN = 45
WINDOW_LEN_MAX = 75
RELEASE_LEAD_MIN = 18
RELEASE_LEAD_MAX = 35

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
    x: int
    y: int


def build_drop_points():
    """
    根据附件中的工位编号布局生成卸货点。

    约束：
    1. 保持物料区 SUPPLY_POS = (18, 10) 不变；
    2. 保持地图 GRID_W = 20, GRID_H = 20 不变；
    3. 只调整工位 drop_point 的布局，使其接近附件中的实际工位编号排布；
    4. 同一工位编号允许重复，会自动生成唯一 name。

    坐标说明：
    - 209 区放在顶部右侧：209F -> 209A；
    - 130 / 140 / 150 区放在左侧三排；
    - 200 区放在右侧多排；
    - 所有坐标避开物料区 (18, 10)、取料点 (17, 9)/(17, 11)、等待区 (20, 7)。
    """
    raw = [
        # 顶部 209 区：对应附件上方 209F, 209E, 209D, 209C, 209B, 209A
        ("209F", 10, 17),
        ("209E", 12, 17),
        ("209D", 14, 17),
        ("209C", 16, 17),
        ("209B", 18, 17),
        ("209A", 20, 17),

        # 右上 200 区：对应附件上方右侧一排 200A, 200B, 200C
        ("200A", 14, 14),
        ("200B", 17, 14),
        ("200C", 20, 14),

        # 中上主通道上侧：左侧 130/140/150 + 右侧 200
        ("130A", 3, 11),
        ("130B", 5, 11),
        ("140A", 7, 11),
        ("140B", 9, 11),
        ("140C", 11, 11),
        ("150", 13, 11),
        ("200A", 15, 11),
        ("200B", 16, 11),
        ("200C", 19, 11),

        # 中下主通道下侧：左侧 130/140/150 + 右侧 200
        ("130A", 3, 7),
        ("130B", 5, 7),
        ("140A", 7, 7),
        ("140B", 9, 7),
        ("140C", 11, 7),
        ("150", 13, 7),
        ("200A", 15, 7),
        ("200B", 17, 7),
        ("200C", 19, 7),

        # 底部一排：左侧 130/140/150 + 右侧 200
        ("130A", 3, 4),
        ("130B", 5, 4),
        ("140A", 7, 4),
        ("140B", 9, 4),
        ("140C", 11, 4),
        ("150", 13, 4),
        ("200A", 15, 4),
        ("200B", 17, 4),
        ("200C", 19, 4),
    ]

    drop_points = []
    used = {}

    for label, x, y in raw:
        if (x, y) == SUPPLY_POS:
            raise ValueError(f"工位 {label} 坐标 {(x, y)} 与物料区 SUPPLY_POS 冲突")

        used[label] = used.get(label, 0) + 1
        name = f"DP-{label}-{used[label]:02d}"
        drop_points.append(DropPoint(name, label, x, y))

    return drop_points


def build_agvs():
    """
    生成候选 AGV 池。

    所有 AGV 使用同一个共享虚拟等待区位置。
    注意：
    - 这些只是候选 AGV；
    - navigation.py 会自动决定最终启用哪些 AGV；
    - 共享等待区内部允许重叠，不按照普通道路网格处理。
    """
    return [
        AGVHome(str(i + 1), SHARED_HOME_X, SHARED_HOME_Y, 180)
        for i in range(MAX_CANDIDATE_AGVS)
    ]


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
    """
    为每个工位生成多轮任务。

    同一工位的任务按时间顺序出现；
    下一轮任务必须在上一轮任务之后出现。
    """
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
        current_start = random.randint(20, 50)

        for _ in range(TASK_ROUNDS_PER_STATION):
            window_len = random.randint(WINDOW_LEN_MIN, WINDOW_LEN_MAX)
            lead = random.randint(RELEASE_LEAD_MIN, RELEASE_LEAD_MAX)
            release_time = max(0, current_start - lead)
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

    write_positions(drop_points, agvs)
    write_tasks(drop_points)

    print("数据生成完成")
    print(f"候选 AGV 数量: {MAX_CANDIDATE_AGVS}")
    print(f"工位数量: {len(drop_points)}")
    print(f"任务数量: {len(drop_points) * TASK_ROUNDS_PER_STATION}")
    print("输出文件: agv_position.csv, agv_task.csv")


if __name__ == "__main__":
    main()