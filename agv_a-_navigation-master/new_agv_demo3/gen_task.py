import csv
import random
from dataclasses import dataclass

GRID_W = 20
GRID_H = 20
RANDOM_SEED = 42

POSITION_FILE = "agv_position.csv"
TASK_FILE = "agv_task.csv"
AGV_COUNT = 8
TASK_ROUNDS_PER_STATION = 1
WINDOW_LEN_MIN = 30
WINDOW_LEN_MAX = 100
NEXT_TASK_GAP_MIN = 8
NEXT_TASK_GAP_MAX = 18
RELEASE_LEAD = 25


@dataclass(frozen=True)
class DropPoint:
    name: str
    label: str
    x: int
    y: int


@dataclass(frozen=True)
class AGVHome:
    agv_id: str
    x: int
    y: int
    pitch: int = 180


def build_drop_points():
    points = []

    for label, x in [
        ("209F", 7),
        ("209E", 9),
        ("209D", 11),
        ("209C", 13),
        ("209B", 15),
        ("209A", 17),
    ]:
        points.append(DropPoint(f"DP-{label}-01", label, x, 18))

    for label, x in [("200A", 14), ("200B", 15), ("200C", 16)]:
        points.append(DropPoint(f"DP-{label}-01", label, x, 15))

    rows = [(11, "02"), (8, "03"), (5, "04")]
    base_row = [
        ("130A", 2),
        ("130B", 4),
        ("140A", 6),
        ("140B", 8),
        ("140C", 10),
        ("150", 12),
    ]

    for y, suffix in rows:
        for label, x in base_row:
            points.append(DropPoint(f"DP-{label}-{suffix}", label, x, y))
        for label, x in [("200A", 14), ("200B", 15), ("200C", 16)]:
            points.append(DropPoint(f"DP-{label}-{suffix}", label, x, y))

    return points


def build_agvs():
    """
    根据 AGV_COUNT 自动生成 AGV 等待区。

    设计原则：
    1. 优先放在最右侧 x=20；
    2. x=19 留作通行道路，避免右侧 AGV 被左侧 home 卡死；
    3. AGV_COUNT 较大时，再使用 x=19 的备用位置；
    4. 等待区避开物料区和取料位附近。
    """
    candidate_homes = []

    # 第一优先级：最右侧一列，左侧 x=19 保持为空，方便所有 AGV 出入
    for y in [7, 8, 9, 11, 12, 13, 6, 14, 5, 15, 4, 16, 3, 17, 2, 18]:
        candidate_homes.append((20, y))

    # 第二优先级：如果 AGV 数量更多，再使用 x=19，但尽量不要与 x=20 同一行完全堵死
    for y in [6, 9, 12, 15, 4, 17, 2, 18]:
        candidate_homes.append((19, y))

    if AGV_COUNT > len(candidate_homes):
        raise ValueError(
            f"AGV_COUNT={AGV_COUNT} 太大，当前自动等待区最多支持 {len(candidate_homes)} 台 AGV。"
        )

    homes = candidate_homes[:AGV_COUNT]

    return [
        AGVHome(str(i + 1), x, y, 180)
        for i, (x, y) in enumerate(homes)
    ]

def validate(drop_points, agvs):
    occupied = {}
    fixed_points = [("SUPPLY-01", 18, 10), ("PICK-1", 17, 10)]

    for name, x, y in fixed_points:
        if not (1 <= x <= GRID_W and 1 <= y <= GRID_H):
            raise ValueError(f"{name} 坐标越界: {(x, y)}")
        occupied[(x, y)] = name

    for point in drop_points:
        if not (1 <= point.x <= GRID_W and 1 <= point.y <= GRID_H):
            raise ValueError(f"{point.name} 坐标越界: {(point.x, point.y)}")
        if (point.x, point.y) in occupied:
            raise ValueError(f"坐标冲突: {point.name} 与 {occupied[(point.x, point.y)]} 都在 {(point.x, point.y)}")
        occupied[(point.x, point.y)] = point.name

    for agv in agvs:
        if not (1 <= agv.x <= GRID_W and 1 <= agv.y <= GRID_H):
            raise ValueError(f"AGV{agv.agv_id} 坐标越界: {(agv.x, agv.y)}")
        if (agv.x, agv.y) in occupied:
            raise ValueError(f"坐标冲突: AGV{agv.agv_id} 与 {occupied[(agv.x, agv.y)]} 都在 {(agv.x, agv.y)}")
        occupied[(agv.x, agv.y)] = f"AGV-{agv.agv_id}"


def write_positions(drop_points, agvs):
    with open(POSITION_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "label", "x", "y", "pitch"])
        writer.writerow(["supply", "SUPPLY-01", "物料区", 18, 10, ""])
        writer.writerow(["pickup_slot", "PICK-1", "P1", 17, 10, ""])

        for point in drop_points:
            writer.writerow(["drop_point", point.name, point.label, point.x, point.y, ""])

        for agv in agvs:
            writer.writerow(["agv", agv.agv_id, f"AGV{agv.agv_id}", agv.x, agv.y, agv.pitch])


def write_tasks(drop_points):
    random.seed(RANDOM_SEED)
    materials = ["M1", "M2", "M3", "M4", "M5"]

    rows = []
    task_index = 1

    for station_index, point in enumerate(drop_points):
        current_start = 30 + (station_index % 6) * 6 + (station_index // 6) * 3

        for round_index in range(TASK_ROUNDS_PER_STATION):
            window_len = random.randint(WINDOW_LEN_MIN, WINDOW_LEN_MAX)
            window_start = current_start
            window_end = window_start + window_len
            release_time = max(0, window_start - RELEASE_LEAD)
            material = materials[(station_index + round_index) % len(materials)]

            rows.append([
                f"T{task_index:04d}",
                point.name,
                point.label,
                material,
                release_time,
                window_start,
                window_end,
                "Normal",
            ])

            task_index += 1
            current_start = window_end + random.randint(NEXT_TASK_GAP_MIN, NEXT_TASK_GAP_MAX)

    rows.sort(key=lambda r: (int(r[4]), int(r[5]), r[1], r[0]))

    with open(TASK_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id",
            "destination_id",
            "destination_label",
            "material",
            "release_time",
            "window_start",
            "window_end",
            "priority",
        ])
        writer.writerows(rows)


def main():
    drop_points = build_drop_points()
    agvs = build_agvs()
    validate(drop_points, agvs)
    write_positions(drop_points, agvs)
    write_tasks(drop_points)

    print("已生成 agv_position.csv 和 agv_task.csv")
    print(f"工位数量: {len(drop_points)}")
    print(f"任务数量: {len(drop_points) * TASK_ROUNDS_PER_STATION}")
    print(f"单任务时间窗长度: {WINDOW_LEN_MIN}~{WINDOW_LEN_MAX} 秒")


if __name__ == "__main__":
    main()
