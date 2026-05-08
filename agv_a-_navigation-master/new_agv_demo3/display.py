"""
display.py

本文件是 AGV 物流配送仿真的可视化程序，主要负责：
1. 读取路径规划算法输出的 CSV 文件；
2. 根据地图、工位、AGV 初始位置和轨迹数据绘制仿真界面；
3. 动态展示 AGV 的移动、取料、配送、卸货和返回等待区过程；
4. 在右侧信息栏显示当前时刻、已完成任务数量、累计延迟、总行驶成本和每台 AGV 状态；
5. 支持键盘交互：空格暂停，左右方向键单步查看。

该文件本身不负责路径规划。
路径规划由 navigation.py 完成，输出 agv_trajectory.csv、agv_task_result.csv 等文件。
本文件只读取这些结果文件并进行动画展示。
"""

import csv
import sys
from collections import defaultdict

import pygame

# 初始化 pygame。
# pygame.init() 会初始化显示、字体、事件等模块。
pygame.init()

# ============================================================
# 1. 基础地图参数
# ============================================================

# 网格地图宽度和高度。
# 当前布局是 20 × 20 的离散网格。
GRID_W = 20
GRID_H = 20

# 动画刷新帧率。
# FPS 越大，动画播放越快；FPS 越小，动画播放越慢。
FPS = 5


# ============================================================
# 2. 颜色配置
# ============================================================
# 以下颜色均使用 RGB 格式。
# 为了便于统一调整界面风格，所有颜色都集中定义在这里。

# 界面背景色。
BG = (246, 246, 244)

# 地图区域和右侧信息面板背景。
PANEL = (252, 252, 250)

# 面板边框颜色。
BORDER = (176, 176, 170)

# 道路网格颜色。
ROAD_FILL = (255, 255, 255)
ROAD_LINE = (214, 214, 210)

# 工位显示颜色。
STATION_FILL = (240, 232, 225)
STATION_BORDER = (118, 102, 91)
STATION_SHADOW = (208, 202, 196)

# 工位当前任务信息框颜色。
# TASK_BG 表示正常状态，TASK_BG_LATE 表示当前任务已经超过开始时间还未送达。
TASK_BG = (233, 245, 236)
TASK_BG_LATE = (247, 224, 224)
TASK_BORDER = (140, 140, 140)

# 物料区颜色。
SUPPLY_FILL = (243, 171, 91)
SUPPLY_BORDER = (165, 107, 57)

# 取料位颜色。
PICK_FILL = (130, 198, 114)
PICK_BORDER = (74, 133, 61)

# 主文字和次级文字颜色。
TEXT = (35, 35, 35)
SUBTEXT = (90, 90, 90)

# AGV 颜色表。
# 如果 AGV 数量超过颜色数量，会通过取模循环使用颜色。
AGV_COLOR_PALETTE = [
    (235, 184, 41),
    (92, 162, 245),
    (94, 194, 117),
    (176, 118, 223),
    (239, 118, 118),
    (80, 200, 200),
    (255, 150, 80),
    (150, 150, 255),
    (120, 210, 150),
    (210, 120, 210),
    (180, 180, 80),
    (80, 180, 180),
]


class DisplayManager:
    """
    AGV 仿真可视化管理类。

    该类负责完整的显示流程：
    1. 初始化 pygame 窗口、地图尺寸和字体；
    2. 加载 CSV 数据；
    3. 根据当前仿真时刻绘制地图、工位、AGV 和右侧状态栏；
    4. 运行 pygame 主循环，实现动画播放和键盘控制。
    """

    def __init__(self, speed=1.0):
        """
        初始化显示窗口、字体和数据。

        参数：
            speed: 动画速度倍率。
                   speed=1.0 表示按 FPS 正常播放；
                   speed=2.0 表示约 2 倍速度播放；
                   speed=0.5 表示约半速播放。
        """
        # 获取当前电脑屏幕信息，用于计算一个不会超过屏幕的窗口大小。
        info = pygame.display.Info()

        # 地图边距、标题区高度、右侧状态栏宽度。
        self.margin = 22
        self.top = 72
        self.sidebar = 320

        # 限制最大窗口尺寸，避免可视化窗口占满整个屏幕。
        max_w = min(1480, info.current_w - 100)
        max_h = min(930, info.current_h - 100)

        # 根据可用窗口大小自动计算单个网格的像素大小。
        # self.cell 是整个界面最关键的比例参数。
        self.cell = max(
            42,
            min(
                56,
                (max_w - self.sidebar - self.margin * 3) // GRID_W,
                (max_h - self.top - self.margin * 2) // GRID_H,
            ),
        )

        # 地图区域像素大小。
        self.map_w = GRID_W * self.cell
        self.map_h = GRID_H * self.cell

        # 创建 pygame 窗口。
        # 窗口宽度 = 地图宽度 + 右侧信息栏 + 边距；
        # 窗口高度 = 地图高度 + 顶部标题区 + 边距。
        self.screen = pygame.display.set_mode(
            (
                self.map_w + self.sidebar + self.margin * 3,
                self.map_h + self.top + self.margin * 2,
            )
        )
        pygame.display.set_caption("AGV Layout Simulation")

        # pygame 时钟对象，用于控制刷新频率。
        self.clock = pygame.time.Clock()
        self.speed = speed

        # ====================================================
        # 字体设置
        # ====================================================
        # 字体大小根据 self.cell 自动缩放，使得不同窗口大小下文字仍较协调。
        self.font_xs = pygame.font.SysFont("arial", max(12, int(self.cell * 0.20)))
        self.font_sm = pygame.font.SysFont("arial", max(14, int(self.cell * 0.24)), bold=True)
        self.font_md = pygame.font.SysFont("arial", max(16, int(self.cell * 0.30)), bold=True)
        self.font_lg = pygame.font.SysFont("arial", max(22, int(self.cell * 0.44)), bold=True)

        # 工位编号字体，例如 140A、209F。
        self.font_station = pygame.font.SysFont(
            "arial",
            max(18, int(self.cell * 0.34)),
            bold=True,
        )

        # 工位任务开始时间字体。
        # 当前设置较小，避免时间数字挤压工位编号和物料编号。
        self.font_time = pygame.font.SysFont(
            "consolas",
            max(5, int(self.cell * 0.22)),
            bold=True,
        )

        # 工位物料编号字体，例如 M1、M2。
        self.font_material = pygame.font.SysFont(
            "arial",
            max(9, int(self.cell * 0.20)),
            bold=True,
        )

        # AGV 载货标签字体，例如 M1->140A。
        self.font_badge = pygame.font.SysFont("arial", max(16, int(self.cell * 0.26)), bold=True)

        # ====================================================
        # 加载数据
        # ====================================================
        # 加载顺序不能随意调整：
        # 1. 先加载位置，确定工位、AGV、物料区、取料位；
        # 2. 再加载任务；
        # 3. 再加载轨迹；
        # 4. 再加载任务结果统计；
        # 5. 最后根据轨迹计算每个任务的送达时刻。
        self.load_positions()
        self.load_tasks()
        self.load_traj()
        self.load_task_results()
        self.compute_task_delivery()

    # ============================================================
    # 3. 数据读取与统计函数
    # ============================================================

    def load_task_results(self):
        """
        读取任务执行结果文件 agv_task_result.csv。

        该文件通常由 navigation.py 输出，包含每个任务的：
        - 执行 AGV；
        - 实际送达时间；
        - 延迟时间；
        - 行驶距离。

        本函数还会计算总延迟 total_delay 和总行驶距离 total_distance，
        用于右侧信息栏展示。

        如果文件不存在，说明当前可能还没有运行 navigation.py，
        此时不报错，只把统计值置为 0。
        """
        self.task_results = []
        self.total_delay = 0
        self.total_distance = 0

        try:
            with open("agv_task_result.csv", "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    # CSV 读入默认是字符串，这里把需要计算的字段转成整数。
                    row["window_start"] = int(row["window_start"])
                    row["window_end"] = int(row["window_end"])
                    row["delivery_time"] = int(row["delivery_time"])
                    row["delay"] = int(row["delay"])
                    row["distance"] = int(row["distance"])
                    self.task_results.append(row)

            # 全局统计指标。
            self.total_delay = sum(r["delay"] for r in self.task_results)
            self.total_distance = sum(r["distance"] for r in self.task_results)

        except FileNotFoundError:
            # 允许没有统计文件，方便只调试界面或轨迹文件。
            self.task_results = []
            self.total_delay = 0
            self.total_distance = 0

    def metrics_until(self, t):
        """
        计算当前时刻 t 之前已经完成任务的累计指标。

        参数：
            t: 当前仿真时间。

        返回：
            delay: 当前时刻前已完成任务的累计延迟；
            distance: 当前时刻前已完成任务的累计行驶成本；
            len(done): 当前时刻前已完成任务数量。
        """
        done = [r for r in self.task_results if r["delivery_time"] <= t]

        delay = sum(r["delay"] for r in done)
        distance = sum(r["distance"] for r in done)

        return delay, distance, len(done)

    def get_agv_color(self, agv_id):
        """
        根据 AGV 编号返回显示颜色。

        如果 AGV 数量多于颜色表长度，则循环使用颜色。
        """
        idx = int(agv_id) - 1
        return AGV_COLOR_PALETTE[idx % len(AGV_COLOR_PALETTE)]

    def load_positions(self):
        """
        读取 agv_position.csv。

        该文件描述地图中的固定位置和 AGV 初始位置，包括：
        - supply: 物料区；
        - pickup_slot: 取料位；
        - drop_point: 工位 / 卸货点；
        - agv: AGV 初始等待区。
        """
        self.supply = None
        self.pickup = None
        self.drop_points = {}
        self.homes = {}
        self.agv_ids = []

        with open("agv_position.csv", "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                item = {
                    "name": row["name"].strip(),
                    "label": row.get("label", "").strip(),
                    "pos": (int(row["x"]), int(row["y"])),
                    "pitch": int(row["pitch"]) if row.get("pitch") not in ("", None) else 0,
                }

                if row["type"] == "supply":
                    self.supply = item
                elif row["type"] == "pickup_slot":
                    self.pickup = item
                elif row["type"] == "drop_point":
                    self.drop_points[item["name"]] = item
                elif row["type"] == "agv":
                    self.homes[item["name"]] = item
                    self.agv_ids.append(item["name"])

        # AGV 编号按数字排序，保证显示顺序稳定。
        self.agv_ids.sort(key=lambda x: int(x))

    def load_tasks(self):
        """
        读取 agv_task.csv。

        本函数把任务按目标工位 destination_id 进行分组，
        存入 self.tasks_by_dest，便于可视化时查询每个工位当前应该显示哪个任务。
        """
        self.tasks_by_dest = defaultdict(list)

        with open("agv_task.csv", "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                row["release_time"] = int(row["release_time"])
                row["window_start"] = int(row["window_start"])
                row["window_end"] = int(row["window_end"])
                self.tasks_by_dest[row["destination_id"]].append(row)

        # 每个工位内部任务按 release_time、window_start、task_id 排序，
        # 保证同一工位任务按时间顺序显示。
        for key in self.tasks_by_dest:
            self.tasks_by_dest[key].sort(key=lambda r: (r["release_time"], r["window_start"], r["task_id"]))

    def load_traj(self):
        """
        读取 agv_trajectory.csv。

        该文件是路径规划结果，记录每台 AGV 在每个 timestamp 的位置和状态。
        本函数将其组织成：
            self.timeline[t][agv_id] = row
        方便在绘制第 t 帧时快速找到所有 AGV 的状态。

        另外，若某些时刻某台 AGV 没有显式记录，本函数会用上一帧状态补齐，
        防止可视化中 AGV 突然消失。
        """
        self.timeline = defaultdict(dict)
        self.max_t = 0

        with open("agv_trajectory.csv", "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            t = int(row["timestamp"])
            self.max_t = max(self.max_t, t)
            row["X"] = int(row["X"])
            row["Y"] = int(row["Y"])
            row["pitch"] = int(row["pitch"])
            self.timeline[t][row["name"]] = row

        # 补齐缺失帧，避免某台 AGV 因某一秒没有记录而从画面消失。
        last = {}
        for t in range(self.max_t + 1):
            for agv_id in self.agv_ids:
                if agv_id in self.timeline[t]:
                    last[agv_id] = self.timeline[t][agv_id]
                elif agv_id in last:
                    self.timeline[t][agv_id] = last[agv_id]

    def compute_task_delivery(self):
        """
        根据轨迹文件推断每个任务的实际送达时刻。

        当某条轨迹记录满足：
            status == "unloading"
        说明 AGV 已经到达目标工位服务点并开始卸货。
        本程序把第一次出现 unloading 的时刻作为任务送达时刻。
        """
        self.delivered_at = {}

        for t, items in self.timeline.items():
            for row in items.values():
                if row["task_id"] and row["status"] == "unloading":
                    self.delivered_at.setdefault(row["task_id"], t)

    # ============================================================
    # 4. 坐标转换与任务显示逻辑
    # ============================================================

    def grid_rect(self, pos, inset=0):
        """
        将网格坐标转换为 pygame 屏幕矩形。

        参数：
            pos: 网格坐标 (x, y)，坐标从 1 开始；
            inset: 内缩像素，用于绘制更小的矩形。

        注意：
            网格坐标 y 越大表示越靠上，
            但 pygame 屏幕坐标 y 越大表示越靠下。
            因此这里使用 GRID_H - y 做一次上下翻转。
        """
        x, y = pos
        sx = self.margin + (x - 1) * self.cell
        sy = self.top + self.margin + (GRID_H - y) * self.cell
        return pygame.Rect(sx + inset, sy + inset, self.cell - inset * 2, self.cell - inset * 2)

    def center(self, pos):
        """
        返回某个网格坐标在屏幕上的中心点。
        """
        return self.grid_rect(pos).center

    def active_task(self, dest_id, t):
        """
        返回当前时刻某个工位应该显示的任务。

        重要规则：
        1. 同一工位的任务必须按顺序显示；
        2. 如果当前任务还没有送达，即使已经超过时间窗，也继续显示该任务；
        3. 只有当前任务送达后，才允许显示该工位的下一个任务；
        4. 如果下一个任务还没有 release，则暂时不显示任何任务信息。

        这样可以避免旧任务还没送到，工位却提前显示下一任务的错误现象。
        """
        tasks = self.tasks_by_dest.get(dest_id, [])

        for task in tasks:
            task_id = task["task_id"]
            delivered_t = self.delivered_at.get(task_id)

            # 已经送达的任务不再显示。
            if delivered_t is not None and delivered_t <= t:
                continue

            # 第一个未送达任务，如果已经释放，则显示。
            if task["release_time"] <= t:
                return task

            # 第一个未送达任务还没释放，则后续任务也不能显示。
            return None

        return None

    def task_state(self, task, t):
        """
        返回任务当前显示状态和背景颜色。

        返回值：
            (状态字符串, 背景颜色)

        WAIT：任务未迟到；
        LATE：任务已经超过开始时间但还没有送达；
        OK：任务准时送达；
        LATE_DONE：任务迟到送达。
        """
        deliver_t = self.delivered_at.get(task["task_id"])

        if deliver_t is None:
            if t <= task["window_start"]:
                return ("WAIT", TASK_BG)
            return ("LATE", TASK_BG_LATE)

        if deliver_t <= task["window_start"]:
            return ("OK", TASK_BG)

        return ("LATE_DONE", TASK_BG_LATE)

    def visual_pitch(self, agv_id, t, row):
        """
        根据相邻帧实际位置计算 AGV 显示方向。

        为什么不完全依赖 CSV 中的 pitch？
        因为 AGV 在等待或补帧时，pitch 可能与实际下一步移动方向不一致。
        为了让箭头更直观，本函数优先使用实际坐标变化判断方向。
        """
        cur = (row["X"], row["Y"])

        # 优先看当前帧相对上一帧的移动方向。
        prev = self.timeline.get(t - 1, {}).get(agv_id)
        if prev:
            p = (prev["X"], prev["Y"])
            if p != cur:
                dx = cur[0] - p[0]
                dy = cur[1] - p[1]
                if dx > 0:
                    return 0
                if dx < 0:
                    return 180
                if dy > 0:
                    return 90
                if dy < 0:
                    return 270

        # 如果当前帧没有移动，再参考下一帧方向。
        nxt = self.timeline.get(t + 1, {}).get(agv_id)
        if nxt:
            n = (nxt["X"], nxt["Y"])
            if n != cur:
                dx = n[0] - cur[0]
                dy = n[1] - cur[1]
                if dx > 0:
                    return 0
                if dx < 0:
                    return 180
                if dy > 0:
                    return 90
                if dy < 0:
                    return 270

        # 如果前后帧都不移动，则使用 CSV 中记录的 pitch。
        return row.get("pitch", 0)

    # ============================================================
    # 5. 绘制函数
    # ============================================================

    def draw_base(self):
        """
        绘制基础界面，包括背景、地图面板、右侧信息栏和标题。
        """
        self.screen.fill(BG)

        # 地图区域面板。
        map_rect = pygame.Rect(self.margin - 6, self.top + self.margin - 6, self.map_w + 12, self.map_h + 12)
        pygame.draw.rect(self.screen, PANEL, map_rect, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, map_rect, 2, border_radius=12)

        # 右侧状态栏面板。
        side = pygame.Rect(self.margin * 2 + self.map_w, self.top, self.sidebar, self.map_h + self.margin)
        pygame.draw.rect(self.screen, PANEL, side, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, side, 2, border_radius=12)

        title = self.font_lg.render("AGV Logistics Simulation", True, TEXT)
        self.screen.blit(title, (self.margin + 8, 18))

        legend = self.font_sm.render("White grids = drivable area", True, SUBTEXT)
        self.screen.blit(legend, (self.margin + 8, 48))

    def draw_roads(self):
        """
        绘制全部道路网格。

        当前可行驶区域以白色网格表示。
        工位、物料区、取料位之后会覆盖在网格上方。
        """
        for x in range(1, GRID_W + 1):
            for y in range(1, GRID_H + 1):
                rect = self.grid_rect((x, y), 1)
                pygame.draw.rect(self.screen, ROAD_FILL, rect, border_radius=3)
                pygame.draw.rect(self.screen, ROAD_LINE, rect, 1, border_radius=3)

    def draw_fixed(self):
        """
        绘制固定设施：物料区 supply 和取料位 pickup_slot。
        """
        if self.supply:
            r = self.grid_rect(self.supply["pos"], 5)
            pygame.draw.rect(self.screen, SUPPLY_FILL, r, border_radius=8)
            pygame.draw.rect(self.screen, SUPPLY_BORDER, r, 2, border_radius=8)
            txt = self.font_sm.render("SUPPLY", True, TEXT)
            self.screen.blit(txt, txt.get_rect(center=r.center))

        if self.pickup:
            r = self.grid_rect(self.pickup["pos"], 8)
            pygame.draw.rect(self.screen, PICK_FILL, r, border_radius=8)
            pygame.draw.rect(self.screen, PICK_BORDER, r, 2, border_radius=8)
            txt = self.font_sm.render(self.pickup["label"], True, TEXT)
            self.screen.blit(txt, txt.get_rect(center=r.center))

    def draw_homes(self):
        """
        绘制 AGV 等待区。

        H1、H2 等表示 AGV 的 home / 初始等待位置。
        """
        for agv_id, item in self.homes.items():
            r = self.grid_rect(item["pos"], 10)
            pygame.draw.rect(self.screen, (235, 238, 244), r, border_radius=6)
            pygame.draw.rect(self.screen, (150, 160, 175), r, 1, border_radius=6)
            txt = self.font_xs.render(f"H{agv_id}", True, SUBTEXT)
            self.screen.blit(txt, txt.get_rect(center=r.center))

    def draw_stations(self, t):
        """
        绘制工位和工位当前任务信息。

        工位显示规则：
        - 工位块尽量填满对应网格；
        - 上半部分显示工位编号，例如 140A；
        - 下半部分显示当前任务信息；
        - 物料编号显示在上方；
        - 任务开始时间显示在下方；
        - 任务送达后，当前任务信息自动消失；
        - 如果任务已经迟到但未送达，信息框会变为 TASK_BG_LATE。
        """
        for dest_id, point in self.drop_points.items():
            cell = self.grid_rect(point["pos"])

            # 工位主体尽量填满整个网格。
            station = pygame.Rect(
                cell.left + int(cell.width * 0.02),
                cell.top + int(cell.height * 0.02),
                int(cell.width * 0.96),
                int(cell.height * 0.96),
            )

            pygame.draw.rect(self.screen, STATION_SHADOW, station.move(2, 3), border_radius=8)
            pygame.draw.rect(self.screen, STATION_FILL, station, border_radius=8)
            pygame.draw.rect(self.screen, STATION_BORDER, station, 2, border_radius=8)

            # 上半部分显示工位编号。
            label_area = pygame.Rect(
                station.left + 2,
                station.top + 2,
                station.width - 4,
                int(station.height * 0.42),
            )
            label = self.font_station.render(point["label"], True, TEXT)
            self.screen.blit(label, label.get_rect(center=label_area.center))

            # 查询当前时刻该工位应该显示的任务。
            task = self.active_task(dest_id, t)
            if task:
                status, bg = self.task_state(task, t)

                # 下半部分任务信息区域。
                info = pygame.Rect(
                    station.left + 4,
                    station.top + int(station.height * 0.43),
                    station.width - 8,
                    int(station.height * 0.50),
                )

                pygame.draw.rect(self.screen, bg, info, border_radius=7)
                pygame.draw.rect(self.screen, TASK_BORDER, info, 1, border_radius=7)

                # 物料编号。
                material = self.font_material.render(task["material"], True, TEXT)
                self.screen.blit(
                    material,
                    material.get_rect(center=(info.centerx, info.top + info.height * 0.32)),
                )

                # 任务开始时间，即 window_start。
                start_number = self.font_time.render(str(task["window_start"]), True, TEXT)
                self.screen.blit(
                    start_number,
                    start_number.get_rect(center=(info.centerx, info.top + info.height * 0.72)),
                )

    def draw_agv(self, agv_id, row, t):
        """
        绘制单台 AGV。

        显示内容包括：
        - AGV 彩色车体；
        - AGV 编号；
        - 运动方向箭头；
        - 如果 AGV 正在载货，则在车体上方显示物料 -> 目标工位标签。
        """
        pos = (row["X"], row["Y"])
        pitch = self.visual_pitch(agv_id, t, row)
        color = self.get_agv_color(agv_id)
        cx, cy = self.center(pos)

        # AGV 主体矩形。
        body = pygame.Rect(0, 0, max(28, int(self.cell * 0.58)), max(22, int(self.cell * 0.42)))
        body.center = (cx, cy)

        # 阴影和车体。
        pygame.draw.rect(self.screen, (170, 170, 170), body.move(2, 2), border_radius=10)
        pygame.draw.rect(self.screen, color, body, border_radius=10)
        pygame.draw.rect(self.screen, (60, 60, 60), body, 2, border_radius=10)

        # 车顶小窗口。
        top = body.inflate(-body.width * 0.34, -body.height * 0.42)
        pygame.draw.rect(self.screen, (255, 247, 219), top, border_radius=6)
        pygame.draw.rect(self.screen, (100, 100, 100), top, 1, border_radius=6)

        # 四个轮子或角点装饰。
        for p in [
            (body.left + 4, body.top + 4),
            (body.right - 4, body.top + 4),
            (body.left + 4, body.bottom - 4),
            (body.right - 4, body.bottom - 4),
        ]:
            pygame.draw.circle(self.screen, (70, 70, 70), p, max(2, int(self.cell * 0.05)))

        # 根据 pitch 绘制方向箭头。
        if pitch == 0:
            pts = [(body.right + 9, body.centery), (body.right - 1, body.centery - 7), (body.right - 1, body.centery + 7)]
        elif pitch == 180:
            pts = [(body.left - 9, body.centery), (body.left + 1, body.centery - 7), (body.left + 1, body.centery + 7)]
        elif pitch == 90:
            pts = [(body.centerx, body.top - 9), (body.centerx - 7, body.top + 1), (body.centerx + 7, body.top + 1)]
        else:
            pts = [(body.centerx, body.bottom + 9), (body.centerx - 7, body.bottom - 1), (body.centerx + 7, body.bottom - 1)]
        pygame.draw.polygon(self.screen, (35, 35, 35), pts)

        # AGV 编号。
        num = self.font_sm.render(agv_id, True, TEXT)
        self.screen.blit(num, num.get_rect(center=body.center))

        # 载货标签，例如 M1->140A。
        if row["loaded"] == "true" and row.get("material"):
            badge_text = f"{row['material']}->{row['destination_label']}"
            badge = self.font_badge.render(badge_text, True, (255, 255, 255))
            br = pygame.Rect(0, 0, badge.get_width() + 16, badge.get_height() + 8)
            br.centerx = cx
            br.bottom = body.top - 4
            pygame.draw.rect(self.screen, (182, 60, 52), br, border_radius=7)
            pygame.draw.rect(self.screen, (120, 40, 35), br, 1, border_radius=7)
            self.screen.blit(badge, badge.get_rect(center=br.center))

    def draw_sidebar(self, t):
        """
        绘制右侧信息栏。

        信息栏包括：
        - 当前仿真时间；
        - 当前已完成任务数；
        - 当前累计延迟；
        - 当前累计行驶成本；
        - 全部任务总延迟和总成本；
        - 当前 AGV 状态统计；
        - 每台 AGV 的详细状态。
        """
        x = self.margin * 2 + self.map_w + 18
        y = self.top + 16

        # 当前时间。
        title = self.font_lg.render(f"Time = {t}s", True, TEXT)
        self.screen.blit(title, (x, y))
        y += 42

        # 当前时刻之前的累计统计指标。
        cur_delay, cur_distance, done_count = self.metrics_until(t)

        metric_lines = [
            f"done tasks: {done_count}",
            f"cum delay: {cur_delay}",
            f"travel cost: {cur_distance}",
            f"total delay: {self.total_delay}",
            f"total cost: {self.total_distance}",
        ]

        for line in metric_lines:
            txt = self.font_md.render(line, True, TEXT)
            self.screen.blit(txt, (x, y))
            y += 24

        y += 10

        # 统计不同状态的 AGV 数量。
        items = self.timeline.get(t, {})
        counters = {"delivering": 0, "pickup": 0, "returning": 0, "idle": 0}

        for row in items.values():
            status = row["status"]
            if status == "delivering":
                counters["delivering"] += 1
            elif status in ("to_pickup", "loading"):
                counters["pickup"] += 1
            elif status == "return_home":
                counters["returning"] += 1
            else:
                counters["idle"] += 1

        for key, value in counters.items():
            txt = self.font_md.render(f"{key}: {value}", True, TEXT)
            self.screen.blit(txt, (x, y))
            y += 24

        y += 12
        sub = self.font_md.render("AGV status", True, TEXT)
        self.screen.blit(sub, (x, y))
        y += 30

        # 每台 AGV 的详细状态卡片。
        for agv_id in self.agv_ids:
            row = items.get(agv_id)
            if not row:
                continue

            rr = pygame.Rect(x, y, self.sidebar - 36, 46)
            pygame.draw.rect(self.screen, (248, 248, 246), rr, border_radius=8)
            pygame.draw.rect(self.screen, (210, 210, 205), rr, 1, border_radius=8)
            pygame.draw.rect(self.screen, self.get_agv_color(agv_id), pygame.Rect(x + 8, y + 13, 16, 16), border_radius=4)

            l1 = self.font_sm.render(f"AGV{agv_id}  {row['status']}", True, TEXT)
            self.screen.blit(l1, (x + 32, y + 5))

            if row["loaded"] == "true":
                msg = f"{row['material']} -> {row['destination_label']}"
            else:
                msg = f"pos ({row['X']}, {row['Y']})"

            l2 = self.font_xs.render(msg, True, SUBTEXT)
            self.screen.blit(l2, (x + 32, y + 25))
            y += 52

        y += 8

        # 操作说明。
        for line in [
            "Notes:",
            "1. White grids are drivable.",
            "2. Arrows are computed from real movement.",
            "3. Station shows label/material/start time.",
            "4. Space: pause, Left/Right: step.",
        ]:
            txt = self.font_xs.render(line, True, TEXT if line == "Notes:" else SUBTEXT)
            self.screen.blit(txt, (x, y))
            y += 18

    def draw(self, t):
        """
        绘制某一个仿真时刻 t 的完整画面。

        绘制顺序很重要：
        1. 先画背景和道路；
        2. 再画工位、等待区、物料区、取料位；
        3. 再画 AGV；
        4. 最后画右侧信息栏。
        """
        self.draw_base()
        self.draw_roads()
        self.draw_stations(t)
        self.draw_homes()
        self.draw_fixed()

        for agv_id in self.agv_ids:
            if agv_id in self.timeline.get(t, {}):
                self.draw_agv(agv_id, self.timeline[t][agv_id], t)

        self.draw_sidebar(t)
        pygame.display.flip()

    # ============================================================
    # 6. 主循环
    # ============================================================

    def run(self):
        """
        运行 pygame 动画主循环。

        控制方式：
        - 关闭窗口：退出程序；
        - Space：暂停 / 继续；
        - 右方向键：单步前进一秒；
        - 左方向键：单步后退一秒。
        """
        t = 0
        paused = False
        running = True

        while running:
            # 处理 pygame 事件，例如关闭窗口、键盘按键。
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_RIGHT:
                        t = min(self.max_t, t + 1)
                    elif event.key == pygame.K_LEFT:
                        t = max(0, t - 1)

            # 绘制当前时刻。
            self.draw(t)

            # 如果未暂停，则时间自动前进。
            if not paused and t < self.max_t:
                t += 1

            # 控制播放速度。
            self.clock.tick(max(1, int(FPS * self.speed)))

        pygame.quit()


if __name__ == "__main__":
    """
    程序入口。

    可以通过命令行参数控制动画速度，例如：
        python display.py        # 正常速度
        python display.py 2.0    # 2 倍速度
        python display.py 0.5    # 半速播放
    """
    speed = 1.0

    if len(sys.argv) > 1:
        try:
            speed = float(sys.argv[1])
        except ValueError:
            pass

    DisplayManager(speed).run()
