import csv
import sys
from collections import defaultdict

import pygame

pygame.init()

# ============================================================
# 基础地图参数
# ============================================================

GRID_W = 32
GRID_H = 18
FPS = 5

# 共享虚拟等待区位置，需要和 navigation.py / generate_data.py 保持一致
SHARED_HOME_POS = (31, 10)

# ============================================================
# 窗口尺寸参数
# ============================================================

WINDOW_CELL = 40
WINDOW_MARGIN = 16
WINDOW_TOP = 58
WINDOW_SIDEBAR = 320

# ============================================================
# 颜色配置
# ============================================================
PICK_OVERLAY_BORDER = (20, 110, 45)
PICK_OVERLAY_TEXT_BG = (255, 255, 235)
BG = (246, 246, 244)
PANEL = (252, 252, 250)
BORDER = (176, 176, 170)

ROAD_FILL = (255, 255, 255)
ROAD_LINE = (214, 214, 210)

STATION_FILL = (240, 232, 225)
STATION_BORDER = (118, 102, 91)
STATION_SHADOW = (208, 202, 196)

TASK_BG = (233, 245, 236)
TASK_BG_LATE = (247, 224, 224)
TASK_BORDER = (140, 140, 140)

SUPPLY_FILL = (243, 171, 91)
SUPPLY_BORDER = (165, 107, 57)

PICK_FILL = (130, 198, 114)
PICK_BORDER = (74, 133, 61)

DEPOT_FILL = (225, 231, 242)
DEPOT_BORDER = (120, 135, 165)

TEXT = (35, 35, 35)
SUBTEXT = (90, 90, 90)

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
    (220, 130, 90),
    (130, 190, 220),
    (170, 120, 90),
    (90, 170, 120),
    (200, 120, 150),
    (120, 150, 200),
    (160, 160, 160),
    (100, 120, 180),
]


class DisplayManager:
    """
    AGV 仿真可视化程序。

    支持：
    1. 多取料点显示；
    2. 动态启用 AGV 显示；
    3. 共享虚拟等待区显示；
    4. 工位当前任务、物料、开始时间显示；
    5. AGV 方向箭头、载货信息显示；
    6. 右侧显示累计延迟、行驶距离、总成本等指标。
    """

    def __init__(self, speed=1.0):
        self.margin = WINDOW_MARGIN
        self.top = WINDOW_TOP
        self.sidebar = WINDOW_SIDEBAR
        self.cell = WINDOW_CELL

        self.map_w = GRID_W * self.cell
        self.map_h = GRID_H * self.cell

        window_w = self.map_w + self.sidebar + self.margin * 3
        window_h = self.map_h + self.top + self.margin * 2

        self.screen = pygame.display.set_mode((window_w, window_h))
        pygame.display.set_caption("Dynamic AGV Simulation - Multi Pickup")

        self.clock = pygame.time.Clock()
        self.speed = speed

        # 字体
        self.font_xs = pygame.font.SysFont("arial", max(11, int(self.cell * 0.20)))
        self.font_sm = pygame.font.SysFont("arial", max(13, int(self.cell * 0.24)), bold=True)
        self.font_md = pygame.font.SysFont("arial", max(15, int(self.cell * 0.30)), bold=True)
        self.font_lg = pygame.font.SysFont("arial", max(21, int(self.cell * 0.44)), bold=True)

        self.font_station = pygame.font.SysFont(
            "arial",
            max(15, int(self.cell * 0.32)),
            bold=True,
        )

        self.font_time = pygame.font.SysFont(
            "consolas",
            max(8, int(self.cell * 0.22)),
            bold=True,
        )

        self.font_material = pygame.font.SysFont(
            "arial",
            max(8, int(self.cell * 0.20)),
            bold=True,
        )

        self.font_badge = pygame.font.SysFont(
            "arial",
            max(12, int(self.cell * 0.24)),
            bold=True,
        )

        # 数据加载
        self.load_positions()
        self.load_tasks()
        self.load_traj()
        self.load_task_results()
        self.compute_task_delivery()

    # ============================================================
    # 数据读取
    # ============================================================

    def load_positions(self):
        """
        读取 agv_position.csv。

        支持多个 pickup_slot：
            self.pickups[pickup_name] = pickup_info
        """
        self.supply = None
        self.pickups = {}
        self.drop_points = {}
        self.homes = {}

        with open("agv_position.csv", "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                item = {
                    "name": row["name"].strip(),
                    "label": row.get("label", "").strip(),
                    "pos": (int(row["x"]), int(row["y"])),
                    "pitch": int(row["pitch"]) if row.get("pitch") not in ("", None) else 0,
                }

                row_type = row["type"].strip()

                if row_type == "supply":
                    self.supply = item
                elif row_type == "pickup_slot":
                    self.pickups[item["name"]] = item
                elif row_type == "drop_point":
                    self.drop_points[item["name"]] = item
                elif row_type == "agv":
                    self.homes[item["name"]] = item
        print("读取到的取料点数量:", len(self.pickups))
        for name, pickup in self.pickups.items():
            print(name, pickup["label"], pickup["pos"])

    def load_tasks(self):
        """
        读取 agv_task.csv，并按工位分组。
        """
        self.tasks_by_dest = defaultdict(list)

        with open("agv_task.csv", "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                row["release_time"] = int(row["release_time"])
                row["window_start"] = int(row["window_start"])
                row["window_end"] = int(row["window_end"])
                self.tasks_by_dest[row["destination_id"]].append(row)

        for dest_id in self.tasks_by_dest:
            self.tasks_by_dest[dest_id].sort(
                key=lambda r: (
                    r["release_time"],
                    r["window_start"],
                    r["task_id"],
                )
            )

    def load_traj(self):
        """
        读取 agv_trajectory.csv。

        只显示实际启用并出现在轨迹文件中的 AGV。
        未启用的候选 AGV 不显示。
        """
        self.timeline = defaultdict(dict)
        self.max_t = 0
        self.agv_ids = set()

        with open("agv_trajectory.csv", "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            t = int(row["timestamp"])
            self.max_t = max(self.max_t, t)

            row["X"] = int(row["X"])
            row["Y"] = int(row["Y"])
            row["pitch"] = int(row["pitch"])

            self.timeline[t][row["name"]] = row
            self.agv_ids.add(row["name"])

        self.agv_ids = sorted(self.agv_ids, key=lambda x: int(x))

        # 补齐缺失帧，避免 AGV 闪烁或消失
        last = {}
        for t in range(self.max_t + 1):
            for agv_id in self.agv_ids:
                if agv_id in self.timeline[t]:
                    last[agv_id] = self.timeline[t][agv_id]
                elif agv_id in last:
                    self.timeline[t][agv_id] = last[agv_id]

    def load_task_results(self):
        """
        读取 agv_task_result.csv 和 agv_summary.csv。
        """
        self.task_results = []
        self.total_delay = 0
        self.total_distance = 0
        self.max_delay = 0
        self.avg_delay = 0
        self.used_agv_count = len(self.agv_ids)
        self.candidate_agv_count = len(self.homes)
        self.unused_candidate_agv_count = max(0, self.candidate_agv_count - self.used_agv_count)
        self.total_cost = 0
        self.pickup_count = len(self.pickups)

        try:
            with open("agv_task_result.csv", "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    row["delivery_time"] = int(row["delivery_time"])
                    row["delay"] = int(row["delay"])
                    row["distance"] = int(row["distance"])
                    self.task_results.append(row)

            self.total_delay = sum(r["delay"] for r in self.task_results)
            self.total_distance = sum(r["distance"] for r in self.task_results)
            self.max_delay = max((r["delay"] for r in self.task_results), default=0)
            self.avg_delay = (
                self.total_delay / len(self.task_results)
                if self.task_results
                else 0
            )
        except FileNotFoundError:
            pass

        try:
            with open("agv_summary.csv", "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    metric = row["metric"]
                    value = row["value"]

                    if metric == "candidate_agv_count":
                        self.candidate_agv_count = int(float(value))
                    elif metric == "used_agv_count":
                        self.used_agv_count = int(float(value))
                    elif metric == "unused_candidate_agv_count":
                        self.unused_candidate_agv_count = int(float(value))
                    elif metric == "pickup_count":
                        self.pickup_count = int(float(value))
                    elif metric == "total_delay":
                        self.total_delay = int(float(value))
                    elif metric == "max_delay":
                        self.max_delay = int(float(value))
                    elif metric == "avg_delay":
                        self.avg_delay = float(value)
                    elif metric == "total_distance":
                        self.total_distance = int(float(value))
                    elif metric == "total_cost":
                        self.total_cost = float(value)
        except FileNotFoundError:
            pass

    def compute_task_delivery(self):
        """
        根据轨迹推断每个任务的送达时刻。

        status == unloading 的第一次出现时刻，作为该任务送达时间。
        """
        self.delivered_at = {}

        for t, items in self.timeline.items():
            for row in items.values():
                if row["task_id"] and row["status"] == "unloading":
                    self.delivered_at.setdefault(row["task_id"], t)

    # ============================================================
    # 辅助函数
    # ============================================================

    def get_agv_color(self, agv_id):
        idx = int(agv_id) - 1
        return AGV_COLOR_PALETTE[idx % len(AGV_COLOR_PALETTE)]

    def grid_rect(self, pos, inset=0):
        """
        网格坐标转屏幕矩形。
        注意 y 轴需要反转。
        """
        x, y = pos
        sx = self.margin + (x - 1) * self.cell
        sy = self.top + self.margin + (GRID_H - y) * self.cell
        return pygame.Rect(
            sx + inset,
            sy + inset,
            self.cell - inset * 2,
            self.cell - inset * 2,
        )

    def center(self, pos):
        return self.grid_rect(pos).center

    def visual_agv_center(self, agv_id, row):
        """
        共享等待区中多个 AGV 可重叠。
        为了可视化清晰，idle 状态下在共享等待区附近做小偏移。
        """
        pos = (row["X"], row["Y"])
        cx, cy = self.center(pos)

        if pos == SHARED_HOME_POS and row["status"] == "idle":
            offsets = [
                (-10, -10),
                (10, -10),
                (-10, 10),
                (10, 10),
                (0, -16),
                (0, 16),
                (-16, 0),
                (16, 0),
                (-18, -18),
                (18, -18),
                (-18, 18),
                (18, 18),
                (-24, 0),
                (24, 0),
                (0, -24),
                (0, 24),
                (-24, -12),
                (24, -12),
                (-24, 12),
                (24, 12),
            ]

            dx, dy = offsets[(int(agv_id) - 1) % len(offsets)]
            return cx + dx, cy + dy

        return cx, cy

    def metrics_until(self, t):
        """
        当前时刻之前已完成任务的累计指标。
        """
        done = [r for r in self.task_results if r["delivery_time"] <= t]
        delay = sum(r["delay"] for r in done)
        distance = sum(r["distance"] for r in done)

        return delay, distance, len(done)

    def active_task(self, dest_id, t):
        """
        工位当前应该显示的任务。

        规则：
        - 同一工位的任务按顺序显示；
        - 当前任务没送达前，不显示下一任务；
        - 当前任务 release_time 到达后才显示。
        """
        for task in self.tasks_by_dest.get(dest_id, []):
            delivered_t = self.delivered_at.get(task["task_id"])

            if delivered_t is not None and delivered_t <= t:
                continue

            if task["release_time"] <= t:
                return task

            return None

        return None

    def task_state(self, task, t):
        delivered_t = self.delivered_at.get(task["task_id"])

        if delivered_t is None:
            if t <= task["window_start"]:
                return "WAIT", TASK_BG
            return "LATE", TASK_BG_LATE

        if delivered_t <= task["window_start"]:
            return "OK", TASK_BG

        return "LATE_DONE", TASK_BG_LATE

    def visual_pitch(self, agv_id, t, row):
        """
        优先根据前后帧坐标计算方向。
        """
        cur = (row["X"], row["Y"])

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

        return row.get("pitch", 0)

    # ============================================================
    # 绘制基础界面
    # ============================================================

    def draw_base(self):
        self.screen.fill(BG)

        map_rect = pygame.Rect(
            self.margin - 6,
            self.top + self.margin - 6,
            self.map_w + 12,
            self.map_h + 12,
        )
        pygame.draw.rect(self.screen, PANEL, map_rect, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, map_rect, 2, border_radius=12)

        side = pygame.Rect(
            self.margin * 2 + self.map_w,
            self.top,
            self.sidebar,
            self.map_h + self.margin,
        )
        pygame.draw.rect(self.screen, PANEL, side, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, side, 2, border_radius=12)

        title = self.font_lg.render("Dynamic AGV Simulation - Multi Pickup", True, TEXT)
        self.screen.blit(title, (self.margin + 8, 18))

        hint = self.font_sm.render(
            "White grids = drivable area, P1/P2 = pickup slots",
            True,
            SUBTEXT,
        )
        self.screen.blit(hint, (self.margin + 8, 43))

    def draw_roads(self):
        for x in range(1, GRID_W + 1):
            for y in range(1, GRID_H + 1):
                rect = self.grid_rect((x, y), 1)
                pygame.draw.rect(self.screen, ROAD_FILL, rect, border_radius=3)
                pygame.draw.rect(self.screen, ROAD_LINE, rect, 1, border_radius=3)

    def draw_fixed(self):
        """
        绘制物料区、多个取料点、共享等待区。
        """
        if self.supply:
            r = self.grid_rect(self.supply["pos"], 5)
            pygame.draw.rect(self.screen, SUPPLY_FILL, r, border_radius=8)
            pygame.draw.rect(self.screen, SUPPLY_BORDER, r, 2, border_radius=8)

            txt = self.font_sm.render("SUPPLY", True, TEXT)
            self.screen.blit(txt, txt.get_rect(center=r.center))

        # 多个取料点
        for pickup in self.pickups.values():
            r = self.grid_rect(pickup["pos"], 8)
            pygame.draw.rect(self.screen, PICK_FILL, r, border_radius=8)
            pygame.draw.rect(self.screen, PICK_BORDER, r, 2, border_radius=8)

            label = pickup["label"] or pickup["name"]
            txt = self.font_sm.render(label, True, TEXT)
            self.screen.blit(txt, txt.get_rect(center=r.center))

        # 共享等待区
        r = self.grid_rect(SHARED_HOME_POS, 4)
        pygame.draw.rect(self.screen, DEPOT_FILL, r, border_radius=8)
        pygame.draw.rect(self.screen, DEPOT_BORDER, r, 2, border_radius=8)

        txt = self.font_xs.render("DEPOT", True, TEXT)
        self.screen.blit(txt, txt.get_rect(center=r.center))

    def draw_pickup_overlay(self):
        """
        在最上层重新绘制取料点标记。

        原因：
        - draw_fixed() 中虽然已经绘制了 P1/P2；
        - 但 AGV 是后绘制的，可能把取料点挡住；
        - 所以这里在 AGV 绘制完成后，再把取料点边框和标签画一次；
        - 这样可以保证 P1/P2 始终可见。
        """
        for pickup in self.pickups.values():
            cell = self.grid_rect(pickup["pos"], 2)

            # 外层明显边框
            pygame.draw.rect(
                self.screen,
                PICK_OVERLAY_BORDER,
                cell,
                3,
                border_radius=8,
            )

            label = pickup["label"] or pickup["name"]

            txt = self.font_sm.render(label, True, TEXT)

            # 标签背景框，放在取料点上方，避免被 AGV 车体完全遮挡
            tag_w = txt.get_width() + 12
            tag_h = txt.get_height() + 6

            tag = pygame.Rect(0, 0, tag_w, tag_h)
            tag.centerx = cell.centerx
            tag.bottom = cell.top + 2

            # 如果标签超出地图上边界，则放到格子下方
            if tag.top < self.top + self.margin:
                tag.top = cell.bottom - 2

            pygame.draw.rect(
                self.screen,
                PICK_OVERLAY_TEXT_BG,
                tag,
                border_radius=6,
            )
            pygame.draw.rect(
                self.screen,
                PICK_OVERLAY_BORDER,
                tag,
                1,
                border_radius=6,
            )

            self.screen.blit(txt, txt.get_rect(center=tag.center))

    def draw_stations(self, t):
        """
        绘制所有工位及其当前任务信息。
        """
        for dest_id, point in self.drop_points.items():
            cell = self.grid_rect(point["pos"])

            station = pygame.Rect(
                cell.left + int(cell.width * 0.02),
                cell.top + int(cell.height * 0.02),
                int(cell.width * 0.96),
                int(cell.height * 0.96),
            )

            pygame.draw.rect(self.screen, STATION_SHADOW, station.move(2, 3), border_radius=8)
            pygame.draw.rect(self.screen, STATION_FILL, station, border_radius=8)
            pygame.draw.rect(self.screen, STATION_BORDER, station, 2, border_radius=8)

            label = self.font_station.render(point["label"], True, TEXT)
            self.screen.blit(
                label,
                label.get_rect(center=(station.centerx, station.top + station.height * 0.25)),
            )

            task = self.active_task(dest_id, t)
            if task:
                _, bg = self.task_state(task, t)

                info = pygame.Rect(
                    station.left + 4,
                    station.top + int(station.height * 0.43),
                    station.width - 8,
                    int(station.height * 0.50),
                )

                pygame.draw.rect(self.screen, bg, info, border_radius=7)
                pygame.draw.rect(self.screen, TASK_BORDER, info, 1, border_radius=7)

                material = self.font_material.render(task["material"], True, TEXT)
                self.screen.blit(
                    material,
                    material.get_rect(center=(info.centerx, info.top + info.height * 0.32)),
                )

                start_number = self.font_time.render(str(task["window_start"]), True, TEXT)
                self.screen.blit(
                    start_number,
                    start_number.get_rect(center=(info.centerx, info.top + info.height * 0.72)),
                )

    # ============================================================
    # 绘制 AGV
    # ============================================================

    def draw_agv(self, agv_id, row, t):
        pos = (row["X"], row["Y"])
        pitch = self.visual_pitch(agv_id, t, row)
        color = self.get_agv_color(agv_id)
        cx, cy = self.visual_agv_center(agv_id, row)

        body = pygame.Rect(
            0,
            0,
            max(24, int(self.cell * 0.55)),
            max(20, int(self.cell * 0.40)),
        )
        body.center = (cx, cy)

        pygame.draw.rect(self.screen, (170, 170, 170), body.move(2, 2), border_radius=8)
        pygame.draw.rect(self.screen, color, body, border_radius=8)
        pygame.draw.rect(self.screen, (60, 60, 60), body, 2, border_radius=8)

        # 车顶窗口
        top = body.inflate(-body.width * 0.35, -body.height * 0.45)
        pygame.draw.rect(self.screen, (255, 247, 219), top, border_radius=5)
        pygame.draw.rect(self.screen, (100, 100, 100), top, 1, border_radius=5)

        # 方向箭头
        if pitch == 0:
            pts = [
                (body.right + 8, body.centery),
                (body.right - 1, body.centery - 6),
                (body.right - 1, body.centery + 6),
            ]
        elif pitch == 180:
            pts = [
                (body.left - 8, body.centery),
                (body.left + 1, body.centery - 6),
                (body.left + 1, body.centery + 6),
            ]
        elif pitch == 90:
            pts = [
                (body.centerx, body.top - 8),
                (body.centerx - 6, body.top + 1),
                (body.centerx + 6, body.top + 1),
            ]
        else:
            pts = [
                (body.centerx, body.bottom + 8),
                (body.centerx - 6, body.bottom - 1),
                (body.centerx + 6, body.bottom - 1),
            ]

        pygame.draw.polygon(self.screen, (35, 35, 35), pts)

        # AGV 编号
        num = self.font_sm.render(agv_id, True, TEXT)
        self.screen.blit(num, num.get_rect(center=body.center))

        # 载货标签
        if row["loaded"] == "true" and row.get("material"):
            badge_text = f"{row['material']}->{row['destination_label']}"
            badge = self.font_badge.render(badge_text, True, (255, 255, 255))

            br = pygame.Rect(
                0,
                0,
                badge.get_width() + 12,
                badge.get_height() + 6,
            )
            br.centerx = cx
            br.bottom = body.top - 3

            pygame.draw.rect(self.screen, (182, 60, 52), br, border_radius=6)
            pygame.draw.rect(self.screen, (120, 40, 35), br, 1, border_radius=6)
            self.screen.blit(badge, badge.get_rect(center=br.center))

    # ============================================================
    # 右侧信息栏
    # ============================================================

    def draw_sidebar(self, t):
        x = self.margin * 2 + self.map_w + 18
        y = self.top + 14

        title = self.font_lg.render(f"Time = {t}s", True, TEXT)
        self.screen.blit(title, (x, y))
        y += 38

        cur_delay, cur_distance, done_count = self.metrics_until(t)

        lines = [
            f"candidate AGV: {self.candidate_agv_count}",
            f"used AGV: {self.used_agv_count}",
            f"pickup slots: {self.pickup_count}",
            f"done tasks: {done_count}",
            f"cum delay: {cur_delay}",
            f"travel cost: {cur_distance}",
            f"total delay: {self.total_delay}",
            f"max delay: {self.max_delay}",
            f"total cost: {round(self.total_cost, 1)}",
        ]

        for line in lines:
            txt = self.font_md.render(line, True, TEXT)
            self.screen.blit(txt, (x, y))
            y += 23

        y += 8

        sub = self.font_md.render("AGV status", True, TEXT)
        self.screen.blit(sub, (x, y))
        y += 28

        items = self.timeline.get(t, {})

        for agv_id in self.agv_ids:
            row = items.get(agv_id)
            if not row:
                continue

            rr = pygame.Rect(x, y, self.sidebar - 36, 44)
            pygame.draw.rect(self.screen, (248, 248, 246), rr, border_radius=8)
            pygame.draw.rect(self.screen, (210, 210, 205), rr, 1, border_radius=8)

            pygame.draw.rect(
                self.screen,
                self.get_agv_color(agv_id),
                pygame.Rect(x + 8, y + 13, 15, 15),
                border_radius=4,
            )

            l1 = self.font_sm.render(f"AGV{agv_id} {row['status']}", True, TEXT)
            self.screen.blit(l1, (x + 30, y + 5))

            if row["loaded"] == "true":
                msg = f"{row['material']} -> {row['destination_label']}"
            else:
                msg = f"pos ({row['X']}, {row['Y']})"

            l2 = self.font_xs.render(msg, True, SUBTEXT)
            self.screen.blit(l2, (x + 30, y + 24))

            y += 48

            # 防止右侧超出窗口
            if y > self.top + self.map_h - 80:
                more = self.font_xs.render("...", True, SUBTEXT)
                self.screen.blit(more, (x + 8, y))
                break

        y = self.top + self.map_h - 72

        notes = [
            "Space: pause / resume",
            "Left / Right: step",
            "P1/P2: pickup slots",
            "DEPOT: shared waiting area",
        ]

        for line in notes:
            txt = self.font_xs.render(line, True, SUBTEXT)
            self.screen.blit(txt, (x, y))
            y += 17

    # ============================================================
    # 主绘制函数
    # ============================================================

    def draw(self, t):
        self.draw_base()
        self.draw_roads()
        self.draw_stations(t)

        # 底层绘制物料区、取料点、等待区
        self.draw_fixed()

        # 绘制 AGV
        for agv_id in self.agv_ids:
            if agv_id in self.timeline.get(t, {}):
                self.draw_agv(agv_id, self.timeline[t][agv_id], t)

        # 关键修改：
        # 在 AGV 之后，再绘制一遍取料点外框和标签，
        # 保证 P1/P2 不会被 AGV 遮挡。
        self.draw_pickup_overlay()

        self.draw_sidebar(t)

        pygame.display.flip()

    # ============================================================
    # 主循环
    # ============================================================

    def run(self):
        t = 0
        paused = False
        running = True

        while running:
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

            self.draw(t)

            if not paused and t < self.max_t:
                t += 1

            self.clock.tick(max(1, int(FPS * self.speed)))

        pygame.quit()


if __name__ == "__main__":
    speed = 1.0

    if len(sys.argv) > 1:
        try:
            speed = float(sys.argv[1])
        except ValueError:
            pass

    DisplayManager(speed).run()