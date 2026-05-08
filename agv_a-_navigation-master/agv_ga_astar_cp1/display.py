import csv
import sys
import argparse
from pathlib import Path
# from collections import defaultdict
from collections import defaultdict
import imageio.v2 as imageio
import cv2
import numpy as np
import pygame

pygame.init()

# ============================================================
# 基础地图参数
# ============================================================
# ============================================================
# 飞机动画参数
# ============================================================

PLANE_ANIM_SECONDS = 30

PLANE_FRAME_PATHS = [
    r"",
    r"",
    r"",
    r"",
    r"",
]

# 动画显示尺寸相对工位大小的缩放
PLANE_SCALE_W = 1.7
PLANE_SCALE_H = 1.7
GRID_W = 52
GRID_H = 52
STATION_W = 3
STATION_H = 3
FPS = 2

SHARED_HOME_POS = (50, 26)
# ============================================================
# 飞机动图素材参数
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PLANE_IMAGE_DIR = BASE_DIR / "image"

PLANE_FRAME_PATHS = [
    PLANE_IMAGE_DIR / "1.png",
    PLANE_IMAGE_DIR / "2.png",
    PLANE_IMAGE_DIR / "3.png",
    PLANE_IMAGE_DIR / "4.png",
    PLANE_IMAGE_DIR / "5.png",
]

# 飞机动画持续时间：30 秒
PLANE_ANIM_SECONDS = 30

# 飞机图最大只占工位宽高的 90%，确保不超出工位范围
PLANE_MAX_W_RATIO = 1.80
PLANE_MAX_H_RATIO = 1.30
# ============================================================
# 窗口尺寸参数
# ============================================================

WINDOW_CELL = 18
WINDOW_MARGIN = 16
WINDOW_TOP = 58
WINDOW_SIDEBAR = 320

# ============================================================
# 颜色配置
# ============================================================

BG = (246, 246, 244)
PANEL = (252, 252, 250)
BORDER = (176, 176, 170)
ROAD_FILL = (255, 255, 255)
ROAD_LINE = (214, 214, 210)

STATION_FILL = (240, 232, 225)
STATION_BORDER = (118, 102, 91)
STATION_SHADOW = (208, 202, 196)

# 卸货点：绿色
UNLOAD_FILL = (80, 210, 120)
UNLOAD_BORDER = (35, 130, 70)

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
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

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
    def __init__(self, speed=1.0):
        self.margin = WINDOW_MARGIN
        self.top = WINDOW_TOP
        self.sidebar = WINDOW_SIDEBAR
        self.cell = WINDOW_CELL
        self.speed = speed

        self.map_w = GRID_W * self.cell
        self.map_h = GRID_H * self.cell

        window_w = self.map_w + self.sidebar + self.margin * 3
        window_h = self.map_h + self.top + self.margin * 2

        self.screen = pygame.display.set_mode((window_w, window_h))
        pygame.display.set_caption("AGV Simulation - 3x3 Stations")

        self.clock = pygame.time.Clock()

        self.font_xs = pygame.font.SysFont("arial", max(9, int(self.cell * 0.48)))
        self.font_sm = pygame.font.SysFont("arial", max(11, int(self.cell * 0.62)), bold=True)
        self.font_md = pygame.font.SysFont("arial", max(14, int(self.cell * 0.78)), bold=True)
        self.font_lg = pygame.font.SysFont("arial", max(20, int(self.cell * 1.05)), bold=True)
        self.font_station = pygame.font.SysFont("arial", max(12, int(self.cell * 0.70)), bold=True)

        self.load_positions()
        self.load_tasks()
        self.load_traj()
        self.load_task_results()
        self.load_summary()
        self.compute_task_delivery()

        self.load_plane_frames()
        self.prepare_station_animation_events()

    # ============================================================
    # 数据读取
    # ============================================================

    def load_plane_frames(self):
        self.plane_frames_raw = []
        self.plane_frame_cache = {}

        for path in PLANE_FRAME_PATHS:
            try:
                img = pygame.image.load(path).convert_alpha()
                self.plane_frames_raw.append(img)
            except Exception as e:
                print(f"[WARN] 无法加载飞机素材: {path} | {e}")

        if not self.plane_frames_raw:
            print("[WARN] 未加载到飞机帧，工位飞机动画将被跳过。")

    def get_scaled_plane_frame(self, frame_idx, width, height):
        key = (frame_idx, width, height)

        if key not in self.plane_frame_cache:
            src = self.plane_frames_raw[frame_idx]
            scaled = pygame.transform.smoothscale(src, (width, height))
            self.plane_frame_cache[key] = scaled

        return self.plane_frame_cache[key]
    def load_positions(self):
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

    def load_tasks(self):
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

        # 补齐缺失帧，避免 AGV 闪烁
        last = {}
        for t in range(self.max_t + 1):
            for agv_id in self.agv_ids:
                if agv_id in self.timeline[t]:
                    last[agv_id] = self.timeline[t][agv_id]
                elif agv_id in last:
                    self.timeline[t][agv_id] = last[agv_id]

    def load_task_results(self):
        self.task_results = {}

        try:
            with open("agv_task_result.csv", "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    row["window_start"] = int(row["window_start"])
                    row["window_end"] = int(row["window_end"])
                    row["delivery_time"] = int(row["delivery_time"])
                    row["delay"] = int(row["delay"])
                    row["distance"] = int(row["distance"])

                    # 如果结果文件里有 arrival_time 就优先用它，否则退化为 delivery_time
                    if "arrival_time" in row and row["arrival_time"] not in ("", None):
                        row["arrival_time"] = int(row["arrival_time"])
                    else:
                        row["arrival_time"] = int(row["delivery_time"])

                    self.task_results[row["task_id"]] = row
        except FileNotFoundError:
            self.task_results = {}

    def prepare_station_animation_events(self):
        """
        为每个工位预先生成动画事件：
        {
            dest_id: [
                {
                    "task_id": ...,
                    "start_t": 到达时间,
                    "end_t": 到达时间 + 10,
                    "window_end": ...,
                    "max_stage": ...,
                },
                ...
            ]
        }
        """
        self.station_anim_events = defaultdict(list)

        if not self.task_results or not self.plane_frames_raw:
            return

        # 取所有 window_end，用于做“越晚越完整”的分级
        all_window_ends = [
            int(r["window_end"])
            for r in self.task_results.values()
        ]

        if not all_window_ends:
            return

        min_we = min(all_window_ends)
        max_we = max(all_window_ends)
        n_frames = len(self.plane_frames_raw)

        def stage_from_window_end(window_end):
            if max_we == min_we:
                return n_frames - 1
            ratio = (window_end - min_we) / (max_we - min_we)
            stage = int(ratio * (n_frames - 1))
            return max(0, min(n_frames - 1, stage))

        for task_id, result in self.task_results.items():
            dest_id = result["destination_id"]
            start_t = int(result.get("arrival_time", result["delivery_time"]))
            end_t = start_t + PLANE_ANIM_SECONDS
            window_end = int(result["window_end"])

            self.station_anim_events[dest_id].append({
                "task_id": task_id,
                "start_t": start_t,
                "end_t": end_t,
                "window_end": window_end,
                "max_stage": stage_from_window_end(window_end),
            })

        for dest_id in self.station_anim_events:
            self.station_anim_events[dest_id].sort(key=lambda e: e["start_t"])

    def current_station_plane_event(self, dest_id, t):
        """
        返回当前时刻工位正在播放的动画事件。
        如果同一工位短时间内有多个事件重叠，优先显示最新触发的那个。
        """
        events = self.station_anim_events.get(dest_id, [])
        active = None

        for ev in events:
            if ev["start_t"] <= t < ev["end_t"]:
                active = ev

        return active

    def draw_station_plane_animations(self, t):
        """
        在 AGV 到达工位后的 30 秒内，在该工位内部显示飞机动画。
        飞机图片会按比例缩放，确保不超出工位范围。
        """
        if not self.plane_frames_raw:
            return

        for dest_id, dp in self.drop_points.items():
            ev = self.current_station_plane_event(dest_id, t)
            if not ev:
                continue

            rect = self.station_rect(*dp["pos"])

            # 动画进度：0 ~ 1
            progress = (t - ev["start_t"]) / PLANE_ANIM_SECONDS
            progress = max(0.0, min(1.0, progress))

            # 当前任务允许显示到的最大完整度
            max_stage = ev["max_stage"]

            # 在 30 秒内逐步从第 0 帧增长到 max_stage
            frame_idx = int(progress * (max_stage + 1))
            frame_idx = min(frame_idx, max_stage)
            frame_idx = max(0, min(frame_idx, len(self.plane_frames_raw) - 1))

            src = self.plane_frames_raw[frame_idx]
            src_w, src_h = src.get_size()

            # 限制飞机图最大不超过工位大小的 90%
            max_w = int(rect.width * PLANE_MAX_W_RATIO)
            max_h = int(rect.height * PLANE_MAX_H_RATIO)

            # 保持长宽比缩放
            if src_w > 0 and src_h > 0:
                scale = min(max_w / src_w, max_h / src_h)
            else:
                scale = 1.0

            # 可选：加入一点轻微呼吸效果，但仍不超过工位范围
            pulse = 1.0 + 0.03 * np.sin(progress * np.pi * 4)
            scale *= pulse

            plane_w = max(1, int(src_w * scale))
            plane_h = max(1, int(src_h * scale))

            # 再做一次保险限制，绝不超过工位
            plane_w = min(plane_w, max_w)
            plane_h = min(plane_h, max_h)

            plane_img = self.get_scaled_plane_frame(frame_idx, plane_w, plane_h).copy()

            # 淡入淡出
            if progress < 0.15:
                alpha = int(255 * (progress / 0.15))
            elif progress > 0.85:
                alpha = int(255 * ((1.0 - progress) / 0.15))
            else:
                alpha = 255

            plane_img.set_alpha(max(0, min(255, alpha)))

            # 严格居中在工位内部
            img_rect = plane_img.get_rect(center=rect.center)

            # 轻微上浮，但不出界
            float_offset = int(2 * np.sin(progress * np.pi))
            img_rect.centery -= float_offset

            # 若超界则拉回工位范围
            if img_rect.left < rect.left:
                img_rect.left = rect.left
            if img_rect.right > rect.right:
                img_rect.right = rect.right
            if img_rect.top < rect.top:
                img_rect.top = rect.top
            if img_rect.bottom > rect.bottom:
                img_rect.bottom = rect.bottom

            self.screen.blit(plane_img, img_rect)

            # 工位上方显示任务号和最晚时间
            tip = f"{ev['task_id']} due={ev['window_end']}"
            tip_surf = self.font_xs.render(tip, True, TEXT)
            tip_rect = tip_surf.get_rect(center=(rect.centerx, rect.top - 8))

            tip_bg = tip_rect.inflate(8, 4)
            pygame.draw.rect(self.screen, WHITE, tip_bg, border_radius=4)
            pygame.draw.rect(self.screen, TASK_BORDER, tip_bg, 1, border_radius=4)
            self.screen.blit(tip_surf, tip_rect)
    def load_summary(self):
        self.summary = {}

        try:
            with open("agv_summary.csv", "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    self.summary[row["metric"]] = row["value"]
        except FileNotFoundError:
            self.summary = {}

    def compute_task_delivery(self):
        self.task_delivery = {}

        for task_id, result in self.task_results.items():
            self.task_delivery[task_id] = result.get("delivery_time", "")

    # ============================================================
    # 坐标与绘图工具
    # ============================================================

    def grid_to_screen(self, x, y):
        sx = self.margin + (x - 1) * self.cell
        sy = self.top + (GRID_H - y) * self.cell
        return sx, sy

    def cell_rect(self, x, y):
        sx, sy = self.grid_to_screen(x, y)
        return pygame.Rect(sx, sy, self.cell, self.cell)

    def station_rect(self, x, y):
        """
        x, y 是工位 3x3 左下角。
        屏幕矩形左上角对应网格 (x, y + STATION_H - 1)。
        """
        sx = self.margin + (x - 1) * self.cell
        sy = self.top + (GRID_H - (y + STATION_H - 1)) * self.cell

        return pygame.Rect(
            sx,
            sy,
            STATION_W * self.cell,
            STATION_H * self.cell,
        )

    def unload_points(self, station_pos):
        """
        卸货点：
        - 工位左下角下方一格；
        - 工位左上角上方一格。

        例如工位左下角为 (3, 4)，工位左上角为 (3, 6)，
        卸货点为 (3, 3) 和 (3, 7)。
        """
        sx, sy = station_pos

        return [
            (sx, sy - 1),
            (sx, sy + STATION_H),
        ]

    def draw_text(self, text, font, color, pos, center=False):
        surf = font.render(str(text), True, color)
        rect = surf.get_rect()

        if center:
            rect.center = pos
        else:
            rect.topleft = pos

        self.screen.blit(surf, rect)
        return rect

    # ============================================================
    # 背景与静态元素
    # ============================================================

    def draw_grid(self):
        map_rect = pygame.Rect(
            self.margin,
            self.top,
            self.map_w,
            self.map_h,
        )

        pygame.draw.rect(self.screen, ROAD_FILL, map_rect)
        pygame.draw.rect(self.screen, BORDER, map_rect, 2)

        for x in range(GRID_W + 1):
            px = self.margin + x * self.cell

            pygame.draw.line(
                self.screen,
                ROAD_LINE,
                (px, self.top),
                (px, self.top + self.map_h),
                1,
            )

        for y in range(GRID_H + 1):
            py = self.top + y * self.cell

            pygame.draw.line(
                self.screen,
                ROAD_LINE,
                (self.margin, py),
                (self.margin + self.map_w, py),
                1,
            )

    def draw_stations(self):
        for dp in self.drop_points.values():
            sx, sy = dp["pos"]
            rect = self.station_rect(sx, sy)

            shadow = rect.move(2, 2)

            pygame.draw.rect(self.screen, STATION_SHADOW, shadow, border_radius=3)
            pygame.draw.rect(self.screen, STATION_FILL, rect, border_radius=3)
            pygame.draw.rect(self.screen, STATION_BORDER, rect, 2, border_radius=3)

            self.draw_text(
                dp["label"],
                self.font_station,
                TEXT,
                rect.center,
                center=True,
            )

            # 画两个绿色卸货点
            for ux, uy in self.unload_points(dp["pos"]):
                if 1 <= ux <= GRID_W and 1 <= uy <= GRID_H:
                    urect = self.cell_rect(ux, uy)

                    pygame.draw.rect(self.screen, UNLOAD_FILL, urect)
                    pygame.draw.rect(self.screen, UNLOAD_BORDER, urect, 2)

                    self.draw_text(
                        "U",
                        self.font_xs,
                        TEXT,
                        urect.center,
                        center=True,
                    )

    def draw_supply_and_pickups(self):
        if self.supply:
            x, y = self.supply["pos"]
            rect = self.cell_rect(x, y)

            pygame.draw.rect(self.screen, SUPPLY_FILL, rect, border_radius=3)
            pygame.draw.rect(self.screen, SUPPLY_BORDER, rect, 2, border_radius=3)

            self.draw_text(
                "S",
                self.font_sm,
                TEXT,
                rect.center,
                center=True,
            )

        for pickup in self.pickups.values():
            x, y = pickup["pos"]
            rect = self.cell_rect(x, y)

            pygame.draw.rect(self.screen, PICK_FILL, rect, border_radius=3)
            pygame.draw.rect(self.screen, PICK_BORDER, rect, 2, border_radius=3)

            self.draw_text(
                pickup["label"],
                self.font_xs,
                TEXT,
                rect.center,
                center=True,
            )

    def draw_depot(self):
        rect = self.cell_rect(*SHARED_HOME_POS)

        pygame.draw.rect(self.screen, DEPOT_FILL, rect, border_radius=3)
        pygame.draw.rect(self.screen, DEPOT_BORDER, rect, 2, border_radius=3)

        self.draw_text(
            "H",
            self.font_sm,
            TEXT,
            rect.center,
            center=True,
        )

    # ============================================================
    # 动态元素
    # ============================================================

    def color_for_agv(self, agv_id):
        idx = (int(agv_id) - 1) % len(AGV_COLOR_PALETTE)
        return AGV_COLOR_PALETTE[idx]

    def draw_agv(self, row):
        agv_id = row["name"]
        x = row["X"]
        y = row["Y"]
        pitch = row["pitch"]
        loaded = row.get("loaded", "false") == "true"
        status = row.get("status", "")

        rect = self.cell_rect(x, y).inflate(-4, -4)
        color = self.color_for_agv(agv_id)

        pygame.draw.ellipse(self.screen, color, rect)
        pygame.draw.ellipse(self.screen, BLACK, rect, 2)

        cx, cy = rect.center
        arrow_len = max(5, self.cell // 2)

        if pitch == 0:
            end = (cx + arrow_len, cy)
        elif pitch == 180:
            end = (cx - arrow_len, cy)
        elif pitch == 90:
            end = (cx, cy - arrow_len)
        elif pitch == 270:
            end = (cx, cy + arrow_len)
        else:
            end = (cx, cy)

        pygame.draw.line(self.screen, BLACK, (cx, cy), end, 2)

        self.draw_text(
            agv_id,
            self.font_xs,
            BLACK,
            rect.center,
            center=True,
        )

        # 载货标签：显示 M1 / M2 / M3 等物料名
        if loaded:
            material = row.get("material", "").strip()
            dest_label = row.get("destination_label", "").strip()

            if material and dest_label:
                label_text = f"{material}->{dest_label}"
            elif material:
                label_text = material
            else:
                label_text = "L"

            label_surf = self.font_sm.render(label_text, True, WHITE)
            label_rect = label_surf.get_rect()

            label_rect.centerx = rect.centerx
            label_rect.bottom = rect.top - 2

            if label_rect.top < self.top:
                label_rect.top = rect.bottom + 2

            bg_rect = label_rect.inflate(8, 5)

            pygame.draw.rect(self.screen, (182, 60, 52), bg_rect, border_radius=4)
            pygame.draw.rect(self.screen, (120, 40, 35), bg_rect, 1, border_radius=4)

            self.screen.blit(label_surf, label_rect)

        if status == "unloading":
            pygame.draw.rect(self.screen, TASK_BORDER, self.cell_rect(x, y), 2)

    def draw_agvs(self, t):
        rows = self.timeline.get(t, {})

        for agv_id in self.agv_ids:
            if agv_id in rows:
                self.draw_agv(rows[agv_id])

    def current_station_task(self, dest_id, t):
        tasks = self.tasks_by_dest.get(dest_id, [])

        for task in tasks:
            delivery = self.task_delivery.get(task["task_id"])

            if delivery == "":
                delivery_t = None
            else:
                delivery_t = int(delivery)

            if t >= task["release_time"] and (delivery_t is None or t <= delivery_t):
                return task, delivery_t

        return None, None

    def draw_station_task_badges(self, t):
        for dest_id, dp in self.drop_points.items():
            task, delivery_t = self.current_station_task(dest_id, t)

            if not task:
                continue

            sx, sy = dp["pos"]
            rect = self.station_rect(sx, sy)

            late = (
                t > task["window_start"]
                and (delivery_t is None or t < delivery_t)
            )

            color = TASK_BG_LATE if late else TASK_BG

            badge = pygame.Rect(
                rect.left,
                rect.bottom - self.cell,
                rect.width,
                self.cell,
            )

            pygame.draw.rect(self.screen, color, badge)
            pygame.draw.rect(self.screen, TASK_BORDER, badge, 1)

            txt = f"{task['material']} {task['window_end']}"

            self.draw_text(
                txt,
                self.font_xs,
                TEXT,
                badge.center,
                center=True,
            )

    # ============================================================
    # 右侧信息栏
    # ============================================================

    def draw_sidebar(self, t):
        x0 = self.margin * 2 + self.map_w
        y0 = self.top

        rect = pygame.Rect(
            x0,
            y0,
            self.sidebar,
            self.map_h,
        )

        pygame.draw.rect(self.screen, PANEL, rect)
        pygame.draw.rect(self.screen, BORDER, rect, 2)

        y = y0 + 16

        self.draw_text(
            "AGV Simulation",
            self.font_lg,
            TEXT,
            (x0 + 16, y),
        )

        y += 36

        self.draw_text(
            f"time: {t} / {self.max_t}",
            self.font_md,
            TEXT,
            (x0 + 16, y),
        )

        y += 30

        self.draw_text(
            f"grid: {GRID_W} x {GRID_H}",
            self.font_sm,
            SUBTEXT,
            (x0 + 16, y),
        )

        y += 24

        self.draw_text(
            f"station: {STATION_W} x {STATION_H}",
            self.font_sm,
            SUBTEXT,
            (x0 + 16, y),
        )

        y += 32

        self.draw_text(
            "Summary",
            self.font_md,
            TEXT,
            (x0 + 16, y),
        )

        y += 28

        for key in [
            "candidate_agv_count",
            "used_agv_count",
            "pickup_count",
            "total_delay",
            "max_delay",
            "avg_delay",
            "total_distance",
            "makespan",
        ]:
            if key in self.summary:
                self.draw_text(
                    f"{key}: {self.summary[key]}",
                    self.font_xs,
                    TEXT,
                    (x0 + 16, y),
                )

                y += 20

        y += 16

        self.draw_text(
            "Legend",
            self.font_md,
            TEXT,
            (x0 + 16, y),
        )

        y += 26

        legends = [
            (STATION_FILL, "3x3 station body"),
            (UNLOAD_FILL, "green unload point"),
            (SUPPLY_FILL, "supply"),
            (PICK_FILL, "pickup slot"),
            (DEPOT_FILL, "shared AGV depot"),
        ]

        for color, label in legends:
            pygame.draw.rect(
                self.screen,
                color,
                (x0 + 16, y + 2, 16, 16),
            )

            pygame.draw.rect(
                self.screen,
                BLACK,
                (x0 + 16, y + 2, 16, 16),
                1,
            )

            self.draw_text(
                label,
                self.font_xs,
                TEXT,
                (x0 + 40, y),
            )

            y += 22

        y += 16

        self.draw_text(
            "Controls",
            self.font_md,
            TEXT,
            (x0 + 16, y),
        )

        y += 26

        for line in [
            "Space: pause/resume",
            "Left/Right: step",
            "Esc/Q: quit",
            "Export: python display.py --save-mp4 agv_simulation.mp4",
        ]:
            self.draw_text(
                line,
                self.font_xs,
                TEXT,
                (x0 + 16, y),
            )

            y += 20

    def draw_header(self):
        self.screen.fill(BG)

        self.draw_text(
            "3x3 Station AGV Layout",
            self.font_lg,
            TEXT,
            (self.margin, 16),
        )

        self.draw_text(
            "AGV is 1x1; each station is 3x3; green cells are unload points.",
            self.font_sm,
            SUBTEXT,
            (self.margin + 340, 22),
        )

    def draw_frame(self, t):
        self.draw_header()
        self.draw_grid()
        self.draw_stations()
        self.draw_supply_and_pickups()
        self.draw_depot()

        # 先画工位任务信息
        self.draw_station_task_badges(t)

        # 再画工位上的飞机动画
        self.draw_station_plane_animations(t)

        # 最后画 AGV，保证 AGV 在最上层
        self.draw_agvs(t)

        self.draw_sidebar(t)
        pygame.display.flip()

    # ============================================================
    # MP4 导出
    # ============================================================

    def save_mp4(self, output_path="agv_simulation.mp4", video_fps=8):
        """
        按时间轴从 0 到 max_t 逐帧调用 draw_frame(t)，
        然后抓取 pygame 当前画面并用 OpenCV 写入 MP4。

        不依赖 imageio / ffmpeg 插件。
        """
        print(f"开始导出 MP4: {output_path}")
        print(f"总帧数: {self.max_t + 1}")
        print(f"视频 FPS: {video_fps}")

        width, height = self.screen.get_size()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(
                "无法创建 MP4 文件。请确认已安装 opencv-python。"
                "如果仍失败，可以改用 .avi 文件并把编码器改为 XVID。"
            )

        for t in range(self.max_t + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    writer.release()
                    pygame.quit()
                    return

            self.draw_frame(t)

            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame)

            if t % 50 == 0:
                print(f"已导出帧: {t}/{self.max_t}")

        writer.release()
        print(f"MP4 导出完成: {output_path}")
        pygame.quit()

    # ============================================================
    # 主循环
    # ============================================================

    def save_gif(self, output_path="agv_simulation.gif", gif_fps=4, frame_step=1):
        """
        导出 GIF 动图。

        参数：
        - output_path: GIF 文件名
        - gif_fps: GIF 播放帧率
        - frame_step: 抽帧间隔。比如 frame_step=2 表示每 2 秒仿真时间取 1 帧，
                      可以显著减小 GIF 文件体积。
        """
        print(f"开始导出 GIF: {output_path}")
        print(f"总仿真帧数: {self.max_t + 1}")
        print(f"GIF FPS: {gif_fps}")
        print(f"抽帧间隔 frame_step: {frame_step}")

        frames = []

        for t in range(0, self.max_t + 1, frame_step):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.draw_frame(t)

            # pygame surface -> numpy RGB frame
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))

            frames.append(frame)

            if t % 50 == 0:
                print(f"已生成 GIF 帧: {t}/{self.max_t}")

        imageio.mimsave(
            output_path,
            frames,
            fps=gif_fps,
            loop=0,
        )

        print(f"GIF 导出完成: {output_path}")
        pygame.quit()
    def run(self):
        t = 0
        paused = False
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

                    elif event.key == pygame.K_SPACE:
                        paused = not paused

                    elif event.key == pygame.K_RIGHT:
                        t = min(self.max_t, t + 1)

                    elif event.key == pygame.K_LEFT:
                        t = max(0, t - 1)

            self.draw_frame(t)

            if not paused:
                t += 1

                if t > self.max_t:
                    t = self.max_t
                    paused = True

            self.clock.tick(max(1, int(FPS * self.speed)))

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="AGV display and MP4 exporter")

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="interactive playback speed multiplier",
    )

    parser.add_argument(
        "--save-mp4",
        type=str,
        default="",
        help="export animation to MP4, for example: agv_simulation.mp4",
    )
    parser.add_argument(
        "--save-gif",
        type=str,
        default="",
        help="export animation to GIF, for example: agv_simulation.gif",
    )

    parser.add_argument(
        "--gif-fps",
        type=int,
        default=8,
        help="FPS of exported GIF",
    )

    parser.add_argument(
        "--gif-frame-step",
        type=int,
        default=1,
        help="Frame sampling step for GIF export. Larger value means smaller GIF.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=8,
        help="FPS of exported video",
    )

    args = parser.parse_args()

    manager = DisplayManager(speed=args.speed)

    if args.save_mp4:
        manager.save_mp4(args.save_mp4, video_fps=args.video_fps)
    elif args.save_gif:
        manager.save_gif(
            args.save_gif,
            gif_fps=args.gif_fps,
            frame_step=max(1, args.gif_frame_step),
        )
    else:
        manager.run()


if __name__ == "__main__":
    main()