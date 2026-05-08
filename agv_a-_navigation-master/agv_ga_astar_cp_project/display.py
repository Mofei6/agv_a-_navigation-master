import csv
import sys
import argparse
from collections import defaultdict

import cv2
import numpy as np
import pygame

pygame.init()

# ============================================================
# 基础地图参数
# ============================================================

GRID_W = 52
GRID_H = 52
STATION_W = 3
STATION_H = 3
FPS = 8

SHARED_HOME_POS = (50, 26)

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

    # ============================================================
    # 数据读取
    # ============================================================

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
                    self.task_results[row["task_id"]] = row
        except FileNotFoundError:
            self.task_results = {}

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

            txt = f"{task['material']} {task['window_start']}"

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
        self.draw_station_task_badges(t)
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
        "--video-fps",
        type=int,
        default=8,
        help="FPS of exported video",
    )

    args = parser.parse_args()

    manager = DisplayManager(speed=args.speed)

    if args.save_mp4:
        manager.save_mp4(args.save_mp4, video_fps=args.video_fps)
    else:
        manager.run()


if __name__ == "__main__":
    main()