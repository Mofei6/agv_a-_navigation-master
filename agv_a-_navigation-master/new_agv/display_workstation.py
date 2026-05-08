"""
Visualization for workstation-based AGV simulation.
Reads:
- agv_position_workstation.csv
- agv_task_workstation.csv
- agv_trajectory_workstation.csv
"""

import csv
import math
import sys
import importlib
from collections import defaultdict

import pygame
import numpy as np

pygame.init()

GRID_SIZE = 40
WINDOW_SIZE = (21 * GRID_SIZE, 21 * GRID_SIZE)
FPS = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (160, 160, 160)
MATERIAL = (46, 125, 50)
WORKSTATION = (245, 222, 179)
WORKSTATION_BUSY = (255, 193, 7)
AGV_COLOR = (135, 206, 250)
CARGO_NORMAL = (255, 192, 203)
CARGO_URGENT = (255, 0, 0)
WAITING = (255, 152, 0)

imageio = None


class DisplayManager:
    def __init__(self, speed=1, record=True, video_filename="agv_simulation_workstation.mp4"):
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("AGV Workstation Simulation")
        self.clock = pygame.time.Clock()
        self.speed = speed
        self.record = record
        self.video_filename = video_filename
        self.video_writer = None
        self.video_saved = False
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)
        self.time_font = pygame.font.Font(None, 34)

        self.material_zones = {}
        self.workstations = {}
        self.agv_initial = {}
        self.tasks = {}
        self.trajectory = []
        self.max_timestamp = 0

        self.load_positions()
        self.load_tasks()
        self.load_trajectory()
        self.setup_video_writer()

    def convert_y(self, y):
        return 21 - y

    def cell_rect(self, pos):
        x, y = pos
        return pygame.Rect((x - 1) * GRID_SIZE, self.convert_y(y) * GRID_SIZE, GRID_SIZE, GRID_SIZE)

    def cell_center(self, pos):
        r = self.cell_rect(pos)
        return r.center

    def load_positions(self):
        with open("agv_position_workstation.csv", newline="") as f:
            for row in csv.DictReader(f):
                typ = row["type"].strip()
                name = row["name"].strip()
                pos = (int(row["x"]), int(row["y"]))
                if typ == "material_zone":
                    self.material_zones[name] = pos
                elif typ in ("workstation", "end_point"):
                    self.workstations[name] = pos
                elif typ == "agv":
                    self.agv_initial[name] = {"pos": pos, "pitch": int(row["pitch"])}

    def load_tasks(self):
        with open("agv_task_workstation.csv", newline="") as f:
            for row in csv.DictReader(f):
                self.tasks[row["task_id"]] = row

    def load_trajectory(self):
        with open("agv_trajectory_workstation.csv", newline="") as f:
            self.trajectory = list(csv.DictReader(f))
        if self.trajectory:
            self.max_timestamp = max(int(r["timestamp"]) for r in self.trajectory)

    def setup_video_writer(self):
        if not self.record:
            return
        global imageio
        try:
            if imageio is None:
                imageio = importlib.import_module("imageio.v2")
            self.video_writer = imageio.get_writer(
                self.video_filename,
                fps=max(1, int(FPS * self.speed)),
                codec="libx264",
                macro_block_size=None,
            )
        except Exception as exc:
            print(f"初始化视频写入器失败: {exc}")
            self.video_writer = None

    def capture_frame(self):
        if not self.video_writer:
            return
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        self.video_writer.append_data(frame)

    def finalize_recording(self, show_message=False):
        if self.video_writer and not self.video_saved:
            self.video_writer.close()
            self.video_writer = None
            self.video_saved = True
            if show_message:
                print(f"可视化已保存至 {self.video_filename}")

    def draw_grid(self):
        for i in range(22):
            pygame.draw.line(self.screen, GRAY, (i * GRID_SIZE, 0), (i * GRID_SIZE, WINDOW_SIZE[1]))
            pygame.draw.line(self.screen, GRAY, (0, i * GRID_SIZE), (WINDOW_SIZE[0], i * GRID_SIZE))

    def draw_arrow(self, pos, pitch):
        cx, cy = self.cell_center(pos)
        angle = math.radians(-pitch)
        length = GRID_SIZE // 2
        end = (cx + length * math.cos(angle), cy + length * math.sin(angle))
        pygame.draw.line(self.screen, BLACK, (cx, cy), end, 2)
        for delta in (-math.pi / 6, math.pi / 6):
            head = (end[0] - 9 * math.cos(angle + delta), end[1] - 9 * math.sin(angle + delta))
            pygame.draw.line(self.screen, BLACK, end, head, 2)

    def draw_label(self, text, center, font=None):
        font = font or self.font
        surf = font.render(text, True, BLACK)
        self.screen.blit(surf, surf.get_rect(center=center))

    def workstation_busy_map(self, timestamp):
        busy = defaultdict(int)
        for r in self.trajectory:
            if int(r["timestamp"]) != timestamp:
                continue
            ws = r.get("workstation", "")
            if ws and r.get("loaded", "false").lower() == "true":
                busy[ws] += 1
        return busy

    def draw_static_objects(self, timestamp):
        busy = self.workstation_busy_map(timestamp)
        for name, pos in self.material_zones.items():
            pygame.draw.rect(self.screen, MATERIAL, self.cell_rect(pos))
            self.draw_label("物料", self.cell_center(pos), self.small_font)

        for name, pos in self.workstations.items():
            color = WORKSTATION_BUSY if busy.get(name) else WORKSTATION
            pygame.draw.rect(self.screen, color, self.cell_rect(pos))
            self.draw_label(name, self.cell_center(pos), self.small_font)

    def draw_cargo(self, pos, urgent=False, waiting=False):
        cx, cy = self.cell_center(pos)
        color = WAITING if waiting else (CARGO_URGENT if urgent else CARGO_NORMAL)
        pygame.draw.circle(self.screen, color, (cx, cy), GRID_SIZE // 4)

    def draw_agvs(self, timestamp):
        current = [r for r in self.trajectory if int(r["timestamp"]) == timestamp]
        for r in current:
            pos = (int(r["X"]), int(r["Y"]))
            pygame.draw.rect(self.screen, AGV_COLOR, self.cell_rect(pos))
            loaded = r.get("loaded", "false").lower() == "true"
            urgent = r.get("Emergency", "false").lower() == "true"
            waiting = r.get("waiting_for_station", "false").lower() == "true"
            if loaded:
                self.draw_cargo(pos, urgent=urgent, waiting=waiting)
            self.draw_arrow(pos, int(r["pitch"]))
            self.draw_label(r["name"][:2], self.cell_center(pos), self.small_font)

    def draw_status(self, timestamp):
        self.screen.blit(self.time_font.render(f"Time: {timestamp}", True, BLACK), (10, 10))
        self.screen.blit(self.small_font.render("绿色=物料区  米色=工位  黄色=被占用/等待处理  红色货物=Urgent", True, BLACK), (10, 42))

    def draw_frame(self, timestamp):
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_static_objects(timestamp)
        self.draw_agvs(timestamp)
        self.draw_status(timestamp)
        pygame.display.flip()
        self.capture_frame()

    def run(self):
        t = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.finalize_recording(show_message=True)
                    running = False
            if t <= self.max_timestamp:
                self.draw_frame(t)
                t += 1
                self.clock.tick(FPS * self.speed)
            else:
                self.finalize_recording(show_message=True)
                running = False
        pygame.quit()


if __name__ == "__main__":
    speed = 1
    record = True
    video_filename = "agv_simulation_workstation.mp4"
    args = sys.argv[1:]
    if args:
        try:
            speed = float(args[0])
            args = args[1:]
        except ValueError:
            pass
    if "--no-record" in args:
        record = False
    elif "--record" in args:
        idx = args.index("--record")
        if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
            video_filename = args[idx + 1]
    DisplayManager(speed=speed, record=record, video_filename=video_filename).run()
