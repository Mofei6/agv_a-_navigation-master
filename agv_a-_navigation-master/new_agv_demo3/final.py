# 本文档包含三个文件：
# 1. generate_data.py
# 2. navigation.py
# 3. display.py
#
# 下面请分别复制到对应文件中。


# =========================
# generate_data.py
# =========================

import csv
import random
from dataclasses import dataclass

GRID_W = 20
GRID_H = 20
RANDOM_SEED = 42
TASK_ROUNDS_PER_STATION = 3
WINDOW_LEN_MIN = 20
WINDOW_LEN_MAX = 50
NEXT_TASK_GAP_MIN = 8
NEXT_TASK_GAP_MAX = 18
RELEASE_LEAD = 25

POSITION_FILE = 'agv_position.csv'
TASK_FILE = 'agv_task.csv'

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
    pts = []
    # 209 top row
    for idx, (label, x) in enumerate([('209F', 7), ('209E', 9), ('209D', 11), ('209C', 13), ('209B', 15), ('209A', 17)], start=1):
        pts.append(DropPoint(f'DP-{label}-01', label, x, 18))
    # upper 200
    for label, x in [('200A', 14), ('200B', 15), ('200C', 16)]:
        pts.append(DropPoint(f'DP-{label}-01', label, x, 15))
    rows = [(11, '02'), (8, '03'), (5, '04')]
    base = [('130A', 2), ('130B', 4), ('140A', 6), ('140B', 8), ('140C', 10), ('150', 12)]
    for y, suffix in rows:
        for label, x in base:
            pts.append(DropPoint(f'DP-{label}-{suffix}', label, x, y))
        for label, x in [('200A', 14), ('200B', 15), ('200C', 16)]:
            pts.append(DropPoint(f'DP-{label}-{suffix}', label, x, y))
    return pts


def build_agvs():
    homes = [(19, 7), (20, 7), (19, 9), (20, 9), (19, 11)]
    return [AGVHome(str(i + 1), x, y) for i, (x, y) in enumerate(homes)]


def validate(drop_points, agvs):
    occupied = {}
    fixed = [('SUPPLY-01', 18, 10), ('PICK-1', 17, 10)]
    for name, x, y in fixed:
        occupied[(x, y)] = name
    for p in drop_points:
        if (p.x, p.y) in occupied:
            raise ValueError(f'坐标冲突: {p.name} 与 {occupied[(p.x,p.y)]}')
        occupied[(p.x, p.y)] = p.name
    for a in agvs:
        if (a.x, a.y) in occupied:
            raise ValueError(f'坐标冲突: AGV{a.agv_id} 与 {occupied[(a.x,a.y)]}')
        occupied[(a.x, a.y)] = f'AGV-{a.agv_id}'


def write_positions(drop_points, agvs):
    with open(POSITION_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['type', 'name', 'label', 'x', 'y', 'pitch'])
        w.writerow(['supply', 'SUPPLY-01', '物料区', 18, 10, ''])
        w.writerow(['pickup_slot', 'PICK-1', 'P1', 17, 10, ''])
        for p in drop_points:
            w.writerow(['drop_point', p.name, p.label, p.x, p.y, ''])
        for a in agvs:
            w.writerow(['agv', a.agv_id, f'AGV{a.agv_id}', a.x, a.y, a.pitch])


def write_tasks(drop_points):
    random.seed(RANDOM_SEED)
    materials = ['A料', 'B料', 'C料', 'D料', 'E料']
    rows = []
    task_idx = 1
    for station_idx, dp in enumerate(drop_points):
        start = 30 + (station_idx % 6) * 6 + station_idx // 6 * 3
        for round_idx in range(TASK_ROUNDS_PER_STATION):
            window_len = random.randint(WINDOW_LEN_MIN, WINDOW_LEN_MAX)
            window_start = start
            window_end = start + window_len
            release_time = max(0, window_start - RELEASE_LEAD)
            material = materials[(station_idx + round_idx) % len(materials)]
            rows.append([
                f'T{task_idx:04d}', dp.name, dp.label, material,
                release_time, window_start, window_end, 'Normal'
            ])
            task_idx += 1
            start = window_end + random.randint(NEXT_TASK_GAP_MIN, NEXT_TASK_GAP_MAX)
    rows.sort(key=lambda r: (int(r[4]), int(r[5]), r[1]))
    with open(TASK_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['task_id', 'destination_id', 'destination_label', 'material', 'release_time', 'window_start', 'window_end', 'priority'])
        w.writerows(rows)


def main():
    dps = build_drop_points()
    agvs = build_agvs()
    validate(dps, agvs)
    write_positions(dps, agvs)
    write_tasks(dps)
    print('generated agv_position.csv and agv_task.csv')
    print(f'drop points={len(dps)}, tasks={len(dps)*TASK_ROUNDS_PER_STATION}')

if __name__ == '__main__':
    main()


# =========================
# navigation.py
# =========================

import csv
import heapq
from collections import defaultdict

GRID_W = 20
GRID_H = 20
POSITION_FILE = 'agv_position.csv'
TASK_FILE = 'agv_task.csv'
TRAJECTORY_FILE = 'agv_trajectory.csv'

LOAD_DURATION = 2
UNLOAD_DURATION = 2
RETURN_HOME_THRESHOLD = 35
MAX_SEARCH_TIME = 1000

DIRS = [(1,0),(-1,0),(0,1),(0,-1),(0,0)]
PITCH_MAP = {(1,0):0, (-1,0):180, (0,1):90, (0,-1):270, (0,0):None}


def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def neighbors(pos):
    for dx, dy in DIRS:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 1 <= nx <= GRID_W and 1 <= ny <= GRID_H:
            yield (nx, ny)


class Planner:
    def __init__(self):
        self.supply = None
        self.pickup = None
        self.drop_points = {}
        self.homes = {}
        self.tasks = []
        self.agv_states = {}
        self.agv_tracks = defaultdict(list)
        self.vertex_res = defaultdict(set)  # t -> set(pos)
        self.edge_res = defaultdict(set)    # t -> set((from,to)) used from t-1->t
        self.load_inputs()

    def load_inputs(self):
        with open(POSITION_FILE, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                t = row['type'].strip()
                item = {
                    'name': row['name'].strip(),
                    'label': row.get('label', '').strip(),
                    'pos': (int(row['x']), int(row['y'])),
                    'pitch': int(row['pitch']) if row.get('pitch') not in ('', None) else 0,
                }
                if t == 'supply':
                    self.supply = item
                elif t == 'pickup_slot':
                    self.pickup = item
                elif t == 'drop_point':
                    self.drop_points[item['name']] = item
                elif t == 'agv':
                    self.homes[item['name']] = item

        with open(TASK_FILE, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                task = {
                    'task_id': row['task_id'],
                    'destination_id': row['destination_id'],
                    'destination_label': row['destination_label'],
                    'material': row.get('material', ''),
                    'release_time': int(row['release_time']),
                    'window_start': int(row['window_start']),
                    'window_end': int(row['window_end']),
                    'priority': row.get('priority', 'Normal'),
                }
                if task['window_end'] <= task['window_start']:
                    raise ValueError(f"任务 {task['task_id']} 时间窗非法")
                self.tasks.append(task)

        self.tasks.sort(key=lambda x: (x['release_time'], x['window_start'], x['task_id']))

        for agv_id, home in self.homes.items():
            self.agv_states[agv_id] = {
                'id': agv_id,
                'pos': home['pos'],
                'time': 0,
                'pitch': home['pitch'],
                'home': home['pos'],
                'last_task_window_start': None,
            }
            self.reserve_point(home['pos'], 0)
            self.agv_tracks[agv_id].append(self.make_row(0, agv_id, home['pos'], home['pitch'], False, '', '', '', 'idle'))

    def static_blocked(self):
        blocked = set()
        if self.supply:
            blocked.add(self.supply['pos'])
        for dp in self.drop_points.values():
            blocked.add(dp['pos'])
        return blocked

    def traversable(self, pos):
        return 1 <= pos[0] <= GRID_W and 1 <= pos[1] <= GRID_H and pos not in self.static_blocked()

    def delivery_goals(self, destination_id):
        target = self.drop_points[destination_id]['pos']
        goals = []
        for nb in neighbors(target):
            if nb != target and self.traversable(nb):
                goals.append(nb)
        if not goals:
            raise RuntimeError(f"任务目的地 {destination_id} 周边无可达道路")
        return goals

    def reserve_point(self, pos, t):
        self.vertex_res[t].add(pos)

    def reserve_edge(self, prev_pos, pos, t):
        self.edge_res[t].add((prev_pos, pos))

    def is_reserved(self, prev_pos, pos, t):
        if pos in self.vertex_res[t]:
            return True
        if (pos, prev_pos) in self.edge_res[t]:
            return True
        return False

    def heuristic(self, pos, goals):
        return min(manhattan(pos, g) for g in goals)

    def astar_time(self, start_pos, start_time, goals, earliest_goal_time=0):
        blocked = self.static_blocked()
        goals = set(goals)
        start = (start_pos, start_time)
        pq = [(self.heuristic(start_pos, goals), 0, start_pos, start_time)]
        parent = {(start_pos, start_time): None}
        gscore = {(start_pos, start_time): 0}

        while pq:
            _, cost, pos, t = heapq.heappop(pq)
            state = (pos, t)
            if pos in goals and t >= earliest_goal_time:
                path = []
                cur = state
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            if t - start_time > MAX_SEARCH_TIME:
                continue

            for nxt in neighbors(pos):
                if nxt in blocked and nxt not in goals:
                    continue
                nt = t + 1
                if self.is_reserved(pos, nxt, nt):
                    continue
                if nxt != pos and nxt in blocked and nxt not in goals:
                    continue
                nstate = (nxt, nt)
                ncost = cost + 1
                if nstate not in gscore or ncost < gscore[nstate]:
                    gscore[nstate] = ncost
                    parent[nstate] = state
                    priority = ncost + self.heuristic(nxt, goals) + max(0, earliest_goal_time - nt)
                    heapq.heappush(pq, (priority, ncost, nxt, nt))
        return None

    def extend_hold(self, path, hold_until_time):
        if not path:
            return path
        pos, t = path[-1]
        while t < hold_until_time:
            t += 1
            path.append((pos, t))
        return path

    def reserve_path(self, path):
        for idx, (pos, t) in enumerate(path):
            self.reserve_point(pos, t)
            if idx > 0:
                prev_pos, _ = path[idx-1]
                self.reserve_edge(prev_pos, pos, t)

    def path_pitch(self, prev_pos, curr_pos, last_pitch):
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        return PITCH_MAP.get((dx, dy), last_pitch) or last_pitch

    def append_path_rows(self, agv_id, path, loaded, material, dest_label, dest_id, task_id, status, skip_first=False):
        state = self.agv_states[agv_id]
        last_pitch = state['pitch']
        rows = self.agv_tracks[agv_id]
        iter_path = path[1:] if skip_first else path
        prev_pos = rows[-1]['pos'] if rows else path[0][0]
        for pos, t in iter_path:
            pitch = self.path_pitch(prev_pos, pos, last_pitch)
            row = self.make_row(t, agv_id, pos, pitch, loaded, material, dest_label, task_id, status, dest_id)
            rows.append(row)
            prev_pos = pos
            last_pitch = pitch
        state['pitch'] = last_pitch
        state['pos'] = path[-1][0]
        state['time'] = path[-1][1]

    def make_row(self, t, agv_id, pos, pitch, loaded, material, dest_label, task_id, status, dest_id=''):
        return {
            'timestamp': t,
            'name': agv_id,
            'X': pos[0],
            'Y': pos[1],
            'pitch': pitch,
            'loaded': 'true' if loaded else 'false',
            'material': material,
            'destination_label': dest_label,
            'destination_id': dest_id,
            'task_id': task_id,
            'status': status,
            'pos': pos,
        }

    def candidate_plan(self, agv_id, task):
        agv = self.agv_states[agv_id]
        cur_pos = agv['pos']
        cur_time = agv['time']
        segments = []
        plan_start_pos = cur_pos
        plan_start_time = cur_time

        if task['window_start'] - cur_time >= RETURN_HOME_THRESHOLD and cur_pos != agv['home']:
            path_home = self.astar_time(cur_pos, cur_time, [agv['home']])
            if path_home is None:
                return None
            segments.append(('return_home', False, '', '', '', '', path_home))
            plan_start_pos, plan_start_time = path_home[-1]

        path_to_pick = self.astar_time(plan_start_pos, plan_start_time, [self.pickup['pos']], earliest_goal_time=task['release_time'])
        if path_to_pick is None:
            return None
        pickup_depart = path_to_pick[-1][1] + LOAD_DURATION
        load_path = list(path_to_pick)
        self.extend_hold(load_path, pickup_depart)
        segments.append(('to_pickup', False, '', '', '', '', path_to_pick))
        segments.append(('loading', True, task['material'], task['destination_label'], task['destination_id'], task['task_id'], load_path[len(path_to_pick)-1:]))

        goals = self.delivery_goals(task['destination_id'])
        path_to_drop = self.astar_time(self.pickup['pos'], pickup_depart, goals)
        if path_to_drop is None:
            return None
        unload_end = path_to_drop[-1][1] + UNLOAD_DURATION
        unload_path = list(path_to_drop)
        self.extend_hold(unload_path, unload_end)
        segments.append(('delivering', True, task['material'], task['destination_label'], task['destination_id'], task['task_id'], path_to_drop))
        segments.append(('unloading', False, '', task['destination_label'], task['destination_id'], task['task_id'], unload_path[len(path_to_drop)-1:]))

        finish_time = unload_path[-1][1]
        finish_pos = unload_path[-1][0]
        return {'segments': segments, 'finish_time': finish_time, 'finish_pos': finish_pos}

    def commit_plan(self, agv_id, plan, task):
        for status, loaded, material, dest_label, dest_id, task_id, path in plan['segments']:
            self.reserve_path(path[1:])
            self.append_path_rows(agv_id, path, loaded, material, dest_label, dest_id, task_id, status, skip_first=True)
        self.agv_states[agv_id]['last_task_window_start'] = task['window_start']

    def choose_best_agv(self, task):
        best = None
        for agv_id in sorted(self.agv_states.keys(), key=lambda x: int(x)):
            plan = self.candidate_plan(agv_id, task)
            if plan is None:
                continue
            arrive = None
            for seg in plan['segments']:
                if seg[0] == 'delivering':
                    arrive = seg[6][-1][1]
            lateness = max(0, arrive - task['window_start']) if arrive is not None else 10**9
            key = (lateness, arrive, plan['finish_time'], int(agv_id))
            if best is None or key < best[0]:
                best = (key, agv_id, plan)
        if best is None:
            raise RuntimeError(f"任务 {task['task_id']} 无法规划可行路径。目标={task['destination_id']}, label={task['destination_label']}")
        return best[1], best[2]

    def fill_idle_until(self, agv_id, target_time, status='idle'):
        rows = self.agv_tracks[agv_id]
        if not rows:
            return
        last = rows[-1]
        pos = last['pos']
        pitch = last['pitch']
        t = last['timestamp']
        while t < target_time:
            t += 1
            rows.append(self.make_row(t, agv_id, pos, pitch, False, '', '', '', status))
            self.reserve_point(pos, t)
            self.reserve_edge(pos, pos, t)
        self.agv_states[agv_id]['time'] = t
        self.agv_states[agv_id]['pos'] = pos
        self.agv_states[agv_id]['pitch'] = pitch

    def finalize_to_home(self):
        for agv_id, agv in self.agv_states.items():
            if agv['pos'] == agv['home']:
                continue
            path = self.astar_time(agv['pos'], agv['time'], [agv['home']])
            if path is None:
                continue
            self.reserve_path(path[1:])
            self.append_path_rows(agv_id, path, False, '', '', '', '', 'return_home', skip_first=True)

    def normalize_tracks(self):
        max_t = max(rows[-1]['timestamp'] for rows in self.agv_tracks.values())
        for agv_id in self.agv_tracks:
            self.fill_idle_until(agv_id, max_t, status='idle')
        return max_t

    def write_trajectory(self):
        fieldnames = ['timestamp', 'name', 'X', 'Y', 'pitch', 'loaded', 'material', 'destination_label', 'destination_id', 'task_id', 'status']
        all_rows = []
        for agv_id, rows in self.agv_tracks.items():
            for r in rows:
                all_rows.append({k: r[k] for k in fieldnames})
        all_rows.sort(key=lambda x: (x['timestamp'], int(x['name'])))
        with open(TRAJECTORY_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    def check_conflicts(self):
        by_t = defaultdict(dict)
        with open(TRAJECTORY_FILE, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                t = int(row['timestamp'])
                by_t[t][row['name']] = (int(row['X']), int(row['Y']))
        has_conflict = False
        for t in sorted(by_t):
            items = list(by_t[t].items())
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a1, p1 = items[i]
                    a2, p2 = items[j]
                    if p1 == p2:
                        print(f'碰撞: t={t}, AGV{a1} 与 AGV{a2} 在 {p1}')
                        has_conflict = True
                    if t + 1 in by_t and a1 in by_t[t+1] and a2 in by_t[t+1]:
                        np1 = by_t[t+1][a1]
                        np2 = by_t[t+1][a2]
                        if p1 == np2 and p2 == np1:
                            print(f'对穿: t={t}->{t+1}, AGV{a1} {p1}->{np1}, AGV{a2} {p2}->{np2}')
                            has_conflict = True
        if not has_conflict:
            print('轨迹检查通过：全程无顶点冲突、无对穿冲突。')
        return not has_conflict

    def run(self):
        for task in self.tasks:
            agv_id, plan = self.choose_best_agv(task)
            self.commit_plan(agv_id, plan, task)
        self.finalize_to_home()
        self.normalize_tracks()
        self.write_trajectory()
        self.check_conflicts()
        print(f'已输出 {TRAJECTORY_FILE}')


def main():
    Planner().run()

if __name__ == '__main__':
    main()


# =========================
# display.py
# =========================

import csv
import sys
from collections import defaultdict
import pygame

pygame.init()

GRID_W = 20
GRID_H = 20
FPS = 2

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
TEXT = (35, 35, 35)
SUBTEXT = (90, 90, 90)
AGV_COLORS = {
    '1': (235, 184, 41),
    '2': (92, 162, 245),
    '3': (94, 194, 117),
    '4': (176, 118, 223),
    '5': (239, 118, 118),
}

class DisplayManager:
    def __init__(self, speed=1.0):
        info = pygame.display.Info()
        self.margin = 22
        self.top = 72
        self.sidebar = 320
        max_w = min(1480, info.current_w - 100)
        max_h = min(930, info.current_h - 100)
        self.cell = max(40, min(54, (max_w - self.sidebar - self.margin * 3)//GRID_W, (max_h - self.top - self.margin*2)//GRID_H))
        self.map_w = GRID_W * self.cell
        self.map_h = GRID_H * self.cell
        self.screen = pygame.display.set_mode((self.map_w + self.sidebar + self.margin*3, self.map_h + self.top + self.margin*2))
        pygame.display.set_caption('AGV Layout Simulation')
        self.clock = pygame.time.Clock()
        self.speed = speed

        self.font_xs = pygame.font.SysFont('arial', max(12, int(self.cell*0.20)))
        self.font_sm = pygame.font.SysFont('arial', max(14, int(self.cell*0.24)), bold=True)
        self.font_md = pygame.font.SysFont('arial', max(16, int(self.cell*0.30)), bold=True)
        self.font_lg = pygame.font.SysFont('arial', max(22, int(self.cell*0.44)), bold=True)
        self.font_station = pygame.font.SysFont('arial', max(20, int(self.cell*0.34)), bold=True)
        self.font_task = pygame.font.SysFont('consolas', max(15, int(self.cell*0.24)), bold=True)
        self.font_badge = pygame.font.SysFont('arial', max(14, int(self.cell*0.22)), bold=True)

        self.load_positions()
        self.load_tasks()
        self.load_traj()
        self.compute_task_delivery()

    def load_positions(self):
        self.supply = None
        self.pickup = None
        self.drop_points = {}
        self.homes = {}
        self.agv_ids = []
        with open('agv_position.csv', 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                item = {
                    'name': row['name'].strip(),
                    'label': row.get('label', '').strip(),
                    'pos': (int(row['x']), int(row['y'])),
                    'pitch': int(row['pitch']) if row.get('pitch') not in ('', None) else 0,
                }
                if row['type'] == 'supply':
                    self.supply = item
                elif row['type'] == 'pickup_slot':
                    self.pickup = item
                elif row['type'] == 'drop_point':
                    self.drop_points[item['name']] = item
                elif row['type'] == 'agv':
                    self.homes[item['name']] = item
                    self.agv_ids.append(item['name'])
        self.agv_ids.sort(key=lambda x: int(x))

    def load_tasks(self):
        self.tasks_by_dest = defaultdict(list)
        with open('agv_task.csv', 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                row['release_time'] = int(row['release_time'])
                row['window_start'] = int(row['window_start'])
                row['window_end'] = int(row['window_end'])
                self.tasks_by_dest[row['destination_id']].append(row)
        for k in self.tasks_by_dest:
            self.tasks_by_dest[k].sort(key=lambda r: (r['release_time'], r['window_start'], r['task_id']))

    def load_traj(self):
        self.timeline = defaultdict(dict)
        self.max_t = 0
        with open('agv_trajectory.csv', 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            t = int(row['timestamp'])
            self.max_t = max(self.max_t, t)
            row['X'] = int(row['X']); row['Y'] = int(row['Y']); row['pitch'] = int(row['pitch'])
            self.timeline[t][row['name']] = row
        last = {}
        for t in range(self.max_t + 1):
            for agv_id in self.agv_ids:
                if agv_id in self.timeline[t]:
                    last[agv_id] = self.timeline[t][agv_id]
                elif agv_id in last:
                    self.timeline[t][agv_id] = last[agv_id]

    def compute_task_delivery(self):
        self.delivered_at = {}
        for t, items in self.timeline.items():
            for row in items.values():
                if row['task_id'] and row['status'] == 'unloading':
                    self.delivered_at.setdefault(row['task_id'], t)

    def grid_rect(self, pos, inset=0):
        x, y = pos
        sx = self.margin + (x - 1) * self.cell
        sy = self.top + self.margin + (GRID_H - y) * self.cell
        return pygame.Rect(sx + inset, sy + inset, self.cell - inset*2, self.cell - inset*2)

    def center(self, pos):
        return self.grid_rect(pos).center

    def active_task(self, dest_id, t):
        for task in self.tasks_by_dest.get(dest_id, []):
            if task['release_time'] <= t <= task['window_end']:
                return task
        return None

    def task_state(self, task, t):
        deliver_t = self.delivered_at.get(task['task_id'])
        if deliver_t is None:
            return ('待料', TASK_BG if t < task['window_start'] else TASK_BG_LATE)
        return ('已到' if deliver_t <= task['window_start'] else '迟到', TASK_BG if deliver_t <= task['window_start'] else TASK_BG_LATE)

    def draw_base(self):
        self.screen.fill(BG)
        map_rect = pygame.Rect(self.margin-6, self.top+self.margin-6, self.map_w+12, self.map_h+12)
        pygame.draw.rect(self.screen, PANEL, map_rect, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, map_rect, 2, border_radius=12)
        side = pygame.Rect(self.margin*2+self.map_w, self.top, self.sidebar, self.map_h+self.margin)
        pygame.draw.rect(self.screen, PANEL, side, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, side, 2, border_radius=12)
        title = self.font_lg.render('AGV 物流配送布局图', True, TEXT)
        self.screen.blit(title, (self.margin+8, 18))
        legend = self.font_sm.render('白色网格 = 可行驶区域', True, SUBTEXT)
        self.screen.blit(legend, (self.margin+8, 48))

    def draw_roads(self):
        for x in range(1, GRID_W+1):
            for y in range(1, GRID_H+1):
                rect = self.grid_rect((x,y), 1)
                pygame.draw.rect(self.screen, ROAD_FILL, rect, border_radius=3)
                pygame.draw.rect(self.screen, ROAD_LINE, rect, 1, border_radius=3)

    def draw_fixed(self):
        if self.supply:
            r = self.grid_rect(self.supply['pos'], 5)
            pygame.draw.rect(self.screen, SUPPLY_FILL, r, border_radius=8)
            pygame.draw.rect(self.screen, SUPPLY_BORDER, r, 2, border_radius=8)
            txt = self.font_sm.render('物料区', True, TEXT)
            self.screen.blit(txt, txt.get_rect(center=r.center))
        if self.pickup:
            r = self.grid_rect(self.pickup['pos'], 8)
            pygame.draw.rect(self.screen, PICK_FILL, r, border_radius=8)
            pygame.draw.rect(self.screen, PICK_BORDER, r, 2, border_radius=8)
            txt = self.font_sm.render(self.pickup['label'], True, TEXT)
            self.screen.blit(txt, txt.get_rect(center=r.center))

    def draw_homes(self):
        for agv_id, item in self.homes.items():
            r = self.grid_rect(item['pos'], 10)
            pygame.draw.rect(self.screen, (235,238,244), r, border_radius=6)
            pygame.draw.rect(self.screen, (150,160,175), r, 1, border_radius=6)
            txt = self.font_xs.render(f'H{agv_id}', True, SUBTEXT)
            self.screen.blit(txt, txt.get_rect(center=r.center))

    def draw_stations(self, t):
        for dest_id, dp in self.drop_points.items():
            cell = self.grid_rect(dp['pos'])
            station = pygame.Rect(0,0,int(cell.width*0.96), int(cell.height*0.95))
            station.center = cell.center
            pygame.draw.rect(self.screen, STATION_SHADOW, station.move(2,3), border_radius=8)
            pygame.draw.rect(self.screen, STATION_FILL, station, border_radius=8)
            pygame.draw.rect(self.screen, STATION_BORDER, station, 2, border_radius=8)
            label = self.font_station.render(dp['label'], True, TEXT)
            self.screen.blit(label, label.get_rect(center=(station.centerx, station.top + station.height*0.23)))

            task = self.active_task(dest_id, t)
            if task:
                status, bg = self.task_state(task, t)
                info = pygame.Rect(station.left+5, station.top+int(station.height*0.42), station.width-10, int(station.height*0.47))
                pygame.draw.rect(self.screen, bg, info, border_radius=7)
                pygame.draw.rect(self.screen, TASK_BORDER, info, 1, border_radius=7)
                line1 = self.font_task.render(task['material'], True, TEXT)
                line2 = self.font_task.render(f"T={task['window_start']:>3}", True, TEXT)
                line3 = self.font_xs.render(status, True, TEXT)
                self.screen.blit(line1, line1.get_rect(midtop=(info.centerx, info.top+3)))
                self.screen.blit(line2, line2.get_rect(center=(info.centerx, info.centery+2)))
                self.screen.blit(line3, line3.get_rect(midbottom=(info.centerx, info.bottom-2)))

    def draw_agv(self, agv_id, row):
        pos = (row['X'], row['Y'])
        pitch = row['pitch']
        color = AGV_COLORS.get(agv_id, (180,180,180))
        cx, cy = self.center(pos)
        body = pygame.Rect(0,0,max(28,int(self.cell*0.58)), max(22,int(self.cell*0.42)))
        body.center = (cx,cy)
        pygame.draw.rect(self.screen, (170,170,170), body.move(2,2), border_radius=10)
        pygame.draw.rect(self.screen, color, body, border_radius=10)
        pygame.draw.rect(self.screen, (60,60,60), body, 2, border_radius=10)
        top = body.inflate(-body.width*0.34, -body.height*0.42)
        pygame.draw.rect(self.screen, (255,247,219), top, border_radius=6)
        pygame.draw.rect(self.screen, (100,100,100), top, 1, border_radius=6)
        for p in [(body.left+4, body.top+4), (body.right-4, body.top+4), (body.left+4, body.bottom-4), (body.right-4, body.bottom-4)]:
            pygame.draw.circle(self.screen, (70,70,70), p, max(2,int(self.cell*0.05)))
        if pitch == 0:
            pts = [(body.right+9, body.centery), (body.right-1, body.centery-7), (body.right-1, body.centery+7)]
        elif pitch == 180:
            pts = [(body.left-9, body.centery), (body.left+1, body.centery-7), (body.left+1, body.centery+7)]
        elif pitch == 90:
            pts = [(body.centerx, body.top-9), (body.centerx-7, body.top+1), (body.centerx+7, body.top+1)]
        else:
            pts = [(body.centerx, body.bottom+9), (body.centerx-7, body.bottom-1), (body.centerx+7, body.bottom-1)]
        pygame.draw.polygon(self.screen, (35,35,35), pts)
        num = self.font_sm.render(agv_id, True, TEXT)
        self.screen.blit(num, num.get_rect(center=body.center))

        if row['loaded'] == 'true' and row.get('material'):
            badge_text = f"{row['material']}->{row['destination_label']}"
            badge = self.font_badge.render(badge_text, True, (255,255,255))
            br = pygame.Rect(0,0,badge.get_width()+14,badge.get_height()+6)
            br.centerx = cx
            br.bottom = body.top - 4
            pygame.draw.rect(self.screen, (182,60,52), br, border_radius=7)
            pygame.draw.rect(self.screen, (120,40,35), br, 1, border_radius=7)
            self.screen.blit(badge, badge.get_rect(center=br.center))

    def draw_sidebar(self, t):
        x = self.margin*2 + self.map_w + 18
        y = self.top + 16
        title = self.font_lg.render(f'Time = {t}s', True, TEXT)
        self.screen.blit(title, (x, y))
        y += 42
        items = self.timeline.get(t, {})
        counters = {'配送中':0,'取料中':0,'回停中':0,'空闲':0}
        for row in items.values():
            st = row['status']
            if st == 'delivering': counters['配送中'] += 1
            elif st in ('to_pickup','loading'): counters['取料中'] += 1
            elif st == 'return_home': counters['回停中'] += 1
            else: counters['空闲'] += 1
        for k,v in counters.items():
            txt = self.font_md.render(f'{k}: {v}', True, TEXT)
            self.screen.blit(txt, (x,y)); y += 24
        y += 12
        sub = self.font_md.render('AGV 状态', True, TEXT)
        self.screen.blit(sub, (x,y)); y += 30
        for agv_id in self.agv_ids:
            row = items.get(agv_id)
            if not row: continue
            rr = pygame.Rect(x, y, self.sidebar-36, 46)
            pygame.draw.rect(self.screen, (248,248,246), rr, border_radius=8)
            pygame.draw.rect(self.screen, (210,210,205), rr, 1, border_radius=8)
            pygame.draw.rect(self.screen, AGV_COLORS.get(agv_id, (150,150,150)), pygame.Rect(x+8,y+13,16,16), border_radius=4)
            l1 = self.font_sm.render(f'AGV{agv_id}  {row["status"]}', True, TEXT)
            self.screen.blit(l1, (x+32, y+5))
            if row['loaded'] == 'true':
                msg = f"{row['material']} -> {row['destination_label']}"
            else:
                msg = f"位置 ({row['X']}, {row['Y']})"
            l2 = self.font_xs.render(msg, True, SUBTEXT)
            self.screen.blit(l2, (x+32, y+25))
            y += 52
        y += 8
        for line in ['说明：', '1. 白色网格表示可行驶区域', '2. 箭头表示朝向', '3. 工位内显示: 编号/物料/开始时间', '4. Space 暂停, ← → 单步查看']:
            txt = self.font_xs.render(line, True, TEXT if line=='说明：' else SUBTEXT)
            self.screen.blit(txt, (x,y)); y += 18

    def draw(self, t):
        self.draw_base()
        self.draw_roads()
        self.draw_stations(t)
        self.draw_homes()
        self.draw_fixed()
        for agv_id in self.agv_ids:
            if agv_id in self.timeline.get(t, {}):
                self.draw_agv(agv_id, self.timeline[t][agv_id])
        self.draw_sidebar(t)
        pygame.display.flip()

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
                        t = min(self.max_t, t+1)
                    elif event.key == pygame.K_LEFT:
                        t = max(0, t-1)
            self.draw(t)
            if not paused and t < self.max_t:
                t += 1
            self.clock.tick(max(1, int(FPS*self.speed)))
        pygame.quit()

if __name__ == '__main__':
    speed = 1.0
    if len(sys.argv) > 1:
        try: speed = float(sys.argv[1])
        except: pass
    DisplayManager(speed).run()
