"""
Workstation version of the AGV planner.

Main changes vs original navigation.py:
1. Replaces multiple pickup stations with one material_zone.
2. Replaces destination outlets with numbered workstations, e.g. 130A/200B/209D/240A.
3. Each task has a release_time and due_time time window.
4. One workstation can process only one task at a time; multiple tasks for the same
   workstation are processed at different times according to workstation availability.
5. AGVs always pick materials from the single material_zone, then deliver to a workstation.

Input files:
- agv_position_workstation.csv
  type,name,x,y,pitch
  material_zone,Material,1,1,
  workstation,240A,5,16,
  agv,Optimus,2,1,90

- agv_task_workstation.csv
  task_id,workstation,release_time,due_time,processing_time,priority,material_zone
  240A-1,240A,0,120,15,Normal,Material

Output:
- agv_trajectory_workstation.csv
"""

import csv
import heapq
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

GRID_SIZE = (21, 21)
POSITION_FILE = "agv_position_workstation.csv"
TASK_FILE = "agv_task_workstation.csv"
TRAJECTORY_FILE = "agv_trajectory_workstation.csv"

State = Tuple[int, int, int, int]  # x, y, timestamp, pitch

MOVES = [
    (1, 0, 0),    # right, +X
    (-1, 0, 180), # left, -X
    (0, 1, 90),   # up, +Y
    (0, -1, 270), # down, -Y
]


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def around(center: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = center
    candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [(cx, cy) for cx, cy in candidates if 1 <= cx <= GRID_SIZE[0] and 1 <= cy <= GRID_SIZE[1]]


@dataclass
class Task:
    task_id: str
    workstation: str
    release_time: int
    due_time: int
    processing_time: int
    priority: str = "Normal"
    material_zone: str = "Material"
    assigned: bool = False
    done: bool = False
    service_start: Optional[int] = None
    service_end: Optional[int] = None
    delivered_at: Optional[int] = None


@dataclass
class AGV:
    name: str
    state: State
    task_id: Optional[str] = None
    path: List[State] = field(default_factory=list)
    steps: List[dict] = field(default_factory=list)

    def is_free(self, now: int) -> bool:
        return self.task_id is None and (not self.path or now >= self.path[-1][2])


class Environment:
    def __init__(self, material_zones: Dict[str, Tuple[int, int]], workstations: Dict[str, Tuple[int, int]]):
        self.material_zones = material_zones
        self.workstations = workstations
        self.moving_obstacles: Dict[str, List[State]] = {}

    def static_obstacles(self) -> Set[Tuple[int, int]]:
        # Centers of material zones and workstations are not passable. AGVs use adjacent cells.
        return set(self.material_zones.values()) | set(self.workstations.values())

    def occupied_at(self, t: int, except_agv: Optional[str] = None) -> Set[Tuple[int, int]]:
        occ = set()
        for name, path in self.moving_obstacles.items():
            if name == except_agv or not path:
                continue
            if t < len(path):
                occ.add(path[t][:2])
            else:
                occ.add(path[-1][:2])
        return occ

    def is_swap(self, agv_name: str, current: State, nxt: State) -> bool:
        cx, cy, ct, _ = current
        nx, ny, nt, _ = nxt
        for name, path in self.moving_obstacles.items():
            if name == agv_name or not path:
                continue
            if ct < len(path):
                other_now = path[ct][:2]
            else:
                other_now = path[-1][:2]
            if nt < len(path):
                other_next = path[nt][:2]
            else:
                other_next = path[-1][:2]
            if other_now == (nx, ny) and other_next == (cx, cy):
                return True
        return False

    def valid_cell(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 1 <= x <= GRID_SIZE[0] and 1 <= y <= GRID_SIZE[1] and pos not in self.static_obstacles()


class Simulation:
    def __init__(self, agvs: List[AGV], tasks: List[Task], env: Environment):
        self.agvs = agvs
        self.tasks = tasks
        self.env = env
        self.time = 0
        self.workstation_available: Dict[str, int] = {name: 0 for name in env.workstations}
        self.task_by_id = {t.task_id: t for t in tasks}

        for agv in self.agvs:
            agv.path = [agv.state]
            agv.steps = [self.make_step(agv, agv.state, loaded=False, task=None)]
            self.env.moving_obstacles[agv.name] = [agv.state]

    def make_step(self, agv: AGV, state: State, loaded: bool, task: Optional[Task], waiting_for_station: bool = False) -> dict:
        return {
            "timestamp": state[2],
            "name": agv.name,
            "X": state[0],
            "Y": state[1],
            "pitch": state[3],
            "loaded": str(bool(loaded)).lower(),
            "destination": task.workstation if task else "",
            "Emergency": str(task.priority == "Urgent").lower() if task else "false",
            "task-id": task.task_id if task else "",
            "workstation": task.workstation if task else "",
            "window_start": task.release_time if task else "",
            "window_end": task.due_time if task else "",
            "service_start": task.service_start if task and task.service_start is not None else "",
            "service_end": task.service_end if task and task.service_end is not None else "",
            "late": str(bool(task and task.delivered_at is not None and task.delivered_at > task.due_time)).lower() if task else "false",
            "waiting_for_station": str(bool(waiting_for_station)).lower(),
        }

    def open_tasks(self) -> List[Task]:
        # Available tasks: released, not assigned, not done.
        # If desired, tasks whose release_time is in the future could also be planned in advance;
        # here we only dispatch released work to keep behavior simple and predictable.
        return [t for t in self.tasks if not t.assigned and not t.done and t.release_time <= self.time]

    def all_done(self) -> bool:
        return all(t.done for t in self.tasks) and all(agv.task_id is None for agv in self.agvs)

    def select_assignment(self, agv: AGV, open_tasks: List[Task]) -> Optional[Task]:
        if not open_tasks:
            return None
        material = next(iter(self.env.material_zones.values()))
        def score(t: Task) -> Tuple[int, int, str]:
            ws = self.env.workstations[t.workstation]
            pickup = min(around(material), key=lambda p: manhattan(agv.state[:2], p))
            delivery = min(around(ws), key=lambda p: manhattan(pickup, p))
            base = manhattan(agv.state[:2], pickup) + manhattan(pickup, delivery)
            urgency_bonus = -20 if t.priority == "Urgent" else 0
            lateness_risk = max(0, self.time + base - t.due_time)
            station_wait = max(0, self.workstation_available[t.workstation] - self.time)
            return (base + urgency_bonus + lateness_risk * 3 + station_wait, t.due_time, t.task_id)
        return min(open_tasks, key=score)

    def neighbors(self, agv_name: str, state: State):
        x, y, t, d = state
        static = self.env.static_obstacles()
        occ_next = self.env.occupied_at(t + 1, except_agv=agv_name)
        occ_two = self.env.occupied_at(t + 2, except_agv=agv_name)

        # Wait in place.
        wait_state = (x, y, t + 1, d)
        if (x, y) not in static and (x, y) not in occ_next:
            yield wait_state, 1, []

        for dx, dy, nd in MOVES:
            nx, ny = x + dx, y + dy
            if not self.env.valid_cell((nx, ny)):
                continue
            if d == nd:
                nxt = (nx, ny, t + 1, nd)
                if (nx, ny) not in occ_next and not self.env.is_swap(agv_name, state, nxt):
                    yield nxt, 1, []
            else:
                # Rotation takes 1 second, then movement takes 1 second.
                turn = (x, y, t + 1, nd)
                move = (nx, ny, t + 2, nd)
                if (x, y) not in occ_next and (nx, ny) not in occ_two:
                    if not self.env.is_swap(agv_name, turn, move):
                        yield move, 2, [turn]

    def astar_to_any(self, agv_name: str, start: State, goals: List[Tuple[int, int]], max_expand: int = 20000) -> List[State]:
        goals_set = set(goals)
        pq = []
        heapq.heappush(pq, (0, 0, start, [start]))
        visited = set()
        expansions = 0
        while pq and expansions < max_expand:
            _, cost, cur, path = heapq.heappop(pq)
            expansions += 1
            key = cur[:3]
            if key in visited:
                continue
            visited.add(key)
            if cur[:2] in goals_set:
                return path
            for nxt, step_cost, extra_states in self.neighbors(agv_name, cur):
                if nxt[:3] in visited:
                    continue
                new_path = path + extra_states + [nxt]
                h = min(manhattan(nxt[:2], g) for g in goals)
                heapq.heappush(pq, (cost + step_cost + h, cost + step_cost, nxt, new_path))
        return []

    def plan_task(self, agv: AGV, task: Task) -> Tuple[List[State], List[dict]]:
        material_pos = self.env.material_zones.get(task.material_zone) or next(iter(self.env.material_zones.values()))
        workstation_pos = self.env.workstations[task.workstation]
        pickup_goals = around(material_pos)
        delivery_goals = around(workstation_pos)

        start = agv.state
        path_to_pickup = self.astar_to_any(agv.name, start, pickup_goals)
        if not path_to_pickup:
            return [], []

        # Loading takes 1 second at pickup point.
        pickup_arrival = path_to_pickup[-1]
        pickup_loaded = (pickup_arrival[0], pickup_arrival[1], pickup_arrival[2] + 1, pickup_arrival[3])

        path_to_delivery = self.astar_to_any(agv.name, pickup_loaded, delivery_goals)
        if not path_to_delivery:
            return [], []

        full_path = path_to_pickup + [pickup_loaded] + path_to_delivery[1:]
        delivery_arrival = full_path[-1]
        arrival_time = delivery_arrival[2]

        # Workstation time-window/resource constraint: one workstation processes only one task at a time.
        service_start = max(arrival_time + 1, task.release_time, self.workstation_available[task.workstation])
        service_end = service_start + task.processing_time
        task.service_start = service_start
        task.service_end = service_end
        task.delivered_at = service_start

        # If AGV arrives before the workstation can accept the task, wait loaded at the unloading cell.
        while full_path[-1][2] < service_start - 1:
            last = full_path[-1]
            full_path.append((last[0], last[1], last[2] + 1, last[3]))

        unload_state = (full_path[-1][0], full_path[-1][1], full_path[-1][2] + 1, full_path[-1][3])
        full_path.append(unload_state)

        steps = []
        pickup_time = pickup_loaded[2]
        unload_time = unload_state[2]
        for st in full_path:
            if st[2] < pickup_time:
                steps.append(self.make_step(agv, st, loaded=False, task=None))
            elif st[2] < unload_time:
                waiting = st[2] >= arrival_time and st[2] < service_start
                steps.append(self.make_step(agv, st, loaded=True, task=task, waiting_for_station=waiting))
            else:
                steps.append(self.make_step(agv, st, loaded=False, task=None))
        return full_path, steps

    def assign_and_plan(self):
        free_agvs = [a for a in self.agvs if a.is_free(self.time)]
        for agv in free_agvs:
            open_tasks = self.open_tasks()
            if not open_tasks:
                return
            task = self.select_assignment(agv, open_tasks)
            if not task:
                return
            path, steps = self.plan_task(agv, task)
            if not path:
                # Avoid trying the same impossible task forever at the same timestamp.
                task.release_time += 1
                continue

            task.assigned = True
            task.done = True
            self.workstation_available[task.workstation] = task.service_end or self.workstation_available[task.workstation]
            agv.task_id = task.task_id
            agv.path = path
            agv.steps.extend(steps[1:])
            self.env.moving_obstacles[agv.name] = path

    def update_agv_states(self):
        for agv in self.agvs:
            path = self.env.moving_obstacles.get(agv.name, agv.path)
            if self.time < len(path):
                agv.state = path[self.time]
            elif path:
                last = path[-1]
                agv.state = (last[0], last[1], self.time, last[3])
                path.append(agv.state)
                self.env.moving_obstacles[agv.name] = path
                agv.steps.append(self.make_step(agv, agv.state, loaded=False, task=None))
            if agv.task_id:
                task = self.task_by_id[agv.task_id]
                if task.service_end is not None and self.time >= task.service_end:
                    agv.task_id = None

    def step(self):
        self.assign_and_plan()
        self.time += 1
        self.update_agv_states()

    def run(self, max_time: int = 2000):
        while not self.all_done() and self.time < max_time:
            self.step()
        if not self.all_done():
            print(f"Warning: simulation stopped at max_time={max_time}; unfinished tasks exist.")

    def write_trajectory(self, filename: str):
        rows = []
        for agv in self.agvs:
            rows.extend(agv.steps)
        rows.sort(key=lambda r: (int(r["timestamp"]), r["name"]))
        fieldnames = [
            "timestamp", "name", "X", "Y", "pitch", "loaded", "destination", "Emergency", "task-id",
            "workstation", "window_start", "window_end", "service_start", "service_end", "late", "waiting_for_station"
        ]
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def load_positions(path: str):
    material_zones = {}
    workstations = {}
    agvs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typ = row["type"].strip()
            name = row["name"].strip()
            x, y = int(row["x"]), int(row["y"])
            if typ == "material_zone":
                material_zones[name] = (x, y)
            elif typ in ("workstation", "end_point"):
                workstations[name] = (x, y)
            elif typ == "agv":
                agvs.append(AGV(name=name, state=(x, y, 0, int(row["pitch"]))))
    if not material_zones:
        raise ValueError("No material_zone found in position file.")
    if not workstations:
        raise ValueError("No workstation found in position file.")
    return material_zones, workstations, agvs


def load_tasks(path: str) -> List[Task]:
    tasks = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(Task(
                task_id=row["task_id"].strip(),
                workstation=row["workstation"].strip(),
                release_time=int(row.get("release_time", 0) or 0),
                due_time=int(row.get("due_time", 999999) or 999999),
                processing_time=int(row.get("processing_time", 1) or 1),
                priority=row.get("priority", "Normal").strip() or "Normal",
                material_zone=row.get("material_zone", "Material").strip() or "Material",
            ))
    tasks.sort(key=lambda t: (t.release_time, t.due_time, t.workstation, t.task_id))
    return tasks


def check_trajectory_conflicts(filename: str) -> bool:
    by_t: Dict[int, Dict[str, Tuple[int, int]]] = {}
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            t = int(row["timestamp"])
            by_t.setdefault(t, {})[row["name"]] = (int(row["X"]), int(row["Y"]))
    has_conflict = False
    for t, positions in sorted(by_t.items()):
        seen = {}
        for name, pos in positions.items():
            if pos in seen:
                print(f"Collision at time {t}: {name} and {seen[pos]} at {pos}")
                has_conflict = True
            seen[pos] = name
        if t + 1 in by_t:
            for a, pos_a in positions.items():
                for b, pos_b in positions.items():
                    if a >= b:
                        continue
                    if a in by_t[t + 1] and b in by_t[t + 1]:
                        if by_t[t + 1][a] == pos_b and by_t[t + 1][b] == pos_a:
                            print(f"Swap conflict at time {t}-{t+1}: {a} <-> {b}")
                            has_conflict = True
    if not has_conflict:
        print("No collision or swap conflict found.")
    return has_conflict


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    pos_file = os.path.join(base, POSITION_FILE)
    task_file = os.path.join(base, TASK_FILE)
    out_file = os.path.join(base, TRAJECTORY_FILE)
    material_zones, workstations, agvs = load_positions(pos_file)
    tasks = load_tasks(task_file)
    env = Environment(material_zones, workstations)
    sim = Simulation(agvs, tasks, env)
    sim.run()
    sim.write_trajectory(out_file)
    check_trajectory_conflicts(out_file)
    print(f"Trajectory saved to {out_file}")


if __name__ == "__main__":
    main()
