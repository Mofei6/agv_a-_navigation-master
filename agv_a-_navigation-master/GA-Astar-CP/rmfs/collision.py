from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from .config import AGV, Task, Params, Coord
from .layout import WarehouseLayout
from .astar import astar

@dataclass
class Segment:
    task_id: int
    agv_id: int
    label: str
    loaded: bool
    path: List[Coord]
    start_time: int = 0
    end_time: int = 0

@dataclass
class EvalResult:
    makespan_picker: int
    makespan_agv: int
    total_cost: float
    agv_task_sequences: Dict[int, List[int]]
    segments: Dict[int, List[Segment]]
    completion_by_task: Dict[int, int]
    arrival_station_by_task: Dict[int, int]
    waiting_time: int

class ReservationTable:
    def __init__(self):
        self.node_time: Set[Tuple[Coord, int]] = set()
        self.edge_time: Set[Tuple[Coord, Coord, int]] = set()

    def can_place(self, path: List[Coord], t0: int) -> bool:
        for k, p in enumerate(path):
            t = t0 + k
            if (p, t) in self.node_time:
                return False
            if k > 0:
                a, b = path[k - 1], p
                # Block same edge and head-on edge conflicts.
                if (a, b, t - 1) in self.edge_time or (b, a, t - 1) in self.edge_time:
                    return False
        return True

    def reserve(self, path: List[Coord], t0: int):
        for k, p in enumerate(path):
            t = t0 + k
            self.node_time.add((p, t))
            if k > 0:
                self.edge_time.add((path[k - 1], p, t - 1))

class CPPlanner:
    """Lower-level CPP oracle: A* paths + reservation-table collision prediction."""
    def __init__(self, layout: WarehouseLayout, params: Params):
        self.layout = layout
        self.params = params
        self._path_cache = {}

    def _path(self, start, goal, loaded, turn_penalty):
        key = (start, goal, loaded, float(turn_penalty))
        if key not in self._path_cache:
            self._path_cache[key] = astar(self.layout, start, goal, loaded=loaded, turn_penalty=turn_penalty)
        return self._path_cache[key]

    def _place_with_wait(self, table: ReservationTable, path: List[Coord], earliest: int) -> Tuple[int, int]:
        t = earliest
        wait = 0
        while not table.can_place(path, t):
            t += 1
            wait += 1
            if wait > self.params.max_wait:
                # Add larger wait window if layout is congested.
                break
        table.reserve(path, t)
        return t, wait

    def evaluate(self, agvs: List[AGV], tasks: List[Task], chromosome: List[int]) -> EvalResult:
        """Backward-compatible evaluator for an AGV-id vector chromosome."""
        seqs: Dict[int, List[int]] = {a.agv_id: [] for a in agvs}
        for task, agv_id in zip(tasks, chromosome):
            seqs[agv_id].append(task.task_id)
        return self.evaluate_sequences(agvs, tasks, seqs)

    def evaluate_sequences(self, agvs: List[AGV], tasks: List[Task], seqs: Dict[int, List[int]]) -> EvalResult:
        """Evaluate a decoded TAS solution with lower-level CPP.

        `seqs` maps each AGV to the ordered list of task IDs it must execute.
        This is the true interface between upper-level TAS and lower-level CPP.
        """
        task_by_id = {t.task_id: t for t in tasks}

        max_len = max((len(v) for v in seqs.values()), default=0)
        table = ReservationTable()
        now = {a.agv_id: 0 for a in agvs}
        pos = {a.agv_id: a.start for a in agvs}
        segments: Dict[int, List[Segment]] = {a.agv_id: [] for a in agvs}
        completion_by_task: Dict[int, int] = {}
        arrival_station_by_task: Dict[int, int] = {}
        waiting_time = 0
        cost = 0.0

        for round_idx in range(max_len):
            candidates = [aid for aid, s in seqs.items() if round_idx < len(s)]
            # Dynamic priority approximation: available AGVs first; loaded-cost proxy second.
            candidates.sort(key=lambda aid: (now[aid], -len(seqs[aid])))
            for aid in candidates:
                task = task_by_id[seqs[aid][round_idx]]
                cur = pos[aid]
                p_task = task.loc
                p_in = self.params.workspace_entrance
                p_out = self.params.workspace_exit

                p1 = self._path(cur, p_task, False, self.params.turn_penalty)
                p2 = self._path(p_task, p_in, True, self.params.turn_penalty)
                fixed = list(self._path(p_in, self.params.station, True, 1.0))
                fixed += self._path(self.params.station, p_out, True, 1.0)[1:]
                p4 = self._path(p_out, p_task, True, self.params.turn_penalty)

                t1, w1 = self._place_with_wait(table, p1, now[aid]); waiting_time += w1
                s1 = Segment(task.task_id, aid, 'S1-S2 no-load', False, p1, t1, t1 + len(p1) - 1)

                # Lifting pod: occupy task node for 3 seconds.
                lift = [p1[-1]] * 4
                t_lift, wl = self._place_with_wait(table, lift, s1.end_time); waiting_time += wl
                lift_end = t_lift + len(lift) - 1

                t2, w2 = self._place_with_wait(table, p2, lift_end); waiting_time += w2
                s2 = Segment(task.task_id, aid, 'S2-S3 loaded', True, p2, t2, t2 + len(p2) - 1)
                arrival_station_by_task[task.task_id] = s2.end_time

                # Queue / picking workspace; task.pick_time approximates picker service.
                workspace_path = fixed + [fixed[-1]] * max(1, task.pick_time)
                t3, w3 = self._place_with_wait(table, workspace_path, s2.end_time); waiting_time += w3
                s3 = Segment(task.task_id, aid, 'S3-S4 workspace', True, workspace_path, t3, t3 + len(workspace_path) - 1)

                t4, w4 = self._place_with_wait(table, p4, s3.end_time); waiting_time += w4
                s4 = Segment(task.task_id, aid, 'S4-S2 return', True, p4, t4, t4 + len(p4) - 1)

                segments[aid].extend([s1, s2, s3, s4])
                completion_by_task[task.task_id] = s4.end_time
                now[aid] = s4.end_time
                pos[aid] = p_task

                # Cost by mode and wait/blocking surrogate.
                cost += (len(p1) - 1) * self.params.cost_noload_per_s
                cost += ((len(p2) - 1) + (len(workspace_path) - 1) + (len(p4) - 1)) * self.params.cost_loaded_per_s
                cost += (w1 + wl + w2 + w3 + w4) * self.params.cost_block_per_s

        makespan_agv = max(now.values()) if now else 0
        makespan_picker = max((arrival_station_by_task.get(t.task_id, 0) + t.pick_time for t in tasks), default=0)
        total_cost = cost + waiting_time * self.params.cost_wait_per_s
        return EvalResult(makespan_picker, makespan_agv, total_cost, seqs, segments,
                          completion_by_task, arrival_station_by_task, waiting_time)

# Attach a fast lower-level evaluator used inside the GA. It still calls improved A*
# for each segment and includes queue/conflict penalties, but avoids expensive
# second-by-second reservation tables for every individual.
def _evaluate_sequences_fast(self, agvs: List[AGV], tasks: List[Task], seqs: Dict[int, List[int]]) -> EvalResult:
    task_by_id = {t.task_id: t for t in tasks}
    now = {a.agv_id: 0 for a in agvs}
    pos = {a.agv_id: a.start for a in agvs}
    segments: Dict[int, List[Segment]] = {a.agv_id: [] for a in agvs}
    completion_by_task: Dict[int, int] = {}
    arrival_station_by_task: Dict[int, int] = {}
    station_available = 0
    recent_workspace_entries: List[int] = []
    waiting_time = 0
    cost = 0.0

    # Round-robin execution keeps the same dynamic-priority spirit as the detailed CPP.
    max_len = max((len(v) for v in seqs.values()), default=0)
    for round_idx in range(max_len):
        candidates = [aid for aid, s in seqs.items() if round_idx < len(s)]
        candidates.sort(key=lambda aid: (now[aid], -len(seqs[aid])))
        for aid in candidates:
            task = task_by_id[seqs[aid][round_idx]]
            p_task = task.loc
            p1 = self._path(pos[aid], p_task, False, self.params.turn_penalty)
            p2 = self._path(p_task, self.params.workspace_entrance, True, self.params.turn_penalty)
            fixed = list(self._path(self.params.workspace_entrance, self.params.station, True, 1.0))
            fixed += self._path(self.params.station, self.params.workspace_exit, True, 1.0)[1:]
            p4 = self._path(self.params.workspace_exit, p_task, True, self.params.turn_penalty)

            t1 = now[aid]
            s1 = Segment(task.task_id, aid, 'S1-S2 no-load', False, p1, t1, t1 + len(p1) - 1)
            lift_end = s1.end_time + 3
            s2_start = lift_end
            s2 = Segment(task.task_id, aid, 'S2-S3 loaded', True, p2, s2_start, s2_start + len(p2) - 1)

            arrival = s2.end_time
            # Queue prediction: single picking workspace capacity plus a small
            # congestion penalty when many AGVs arrive in a short time window.
            queue_wait = max(0, station_available - arrival)
            burst = sum(1 for t in recent_workspace_entries if abs(arrival - t) <= 4)
            conflict_wait = max(0, burst - 1) * 2
            wait = queue_wait + conflict_wait
            waiting_time += wait
            service_start = arrival + wait
            arrival_station_by_task[task.task_id] = service_start
            station_available = service_start + task.pick_time
            recent_workspace_entries.append(arrival)
            if len(recent_workspace_entries) > 20:
                recent_workspace_entries = recent_workspace_entries[-20:]

            workspace_path = fixed + [fixed[-1]] * max(1, task.pick_time)
            s3 = Segment(task.task_id, aid, 'S3-S4 workspace', True, workspace_path,
                         service_start, service_start + len(workspace_path) - 1)
            s4 = Segment(task.task_id, aid, 'S4-S2 return', True, p4,
                         s3.end_time, s3.end_time + len(p4) - 1)

            segments[aid].extend([s1, s2, s3, s4])
            completion_by_task[task.task_id] = s4.end_time
            now[aid] = s4.end_time
            pos[aid] = p_task
            cost += (len(p1) - 1) * self.params.cost_noload_per_s
            cost += ((len(p2) - 1) + (len(workspace_path) - 1) + (len(p4) - 1)) * self.params.cost_loaded_per_s
            cost += wait * (self.params.cost_wait_per_s + self.params.cost_block_per_s)

    makespan_agv = max(now.values()) if now else 0
    makespan_picker = max((arrival_station_by_task.get(t.task_id, 0) + t.pick_time for t in tasks), default=0)
    return EvalResult(makespan_picker, makespan_agv, cost, seqs, segments,
                      completion_by_task, arrival_station_by_task, waiting_time)

CPPlanner.evaluate_sequences_fast = _evaluate_sequences_fast
