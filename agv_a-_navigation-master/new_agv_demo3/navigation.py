import csv
import heapq
from collections import defaultdict
from copy import deepcopy

# 公平调度参数
LOOKAHEAD_TIME = 80

# 延迟平方惩罚，避免单个任务延迟过大
DELAY_SQUARE_WEIGHT = 2000

# 普通总延迟惩罚
DELAY_WEIGHT = 100000

# 任务老化奖励，任务越久越优先
TASK_AGING_WEIGHT = 3000

# 超过该延迟后，额外强惩罚
MAX_DELAY_SOFT_LIMIT = 80
MAX_DELAY_EXTRA_WEIGHT = 5000

# 行驶距离权重
DISTANCE_WEIGHT = 30

# 完成时间权重
FINISH_TIME_WEIGHT = 5

# 空闲 AGV 奖励
IDLE_AGV_BONUS = 500000
GRID_W = 20
GRID_H = 20
POSITION_FILE = "agv_position.csv"
TASK_FILE = "agv_task.csv"
TRAJECTORY_FILE = "agv_trajectory.csv"
PICKUP_MAX_WAIT_AT_PICKUP = 5
LOAD_DURATION = 2
UNLOAD_DURATION = 2
PICKUP_MAX_EARLY_ARRIVAL = 5

# 调度目标权重
DELAY_WEIGHT = 100000
DELIVERY_TIME_WEIGHT = 1000
IDLE_AGV_BONUS = 500000
DISTANCE_WEIGHT = 50
AGV_AVAILABLE_TIME_WEIGHT = 500
FINISH_TIME_WEIGHT = 10

# 调度时向前看的任务数量，避免只按任务表顺序导致空闲 AGV 没任务可接
SCHEDULING_LOOKAHEAD = 30
RETURN_HOME_THRESHOLD = 35
MAX_SEARCH_TIME = 1200
RESERVE_HORIZON = 5000
CBS_MAX_NODES = 120
PICKUP_MAX_EARLY_ARRIVAL = 5
LATE_PENALTY_WEIGHT = 100000
DISTANCE_WEIGHT = 100
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
PITCH_MAP = {(1, 0): 0, (-1, 0): 180, (0, 1): 90, (0, -1): 270, (0, 0): None}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(pos):
    for dx, dy in DIRS:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 1 <= nx <= GRID_W and 1 <= ny <= GRID_H:
            yield (nx, ny)


def empty_constraints():
    return defaultdict(lambda: {"vertex": set(), "edge": set()})


def clone_constraints(constraints):
    copied = empty_constraints()
    for agv_id, value in constraints.items():
        copied[agv_id]["vertex"] = set(value["vertex"])
        copied[agv_id]["edge"] = set(value["edge"])
    return copied


def constraints_signature(constraints):
    items = []
    for agv_id in sorted(constraints.keys(), key=lambda x: int(x)):
        vertices = tuple(sorted(constraints[agv_id]["vertex"], key=lambda x: (x[1], x[0][0], x[0][1])))
        edges = tuple(sorted(constraints[agv_id]["edge"], key=lambda x: (x[2], x[0][0], x[0][1], x[1][0], x[1][1])))
        items.append((agv_id, vertices, edges))
    return tuple(items)


class ReservationTable:
    """
    时空预约表。
    vertex[t][pos] = agv_id
    edge[t][(from_pos, to_pos)] = agv_id，其中 t 表示从 t-1 到 t 这个时间步。
    它用于单次排程中的动态障碍；CBS 约束则用于冲突修复分支。
    """
    def __init__(self, constraints=None):
        self.vertex = defaultdict(dict)
        self.edge = defaultdict(dict)
        self.constraints = constraints if constraints is not None else empty_constraints()

    def clone(self):
        copied = ReservationTable(self.constraints)
        copied.vertex = defaultdict(dict, {t: dict(v) for t, v in self.vertex.items()})
        copied.edge = defaultdict(dict, {t: dict(v) for t, v in self.edge.items()})
        return copied

    def clear_agv_from(self, agv_id, start_t, end_t=RESERVE_HORIZON):
        for t in list(self.vertex.keys()):
            if start_t <= t <= end_t:
                for pos in list(self.vertex[t].keys()):
                    if self.vertex[t][pos] == agv_id:
                        del self.vertex[t][pos]
                if not self.vertex[t]:
                    del self.vertex[t]
        for t in list(self.edge.keys()):
            if start_t <= t <= end_t:
                for edge in list(self.edge[t].keys()):
                    if self.edge[t][edge] == agv_id:
                        del self.edge[t][edge]
                if not self.edge[t]:
                    del self.edge[t]

    def violates_constraint(self, agv_id, prev_pos, pos, t):
        if (pos, t) in self.constraints[agv_id]["vertex"]:
            return True
        if (prev_pos, pos, t) in self.constraints[agv_id]["edge"]:
            return True
        return False

    def is_free(self, agv_id, prev_pos, pos, t):
        if self.violates_constraint(agv_id, prev_pos, pos, t):
            return False

        owner = self.vertex.get(t, {}).get(pos)
        if owner is not None and owner != agv_id:
            return False

        same_edge_owner = self.edge.get(t, {}).get((prev_pos, pos))
        if same_edge_owner is not None and same_edge_owner != agv_id:
            return False

        reverse_edge_owner = self.edge.get(t, {}).get((pos, prev_pos))
        if reverse_edge_owner is not None and reverse_edge_owner != agv_id:
            return False

        return True

    def can_reserve_path(self, agv_id, path):
        for idx, (pos, t) in enumerate(path):
            prev_pos = path[idx - 1][0] if idx > 0 else pos
            if not self.is_free(agv_id, prev_pos, pos, t):
                return False
        return True

    def reserve_path(self, agv_id, path):
        if not self.can_reserve_path(agv_id, path):
            raise RuntimeError(f"预约冲突: AGV{agv_id}, path={path[:3]}...{path[-3:]}")

        for idx, (pos, t) in enumerate(path):
            self.vertex[t][pos] = agv_id
            if idx > 0:
                prev_pos, _ = path[idx - 1]
                self.edge[t][(prev_pos, pos)] = agv_id

    def reserve_wait(self, agv_id, pos, start_t, end_t):
        path = [(pos, t) for t in range(start_t, end_t + 1)]
        self.reserve_path(agv_id, path)
        return path

    def can_hold(self, agv_id, pos, start_t, duration):
        for t in range(start_t, start_t + duration + 1):
            if not self.is_free(agv_id, pos, pos, t):
                return False
        return True


class Planner:
    def __init__(self, constraints=None):
        self.constraints = constraints if constraints is not None else empty_constraints()
        self.supply = None
        self.pickup = None
        self.drop_points = {}
        self.homes = {}
        self.tasks = []
        self.agv_states = {}
        self.task_results = []
        self.total_delay = 0
        self.total_distance = 0
        self.agv_tracks = defaultdict(list)

        # 任务执行统计
        self.task_results = []
        self.total_delay = 0
        self.total_distance = 0

        self.res = ReservationTable(self.constraints)
        self.load_inputs()

    def task_age(self, task):
        """
        任务老化时间。

        如果当前时间已经超过任务开始时间 window_start，
        且任务还未被调度，则说明该任务已经在等待物料。
        """
        now = self.current_system_time()
        return max(0, now - task["window_start"])

    def current_system_time(self):
        """
        当前系统参考时间。
        用最早可用 AGV 的时间作为调度参考时间。
        """
        return min(state["time"] for state in self.agv_states.values())

    def write_task_results(self):
        fields = [
            "task_id",
            "destination_id",
            "destination_label",
            "material",
            "agv_id",
            "window_start",
            "window_end",
            "delivery_time",
            "delay",
            "distance",
        ]


        with open("agv_task_result.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.task_results)

        with open("agv_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_delay", self.total_delay])
            writer.writerow(["total_distance", self.total_distance])
            writer.writerow(["task_count", len(self.task_results)])

            max_delay = max((r["delay"] for r in self.task_results), default=0)
            avg_delay = (
                sum(r["delay"] for r in self.task_results) / len(self.task_results)
                if self.task_results else 0
            )

            writer.writerow(["max_delay", max_delay])
            writer.writerow(["avg_delay", round(avg_delay, 2)])

    def is_agv_idle(self, agv_id):
        """
        判断 AGV 当前是否处于等待区空闲状态。
        """
        state = self.agv_states[agv_id]
        return state["pos"] == state["home"]
    def path_distance(self, path):
        """
        计算路径实际行驶距离。
        原地等待不算行驶距离。
        """
        if not path or len(path) <= 1:
            return 0

        dist = 0
        for i in range(1, len(path)):
            if path[i][0] != path[i - 1][0]:
                dist += 1
        return dist

    def plan_distance(self, segments):
        """
        计算一个任务方案的总行驶成本。
        """
        total = 0
        for _, _, _, _, _, _, path in segments:
            total += self.path_distance(path)
        return total
    def load_inputs(self):
        with open(POSITION_FILE, "r", encoding="utf-8-sig") as f:
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

        if self.supply is None:
            raise ValueError("agv_position.csv 缺少 supply")
        if self.pickup is None:
            raise ValueError("agv_position.csv 缺少 pickup_slot")
        if len(self.homes) < 1:
            raise ValueError("当前版本要求 5 台 AGV")

        self.home_owner = {
            home["pos"]: agv_id
            for agv_id, home in self.homes.items()
        }

        with open(TASK_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                task = {
                    "task_id": row["task_id"].strip(),
                    "destination_id": row["destination_id"].strip(),
                    "destination_label": row["destination_label"].strip(),
                    "material": row.get("material", "").strip(),
                    "release_time": int(row["release_time"]),
                    "window_start": int(row["window_start"]),
                    "window_end": int(row["window_end"]),
                    "priority": row.get("priority", "Normal").strip() or "Normal",
                }
                if task["destination_id"] not in self.drop_points:
                    raise ValueError(f"任务 {task['task_id']} 的目的地不存在: {task['destination_id']}")
                if task["window_end"] <= task["window_start"]:
                    raise ValueError(f"任务 {task['task_id']} 时间窗非法")
                self.tasks.append(task)

        self.tasks.sort(key=lambda x: (x["release_time"], x["window_start"], x["task_id"]))

        for agv_id, home in self.homes.items():
            self.agv_states[agv_id] = {
                "id": agv_id,
                "pos": home["pos"],
                "time": 0,
                "pitch": home["pitch"],
                "home": home["pos"],
            }
            self.agv_tracks[agv_id].append(
                self.make_row(0, agv_id, home["pos"], home["pitch"], False, "", "", "", "idle")
            )
            self.res.reserve_wait(agv_id, home["pos"], 0, RESERVE_HORIZON)

    def static_blocked(self, extra_blocked=None):
        """
        静态障碍：
        1. 物料区不可穿越；
        2. 工位本体不可进入；
        3. home 不作为静态障碍，是否占用由预约表决定。
        """
        blocked = {self.supply["pos"]}

        for point in self.drop_points.values():
            blocked.add(point["pos"])

        if extra_blocked:
            blocked.update(extra_blocked)

        return blocked

    def traversable(self, pos, extra_blocked=None):
        return (
                1 <= pos[0] <= GRID_W
                and 1 <= pos[1] <= GRID_H
                and pos not in self.static_blocked(extra_blocked)
        )

    def delivery_goals(self, destination_id, extra_blocked=None):
        """
        卸货目标不是工位格子本身，而是工位旁边的可行驶道路格。
        这里也要排除其它 AGV 的 home。
        """
        target = self.drop_points[destination_id]["pos"]
        goals = []

        for nb in neighbors(target):
            if nb != target and self.traversable(nb, extra_blocked):
                goals.append(nb)

        if not goals:
            raise RuntimeError(f"任务目的地 {destination_id} 周边无可达道路")

        return goals

    def heuristic(self, pos, goals):
        return min(manhattan(pos, goal) for goal in goals)

    def astar_time(self, res, agv_id, start_pos, start_time, goals, earliest_goal_time=0, extra_blocked=None,
                   goal_hold=0, max_early_arrival=None):
        """
        时空 A*。
        max_wait_at_goal 用于避免 AGV 过早进入目标格等待。
        例如去取料位时：
            earliest_goal_time = release_time
            max_wait_at_goal = 5
        表示 AGV 最多只能提前 5 秒到达取料位；
        如果会更早到达，则 A* 会在路上/等待区等待，而不是进入取料位堵住取料口。
        """
        blocked = self.static_blocked(extra_blocked)
        goals = set(goals)
        start = (start_pos, start_time)
        pq = [(self.heuristic(start_pos, goals), 0, start_pos, start_time)]
        parent = {start: None}
        gscore = {start: 0}
        while pq:
            _, cost, pos, t = heapq.heappop(pq)
            state = (pos, t)

            if pos in goals and t >= earliest_goal_time and res.can_hold(agv_id, pos, t, goal_hold):
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
                nt = t + 1
                if nxt in blocked and nxt not in goals:
                    continue
                    # 关键修改：不允许太早进入目标格。
                    # 对取料位来说，这能防止 AGV 提前很久占住取料区。
                if nxt in goals and max_early_arrival is not None:
                    if nt < earliest_goal_time - max_early_arrival:
                        continue

                if not res.is_free(agv_id, pos, nxt, nt):
                    continue

                nstate = (nxt, nt)
                ncost = cost + 1
                if nstate not in gscore or ncost < gscore[nstate]:
                    gscore[nstate] = ncost
                    parent[nstate] = state
                    priority = ncost + self.heuristic(nxt, goals) + max(0, earliest_goal_time - nt)
                    heapq.heappush(pq, (priority, ncost, nxt, nt))
        return None

    def wait_path(self, pos, start_t, duration):
        return [(pos, t) for t in range(start_t, start_t + duration + 1)]

    def path_pitch(self, prev_pos, curr_pos, last_pitch):
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        return PITCH_MAP.get((dx, dy), last_pitch) or last_pitch

    def make_row(self, t, agv_id, pos, pitch, loaded, material, dest_label, task_id, status, dest_id=""):
        return {
            "timestamp": t,
            "name": agv_id,
            "X": pos[0],
            "Y": pos[1],
            "pitch": pitch,
            "loaded": "true" if loaded else "false",
            "material": material,
            "destination_label": dest_label,
            "destination_id": dest_id,
            "task_id": task_id,
            "status": status,
            "pos": pos,
        }

    def append_path_rows(self, agv_id, path, loaded, material, dest_label, dest_id, task_id, status, skip_first=True):
        state = self.agv_states[agv_id]
        rows = self.agv_tracks[agv_id]
        last_pitch = state["pitch"]
        prev_pos = rows[-1]["pos"] if rows else path[0][0]
        iterable = path[1:] if skip_first else path

        for pos, t in iterable:
            pitch = self.path_pitch(prev_pos, pos, last_pitch)
            rows.append(self.make_row(t, agv_id, pos, pitch, loaded, material, dest_label, task_id, status, dest_id))
            prev_pos = pos
            last_pitch = pitch
        state["pitch"] = last_pitch
        state["pos"] = path[-1][0]
        state["time"] = path[-1][1]

    def candidate_plan(self, agv_id, task):
        agv = self.agv_states[agv_id]
        cur_pos = agv["pos"]
        cur_time = agv["time"]

        temp_res = self.res.clone()
        temp_res.clear_agv_from(agv_id, cur_time, RESERVE_HORIZON)

        segments = []
        plan_start_pos = cur_pos
        plan_start_time = cur_time

        if (
                cur_pos != agv["home"]
                and (
                task["release_time"] - cur_time > PICKUP_MAX_WAIT_AT_PICKUP
                or task["window_start"] - cur_time >= RETURN_HOME_THRESHOLD
        )
        ):
            path_home = self.astar_time(temp_res, agv_id, cur_pos, cur_time, [agv["home"]])
            if path_home is None:
                return None
            temp_res.reserve_path(agv_id, path_home)
            segments.append(("return_home", False, "", "", "", "", path_home))
            plan_start_pos, plan_start_time = path_home[-1]
        # 如果距离任务释放还有较长时间，则不要在取料区附近乱动等待。
        # 优先回等待区；如果已经在等待区，则在等待区等待。
        if task["release_time"] - plan_start_time > PICKUP_MAX_EARLY_ARRIVAL:
            if plan_start_pos != agv["home"]:
                path_home = self.astar_time(
                    temp_res,
                    agv_id,
                    plan_start_pos,
                    plan_start_time,
                    [agv["home"]],
                )
                if path_home is None:
                    return None

                temp_res.reserve_path(agv_id, path_home)
                segments.append(("return_home", False, "", "", "", "", path_home))
                plan_start_pos, plan_start_time = path_home[-1]

            # 在等待区等到接近 release_time，再出发。
            # 粗略估算去取料位距离，避免太早出发。
            dist_to_pickup = manhattan(plan_start_pos, self.pickup["pos"])
            desired_depart = max(
                plan_start_time,
                task["release_time"] - PICKUP_MAX_EARLY_ARRIVAL - dist_to_pickup
            )

            if desired_depart > plan_start_time:
                wait_path = [(plan_start_pos, t) for t in range(plan_start_time, desired_depart + 1)]
                if not temp_res.can_reserve_path(agv_id, wait_path):
                    return None

                temp_res.reserve_path(agv_id, wait_path)
                segments.append(("idle", False, "", "", "", "", wait_path))
                plan_start_time = desired_depart
        path_to_pick = self.astar_time(
            temp_res,
            agv_id,
            plan_start_pos,
            plan_start_time,
            [self.pickup["pos"]],
            earliest_goal_time=task["release_time"],
            goal_hold=LOAD_DURATION,
            max_early_arrival=PICKUP_MAX_EARLY_ARRIVAL,
        )
        if path_to_pick is None:
            return None
        temp_res.reserve_path(agv_id, path_to_pick)

        pickup_arrive = path_to_pick[-1][1]
        pickup_depart = pickup_arrive + LOAD_DURATION
        load_path = self.wait_path(self.pickup["pos"], pickup_arrive, LOAD_DURATION)
        temp_res.reserve_path(agv_id, load_path)

        segments.append(("to_pickup", False, "", "", "", "", path_to_pick))
        segments.append(("loading", False, "", "", "", "", load_path))

        extra_after_pickup = {self.pickup["pos"]}
        goals = self.delivery_goals(task["destination_id"], extra_after_pickup)
        path_to_drop = self.astar_time(
            temp_res,
            agv_id,
            self.pickup["pos"],
            pickup_depart,
            goals,
            extra_blocked=extra_after_pickup,
            goal_hold=UNLOAD_DURATION,
        )
        if path_to_drop is None:
            return None
        temp_res.reserve_path(agv_id, path_to_drop)

        drop_arrive = path_to_drop[-1][1]
        unload_path = self.wait_path(path_to_drop[-1][0], drop_arrive, UNLOAD_DURATION)
        temp_res.reserve_path(agv_id, unload_path)

        segments.append(("delivering", True, task["material"], task["destination_label"], task["destination_id"], task["task_id"], path_to_drop))
        segments.append(("unloading", True, task["material"], task["destination_label"], task["destination_id"], task["task_id"], unload_path))
        distance = self.plan_distance(segments)
        return {
            "segments": segments,
            "finish_time": unload_path[-1][1],
            "finish_pos": unload_path[-1][0],
            "delivery_time": drop_arrive,
            "distance": distance,
        }

    def commit_plan(self, agv_id, plan, task):
        cur_time = self.agv_states[agv_id]["time"]
        self.res.clear_agv_from(agv_id, cur_time, RESERVE_HORIZON)

        for status, loaded, material, dest_label, dest_id, task_id, path in plan["segments"]:
            self.res.reserve_path(agv_id, path)
            self.append_path_rows(
                agv_id,
                path,
                loaded,
                material,
                dest_label,
                dest_id,
                task_id,
                status,
                skip_first=True
            )

        final_pos = self.agv_states[agv_id]["pos"]
        final_time = self.agv_states[agv_id]["time"]
        self.res.reserve_wait(agv_id, final_pos, final_time, RESERVE_HORIZON)

        delay = max(0, plan["delivery_time"] - task["window_start"])
        distance = plan.get("distance", 0)

        self.total_delay += delay
        self.total_distance += distance

        self.task_results.append({
            "task_id": task["task_id"],
            "destination_id": task["destination_id"],
            "destination_label": task["destination_label"],
            "material": task["material"],
            "agv_id": agv_id,
            "window_start": task["window_start"],
            "window_end": task["window_end"],
            "delivery_time": plan["delivery_time"],
            "delay": delay,
            "distance": distance,
        })

    def candidate_score(self, agv_id, task, plan):
        """
        公平调度评分函数。

        目标：
        1. 避免单个任务延迟特别大；
        2. 优先处理已经等待很久的任务；
        3. 在延迟接近时减少行驶距离；
        4. 鼓励空闲 AGV 参与执行紧急任务。
        """
        delivery_time = plan["delivery_time"]
        finish_time = plan["finish_time"]
        distance = plan.get("distance", 0)

        # 预测延迟：送达时间晚于任务开始时间的部分
        delay = max(0, delivery_time - task["window_start"])

        # 当前任务已经等待了多久
        age = self.task_age(task)

        # 延迟平方惩罚：避免出现某些任务延迟几百秒
        delay_square_penalty = delay * delay * DELAY_SQUARE_WEIGHT

        # 普通延迟惩罚：仍然保持总延迟尽量小
        delay_penalty = delay * DELAY_WEIGHT

        # 超过软上限后的额外惩罚
        extra_delay = max(0, delay - MAX_DELAY_SOFT_LIMIT)
        extra_delay_penalty = extra_delay * extra_delay * MAX_DELAY_EXTRA_WEIGHT

        # 任务老化奖励：任务越久没被处理，分数越低，越容易被选中
        aging_bonus = age * TASK_AGING_WEIGHT

        score = (
                delay_square_penalty
                + delay_penalty
                + extra_delay_penalty
                + distance * DISTANCE_WEIGHT
                + finish_time * FINISH_TIME_WEIGHT
                - aging_bonus
        )

        # 如果任务已经延迟，且当前 AGV 空闲，鼓励马上派空闲 AGV 去做
        if age > 0 and self.is_agv_idle(agv_id):
            score -= IDLE_AGV_BONUS

        return score

    def choose_best_task_and_agv(self, unscheduled_tasks):
        """
        从未调度任务中选择一个 task-agv 组合。

        改进点：
        1. 已延迟任务必须优先进入候选集合；
        2. 任务越早出现，越优先；
        3. 避免新任务不断插队，导致旧任务延迟过大。
        """
        if not unscheduled_tasks:
            return None, None, None

        now = self.current_system_time()

        # 已经延迟的任务
        overdue_tasks = [
            task for task in unscheduled_tasks
            if task["window_start"] <= now
        ]

        # 即将到期或已经释放的任务
        near_tasks = [
            task for task in unscheduled_tasks
            if task["release_time"] <= now + LOOKAHEAD_TIME
               or task["window_start"] <= now + LOOKAHEAD_TIME
        ]

        # 如果存在已经延迟的任务，则优先只在这些任务里选
        if overdue_tasks:
            candidate_tasks = sorted(
                overdue_tasks,
                key=lambda x: (
                    x["window_start"],  # 越早出现越优先
                    x["release_time"],
                    x["task_id"]
                )
            )
        else:
            candidate_tasks = sorted(
                near_tasks if near_tasks else unscheduled_tasks,
                key=lambda x: (
                    x["window_start"],
                    x["release_time"],
                    x["task_id"]
                )
            )

        # 防止候选任务过多导致计算太慢
        candidate_tasks = candidate_tasks[:SCHEDULING_LOOKAHEAD]

        best = None

        # AGV 顺序：空闲 AGV 优先，其次当前可用时间早的 AGV
        agv_order = sorted(
            self.agv_states.keys(),
            key=lambda aid: (
                0 if self.is_agv_idle(aid) else 1,
                self.agv_states[aid]["time"],
                int(aid)
            )
        )

        for task in candidate_tasks:
            for agv_id in agv_order:
                plan = self.candidate_plan(agv_id, task)
                if plan is None:
                    continue

                delay = max(0, plan["delivery_time"] - task["window_start"])
                age = self.task_age(task)
                score = self.candidate_score(agv_id, task, plan)

                key = (
                    score,
                    delay,
                    -age,  # 等待越久越优先
                    task["window_start"],  # 越早出现越优先
                    plan.get("distance", 0),
                    plan["finish_time"],
                    int(agv_id),
                )

                if best is None or key < best[0]:
                    best = (key, task, agv_id, plan)

        if best is None:
            raise RuntimeError("当前所有候选任务均无法规划可行路径")

        _, task, agv_id, plan = best
        return task, agv_id, plan

    def finalize_to_home(self):
        for agv_id, agv in self.agv_states.items():
            if agv["pos"] == agv["home"]:
                continue
            self.res.clear_agv_from(agv_id, agv["time"], RESERVE_HORIZON)
            path = self.astar_time(
                self.res,
                agv_id,
                agv["pos"],
                agv["time"],
                [agv["home"]],
                extra_blocked={self.pickup["pos"]},
            )
            if path is None:
                continue
            self.res.reserve_path(agv_id, path)
            self.append_path_rows(agv_id, path, False, "", "", "", "", "return_home", skip_first=True)
            self.res.reserve_wait(agv_id, agv["home"], self.agv_states[agv_id]["time"], RESERVE_HORIZON)

    def fill_idle_until(self, target_time):
        for agv_id, rows in self.agv_tracks.items():
            last = rows[-1]
            t = last["timestamp"]
            pos = last["pos"]
            pitch = last["pitch"]
            while t < target_time:
                t += 1
                rows.append(self.make_row(t, agv_id, pos, pitch, False, "", "", "", "idle"))

    def build_rows(self):
        fields = ["timestamp", "name", "X", "Y", "pitch", "loaded", "material", "destination_label", "destination_id", "task_id", "status"]
        all_rows = []
        for rows in self.agv_tracks.values():
            seen = {}
            for row in rows:
                seen[row["timestamp"]] = row
            for row in seen.values():
                all_rows.append({k: row[k] for k in fields})
        all_rows.sort(key=lambda x: (x["timestamp"], int(x["name"])))
        return all_rows

    def write_trajectory(self):
        all_rows = self.build_rows()
        fields = ["timestamp", "name", "X", "Y", "pitch", "loaded", "material", "destination_label", "destination_id", "task_id", "status"]
        with open(TRAJECTORY_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)

    def detect_first_conflict(self):
        by_t = defaultdict(dict)
        for row in self.build_rows():
            t = int(row["timestamp"])
            by_t[t][row["name"]] = (int(row["X"]), int(row["Y"]))

        for t in sorted(by_t):
            seen = {}
            for agv_id, pos in by_t[t].items():
                if pos in seen:
                    return {
                        "type": "vertex",
                        "time": t,
                        "pos": pos,
                        "a1": seen[pos],
                        "a2": agv_id,
                    }
                seen[pos] = agv_id

            if t + 1 in by_t:
                ids = sorted(set(by_t[t].keys()) & set(by_t[t + 1].keys()), key=lambda x: int(x))
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        a1, a2 = ids[i], ids[j]
                        p1, p2 = by_t[t][a1], by_t[t][a2]
                        n1, n2 = by_t[t + 1][a1], by_t[t + 1][a2]
                        if p1 != p2 and p1 == n2 and p2 == n1:
                            return {
                                "type": "edge",
                                "time": t + 1,
                                "a1": a1,
                                "a2": a2,
                                "edge1": (p1, n1),
                                "edge2": (p2, n2),
                            }
        return None

    def solution_cost(self):
        all_rows = self.build_rows()
        makespan = max((int(r["timestamp"]) for r in all_rows), default=0)

        constraint_penalty = sum(
            len(v["vertex"]) + len(v["edge"])
            for v in self.constraints.values()
        )

        return (
                self.total_delay * DELAY_WEIGHT
                + self.total_distance * DISTANCE_WEIGHT
                + makespan
                + constraint_penalty * 10
        )

    def run_once(self):
        unscheduled_tasks = list(self.tasks)

        while unscheduled_tasks:
            task, agv_id, plan = self.choose_best_task_and_agv(unscheduled_tasks)
            self.commit_plan(agv_id, plan, task)
            unscheduled_tasks.remove(task)

        self.finalize_to_home()
        max_t = max(rows[-1]["timestamp"] for rows in self.agv_tracks.values())
        self.fill_idle_until(max_t)
        return self.detect_first_conflict()


def plan_with_cbs():
    root_constraints = empty_constraints()
    open_list = []
    counter = 0

    root = Planner(root_constraints)
    try:
        conflict = root.run_once()
    except RuntimeError as exc:
        raise RuntimeError(f"根节点无法规划: {exc}")

    heapq.heappush(open_list, (root.solution_cost(), counter, root_constraints, conflict, root))
    visited = {constraints_signature(root_constraints)}

    while open_list and counter < CBS_MAX_NODES:
        _, _, constraints, conflict, planner = heapq.heappop(open_list)

        if conflict is None:
            planner.write_trajectory()
            planner.write_task_results()
            print("CBS 修复完成：最终轨迹无顶点冲突、无对穿冲突。")
            print(f"累计延迟时间: {planner.total_delay}")
            print(f"总行驶成本: {planner.total_distance}")
            print(f"已输出 {TRAJECTORY_FILE}")
            print("已输出 agv_task_result.csv")
            print("已输出 agv_summary.csv")
            return

        print(f"发现冲突，生成 CBS 分支: {conflict}")

        for agv_id in [conflict["a1"], conflict["a2"]]:
            new_constraints = clone_constraints(constraints)

            if conflict["type"] == "vertex":
                new_constraints[agv_id]["vertex"].add((conflict["pos"], conflict["time"]))
            else:
                if agv_id == conflict["a1"]:
                    from_pos, to_pos = conflict["edge1"]
                else:
                    from_pos, to_pos = conflict["edge2"]
                new_constraints[agv_id]["edge"].add((from_pos, to_pos, conflict["time"]))

            sig = constraints_signature(new_constraints)
            if sig in visited:
                continue
            visited.add(sig)

            try:
                child = Planner(new_constraints)
                child_conflict = child.run_once()
            except RuntimeError:
                continue

            counter += 1
            heapq.heappush(open_list, (child.solution_cost(), counter, new_constraints, child_conflict, child))

    raise RuntimeError("CBS 修复失败：超过搜索节点上限，仍未找到无冲突轨迹。可以增大 CBS_MAX_NODES 或放宽任务时间窗。")


def main():
    plan_with_cbs()


if __name__ == "__main__":
    main()
