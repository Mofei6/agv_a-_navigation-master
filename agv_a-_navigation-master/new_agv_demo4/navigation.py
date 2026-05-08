import csv
import heapq
from collections import defaultdict

# ============================================================
# 基础参数
# ============================================================

GRID_W = 20
GRID_H = 20

POSITION_FILE = "agv_position.csv"
TASK_FILE = "agv_task.csv"
TRAJECTORY_FILE = "agv_trajectory.csv"
TASK_RESULT_FILE = "agv_task_result.csv"
SUMMARY_FILE = "agv_summary.csv"

LOAD_DURATION = 2
UNLOAD_DURATION = 2

MAX_SEARCH_TIME = 1200
RESERVE_HORIZON = 4000

# ============================================================
# 共享虚拟等待区
# ============================================================
# 所有候选 AGV 共享同一个等待区。
# 等待区内部允许重叠，不参与碰撞检测。
# 只有从 SHARED_HOME_POS 出发进入道路时，才参与预约表。
SHARED_HOME_ENABLED = True
SHARED_HOME_POS = (20, 7)

# ============================================================
# 动态 AGV 数量优化参数
# ============================================================

# 每启用一台 AGV 的固定成本。
# 越大越倾向少用 AGV；越小越倾向多用 AGV。
AGV_FIXED_COST = 18000

# 总延迟权重。
TOTAL_DELAY_WEIGHT = 1500

# 延迟平方惩罚，避免单个任务延迟特别大。
DELAY_SQUARE_WEIGHT = 60

# 最大延迟权重。
MAX_DELAY_WEIGHT = 3500

# 行驶距离成本。
DISTANCE_WEIGHT = 8

# 完成时间只作为很弱的辅助项。
FINISH_TIME_WEIGHT = 0.2

# 任务老化权重。
TASK_AGING_WEIGHT = 400

# 启用新 AGV 至少需要带来的延迟下降。
MIN_ACTIVATION_DELAY_REDUCTION = 8

# 如果新 AGV 没有减少延迟，则不启用。
REQUIRE_DELAY_REDUCTION_FOR_ACTIVATION = True

# ============================================================
# 调度候选参数
# ============================================================

SCHEDULING_LOOKAHEAD = 40
LOOKAHEAD_TIME = 90

# ============================================================
# 取料区等待控制参数
# ============================================================

# 如果距离下一次可取货时间超过该阈值，则 AGV 不应在通道等待。
DEPOT_WAIT_THRESHOLD = 5

# 如果 AGV 当前不在等待区，且 5 秒内无法到取料区，则先回等待区。
MAX_ROAD_WAIT_BEFORE_PICKUP = 5

# AGV 最多提前多少秒到达取料点。
PICKUP_MAX_EARLY_ARRIVAL = 5

# 返回等待区时禁止经过取料点。
RETURN_HOME_BLOCK_PICKUP = True

# 不允许 AGV 在取料区附近长时间等待。
STRICT_PICKUP_STAGING = True

# 是否每次完成任务后都强制回等待区。
# 如果你希望道路最干净，可以改成 True。
FORCE_RETURN_HOME_AFTER_EACH_TASK = False

# ============================================================
# 移动方向
# ============================================================

DIRS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (0, 0),
]

PITCH_MAP = {
    (1, 0): 0,
    (-1, 0): 180,
    (0, 1): 90,
    (0, -1): 270,
    (0, 0): None,
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(pos):
    for dx, dy in DIRS:
        nx = pos[0] + dx
        ny = pos[1] + dy

        if 1 <= nx <= GRID_W and 1 <= ny <= GRID_H:
            yield (nx, ny)


class ReservationTable:
    """
    时空预约表。

    vertex[t][pos] = agv_id
        表示 t 时刻 pos 被某台 AGV 占用。

    edge[t][(from_pos, to_pos)] = agv_id
        表示 t-1 到 t 该 AGV 从 from_pos 到 to_pos。

    用于避免：
    1. 顶点冲突：两台 AGV 同时占用同一网格；
    2. 对穿冲突：两台 AGV 在同一时间步交换位置。
    """

    def __init__(self):
        self.vertex = defaultdict(dict)
        self.edge = defaultdict(dict)

    def clone(self):
        other = ReservationTable()
        other.vertex = defaultdict(dict, {t: dict(v) for t, v in self.vertex.items()})
        other.edge = defaultdict(dict, {t: dict(v) for t, v in self.edge.items()})
        return other

    def clear_agv_from(self, agv_id, start_t, end_t=RESERVE_HORIZON):
        """
        清除某台 AGV 从 start_t 开始的未来预约。
        用于重新规划该 AGV 的后续动作。
        """
        for t in list(self.vertex.keys()):
            if start_t <= t <= end_t:
                for pos in list(self.vertex[t].keys()):
                    if self.vertex[t][pos] == agv_id:
                        del self.vertex[t][pos]

                if not self.vertex[t]:
                    del self.vertex[t]

        for t in list(self.edge.keys()):
            if start_t <= t <= end_t:
                for edge_key in list(self.edge[t].keys()):
                    if self.edge[t][edge_key] == agv_id:
                        del self.edge[t][edge_key]

                if not self.edge[t]:
                    del self.edge[t]

    def is_free(self, agv_id, prev_pos, pos, t):
        """
        判断 agv_id 在 t 时刻从 prev_pos 到 pos 是否可用。
        """
        owner = self.vertex.get(t, {}).get(pos)
        if owner is not None and owner != agv_id:
            return False

        same_owner = self.edge.get(t, {}).get((prev_pos, pos))
        if same_owner is not None and same_owner != agv_id:
            return False

        reverse_owner = self.edge.get(t, {}).get((pos, prev_pos))
        if reverse_owner is not None and reverse_owner != agv_id:
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
            for idx, (pos, t) in enumerate(path):
                prev_pos = path[idx - 1][0] if idx > 0 else pos

                owner = self.vertex.get(t, {}).get(pos)
                if owner is not None and owner != agv_id:
                    raise RuntimeError(
                        f"预约冲突: AGV{agv_id} 在 t={t} 想占用 {pos}, "
                        f"但该位置已被 AGV{owner} 占用"
                    )

                reverse_owner = self.edge.get(t, {}).get((pos, prev_pos))
                if reverse_owner is not None and reverse_owner != agv_id:
                    raise RuntimeError(
                        f"对穿冲突: AGV{agv_id} t={t} {prev_pos}->{pos}, "
                        f"但 AGV{reverse_owner} 正在反向通过"
                    )

            raise RuntimeError(f"预约冲突: AGV{agv_id}, path={path[:3]}...{path[-3:]}")

        for idx, (pos, t) in enumerate(path):
            self.vertex[t][pos] = agv_id

            if idx > 0:
                prev_pos = path[idx - 1][0]
                self.edge[t][(prev_pos, pos)] = agv_id

    def can_hold(self, agv_id, pos, start_t, duration):
        """
        判断 AGV 能否从 start_t 开始在 pos 连续停留 duration 秒。
        """
        for t in range(start_t, start_t + duration + 1):
            if not self.is_free(agv_id, pos, pos, t):
                return False

        return True


class Planner:
    def __init__(self):
        self.supply = None

        # 多取料点。
        # self.pickups[name] = {"name":..., "label":..., "pos":...}
        self.pickups = {}

        self.drop_points = {}
        self.homes = {}
        self.tasks = []

        # 动态启用 AGV。
        self.active_agvs = set()
        self.inactive_agvs = set()

        self.agv_states = {}
        self.agv_tracks = defaultdict(list)

        self.task_results = []
        self.total_delay = 0
        self.total_distance = 0
        self.current_max_delay = 0

        self.res = ReservationTable()

        self.load_inputs()

    # ============================================================
    # 数据读取
    # ============================================================

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
                    self.pickups[item["name"]] = item
                elif row["type"] == "drop_point":
                    self.drop_points[item["name"]] = item
                elif row["type"] == "agv":
                    self.homes[item["name"]] = item

        if not self.homes:
            raise ValueError("至少需要 1 台候选 AGV")

        if self.supply is None or not self.pickups:
            raise ValueError("agv_position.csv 缺少 supply 或 pickup_slot")

        with open(TASK_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                task = {
                    "task_id": row["task_id"].strip(),
                    "destination_id": row["destination_id"].strip(),
                    "destination_label": row["destination_label"].strip(),
                    "material": row["material"].strip(),
                    "release_time": int(row["release_time"]),
                    "window_start": int(row["window_start"]),
                    "window_end": int(row["window_end"]),
                    "priority": row.get("priority", "Normal"),
                }

                if task["destination_id"] not in self.drop_points:
                    raise ValueError(f"任务目的地不存在: {task['destination_id']}")

                self.tasks.append(task)

        self.tasks.sort(key=lambda x: (x["release_time"], x["window_start"], x["task_id"]))

        for agv_id, home in self.homes.items():
            home_pos = SHARED_HOME_POS if SHARED_HOME_ENABLED else home["pos"]

            self.agv_states[agv_id] = {
                "id": agv_id,
                "pos": home_pos,
                "time": 0,
                "pitch": home["pitch"],
                "home": home_pos,
                "activated": False,
            }

            self.inactive_agvs.add(agv_id)

        print(f"候选 AGV 数量: {len(self.homes)}")
        print(f"取料点数量: {len(self.pickups)}")

    # ============================================================
    # 地图与辅助函数
    # ============================================================

    def pickup_positions(self):
        return [p["pos"] for p in self.pickups.values()]

    def nearest_pickup_distance(self, start_pos):
        return min(manhattan(start_pos, p["pos"]) for p in self.pickups.values())

    def static_blocked(self, extra_blocked=None):
        """
        静态障碍：
        1. 物料区本体；
        2. 工位本体；
        3. extra_blocked 中指定的临时障碍。

        注意：
        取料点不是静态障碍，AGV 需要进入取料点取料。
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
        工位本体不可进入，AGV 只到工位相邻道路格卸货。
        """
        target = self.drop_points[destination_id]["pos"]
        goals = []

        for nb in neighbors(target):
            if nb != target and self.traversable(nb, extra_blocked):
                goals.append(nb)

        if not goals:
            raise RuntimeError(f"工位 {destination_id} 周边没有可用服务点")

        return goals

    def heuristic(self, pos, goals):
        return min(manhattan(pos, g) for g in goals)

    def wait_path(self, pos, start_t, duration):
        return [(pos, t) for t in range(start_t, start_t + duration + 1)]

    def path_distance(self, path):
        if not path or len(path) <= 1:
            return 0

        return sum(
            1
            for i in range(1, len(path))
            if path[i][0] != path[i - 1][0]
        )

    def plan_distance(self, segments):
        return sum(self.path_distance(seg[-1]) for seg in segments)

    def current_system_time(self):
        if not self.active_agvs:
            return 0

        return min(self.agv_states[aid]["time"] for aid in self.active_agvs)

    def task_age(self, task):
        return max(0, self.current_system_time() - task["window_start"])

    def shortest_distance_estimate(self, start, goal):
        return manhattan(start, goal)

    def should_stage_from_depot(self, agv_id, task, cur_pos, cur_time):
        """
        如果 AGV 当前在道路上，但短期内不能取料，则先回等待区。
        """
        home = self.agv_states[agv_id]["home"]

        if cur_pos == home:
            return False

        if task["release_time"] - cur_time > DEPOT_WAIT_THRESHOLD:
            return True

        dist_to_pickup = self.nearest_pickup_distance(cur_pos)

        if dist_to_pickup > MAX_ROAD_WAIT_BEFORE_PICKUP:
            return True

        return False

    def compute_depot_depart_time(self, start_pos, task, earliest_time):
        """
        从等待区出发的合适时间。
        目标是不让 AGV 提前聚集到取料区附近。
        """
        dist = self.nearest_pickup_distance(start_pos)
        desired_depart = task["release_time"] - dist - PICKUP_MAX_EARLY_ARRIVAL
        return max(earliest_time, desired_depart, 0)

    def append_return_home_stage(self, temp_res, agv_id, cur_pos, cur_time, segments):
        """
        追加回共享等待区路径。
        返回 (new_pos, new_time)。
        """
        home = self.agv_states[agv_id]["home"]

        if cur_pos == home:
            return cur_pos, cur_time

        extra_blocked = set()

        if RETURN_HOME_BLOCK_PICKUP:
            for pickup in self.pickups.values():
                extra_blocked.add(pickup["pos"])

        return_path = self.astar_time(
            temp_res,
            agv_id,
            cur_pos,
            cur_time,
            [home],
            extra_blocked=extra_blocked,
        )

        if return_path is None:
            return None, None

        if not temp_res.can_reserve_path(agv_id, return_path):
            return None, None

        temp_res.reserve_path(agv_id, return_path)

        segments.append((
            "return_home",
            False,
            "",
            "",
            "",
            "",
            return_path,
        ))

        return return_path[-1][0], return_path[-1][1]

    def has_short_term_task_after(self, current_task, finish_time, remaining_tasks):
        """
        判断任务完成后，短期内是否值得留在道路上等待。
        """
        if FORCE_RETURN_HOME_AFTER_EACH_TASK:
            return False

        if not remaining_tasks:
            return False

        for task in remaining_tasks:
            if task["task_id"] == current_task["task_id"]:
                continue

            if task["release_time"] <= finish_time + DEPOT_WAIT_THRESHOLD:
                return True

        return False

    def should_return_home_after_task(self, current_task, finish_time, remaining_tasks):
        return not self.has_short_term_task_after(
            current_task=current_task,
            finish_time=finish_time,
            remaining_tasks=remaining_tasks,
        )

    # ============================================================
    # 时空 A*
    # ============================================================

    def astar_time(
        self,
        res,
        agv_id,
        start_pos,
        start_time,
        goals,
        earliest_goal_time=0,
        extra_blocked=None,
        goal_hold=0,
        max_early_arrival=None,
    ):
        blocked = self.static_blocked(extra_blocked)
        goals = set(goals)

        start_state = (start_pos, start_time)

        pq = [
            (
                self.heuristic(start_pos, goals),
                0,
                start_pos,
                start_time,
            )
        ]

        parent = {start_state: None}
        gscore = {start_state: 0}

        while pq:
            _, cost, pos, t = heapq.heappop(pq)
            state = (pos, t)

            if (
                pos in goals
                and t >= earliest_goal_time
                and res.can_hold(agv_id, pos, t, goal_hold)
            ):
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

                    priority = (
                        ncost
                        + self.heuristic(nxt, goals)
                        + max(0, earliest_goal_time - nt)
                    )

                    heapq.heappush(pq, (priority, ncost, nxt, nt))

        return None

    def find_path_with_start_offsets(
        self,
        res,
        agv_id,
        start_pos,
        goals,
        start_time,
        max_offset=60,
        **kwargs,
    ):
        """
        如果某个出发时刻被占用，则尝试延迟出发。
        """
        for offset in range(max_offset + 1):
            path = self.astar_time(
                res,
                agv_id,
                start_pos,
                start_time + offset,
                goals,
                **kwargs,
            )

            if path is not None:
                return path

        return None

    # ============================================================
    # 多取料点选择
    # ============================================================

    def plan_to_best_pickup(self, temp_res, agv_id, cur_pos, cur_time, task):
        """
        在多个取料点中选择一个最优取料点。

        返回：
            {
                "pickup": pickup_item,
                "path": path_to_pick,
                "arrive_time": pickup_arrive,
                "depart_time": pickup_depart,
                "score": score,
                "road_wait": road_wait,
            }

        如果没有任何取料点可行，返回 None。
        """
        best = None

        for pickup in self.pickups.values():
            pickup_pos = pickup["pos"]

            path_to_pick = self.find_path_with_start_offsets(
                temp_res,
                agv_id,
                cur_pos,
                goals=[pickup_pos],
                start_time=cur_time,
                earliest_goal_time=task["release_time"],
                goal_hold=LOAD_DURATION,
                max_early_arrival=PICKUP_MAX_EARLY_ARRIVAL,
            )

            if path_to_pick is None:
                continue

            if not temp_res.can_reserve_path(agv_id, path_to_pick):
                continue

            pickup_arrive = path_to_pick[-1][1]
            pickup_depart = pickup_arrive + LOAD_DURATION

            min_travel = manhattan(cur_pos, pickup_pos)
            road_wait = max(0, pickup_arrive - cur_time - min_travel)
            dist = self.path_distance(path_to_pick)

            score = (
                pickup_arrive,
                road_wait,
                dist,
                pickup["name"],
            )

            if best is None or score < best["score"]:
                best = {
                    "pickup": pickup,
                    "path": path_to_pick,
                    "arrive_time": pickup_arrive,
                    "depart_time": pickup_depart,
                    "score": score,
                    "road_wait": road_wait,
                }

        return best

    # ============================================================
    # 候选方案评估
    # ============================================================

    def candidate_plan(self, agv_id, task, activate=False, remaining_tasks=None):
        """
        评估某台 AGV 执行某个任务的完整路径。

        新增支持：
        1. 两个或多个取料点；
        2. AGV 自动选择最优取料点；
        3. 如果短期内无法取料，AGV 先回等待区；
        4. 避免 AGV 在取料区附近或通道内长时间等待。
        """
        if remaining_tasks is None:
            remaining_tasks = []

        agv = self.agv_states[agv_id]
        temp_res = self.res.clone()
        segments = []

        # --------------------------------------------------------
        # 1. 确定起点
        # --------------------------------------------------------
        if activate:
            cur_pos = agv["home"]
            cur_time = 0
        else:
            cur_pos = agv["pos"]
            cur_time = agv["time"]
            temp_res.clear_agv_from(agv_id, cur_time, RESERVE_HORIZON)

        # --------------------------------------------------------
        # 2. 如果当前在道路上，但短期内不能取料，先回等待区
        # --------------------------------------------------------
        if self.should_stage_from_depot(agv_id, task, cur_pos, cur_time):
            cur_pos, cur_time = self.append_return_home_stage(
                temp_res=temp_res,
                agv_id=agv_id,
                cur_pos=cur_pos,
                cur_time=cur_time,
                segments=segments,
            )

            if cur_pos is None:
                return None

        # --------------------------------------------------------
        # 3. 在共享等待区等待到合适出发时间
        # --------------------------------------------------------
        if cur_pos == agv["home"]:
            depart_time = self.compute_depot_depart_time(
                start_pos=cur_pos,
                task=task,
                earliest_time=cur_time,
            )

            # 不生成等待路径，不预约等待区。
            cur_time = depart_time

        # --------------------------------------------------------
        # 4. 选择最优取料点
        # --------------------------------------------------------
        pickup_plan = self.plan_to_best_pickup(
            temp_res=temp_res,
            agv_id=agv_id,
            cur_pos=cur_pos,
            cur_time=cur_time,
            task=task,
        )

        if pickup_plan is None:
            return None

        pickup = pickup_plan["pickup"]
        pickup_pos = pickup["pos"]
        path_to_pick = pickup_plan["path"]
        pickup_arrive = pickup_plan["arrive_time"]
        pickup_depart = pickup_plan["depart_time"]

        # --------------------------------------------------------
        # 5. 严格检查：不允许在道路或取料区附近长期等待
        # --------------------------------------------------------
        if STRICT_PICKUP_STAGING:
            min_travel = self.shortest_distance_estimate(cur_pos, pickup_pos)
            road_wait = pickup_arrive - cur_time - min_travel

            if road_wait > DEPOT_WAIT_THRESHOLD and cur_pos != agv["home"]:
                return None

            if road_wait > DEPOT_WAIT_THRESHOLD and cur_pos == agv["home"]:
                delayed_start_time = cur_time + road_wait

                pickup_plan = self.plan_to_best_pickup(
                    temp_res=temp_res,
                    agv_id=agv_id,
                    cur_pos=cur_pos,
                    cur_time=delayed_start_time,
                    task=task,
                )

                if pickup_plan is None:
                    return None

                pickup = pickup_plan["pickup"]
                pickup_pos = pickup["pos"]
                path_to_pick = pickup_plan["path"]
                pickup_arrive = pickup_plan["arrive_time"]
                pickup_depart = pickup_plan["depart_time"]

        if not temp_res.can_reserve_path(agv_id, path_to_pick):
            return None

        temp_res.reserve_path(agv_id, path_to_pick)

        segments.append((
            "to_pickup",
            False,
            "",
            "",
            "",
            "",
            path_to_pick,
        ))

        # --------------------------------------------------------
        # 6. 在选中的取料点取料
        # --------------------------------------------------------
        load_path = self.wait_path(pickup_pos, pickup_arrive, LOAD_DURATION)

        if not temp_res.can_reserve_path(agv_id, load_path):
            return None

        temp_res.reserve_path(agv_id, load_path)

        segments.append((
            "loading",
            False,
            "",
            "",
            "",
            "",
            load_path,
        ))

        # --------------------------------------------------------
        # 7. 取货后配送
        # --------------------------------------------------------
        # 取完料后，配送阶段不允许穿过任何取料点。
        extra_after_pickup = {p["pos"] for p in self.pickups.values()}

        goals = self.delivery_goals(task["destination_id"], extra_after_pickup)

        path_to_drop = self.astar_time(
            temp_res,
            agv_id,
            pickup_pos,
            pickup_depart,
            goals,
            extra_blocked=extra_after_pickup,
            goal_hold=UNLOAD_DURATION,
        )

        if path_to_drop is None:
            return None

        if not temp_res.can_reserve_path(agv_id, path_to_drop):
            return None

        temp_res.reserve_path(agv_id, path_to_drop)

        segments.append((
            "delivering",
            True,
            task["material"],
            task["destination_label"],
            task["destination_id"],
            task["task_id"],
            path_to_drop,
        ))

        drop_arrive = path_to_drop[-1][1]

        # --------------------------------------------------------
        # 8. 卸货
        # --------------------------------------------------------
        unload_path = self.wait_path(path_to_drop[-1][0], drop_arrive, UNLOAD_DURATION)

        if not temp_res.can_reserve_path(agv_id, unload_path):
            return None

        temp_res.reserve_path(agv_id, unload_path)

        segments.append((
            "unloading",
            True,
            task["material"],
            task["destination_label"],
            task["destination_id"],
            task["task_id"],
            unload_path,
        ))

        final_pos = unload_path[-1][0]
        final_time = unload_path[-1][1]

        # --------------------------------------------------------
        # 9. 任务完成后是否回等待区
        # --------------------------------------------------------
        need_return_home = self.should_return_home_after_task(
            current_task=task,
            finish_time=final_time,
            remaining_tasks=remaining_tasks,
        )

        if need_return_home and final_pos != agv["home"]:
            final_pos, final_time = self.append_return_home_stage(
                temp_res=temp_res,
                agv_id=agv_id,
                cur_pos=final_pos,
                cur_time=final_time,
                segments=segments,
            )

            if final_pos is None:
                return None

        # --------------------------------------------------------
        # 10. 最终等待预约
        # --------------------------------------------------------
        # 如果最终回到共享等待区，则不长期预约道路。
        # 如果停在道路上，则必须长期预约，防止其它 AGV 穿过。
        if not (SHARED_HOME_ENABLED and final_pos == agv["home"]):
            hold_path = [
                (final_pos, t)
                for t in range(final_time, RESERVE_HORIZON + 1)
            ]

            if not temp_res.can_reserve_path(agv_id, hold_path):
                return None

            temp_res.reserve_path(agv_id, hold_path)

        distance = self.plan_distance(segments)

        return {
            "segments": segments,
            "delivery_time": drop_arrive,
            "finish_time": final_time,
            "finish_pos": final_pos,
            "distance": distance,
            "activate": activate,
            "res_after": temp_res,
            "return_home": need_return_home,
            "pickup_name": pickup["name"],
            "pickup_label": pickup["label"],
        }

    # ============================================================
    # 任务选择与 AGV 启用决策
    # ============================================================

    def select_candidate_tasks(self, unscheduled_tasks):
        now = self.current_system_time()

        overdue = [
            task
            for task in unscheduled_tasks
            if task["window_start"] <= now
        ]

        if overdue:
            tasks = sorted(
                overdue,
                key=lambda x: (x["window_start"], x["release_time"], x["task_id"]),
            )
        else:
            near = [
                task
                for task in unscheduled_tasks
                if task["release_time"] <= now + LOOKAHEAD_TIME
                or task["window_start"] <= now + LOOKAHEAD_TIME
            ]

            tasks = sorted(
                near if near else unscheduled_tasks,
                key=lambda x: (x["window_start"], x["release_time"], x["task_id"]),
            )

        return tasks[:SCHEDULING_LOOKAHEAD]

    def operating_cost(self, task, plan):
        delay = max(0, plan["delivery_time"] - task["window_start"])
        distance = plan.get("distance", 0)
        finish_time = plan["finish_time"]

        max_delay_increase = max(0, delay - self.current_max_delay)
        age = self.task_age(task)

        cost = (
            delay * TOTAL_DELAY_WEIGHT
            + delay * delay * DELAY_SQUARE_WEIGHT
            + max_delay_increase * MAX_DELAY_WEIGHT
            + distance * DISTANCE_WEIGHT
            + finish_time * FINISH_TIME_WEIGHT
            - age * TASK_AGING_WEIGHT
        )

        return cost

    def choose_best_action(self, unscheduled_tasks):
        candidate_tasks = self.select_candidate_tasks(unscheduled_tasks)
        best_global = None

        active_order = sorted(
            self.active_agvs,
            key=lambda aid: (self.agv_states[aid]["time"], int(aid)),
        )

        inactive_order = sorted(
            self.inactive_agvs,
            key=lambda aid: int(aid),
        )

        for task in candidate_tasks:
            best_active = None

            for agv_id in active_order:
                plan = self.candidate_plan(
                    agv_id,
                    task,
                    activate=False,
                    remaining_tasks=unscheduled_tasks,
                )

                if plan is None:
                    continue

                delay = max(0, plan["delivery_time"] - task["window_start"])
                cost = self.operating_cost(task, plan)

                key = (
                    cost,
                    delay,
                    plan.get("distance", 0),
                    plan["finish_time"],
                    int(agv_id),
                )

                if best_active is None or key < best_active[0]:
                    best_active = (key, agv_id, plan, cost, delay)

            best_new = None

            for agv_id in inactive_order:
                plan = self.candidate_plan(
                    agv_id,
                    task,
                    activate=True,
                    remaining_tasks=unscheduled_tasks,
                )

                if plan is None:
                    continue

                delay = max(0, plan["delivery_time"] - task["window_start"])
                op_cost = self.operating_cost(task, plan)
                total_cost = op_cost + AGV_FIXED_COST

                key = (
                    total_cost,
                    delay,
                    plan.get("distance", 0),
                    plan["finish_time"],
                    int(agv_id),
                )

                if best_new is None or key < best_new[0]:
                    best_new = (key, agv_id, plan, op_cost, total_cost, delay)

            chosen = None

            if best_active is None and best_new is not None:
                _, agv_id, plan, op_cost, total_cost, delay = best_new

                chosen = {
                    "score": total_cost,
                    "type": "activate",
                    "task": task,
                    "agv_id": agv_id,
                    "plan": plan,
                    "delay": delay,
                    "distance": plan.get("distance", 0),
                    "finish_time": plan["finish_time"],
                }

            elif best_active is not None and best_new is None:
                _, agv_id, plan, cost, delay = best_active

                chosen = {
                    "score": cost,
                    "type": "use_active",
                    "task": task,
                    "agv_id": agv_id,
                    "plan": plan,
                    "delay": delay,
                    "distance": plan.get("distance", 0),
                    "finish_time": plan["finish_time"],
                }

            elif best_active is not None and best_new is not None:
                _, active_agv, active_plan, active_cost, active_delay = best_active
                _, new_agv, new_plan, new_op_cost, new_total_cost, new_delay = best_new

                delay_reduction = active_delay - new_delay
                cost_reduction = active_cost - new_op_cost

                should_activate = False

                if delay_reduction >= MIN_ACTIVATION_DELAY_REDUCTION:
                    if cost_reduction > AGV_FIXED_COST:
                        should_activate = True

                if REQUIRE_DELAY_REDUCTION_FOR_ACTIVATION and delay_reduction <= 0:
                    should_activate = False

                if should_activate:
                    chosen = {
                        "score": new_total_cost,
                        "type": "activate",
                        "task": task,
                        "agv_id": new_agv,
                        "plan": new_plan,
                        "delay": new_delay,
                        "distance": new_plan.get("distance", 0),
                        "finish_time": new_plan["finish_time"],
                    }
                else:
                    chosen = {
                        "score": active_cost,
                        "type": "use_active",
                        "task": task,
                        "agv_id": active_agv,
                        "plan": active_plan,
                        "delay": active_delay,
                        "distance": active_plan.get("distance", 0),
                        "finish_time": active_plan["finish_time"],
                    }

            if chosen is None:
                continue

            global_key = (
                chosen["score"],
                chosen["delay"],
                chosen["distance"],
                chosen["finish_time"],
                0 if chosen["type"] == "use_active" else 1,
                int(chosen["agv_id"]),
            )

            if best_global is None or global_key < best_global[0]:
                best_global = (global_key, chosen)

        if best_global is None:
            raise RuntimeError("当前没有可行的任务-AGV 动作")

        return best_global[1]

    # ============================================================
    # 轨迹写入与状态更新
    # ============================================================

    def path_pitch(self, prev_pos, curr_pos, last_pitch):
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]

        return PITCH_MAP.get((dx, dy), last_pitch) or last_pitch

    def make_row(
        self,
        t,
        agv_id,
        pos,
        pitch,
        loaded,
        material,
        dest_label,
        dest_id,
        task_id,
        status,
    ):
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

    def fill_display_idle(self, agv_id, until_t):
        rows = self.agv_tracks[agv_id]

        if not rows:
            return

        last = rows[-1]
        t = last["timestamp"]

        while t < until_t - 1:
            t += 1
            rows.append(
                self.make_row(
                    t,
                    agv_id,
                    last["pos"],
                    last["pitch"],
                    False,
                    "",
                    "",
                    "",
                    "",
                    "idle",
                )
            )

    def append_path_rows(
        self,
        agv_id,
        path,
        loaded,
        material,
        dest_label,
        dest_id,
        task_id,
        status,
    ):
        rows = self.agv_tracks[agv_id]
        state = self.agv_states[agv_id]

        if not rows:
            start_pos = path[0][0]
            start_t = path[0][1]

            rows.append(
                self.make_row(
                    start_t,
                    agv_id,
                    start_pos,
                    state["pitch"],
                    False,
                    "",
                    "",
                    "",
                    "",
                    "idle",
                )
            )

        self.fill_display_idle(agv_id, path[0][1])

        last_pitch = rows[-1]["pitch"]
        prev_pos = rows[-1]["pos"]

        for pos, t in path[1:]:
            pitch = self.path_pitch(prev_pos, pos, last_pitch)

            rows.append(
                self.make_row(
                    t,
                    agv_id,
                    pos,
                    pitch,
                    loaded,
                    material,
                    dest_label,
                    dest_id,
                    task_id,
                    status,
                )
            )

            prev_pos = pos
            last_pitch = pitch

        state["pos"] = path[-1][0]
        state["time"] = path[-1][1]
        state["pitch"] = last_pitch

    def commit_action(self, action):
        """
        提交当前选中的任务-AGV 动作。

        注意：
        candidate_plan() 里已经基于 temp_res 完整验证并预约路径。
        所以这里不再二次 reserve_path，而是直接使用 plan["res_after"]。
        """
        agv_id = action["agv_id"]
        task = action["task"]
        plan = action["plan"]

        if action["type"] == "activate":
            self.active_agvs.add(agv_id)
            self.inactive_agvs.remove(agv_id)

            self.agv_states[agv_id]["activated"] = True

            self.agv_tracks[agv_id].append(
                self.make_row(
                    0,
                    agv_id,
                    self.agv_states[agv_id]["home"],
                    self.agv_states[agv_id]["pitch"],
                    False,
                    "",
                    "",
                    "",
                    "",
                    "idle",
                )
            )

        for status, loaded, material, dest_label, dest_id, task_id, path in plan["segments"]:
            self.append_path_rows(
                agv_id,
                path,
                loaded,
                material,
                dest_label,
                dest_id,
                task_id,
                status,
            )

        self.res = plan["res_after"]

        self.agv_states[agv_id]["pos"] = plan["finish_pos"]
        self.agv_states[agv_id]["time"] = plan["finish_time"]

        delay = max(0, plan["delivery_time"] - task["window_start"])

        self.total_delay += delay
        self.total_distance += plan["distance"]
        self.current_max_delay = max(self.current_max_delay, delay)

        self.task_results.append(
            {
                "task_id": task["task_id"],
                "destination_id": task["destination_id"],
                "destination_label": task["destination_label"],
                "material": task["material"],
                "agv_id": agv_id,
                "window_start": task["window_start"],
                "window_end": task["window_end"],
                "delivery_time": plan["delivery_time"],
                "delay": delay,
                "distance": plan["distance"],
                "pickup": plan.get("pickup_label", ""),
            }
        )

    # ============================================================
    # 收尾、冲突检查与输出
    # ============================================================

    def finalize_to_home(self):
        """
        所有任务结束后，将仍在道路上的 AGV 返回等待区。
        """
        for agv_id in list(self.active_agvs):
            st = self.agv_states[agv_id]
            home = st["home"]

            if st["pos"] == home:
                continue

            self.res.clear_agv_from(agv_id, st["time"], RESERVE_HORIZON)

            extra_blocked = set()
            for pickup in self.pickups.values():
                extra_blocked.add(pickup["pos"])

            path = self.astar_time(
                self.res,
                agv_id,
                st["pos"],
                st["time"],
                [home],
                extra_blocked=extra_blocked,
            )

            if path is None:
                continue

            self.res.reserve_path(agv_id, path)

            self.append_path_rows(
                agv_id,
                path,
                False,
                "",
                "",
                "",
                "",
                "return_home",
            )

            self.total_distance += self.path_distance(path)

            st["pos"] = home
            st["time"] = path[-1][1]

    def fill_idle_until(self, max_t):
        for agv_id in self.active_agvs:
            rows = self.agv_tracks[agv_id]

            if not rows:
                continue

            last = rows[-1]
            t = last["timestamp"]

            while t < max_t:
                t += 1

                rows.append(
                    self.make_row(
                        t,
                        agv_id,
                        last["pos"],
                        last["pitch"],
                        False,
                        "",
                        "",
                        "",
                        "",
                        "idle",
                    )
                )

    def build_rows(self):
        fields = [
            "timestamp",
            "name",
            "X",
            "Y",
            "pitch",
            "loaded",
            "material",
            "destination_label",
            "destination_id",
            "task_id",
            "status",
        ]

        all_rows = []

        for agv_id in sorted(self.active_agvs, key=lambda x: int(x)):
            rows = self.agv_tracks[agv_id]
            seen = {}

            for row in rows:
                seen[row["timestamp"]] = row

            for row in seen.values():
                all_rows.append({k: row[k] for k in fields})

        all_rows.sort(key=lambda r: (int(r["timestamp"]), int(r["name"])))

        return all_rows

    def detect_conflict(self):
        """
        检查最终轨迹是否仍存在冲突。

        共享等待区 idle 状态允许重叠。
        其它状态都参与冲突检查。
        """
        rows = self.build_rows()
        by_t = defaultdict(dict)

        for row in rows:
            t = int(row["timestamp"])
            pos = (int(row["X"]), int(row["Y"]))
            agv_id = row["name"]
            status = row["status"]

            if pos == SHARED_HOME_POS and status == "idle":
                continue

            by_t[t][agv_id] = pos

        for t in sorted(by_t):
            seen = {}

            for agv_id, pos in by_t[t].items():
                if pos in seen:
                    return f"碰撞: t={t}, AGV{seen[pos]} 与 AGV{agv_id} 在 {pos}"

                seen[pos] = agv_id

            if t + 1 in by_t:
                ids = sorted(set(by_t[t]) & set(by_t[t + 1]), key=lambda x: int(x))

                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        a1 = ids[i]
                        a2 = ids[j]

                        p1 = by_t[t][a1]
                        p2 = by_t[t][a2]
                        n1 = by_t[t + 1][a1]
                        n2 = by_t[t + 1][a2]

                        if p1 != p2 and p1 == n2 and p2 == n1:
                            return (
                                f"对穿: t={t}->{t + 1}, "
                                f"AGV{a1} {p1}->{n1}, "
                                f"AGV{a2} {p2}->{n2}"
                            )

        return None

    def write_trajectory(self):
        rows = self.build_rows()

        fields = [
            "timestamp",
            "name",
            "X",
            "Y",
            "pitch",
            "loaded",
            "material",
            "destination_label",
            "destination_id",
            "task_id",
            "status",
        ]

        with open(TRAJECTORY_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

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
            "pickup",
        ]

        with open(TASK_RESULT_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.task_results)

        max_delay = max((r["delay"] for r in self.task_results), default=0)

        avg_delay = (
            sum(r["delay"] for r in self.task_results) / len(self.task_results)
            if self.task_results else 0
        )

        makespan = max((r["delivery_time"] for r in self.task_results), default=0)

        total_cost = (
            len(self.active_agvs) * AGV_FIXED_COST
            + self.total_delay * TOTAL_DELAY_WEIGHT
            + max_delay * MAX_DELAY_WEIGHT
            + self.total_distance * DISTANCE_WEIGHT
            + makespan * FINISH_TIME_WEIGHT
        )

        with open(SUMMARY_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["candidate_agv_count", len(self.homes)])
            writer.writerow(["used_agv_count", len(self.active_agvs)])
            writer.writerow(["unused_candidate_agv_count", len(self.inactive_agvs)])
            writer.writerow(["pickup_count", len(self.pickups)])
            writer.writerow(["total_delay", self.total_delay])
            writer.writerow(["max_delay", max_delay])
            writer.writerow(["avg_delay", round(avg_delay, 2)])
            writer.writerow(["total_distance", self.total_distance])
            writer.writerow(["makespan", makespan])
            writer.writerow(["total_cost", round(total_cost, 2)])
            writer.writerow(["agv_fixed_cost", AGV_FIXED_COST])

    # ============================================================
    # 主流程
    # ============================================================

    def run(self):
        unscheduled_tasks = list(self.tasks)

        while unscheduled_tasks:
            action = self.choose_best_action(unscheduled_tasks)
            self.commit_action(action)
            unscheduled_tasks.remove(action["task"])

            print(
                f"{action['type']} AGV{action['agv_id']} -> "
                f"{action['task']['task_id']} "
                f"{action['task']['destination_label']}, "
                f"delay={action.get('delay', 0)}, "
                f"pickup={action['plan'].get('pickup_label', '')}, "
                f"return_home={action['plan'].get('return_home', False)}, "
                f"active={len(self.active_agvs)}, "
                f"remaining={len(unscheduled_tasks)}"
            )

        self.finalize_to_home()

        max_t = 0

        for agv_id in self.active_agvs:
            if self.agv_tracks[agv_id]:
                max_t = max(max_t, self.agv_tracks[agv_id][-1]["timestamp"])

        self.fill_idle_until(max_t)

        conflict = self.detect_conflict()

        if conflict:
            raise RuntimeError(f"最终轨迹仍存在冲突: {conflict}")

        self.write_trajectory()
        self.write_task_results()

        print("=" * 60)
        print("动态 AGV 数量 + 多取料点路径规划完成")
        print(f"候选 AGV 数量: {len(self.homes)}")
        print(f"实际启用 AGV 数量: {len(self.active_agvs)}")
        print(f"取料点数量: {len(self.pickups)}")
        print(f"累计延迟: {self.total_delay}")
        print(f"最大延迟: {self.current_max_delay}")
        print(f"总行驶距离: {self.total_distance}")
        print("输出文件:")
        print(f"  {TRAJECTORY_FILE}")
        print(f"  {TASK_RESULT_FILE}")
        print(f"  {SUMMARY_FILE}")


def main():
    Planner().run()


if __name__ == "__main__":
    main()