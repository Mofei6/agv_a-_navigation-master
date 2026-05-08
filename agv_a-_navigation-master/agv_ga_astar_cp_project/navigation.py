import csv
import heapq
from collections import defaultdict

# ============================================================
# 基础参数
# ============================================================

GRID_W = 52
GRID_H = 52
STATION_W = 3
STATION_H = 3

POSITION_FILE = "agv_position.csv"
TASK_FILE = "agv_task.csv"
TRAJECTORY_FILE = "agv_trajectory.csv"
TASK_RESULT_FILE = "agv_task_result.csv"
SUMMARY_FILE = "agv_summary.csv"

LOAD_DURATION = 2
UNLOAD_DURATION = 2
MAX_SEARCH_TIME = 1200

# 共享虚拟等待区
SHARED_HOME_POS = (50, 26)

# 这里控制实际启用的 AGV 数量。
# 如果你希望更多 AGV 同时参与，把这个数调大即可。
MAX_ACTIVE_AGVS = 8

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
    x, y = pos

    for dx, dy in DIRS:
        nx = x + dx
        ny = y + dy

        if 1 <= nx <= GRID_W and 1 <= ny <= GRID_H:
            yield nx, ny


class ReservationTable:
    """
    时空预约表。

    vertex[t][pos] = agv_id
        表示 t 时刻 pos 被某台 AGV 占用。

    edge[t][(from_pos, to_pos)] = agv_id
        表示 t-1 到 t 该 AGV 从 from_pos 到 to_pos。

    共享等待区 SHARED_HOME_POS 是虚拟区域，允许 AGV 重叠。
    """

    def __init__(self):
        self.vertex = defaultdict(dict)
        self.edge = defaultdict(dict)

    def clone(self):
        other = ReservationTable()
        other.vertex = defaultdict(dict, {t: dict(v) for t, v in self.vertex.items()})
        other.edge = defaultdict(dict, {t: dict(v) for t, v in self.edge.items()})
        return other

    def is_free(self, agv_id, prev_pos, pos, t):
        """
        判断 AGV 在 t 时刻从 prev_pos 到 pos 是否可用。

        共享等待区允许多车重叠。
        普通道路仍然禁止顶点冲突和对穿冲突。
        """
        if pos != SHARED_HOME_POS:
            owner = self.vertex.get(t, {}).get(pos)
            if owner is not None and owner != agv_id:
                return False

        if pos != SHARED_HOME_POS and prev_pos != SHARED_HOME_POS:
            reverse_owner = self.edge.get(t, {}).get((pos, prev_pos))
            if reverse_owner is not None and reverse_owner != agv_id:
                return False

        return True

    def can_hold(self, agv_id, pos, start_t, duration):
        for t in range(start_t, start_t + duration + 1):
            if not self.is_free(agv_id, pos, pos, t):
                return False

        return True

    def reserve_path(self, agv_id, path):
        for idx, (pos, t) in enumerate(path):
            prev_pos = path[idx - 1][0] if idx > 0 else pos

            if not self.is_free(agv_id, prev_pos, pos, t):
                raise RuntimeError(f"预约冲突: AGV{agv_id}, t={t}, pos={pos}")

        for idx, (pos, t) in enumerate(path):
            self.vertex[t][pos] = agv_id

            if idx > 0:
                prev_pos = path[idx - 1][0]
                self.edge[t][(prev_pos, pos)] = agv_id


class Planner:
    def __init__(self):
        self.supply = None
        self.pickups = {}
        self.drop_points = {}
        self.homes = {}
        self.tasks = []

        self.res = ReservationTable()

        self.agv_states = {}
        self.active_agvs = []
        self.agv_tracks = defaultdict(list)

        self.task_results = []
        self.total_distance = 0
        self.total_delay = 0

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
                    "pitch": int(row["pitch"]) if row.get("pitch") not in ("", None) else 180,
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

        with open(TASK_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                self.tasks.append({
                    "task_id": row["task_id"].strip(),
                    "destination_id": row["destination_id"].strip(),
                    "destination_label": row["destination_label"].strip(),
                    "material": row["material"].strip(),
                    "release_time": int(row["release_time"]),
                    "window_start": int(row["window_start"]),
                    "window_end": int(row["window_end"]),
                    "priority": row.get("priority", "Normal"),
                })

        self.tasks.sort(key=lambda r: (r["release_time"], r["window_start"], r["task_id"]))

        for agv_id in sorted(self.homes, key=lambda x: int(x))[:MAX_ACTIVE_AGVS]:
            self.active_agvs.append(agv_id)

            self.agv_states[agv_id] = {
                "id": agv_id,
                "pos": SHARED_HOME_POS,
                "time": 0,
                "pitch": self.homes[agv_id]["pitch"],
                "home": SHARED_HOME_POS,
            }

            self.agv_tracks[agv_id].append(
                self.make_row(
                    0,
                    agv_id,
                    SHARED_HOME_POS,
                    180,
                    False,
                    "",
                    "",
                    "",
                    "",
                    "idle",
                )
            )

        if self.supply is None:
            raise ValueError("agv_position.csv 缺少物料区 supply")

        if not self.pickups:
            raise ValueError("agv_position.csv 缺少取料点 pickup_slot")

        if not self.drop_points:
            raise ValueError("agv_position.csv 缺少工位 drop_point")

        print(f"地图尺寸: {GRID_W} x {GRID_H}")
        print(f"工位尺寸: {STATION_W} x {STATION_H}")
        print(f"实际启用 AGV 数量: {len(self.active_agvs)}")
        print(f"取料点数量: {len(self.pickups)}")
        print(f"工位数量: {len(self.drop_points)}")

    # ============================================================
    # 地图与工位函数
    # ============================================================

    def station_cells(self, station_pos):
        """
        station_pos 是工位 3x3 区域左下角。
        返回该工位本体占用的全部网格。
        """
        sx, sy = station_pos

        return {
            (sx + dx, sy + dy)
            for dx in range(STATION_W)
            for dy in range(STATION_H)
        }

    def station_delivery_points(self, station_pos):
        """
        每个工位只有两个合法卸货点：
        1. 工位左下角下方一格
        2. 工位左上角上方一格

        例如工位左下角为 (3, 4)，左上角为 (3, 6)，
        则卸货点为 (3, 3) 和 (3, 7)。
        """
        sx, sy = station_pos

        return [
            (sx, sy - 1),
            (sx, sy + STATION_H),
        ]

    def static_blocked(self, extra_blocked=None):
        """
        静态障碍：
        1. 物料区本体；
        2. 每个工位 3x3 本体；
        3. extra_blocked 中指定的临时障碍。

        注意：
        - 取料点不是静态障碍，AGV 需要进入取料点取料。
        - 卸货点不是静态障碍，AGV 需要进入卸货点卸货。
        """
        blocked = {self.supply["pos"]}

        for dp in self.drop_points.values():
            blocked.update(self.station_cells(dp["pos"]))

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
        AGV 只能到达该工位左下方或左上方卸货点。
        不再允许到工位周边任意格卸货。
        """
        station_pos = self.drop_points[destination_id]["pos"]
        candidates = self.station_delivery_points(station_pos)

        goals = [
            p
            for p in candidates
            if self.traversable(p, extra_blocked)
        ]

        if not goals:
            raise RuntimeError(f"工位 {destination_id} 没有可用卸货点: {candidates}")

        return goals

    # ============================================================
    # 路径规划
    # ============================================================

    def heuristic(self, pos, goals):
        return min(manhattan(pos, g) for g in goals)

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
        best = {start_state: 0}

        while pq:
            _, cost, pos, t = heapq.heappop(pq)
            state = (pos, t)

            if (
                pos in goals
                and t >= earliest_goal_time
                and res.can_hold(agv_id, pos, t, goal_hold)
            ):
                path = []

                while state is not None:
                    path.append(state)
                    state = parent[state]

                path.reverse()
                return path

            if t - start_time > MAX_SEARCH_TIME:
                continue

            for nxt in neighbors(pos):
                nt = t + 1

                if nxt in blocked and nxt not in goals:
                    continue

                if not res.is_free(agv_id, pos, nxt, nt):
                    continue

                nstate = (nxt, nt)
                ncost = cost + 1

                if nstate not in best or ncost < best[nstate]:
                    best[nstate] = ncost
                    parent[nstate] = state

                    priority = (
                        ncost
                        + self.heuristic(nxt, goals)
                        + max(0, earliest_goal_time - nt)
                    )

                    heapq.heappush(pq, (priority, ncost, nxt, nt))

        return None

    def wait_path(self, pos, start_t, duration):
        return [
            (pos, t)
            for t in range(start_t, start_t + duration + 1)
        ]

    def path_distance(self, path):
        if not path or len(path) <= 1:
            return 0

        return sum(
            1
            for i in range(1, len(path))
            if path[i][0] != path[i - 1][0]
        )

    def choose_best_pickup_path(self, res, agv_id, start_pos, start_time, release_time):
        best = None

        for pickup in self.pickups.values():
            path = self.astar_time(
                res,
                agv_id,
                start_pos,
                start_time,
                [pickup["pos"]],
                earliest_goal_time=release_time,
                goal_hold=LOAD_DURATION,
            )

            if path is None:
                continue

            score = (
                path[-1][1],
                self.path_distance(path),
                pickup["name"],
            )

            if best is None or score < best[0]:
                best = (score, pickup, path)

        if best is None:
            return None

        return best[1], best[2]

    def build_candidate_plan(self, agv_id, task):
        st = self.agv_states[agv_id]

        temp_res = self.res.clone()
        cur_pos = st["pos"]
        cur_time = st["time"]

        segments = []

        # 如果 AGV 在共享等待区，不预约等待，只从合适时间出发，避免过早堵在取料区。
        if cur_pos == SHARED_HOME_POS:
            dist_to_pick = min(
                manhattan(cur_pos, pickup["pos"])
                for pickup in self.pickups.values()
            )

            cur_time = max(
                cur_time,
                task["release_time"] - dist_to_pick - 5,
                0,
            )

        # 1. 去取料点
        pickup_plan = self.choose_best_pickup_path(
            temp_res,
            agv_id,
            cur_pos,
            cur_time,
            task["release_time"],
        )

        if pickup_plan is None:
            return None

        pickup, path_to_pick = pickup_plan

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

        # 2. 取料
        load_path = self.wait_path(
            pickup["pos"],
            path_to_pick[-1][1],
            LOAD_DURATION,
        )

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

        # 3. 配送
        # 配送阶段禁止穿过取料点。
        extra_blocked = {
            pickup_item["pos"]
            for pickup_item in self.pickups.values()
        }

        goals = self.delivery_goals(task["destination_id"], extra_blocked)

        path_to_drop = self.astar_time(
            temp_res,
            agv_id,
            pickup["pos"],
            load_path[-1][1],
            goals,
            extra_blocked=extra_blocked,
            goal_hold=UNLOAD_DURATION,
        )

        if path_to_drop is None:
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

        unload_pos = path_to_drop[-1][0]
        delivery_time = path_to_drop[-1][1]

        # 4. 卸货
        unload_path = self.wait_path(
            unload_pos,
            delivery_time,
            UNLOAD_DURATION,
        )

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

        # 5. 回共享等待区
        path_home = self.astar_time(
            temp_res,
            agv_id,
            unload_path[-1][0],
            unload_path[-1][1],
            [SHARED_HOME_POS],
            extra_blocked=extra_blocked,
        )

        if path_home is None:
            return None

        temp_res.reserve_path(agv_id, path_home)

        segments.append((
            "return_home",
            False,
            "",
            "",
            "",
            "",
            path_home,
        ))

        distance = sum(
            self.path_distance(seg[-1])
            for seg in segments
        )

        delay = max(0, delivery_time - task["window_start"])
        finish_time = path_home[-1][1]

        return {
            "agv_id": agv_id,
            "segments": segments,
            "res": temp_res,
            "delivery_time": delivery_time,
            "finish_time": finish_time,
            "finish_pos": SHARED_HOME_POS,
            "distance": distance,
            "delay": delay,
            "pickup": pickup,
            "unload_pos": unload_pos,
        }

    def choose_agv_for_task(self, task):
        best = None

        for agv_id in self.active_agvs:
            plan = self.build_candidate_plan(agv_id, task)

            if plan is None:
                continue

            score = (
                plan["delay"] * 1000 + plan["finish_time"] + plan["distance"],
                plan["delay"],
                plan["finish_time"],
                int(agv_id),
            )

            if best is None or score < best[0]:
                best = (score, plan)

        if best is None:
            raise RuntimeError(f"任务 {task['task_id']} 找不到可行 AGV")

        return best[1]

    # ============================================================
    # 轨迹输出
    # ============================================================

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

    def path_pitch(self, prev_pos, curr_pos, old_pitch):
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]

        return PITCH_MAP.get((dx, dy), old_pitch) or old_pitch

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

        if not rows:
            rows.append(
                self.make_row(
                    path[0][1],
                    agv_id,
                    path[0][0],
                    180,
                    False,
                    "",
                    "",
                    "",
                    "",
                    "idle",
                )
            )

        while rows[-1]["timestamp"] < path[0][1] - 1:
            last = rows[-1]

            rows.append(
                self.make_row(
                    last["timestamp"] + 1,
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

        prev_pos = rows[-1]["pos"]
        pitch = rows[-1]["pitch"]

        for pos, t in path[1:]:
            pitch = self.path_pitch(prev_pos, pos, pitch)

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

    def commit_plan(self, task, plan):
        agv_id = plan["agv_id"]

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

        self.res = plan["res"]

        self.agv_states[agv_id]["pos"] = plan["finish_pos"]
        self.agv_states[agv_id]["time"] = plan["finish_time"]

        self.total_distance += plan["distance"]
        self.total_delay += plan["delay"]

        self.task_results.append({
            "task_id": task["task_id"],
            "destination_id": task["destination_id"],
            "destination_label": task["destination_label"],
            "material": task["material"],
            "agv_id": agv_id,
            "window_start": task["window_start"],
            "window_end": task["window_end"],
            "delivery_time": plan["delivery_time"],
            "delay": plan["delay"],
            "distance": plan["distance"],
            "pickup": plan["pickup"]["label"],
            "unload_x": plan["unload_pos"][0],
            "unload_y": plan["unload_pos"][1],
        })

    def fill_idle_until_end(self):
        max_t = 0

        for agv_id in self.active_agvs:
            if self.agv_tracks[agv_id]:
                max_t = max(max_t, self.agv_tracks[agv_id][-1]["timestamp"])

        for agv_id in self.active_agvs:
            rows = self.agv_tracks[agv_id]

            while rows and rows[-1]["timestamp"] < max_t:
                last = rows[-1]

                rows.append(
                    self.make_row(
                        last["timestamp"] + 1,
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

        rows = []

        for agv_id in self.active_agvs:
            last_by_time = {
                r["timestamp"]: r
                for r in self.agv_tracks[agv_id]
            }

            rows.extend(
                {k: r[k] for k in fields}
                for r in last_by_time.values()
            )

        rows.sort(key=lambda r: (int(r["timestamp"]), int(r["name"])))

        return rows

    def write_outputs(self):
        self.fill_idle_until_end()

        rows = self.build_rows()

        trajectory_fields = [
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
            writer = csv.DictWriter(f, fieldnames=trajectory_fields)
            writer.writeheader()
            writer.writerows(rows)

        result_fields = [
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
            "unload_x",
            "unload_y",
        ]

        with open(TASK_RESULT_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=result_fields)
            writer.writeheader()
            writer.writerows(self.task_results)

        max_delay = max((r["delay"] for r in self.task_results), default=0)

        avg_delay = (
            sum(r["delay"] for r in self.task_results) / len(self.task_results)
            if self.task_results
            else 0
        )

        makespan = max((int(r["timestamp"]) for r in rows), default=0)

        with open(SUMMARY_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["candidate_agv_count", len(self.homes)])
            writer.writerow(["used_agv_count", len(self.active_agvs)])
            writer.writerow(["pickup_count", len(self.pickups)])
            writer.writerow(["station_size", f"{STATION_W}x{STATION_H}"])
            writer.writerow(["total_delay", self.total_delay])
            writer.writerow(["max_delay", max_delay])
            writer.writerow(["avg_delay", round(avg_delay, 2)])
            writer.writerow(["total_distance", self.total_distance])
            writer.writerow(["makespan", makespan])
            if hasattr(self, "best_score"):
                writer.writerow(["best_score", round(self.best_score, 4)])
            if hasattr(self, "total_tardiness"):
                writer.writerow(["total_tardiness", self.total_tardiness])
            if hasattr(self, "max_tardiness"):
                writer.writerow(["max_tardiness", self.max_tardiness])
            if hasattr(self, "total_wait"):
                writer.writerow(["total_wait", self.total_wait])
            if hasattr(self, "workload_imbalance"):
                writer.writerow(["workload_imbalance", self.workload_imbalance])

    # ============================================================
    # 主流程：GA-A*-CP 全局优化
    # ============================================================

    def run(self):
        """
        用 GA-A*-CP 替换原来的逐任务贪心:
        1) GA 在上层搜索所有任务的分配与排序;
        2) A*-CP 在下层对每个完整染色体做时空路径规划与碰撞预约;
        3) 最终只把最优染色体的 detailed evaluation 写入原有 CSV 输出。
        """
        from ga_optimizer import GAAStarCPOptimizer

        optimizer = GAAStarCPOptimizer(
            planner=self,
            population_size=50,
            generations=100,
            crossover_rate=0.85,
            mutation_rate=0.25,
            elite_size=3,
            seed=42,
        )

        best_chromosome, best_eval = optimizer.solve()

        self.agv_tracks = best_eval.agv_tracks
        self.task_results = best_eval.task_results
        self.total_distance = best_eval.total_distance
        self.total_delay = best_eval.total_tardiness

        # 额外统计字段，write_outputs 会保持原有 CSV 兼容；
        # GA 相关输出由 optimizer 写入 ga_history.csv / best_chromosome.csv。
        self.best_score = best_eval.score
        self.total_tardiness = best_eval.total_tardiness
        self.max_tardiness = best_eval.max_tardiness
        self.total_wait = best_eval.total_wait
        self.workload_imbalance = best_eval.workload_imbalance

        self.write_outputs()

        print("=" * 60)
        print("GA-A*-CP 全局优化完成")
        print(f"实际启用 AGV 数量: {len(self.active_agvs)}")
        print(f"最优 score: {best_eval.score:.2f}")
        print(f"累计时间窗迟到 total_tardiness: {best_eval.total_tardiness}")
        print(f"最大时间窗迟到 max_tardiness: {best_eval.max_tardiness}")
        print(f"总行驶距离 total_distance: {best_eval.total_distance}")
        print(f"总等待时间 total_wait: {best_eval.total_wait}")
        print(f"完工时间 makespan: {best_eval.makespan}")
        print("输出文件: agv_trajectory.csv, agv_task_result.csv, agv_summary.csv")
        print("额外输出: ga_history.csv, best_chromosome.csv")


def main():
    Planner().run()


if __name__ == "__main__":
    main()
