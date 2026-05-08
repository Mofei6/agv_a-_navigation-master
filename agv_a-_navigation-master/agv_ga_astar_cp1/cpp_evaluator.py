
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


Coord = Tuple[int, int]


@dataclass
class Chromosome:
    """Upper-level TAS chromosome.

    task_order:
        A permutation of all task IDs. This controls global sequencing pressure.
    agv_assign:
        A mapping task_id -> agv_id. This controls task allocation.
    """
    task_order: List[str]
    agv_assign: Dict[str, str]


@dataclass
class EvaluationResult:
    feasible: bool
    score: float
    total_tardiness: int
    max_tardiness: int
    late_task_count: int
    makespan: int
    total_distance: int
    total_wait: int
    workload_imbalance: float
    task_results: List[dict]
    agv_tracks: Dict[str, List[dict]]
    agv_task_sequences: Dict[str, List[str]]
    infeasible_count: int = 0


class CPPEvaluator:
    """Lower-level A*-CP evaluator.

    Given a complete TAS chromosome, it rebuilds a fresh reservation table and
    simulates all AGVs from scratch. This avoids the old greedy problem where a
    committed early decision cannot be repaired by later tasks.
    """

    def __init__(self, planner):
        self.planner = planner
        self.home_pos = (50, 26)
        self.load_duration = 2
        self.unload_duration = 2
        self.big_m = 10**9

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, chrom: Chromosome) -> Dict[str, List[dict]]:
        task_by_id = {t["task_id"]: t for t in self.planner.tasks}
        seqs = {agv_id: [] for agv_id in self.planner.active_agvs}

        # task_order determines each AGV's internal sequence after assignment.
        for tid in chrom.task_order:
            task = task_by_id[tid]
            agv_id = chrom.agv_assign.get(tid)
            if agv_id not in seqs:
                # Repair invalid assignment deterministically.
                agv_id = self.planner.active_agvs[hash(tid) % len(self.planner.active_agvs)]
                chrom.agv_assign[tid] = agv_id
            seqs[agv_id].append(task)

        return seqs

    # ------------------------------------------------------------------
    # Local track utilities; these mirror Planner.append_path_rows but do
    # not mutate the Planner during chromosome evaluation.
    # ------------------------------------------------------------------

    def _append_path_rows(
        self,
        tracks,
        agv_id,
        path,
        loaded,
        material,
        dest_label,
        dest_id,
        task_id,
        status,
    ):
        rows = tracks[agv_id]

        if not rows:
            rows.append(
                self.planner.make_row(
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

        # Fill idle gap before the new segment starts.
        while rows[-1]["timestamp"] < path[0][1] - 1:
            last = rows[-1]
            rows.append(
                self.planner.make_row(
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
            pitch = self.planner.path_pitch(prev_pos, pos, pitch)
            rows.append(
                self.planner.make_row(
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

    def _fill_idle_until_end(self, tracks):
        max_t = 0
        for agv_id in self.planner.active_agvs:
            if tracks[agv_id]:
                max_t = max(max_t, tracks[agv_id][-1]["timestamp"])

        for agv_id in self.planner.active_agvs:
            rows = tracks[agv_id]
            while rows and rows[-1]["timestamp"] < max_t:
                last = rows[-1]
                rows.append(
                    self.planner.make_row(
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

    # ------------------------------------------------------------------
    # One task planning under A*-CP
    # ------------------------------------------------------------------

    def _plan_one_task(self, res, agv_state, task, return_home_after_task: bool):
        agv_id = agv_state["id"]
        cur_pos = agv_state["pos"]
        cur_time = agv_state["time"]
        segments = []
        planned_wait = 0

        # If AGV starts at shared home, do not reserve long idle time there.
        # Leave just in time for the pickup release.
        if cur_pos == self.home_pos:
            dist_to_pick = min(
                self.planner.manhattan(cur_pos, pickup["pos"]) if hasattr(self.planner, "manhattan") else abs(cur_pos[0]-pickup["pos"][0])+abs(cur_pos[1]-pickup["pos"][1])
                for pickup in self.planner.pickups.values()
            )
            cur_time = max(cur_time, task["release_time"] - dist_to_pick - 5, 0)

        # 1. To pickup. The old project already supports multiple pickup slots.
        pickup_plan = self.planner.choose_best_pickup_path(
            res,
            agv_id,
            cur_pos,
            cur_time,
            task["release_time"],
        )
        if pickup_plan is None:
            return None

        pickup, path_to_pick = pickup_plan
        res.reserve_path(agv_id, path_to_pick)
        planned_wait += max(0, self.planner.path_distance(path_to_pick) - (path_to_pick[-1][1] - path_to_pick[0][1]))

        segments.append(("to_pickup", False, "", "", "", "", path_to_pick))

        # 2. Loading.
        load_path = self.planner.wait_path(
            pickup["pos"],
            path_to_pick[-1][1],
            self.load_duration,
        )
        res.reserve_path(agv_id, load_path)
        segments.append(("loading", False, "", "", "", "", load_path))

        # 3. Loaded delivery. Loaded AGV cannot pass through pickup cells.
        extra_blocked = {pickup_item["pos"] for pickup_item in self.planner.pickups.values()}
        goals = self.planner.delivery_goals(task["destination_id"], extra_blocked)

        # Time-window handling:
        # - release is already enforced at pickup;
        # - delivery should not be earlier than window_start;
        # - tardiness is measured against window_end.
        path_to_drop = self.planner.astar_time(
            res,
            agv_id,
            pickup["pos"],
            load_path[-1][1],
            goals,
            earliest_goal_time=task["window_start"],
            extra_blocked=extra_blocked,
            goal_hold=self.unload_duration,
        )
        if path_to_drop is None:
            return None

        res.reserve_path(agv_id, path_to_drop)
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

        arrival_time = path_to_drop[-1][1]

        # 如果早于 window_start 到达，可以等待到 window_start 再卸货；
        # 但延迟只看是否超过 window_end。
        window_start = int(task["window_start"])
        window_end = int(task["window_end"])
        unload_start = max(arrival_time, window_start)
        unload_finish = unload_start + self.unload_duration
        delay = max(0, unload_finish - window_end)
        tardiness = delay
        # # 延迟按 AGV 到达时间是否超过最晚时间窗计算
        # delay = max(0, arrival_time - window_end)
        # tardiness = delay

        # 4. Unloading.
        unload_path = self.planner.wait_path(
            unload_pos,
            arrival_time,
            self.unload_duration,
        )
        res.reserve_path(agv_id, unload_path)
        segments.append((
            "unloading",
            True,
            task["material"],
            task["destination_label"],
            task["destination_id"],
            task["task_id"],
            unload_path,
        ))

        # 5. 是否返回共享等候区。
        # 如果该 AGV 后续还有任务，则不返回等候区，直接从当前卸货点出发去下一个取货点。
        # 如果该 AGV 后续没有任务，则返回等候区，避免最终停在工位卸货点附近占用通道。
        if return_home_after_task:
            path_home = self.planner.astar_time(
                res,
                agv_id,
                unload_path[-1][0],
                unload_path[-1][1],
                [self.home_pos],
                extra_blocked=extra_blocked,
            )
            if path_home is None:
                return None

            res.reserve_path(agv_id, path_home)
            segments.append(("return_home", False, "", "", "", "", path_home))

            finish_pos = self.home_pos
            finish_time = path_home[-1][1]
        else:
            # 不返航：任务完成后，AGV 的当前位置就是卸货点。
            # 下一次 _plan_one_task 会从该卸货点和当前时间继续做 A*-CP 路径规划。
            finish_pos = unload_pos
            finish_time = unload_path[-1][1]

        distance = sum(self.planner.path_distance(seg[-1]) for seg in segments)
        # tardiness = max(0, arrival_time - task["window_end"])

        # Wait estimate: elapsed time minus movement and fixed service.
        elapsed = finish_time - cur_time
        fixed_service = self.load_duration + self.unload_duration
        estimated_wait = max(0, elapsed - distance - fixed_service)

        return {
            "agv_id": agv_id,
            "segments": segments,
            "delivery_time": arrival_time,
            "finish_time": finish_time,
            "finish_pos": finish_pos,
            "distance": distance,
            "tardiness": tardiness,
            "estimated_wait": estimated_wait,
            "pickup": pickup,
            "unload_pos": unload_pos,
            "arrival_time": arrival_time,
            # "tardiness": tardiness,
        }
    # ------------------------------------------------------------------
    # Full chromosome evaluation
    # ------------------------------------------------------------------

    def evaluate(self, chrom: Chromosome, detailed: bool = False) -> EvaluationResult:
        seqs = self.decode(chrom)

        # Fresh lower-level state for this chromosome.
        res = self.planner.res.__class__()
        agv_states = {
            agv_id: {
                "id": agv_id,
                "pos": self.home_pos,
                "time": 0,
                "pitch": self.planner.homes[agv_id]["pitch"],
                "home": self.home_pos,
            }
            for agv_id in self.planner.active_agvs
        }

        tracks = defaultdict(list)
        for agv_id in self.planner.active_agvs:
            tracks[agv_id].append(
                self.planner.make_row(
                    0,
                    agv_id,
                    self.home_pos,
                    180,
                    False,
                    "",
                    "",
                    "",
                    "",
                    "idle",
                )
            )

        indices = {agv_id: 0 for agv_id in self.planner.active_agvs}
        remaining = sum(len(v) for v in seqs.values())

        task_results = []
        total_distance = 0
        total_tardiness = 0
        max_tardiness = 0
        total_wait = 0
        infeasible_count = 0

        while remaining > 0:
            candidates = [
                agv_id
                for agv_id in self.planner.active_agvs
                if indices[agv_id] < len(seqs[agv_id])
            ]

            # Dynamic CP priority approximation:
            # earliest available AGV first; among ties, task with earlier window_end first.
            candidates.sort(
                key=lambda aid: (
                    agv_states[aid]["time"],
                    seqs[aid][indices[aid]]["window_end"],
                    int(aid),
                )
            )

            agv_id = candidates[0]
            task = seqs[agv_id][indices[agv_id]]

            # 该 AGV 后面是否还有任务。
            # 如果还有任务，则不回等候区；
            # 如果没有任务，则最后一次任务完成后返回等候区。
            has_next_task = indices[agv_id] + 1 < len(seqs[agv_id])
            return_home_after_task = not has_next_task

            plan = self._plan_one_task(
                res,
                agv_states[agv_id],
                task,
                return_home_after_task=return_home_after_task,
            )
            if plan is None:
                infeasible_count += 1
                # Penalize but keep schedule moving to avoid infinite loops.
                agv_states[agv_id]["time"] += 100
                indices[agv_id] += 1
                remaining -= 1
                continue

            for status, loaded, material, dest_label, dest_id, task_id, path in plan["segments"]:
                self._append_path_rows(
                    tracks,
                    agv_id,
                    path,
                    loaded,
                    material,
                    dest_label,
                    dest_id,
                    task_id,
                    status,
                )

            agv_states[agv_id]["pos"] = plan["finish_pos"]
            agv_states[agv_id]["time"] = plan["finish_time"]

            total_distance += plan["distance"]
            total_tardiness += plan["tardiness"]
            max_tardiness = max(max_tardiness, plan["tardiness"])
            total_wait += plan["estimated_wait"]

            task_results.append({
                "task_id": task["task_id"],
                "destination_id": task["destination_id"],
                "destination_label": task["destination_label"],
                "material": task["material"],
                "agv_id": agv_id,
                "window_start": task["window_start"],
                "window_end": task["window_end"],

                # AGV 到达卸货点时间
                "arrival_time": plan["arrival_time"],

                # 兼容旧代码，delivery_time 也设为到达时间
                "delivery_time": plan["arrival_time"],

                # 延迟 = max(0, arrival_time - window_end)
                "delay": plan["tardiness"],

                "distance": plan["distance"],
                "pickup": plan["pickup"]["label"],
                "unload_x": plan["unload_pos"][0],
                "unload_y": plan["unload_pos"][1],
            })
            indices[agv_id] += 1
            remaining -= 1

        self._fill_idle_until_end(tracks)

        makespan = max((tracks[aid][-1]["timestamp"] for aid in self.planner.active_agvs if tracks[aid]), default=0)
        late_task_count = sum(1 for r in task_results if r["delay"] > 0)
        workloads = [sum(1 for r in task_results if r["agv_id"] == aid) for aid in self.planner.active_agvs]
        workload_imbalance = max(workloads) - min(workloads) if workloads else 0
        unused_agv_count = sum(1 for v in workloads if v == 0)
        # Lexicographic objective encoded as weighted scalar:
        # feasibility > time-window violations > makespan > distance > waiting > balance.
        score = (
                10000 * infeasible_count
                # + 100_000 * late_task_count
                + 100 * total_tardiness
                + 100 * max_tardiness
                + 100 * makespan
                # + 1 * total_distance
                # + 2 * total_wait
                # + 500 * workload_imbalance
                # + 2_000 * unused_agv_count
        )

        seq_out = {aid: [task["task_id"] for task in seq] for aid, seq in seqs.items()}

        return EvaluationResult(
            feasible=(infeasible_count == 0),
            score=score,
            total_tardiness=total_tardiness,
            max_tardiness=max_tardiness,
            late_task_count=late_task_count,
            makespan=makespan,
            total_distance=total_distance,
            total_wait=total_wait,
            workload_imbalance=workload_imbalance,
            task_results=task_results,
            agv_tracks=tracks,
            agv_task_sequences=seq_out,
            infeasible_count=infeasible_count,
        )
