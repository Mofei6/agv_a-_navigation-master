import csv
import math


class AGVNavigator:
    def __init__(self):
        self.supply_station = (19, 10)
        self.station_capacity = 3  # 供货台最大同时容纳AGV数量

        self.agvs = {}  # {agv_id: {"pos": (x,y), "free_time": 0}}
        self.tasks = []  # [(task_id, dest_pos, tw_start, tw_end)]
        self.trajectories = []  # 记录轨迹准备输出

        # 时空预约表：用于避障 reservation_table[(x, y, t)] = agv_id
        self.reservation_table = {}
        # 供货台占用表：station_occupancy[t] = count
        self.station_occupancy = {}

    def load_data(self):
        # 加载AGV位置
        with open('agv_position.csv', 'r') as f:
            reader = csv.DictReader(f)
            self.destinations = {}
            for row in reader:
                if row['type'] == 'agv':
                    self.agvs[row['name']] = {"pos": (int(row['X']), int(row['Y'])), "free_time": 0}
                elif row['type'] == 'end_point':
                    self.destinations[row['name']] = (int(row['X']), int(row['Y']))

        # 加载任务 (按时间窗开始时间排序)
        with open('agv_task.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.tasks.append({
                    "id": row['task_id'],
                    "dest": self.destinations[row['end_point']],
                    "tw_start": int(row['tw_start']),
                    "tw_end": int(row['tw_end'])
                })
        self.tasks.sort(key=lambda x: x['tw_start'])

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def space_time_a_star(self, start_pos, goal_pos, start_time, agv_id, is_pickup=False):
        """带有容量限制和时空避障的A*算法"""
        open_set = [(0, start_time, start_pos, [])]  # (f, time, pos, path)
        closed_set = set()

        while open_set:
            open_set.sort(key=lambda x: x[0])
            _, curr_t, curr_pos, path = open_set.pop(0)

            state = (curr_pos[0], curr_pos[1], curr_t)
            if state in closed_set:
                continue
            closed_set.add(state)

            if curr_pos == goal_pos:
                return path + [curr_pos]

            # 扩展邻居：上下左右 + 原地等待
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
            for dx, dy in neighbors:
                nxt_pos = (curr_pos[0] + dx, curr_pos[1] + dy)
                nxt_t = curr_t + 1

                # 1. 边界检测 (假设网格 21x21)
                if not (0 <= nxt_pos[0] <= 20 and 0 <= nxt_pos[1] <= 20): continue

                # 2. 节点碰撞检测
                if self.reservation_table.get((nxt_pos[0], nxt_pos[1], nxt_t), agv_id) != agv_id: continue

                # 3. 对穿碰撞检测 (边碰撞)
                if self.reservation_table.get((nxt_pos[0], nxt_pos[1], curr_t)) is not None and \
                        self.reservation_table.get((curr_pos[0], curr_pos[1], nxt_t)) == self.reservation_table.get(
                    (nxt_pos[0], nxt_pos[1], curr_t)):
                    continue

                # 4. **核心需求：供货台容量检测**
                if nxt_pos == self.supply_station:
                    if self.station_occupancy.get(nxt_t, 0) >= self.station_capacity:
                        continue  # 供货台满了，不能进入

                g = nxt_t - start_time
                h = self.heuristic(nxt_pos, goal_pos)
                open_set.append((g + h, nxt_t, nxt_pos, path + [curr_pos]))

        return None  # 找不到路径

    def record_path(self, agv_id, task_id, path, start_time, is_loading=False):
        curr_t = start_time
        for pos in path:
            self.reservation_table[(pos[0], pos[1], curr_t)] = agv_id
            if pos == self.supply_station:
                self.station_occupancy[curr_t] = self.station_occupancy.get(curr_t, 0) + 1

            self.trajectories.append({
                "timestamp": curr_t, "name": agv_id, "X": pos[0], "Y": pos[1],
                "task_id": task_id, "status": "Loading" if is_loading else "Moving"
            })
            curr_t += 1
        return curr_t - 1

    def solve(self):
        self.load_data()

        # 遍历分配任务
        for task in self.tasks:
            # 找最早有空的AGV
            agv_id = min(self.agvs.keys(), key=lambda k: self.agvs[k]['free_time'])
            agv = self.agvs[agv_id]

            curr_t = agv['free_time']
            curr_pos = agv['pos']

            # --- 解决时间窗问题 ---
            # 预估到达卸货点的时间 = 当前时间 + 到供货台距离 + 供货台到卸货点距离
            est_arrival = curr_t + self.heuristic(curr_pos, self.supply_station) + self.heuristic(self.supply_station,
                                                                                                  task['dest'])

            # 如果预估到达时间比时间窗早，让AGV在原位置待机快进时间
            if est_arrival < task['tw_start']:
                wait_time = task['tw_start'] - est_arrival
                # 记录待机路径
                wait_path = [curr_pos] * wait_time
                curr_t = self.record_path(agv_id, "WAIT", wait_path, curr_t) + 1

            # 1. 前往供货台 (取货)
            path_to_station = self.space_time_a_star(curr_pos, self.supply_station, curr_t, agv_id, is_pickup=True)
            if not path_to_station: print(f"Warning: {agv_id} 无法到达供货台"); continue
            curr_t = self.record_path(agv_id, task['id'], path_to_station, curr_t) + 1

            # 模拟在供货台装货停留 2 秒
            curr_t = self.record_path(agv_id, task['id'], [self.supply_station] * 2, curr_t, is_loading=True) + 1

            # 2. 前往卸货点 (送货)
            path_to_dest = self.space_time_a_star(self.supply_station, task['dest'], curr_t, agv_id)
            if not path_to_dest: print(f"Warning: {agv_id} 无法到达卸货点"); continue
            curr_t = self.record_path(agv_id, task['id'], path_to_dest, curr_t) + 1

            # 更新AGV状态
            self.agvs[agv_id]['pos'] = task['dest']
            self.agvs[agv_id]['free_time'] = curr_t

        self.export_trajectory()

    def export_trajectory(self):
        with open('agv_trajectory.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'name', 'X', 'Y', 'task_id', 'status'])
            for t in sorted(self.trajectories, key=lambda x: x['timestamp']):
                writer.writerow([t['timestamp'], t['name'], t['X'], t['Y'], t['task_id'], t['status']])
        print("调度完成，已生成 agv_trajectory.csv")


if __name__ == '__main__':
    nav = AGVNavigator()
    nav.solve()