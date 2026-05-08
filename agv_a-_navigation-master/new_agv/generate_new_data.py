import csv
import random


# 1. 生成新的 agv_position.csv
def generate_positions():
    with open('agv_position.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'name', 'X', 'Y', 'pitch'])

        # 唯一的供货台 (右侧中间)
        writer.writerow(['start_point', 'Supply_Station', 19, 10, 0])

        # 卸货点 (左侧和上下两侧分布)
        for i in range(1, 16):
            writer.writerow(['end_point', f'Dest_{i}', 2, i * 2, 0])

        # 12台AGV (初始停靠在供货台附近的空地上)
        agv_starts = [(18, 8), (18, 9), (18, 11), (18, 12),
                      (17, 8), (17, 9), (17, 10), (17, 11), (17, 12),
                      (16, 9), (16, 10), (16, 11)]
        for i in range(12):
            writer.writerow(['agv', f'AGV_{i + 1}', agv_starts[i][0], agv_starts[i][1], 90])


# 2. 生成新的 agv_task.csv
def generate_tasks():
    with open('agv_task.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'start_point', 'end_point', 'tw_start', 'tw_end'])

        # 生成102个任务
        for i in range(1, 103):
            dest_id = random.randint(1, 10)
            # 时间窗设置：随机开始时间，结束时间至少为开始时间 + 3600秒(1小时)
            tw_start = random.randint(0, 5000)
            tw_end = tw_start + random.randint(3600, 7200)
            writer.writerow([f'Task_{i}', 'Supply_Station', f'Dest_{dest_id}', tw_start, tw_end])


if __name__ == '__main__':
    generate_positions()
    generate_tasks()
    print("新的位置和任务数据生成完毕！")