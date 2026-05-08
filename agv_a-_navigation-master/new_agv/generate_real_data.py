import csv
import random


def generate_exact_layout():
    with open('agv_position.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'name', 'X', 'Y', 'pitch'])

        # 1. 唯一供货台 (右侧中间过道)
        writer.writerow(['start_point', 'Supply_Station', 19, 10, 0])

        # 2. 精确设置卸货点 (110A, 110B... 440B)
        # 巷道所在列: 3, 7, 11, 15
        # 纵向位置: 分为4组(10, 20, 30, 40)，每组包含A/B两个紧邻位置
        x_columns = [3, 7, 11, 15]
        y_groups = [(2, 3), (6, 7), (10, 11), (14, 15)]  # 每组A/B的Y坐标

        for col_idx, x in enumerate(x_columns):
            lane_num = col_idx + 1  # 1xx, 2xx, 3xx, 4xx
            for group_idx, (y_a, y_b) in enumerate(y_groups):
                group_num = (group_idx + 1) * 10  # 10, 20, 30, 40

                # A位
                writer.writerow(['end_point', f"{lane_num}{group_num}A", x, y_a, 0])
                # B位
                writer.writerow(['end_point', f"{lane_num}{group_num}B", x, y_b, 0])

        # 3. 设置货架 (障碍物) - 位于卸货点列的两侧
        for x in [2, 4, 6, 8, 10, 12, 14, 16]:
            for y in range(1, 20):
                # 在 Y=4, 5, 8, 9, 12, 13 (组间距) 留出横向过道
                if y not in [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]:
                    # 只有不在卸货点坐标上的才是纯货架
                    writer.writerow(['obstacle', f'Shelf_{x}_{y}', x, y, 0])

        # 4. AGV 初始位置
        for i in range(12):
            writer.writerow(['agv', f'AGV_{i + 1}', 18, 5 + i, 90])


def generate_tasks():
    # 生成102个任务，目的地随机选择上述定义的库位
    dest_names = []
    for lane in [1, 2, 3, 4]:
        for group in [10, 20, 30, 40]:
            dest_names.extend([f"{lane}{group}A", f"{lane}{group}B"])

    with open('agv_task.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'start_point', 'end_point', 'tw_start', 'tw_end'])
        for i in range(1, 103):
            dest = random.choice(dest_names)
            tw_start = random.randint(0, 1000)
            writer.writerow([f'Task_{i}', 'Supply_Station', dest, tw_start, tw_start + 3600])


if __name__ == '__main__':
    generate_exact_layout()
    generate_tasks()
    print("已按照图片（130A, 130B）逻辑完成精确布局生成！")