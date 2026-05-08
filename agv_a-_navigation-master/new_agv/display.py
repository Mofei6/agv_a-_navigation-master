import pygame
import csv
import sys

# 初始化配置
GRID_SIZE = 30
MAP_WIDTH, MAP_HEIGHT = 21, 21
WINDOW_SIZE = (MAP_WIDTH * GRID_SIZE, MAP_HEIGHT * GRID_SIZE)

# 颜色定义
COLORS = {
    'bg': (240, 240, 240),
    'grid': (200, 200, 200),
    'supply': (255, 100, 100),  # 供货台红色
    'dest': (100, 200, 100),  # 卸货点绿色
    'agv': (50, 50, 255),  # AGV蓝色
    'agv_loading': (255, 215, 0)  # 正在装货的AGV金色
}


def load_static_map():
    supply = None
    dests = []
    with open('agv_position.csv', 'r') as f:
        for row in csv.DictReader(f):
            if row['type'] == 'start_point':
                supply = (int(row['X']), int(row['Y']))
            elif row['type'] == 'end_point':
                dests.append((int(row['X']), int(row['Y'])))
    return supply, dests


def load_trajectory():
    frames = {}
    max_time = 0
    with open('agv_trajectory.csv', 'r') as f:
        for row in csv.DictReader(f):
            t = int(row['timestamp'])
            if t not in frames: frames[t] = []
            frames[t].append({
                'name': row['name'],
                'pos': (int(row['X']), int(row['Y'])),
                'status': row['status']
            })
            if t > max_time: max_time = t
    return frames, max_time


def draw_map(screen, supply, dests):
    screen.fill(COLORS['bg'])
    # 画网格
    for x in range(MAP_WIDTH):
        for y in range(MAP_HEIGHT):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, COLORS['grid'], rect, 1)

    # 画卸货点
    for d in dests:
        pygame.draw.rect(screen, COLORS['dest'], (d[0] * GRID_SIZE, d[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # 画唯一的供货台(放大一点)
    if supply:
        pygame.draw.rect(screen, COLORS['supply'], (supply[0] * GRID_SIZE, supply[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("AGV Time-Window & Capacity Scheduler")
    clock = pygame.time.Clock()

    supply, dests = load_static_map()
    frames, max_time = load_trajectory()

    font = pygame.font.SysFont("Arial", 16)

    current_time = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_map(screen, supply, dests)

        # 画当前帧的AGV
        if current_time in frames:
            for agv in frames[current_time]:
                x, y = agv['pos']
                color = COLORS['agv_loading'] if agv['status'] == 'Loading' else COLORS['agv']
                # 用圆形代表AGV
                pygame.draw.circle(screen, color, (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2),
                                   GRID_SIZE // 2 - 2)

        # 显示时间
        time_text = font.render(f"Time: {current_time} s", True, (0, 0, 0))
        screen.blit(time_text, (10, 10))

        pygame.display.flip()

        if current_time <= max_time:
            current_time += 1
            # 可以调整这里的播放速度，比如60帧/秒模拟快进
            clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()