import pygame
import csv
import sys
import os

# === 界面配置 ===
GRID_SIZE = 38
MAP_SIZE = 21
WIN_W, WIN_H = MAP_SIZE * GRID_SIZE, MAP_SIZE * GRID_SIZE + 60
FPS = 12

# === 颜色方案 ===
THEME = {
    'bg': (245, 245, 245),
    'grid': (220, 220, 220),
    'shelf': (100, 116, 139),  # 工业灰蓝色
    'station': (239, 68, 68),  # 供货台红色
    'target': (34, 197, 94),  # 卸货点绿色
    'agv': (59, 130, 246),  # AGV蓝色
    'agv_load': (245, 158, 11),  # 装货金色
    'text_light': (255, 255, 255),
    'text_dark': (30, 41, 59)
}


def load_data():
    static = {'supply': None, 'dests': {}, 'shelves': []}
    if os.path.exists('agv_position.csv'):
        with open('agv_position.csv', 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                p = (int(row['X']), int(row['Y']))
                if row['type'] == 'start_point':
                    static['supply'] = p
                elif row['type'] == 'end_point':
                    static['dests'][p] = row['name']
                elif row['type'] == 'obstacle':
                    static['shelves'].append(p)

    frames = {}
    max_t = 0
    if os.path.exists('agv_trajectory.csv'):
        with open('agv_trajectory.csv', 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                t = int(row['timestamp'])
                if t not in frames: frames[t] = []
                frames[t].append({
                    'pos': (int(row['X']), int(row['Y'])),
                    'name': row['name'],
                    'status': row.get('status', 'Moving')
                })
                max_t = max(max_t, t)
    return static, frames, max_t


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("仓库 AGV 精确布局监控系统")
    clock = pygame.time.Clock()

    # 尝试加载中文字体，如果失败则使用系统默认
    try:
        font_id = pygame.font.SysFont("Microsoft YaHei", 12, bold=True)
        font_ui = pygame.font.SysFont("Microsoft YaHei", 18)
    except:
        font_id = pygame.font.SysFont("Arial", 11, bold=True)
        font_ui = pygame.font.SysFont("Arial", 18)

    static_map, trajectory, total_t = load_data()
    cur_t, paused = 0, False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: paused = not paused

        screen.fill(THEME['bg'])

        # 1. 绘制网格与货架
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                rect = (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, THEME['grid'], rect, 1)

        for sx, sy in static_map['shelves']:
            pygame.draw.rect(screen, THEME['shelf'],
                             (sx * GRID_SIZE + 1, sy * GRID_SIZE + 1, GRID_SIZE - 2, GRID_SIZE - 2))

        # 2. 绘制卸货工位及编号（如 130A）
        for (dx, dy), name in static_map['dests'].items():
            rect = (dx * GRID_SIZE + 1, dy * GRID_SIZE + 1, GRID_SIZE - 2, GRID_SIZE - 2)
            pygame.draw.rect(screen, THEME['target'], rect)
            txt = font_id.render(name, True, THEME['text_light'])
            screen.blit(txt, txt.get_rect(center=(dx * GRID_SIZE + GRID_SIZE // 2, dy * GRID_SIZE + GRID_SIZE // 2)))

        # 3. 绘制供货台
        if static_map['supply']:
            sp = static_map['supply']
            pygame.draw.rect(screen, THEME['station'], (sp[0] * GRID_SIZE, sp[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            lbl = font_id.render("供货", True, THEME['text_light'])
            screen.blit(lbl,
                        lbl.get_rect(center=(sp[0] * GRID_SIZE + GRID_SIZE // 2, sp[1] * GRID_SIZE + GRID_SIZE // 2)))

        # 4. 绘制 AGV
        if cur_t in trajectory:
            for agv in trajectory[cur_t]:
                ax, ay = agv['pos']
                color = THEME['agv_load'] if agv['status'] == 'Loading' else THEME['agv']
                center = (ax * GRID_SIZE + GRID_SIZE // 2, ay * GRID_SIZE + GRID_SIZE // 2)
                pygame.draw.circle(screen, color, center, GRID_SIZE // 2 - 4)
                pygame.draw.circle(screen, THEME['text_dark'], center, GRID_SIZE // 2 - 4, 1)

        # 5. UI 信息栏
        pygame.draw.rect(screen, (255, 255, 255), (0, WIN_H - 60, WIN_W, 60))
        info = font_ui.render(f"当前时间: {cur_t}s | 任务进度: {min(100, int(cur_t / max(1, total_t) * 100))}%", True,
                              THEME['text_dark'])
        screen.blit(info, (20, WIN_H - 45))

        pygame.display.flip()
        if not paused and cur_t < total_t: cur_t += 1
        clock.tick(FPS)


if __name__ == "__main__":
    main()