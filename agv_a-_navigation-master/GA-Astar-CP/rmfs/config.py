from dataclasses import dataclass
from typing import List, Tuple

Coord = Tuple[int, int]  # 1-based (x, y)

@dataclass(frozen=True)
class Task:
    task_id: int
    grid_no: int
    loc: Coord
    pick_time: int = 2

@dataclass(frozen=True)
class AGV:
    agv_id: int
    grid_no: int
    start: Coord

@dataclass
class Params:
    width: int = 20
    height: int = 20
    station: Coord = (6, 1)
    workspace_entrance: Coord = (3, 10)
    workspace_exit: Coord = (6, 7)

    # GA parameters. The previous demo used only 6 individuals and 6 generations,
    # which made the best-so-far curve frequently become a straight line. These
    # defaults are still fast, but large enough for visible TAS optimization.
    population_size: int = 100
    generations: int = 100
    crossover_prob: float = 0.85
    mutation_prob: float = 0.20
    elite_size: int = 2
    tournament_k: int = 3

    turn_penalty: float = 2.0
    max_wait: int = 30
    seed: int = 7

    # The original article reports costs per hour. Here we convert an approximate
    # loaded/no-load running cost into CNY per second for a runnable simulation.
    cost_noload_per_s: float = 3.258 / 3600.0
    cost_loaded_per_s: float = 4.554 / 3600.0
    cost_wait_per_s: float = 0.8 / 3600.0
    cost_block_per_s: float = 1.0 / 3600.0

AGVS: List[AGV] = [
    AGV(1, 130, (7, 10)),
    AGV(2, 251, (13, 11)),
    AGV(3, 196, (10, 16)),
    AGV(4, 311, (16, 11)),
    AGV(5, 190, (10, 10)),
]

TASKS: List[Task] = [
    Task(1, 171, (9, 11)), Task(2, 269, (14, 9)), Task(3, 147, (8, 7)),
    Task(4, 232, (12, 12)), Task(5, 173, (9, 13)), Task(6, 86, (5, 6)),
    Task(7, 287, (15, 7)), Task(8, 208, (11, 8)), Task(9, 46, (3, 6)),
    Task(10, 289, (15, 9)), Task(11, 266, (14, 6)), Task(12, 206, (11, 6)),
    Task(13, 219, (11, 19)), Task(14, 267, (14, 7)), Task(15, 337, (17, 17)),
    Task(16, 153, (8, 13)), Task(17, 214, (11, 14)), Task(18, 227, (12, 7)),
    Task(19, 51, (3, 11)), Task(20, 234, (12, 14)), Task(21, 107, (6, 7)),
    Task(22, 233, (12, 13)), Task(23, 171, (9, 11)), Task(24, 167, (9, 7)),
    Task(25, 271, (14, 11)), Task(26, 292, (15, 12)), Task(27, 233, (12, 13)),
    Task(28, 274, (14, 14)), Task(29, 109, (6, 9)), Task(30, 174, (9, 14)),
]
