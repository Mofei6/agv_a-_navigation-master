import heapq
from typing import Dict, List, Optional, Tuple
from .config import Coord
from .layout import WarehouseLayout

DIRS = [(1,0), (-1,0), (0,1), (0,-1)]

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def direction(a: Coord, b: Coord) -> Tuple[int, int]:
    return (b[0] - a[0], b[1] - a[1])

def astar(layout: WarehouseLayout, start: Coord, goal: Coord, loaded: bool, turn_penalty: float = 2.0) -> List[Coord]:
    """Improved A*: Manhattan heuristic + turn penalty F=G+H+HT."""
    if start == goal:
        return [start]
    if loaded:
        start = layout.nearest_loaded_free(start)
        goal = layout.nearest_loaded_free(goal)
    open_heap = []
    # state = (node, previous direction)
    start_state = (start, (0, 0))
    heapq.heappush(open_heap, (manhattan(start, goal), 0.0, start_state))
    parent: Dict[Tuple[Coord, Tuple[int, int]], Optional[Tuple[Coord, Tuple[int, int]]]] = {start_state: None}
    best_g: Dict[Tuple[Coord, Tuple[int, int]], float] = {start_state: 0.0}

    while open_heap:
        _, g, (u, prev_dir) = heapq.heappop(open_heap)
        if u == goal:
            path = [u]
            state = (u, prev_dir)
            while parent[state] is not None:
                state = parent[state]
                path.append(state[0])
            return list(reversed(path))
        # Directional expansion: move toward goal first, then other directions.
        cand = list(layout.neighbors(u, loaded=loaded))
        cand.sort(key=lambda q: manhattan(q, goal))
        for v in cand:
            nd = direction(u, v)
            turn = 0.0 if prev_dir == (0, 0) or prev_dir == nd else turn_penalty
            ng = g + 1.0 + turn
            ns = (v, nd)
            if ng < best_g.get(ns, float("inf")):
                best_g[ns] = ng
                parent[ns] = (u, prev_dir)
                heapq.heappush(open_heap, (ng + manhattan(v, goal), ng, ns))
    raise RuntimeError(f"A* failed from {start} to {goal}, loaded={loaded}")
