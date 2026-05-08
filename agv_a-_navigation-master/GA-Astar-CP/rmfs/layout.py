from typing import Iterable, Set, Tuple
from .config import Params, Coord

class WarehouseLayout:
    """20x20 RMFS layout with back-to-back rack blocks approximating the paper."""
    def __init__(self, params: Params):
        self.params = params
        self.width = params.width
        self.height = params.height
        self.pods: Set[Coord] = self._build_pods()
        self.special = {params.station, params.workspace_entrance, params.workspace_exit}

    def _build_pods(self) -> Set[Coord]:
        # Six rack rows (two-cell height) by three rack columns (four-cell width),
        # giving 6 * 3 * 4 * 2 = 144 pod cells, as described in the article.
        pods = set()
        x_blocks = [(7, 10), (12, 15), (17, 20)]
        y_blocks = [(2, 3), (5, 6), (8, 9), (11, 12), (14, 15), (17, 18)]
        for xs in x_blocks:
            for ys in y_blocks:
                for x in range(xs[0], xs[1] + 1):
                    for y in range(ys[0], ys[1] + 1):
                        pods.add((x, y))
        return pods

    def in_bounds(self, p: Coord) -> bool:
        x, y = p
        return 1 <= x <= self.width and 1 <= y <= self.height

    def is_free_loaded(self, p: Coord) -> bool:
        # Loaded AGVs cannot move below pods; no-load AGVs may traverse pod cells.
        return self.in_bounds(p) and p not in self.pods

    def is_free_noload(self, p: Coord) -> bool:
        return self.in_bounds(p)

    def neighbors(self, p: Coord, loaded: bool) -> Iterable[Coord]:
        x, y = p
        for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if loaded:
                if self.is_free_loaded(q):
                    yield q
            else:
                if self.is_free_noload(q):
                    yield q

    def nearest_loaded_free(self, p: Coord) -> Coord:
        if self.is_free_loaded(p):
            return p
        frontier = [p]
        seen = {p}
        while frontier:
            new = []
            for u in frontier:
                for v in self.neighbors(u, loaded=False):
                    if v in seen:
                        continue
                    if self.is_free_loaded(v):
                        return v
                    seen.add(v)
                    new.append(v)
            frontier = new
        raise RuntimeError(f"No loaded-free neighbor found for {p}")
