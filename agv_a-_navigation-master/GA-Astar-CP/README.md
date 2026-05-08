# RMFS Bi-level GA-A*-CP reproduction

This project is a runnable reproduction-style implementation of the paper's bi-level TAS-CPP idea:

- Upper level TAS: a genetic algorithm searches task allocation and sequencing.
- Lower level CPP: improved A* computes loaded/no-load paths with turn penalties.
- CP feedback: queue/conflict waiting and AGV cost are fed back to GA fitness.
- Visualizations: iteration curve, grid-based TAS result, and one AGV path-planning result.

## Important fix in this version

The first demo version could produce a straight convergence plot. This version fixes the core causes:

1. The chromosome is now a task permutation, not only an AGV-ID vector. Decoding by round-robin AGV assignment changes both task allocation and AGV task order.
2. The GA uses order crossover and swap/inversion mutation for a real permutation search space.
3. The A* cached path list is copied before concatenation. The previous code used `fixed += ...` on a cached list, which mutated cached paths and corrupted repeated evaluations.
4. The history records best-so-far metrics over 25 generations, so the convergence curve shows actual improvement.

## Run

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

Outputs are written to `output/`:

- `fig8_iteration_curve.png`
- `fig9_tas_grid.png`
- `fig10_path_planning.png`
- `results.json`

Typical runtime is about 10-20 seconds on this small 20x20 example.
