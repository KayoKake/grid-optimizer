# Grid Optimizer

Finds the cheapest-to-build module-grid layout that reaches a target bits/sec,
minimizing the number of manual merges (the real build-time cost). Renders the
best layout to an HTML page that mirrors the in-game grid.

## Run

- **Windows:** double-click `Grid Optimizer.pyw` (opens a small GUI, no console window).
- **Any OS:** `python "Grid Optimizer.pyw"`

Requires **Python 3.8+**.

For the best results, also install OR-Tools (the exact CP-SAT solver):

```
pip install ortools
```

Without OR-Tools it still runs, but uses a heuristic only (slightly worse layouts).

## How it works

Set the target bits/sec, grid size, bits multiplier, max tier, and a time limit,
then Run. It warm-starts a heuristic (simulated annealing + local search + a
memetic population), feeds that into the CP-SAT exact solver, and keeps the best
of the two plus any previous champion — so a rerun never comes back worse.

## Build a standalone .exe (optional)

```
pip install pyinstaller ortools
pyinstaller --onefile --windowed --collect-all ortools ^
  --exclude-module pandas --exclude-module matplotlib --exclude-module scipy ^
  "Grid Optimizer.pyw"
```

Produces `dist/Grid Optimizer.exe` (~44 MB, self-contained, no install needed).
