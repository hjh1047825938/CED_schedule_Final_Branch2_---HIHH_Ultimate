# CED Schedule - Cloud-Edge-Device Task Scheduling Optimization

A multi-method metaheuristic optimization solver for Cloud-Edge-Device (CED) task scheduling.

## Building on Windows

### Prerequisites
- Visual Studio 2022 (or 2019) with C++ Desktop development workload
- CMake 3.15 or higher

### Build Commands

```powershell
cd CED_schedule_Final_Branch2_分块+HIHH_CLEAN
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Quick Start (Windows)

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\Release\CED_Schedule.exe --solver GDE --seed 1 --data_dir .\data --data_file data_matrix_100.txt --generations 5
```

Sample output (short):

```
=== CED_Schedule Configuration ===
Data directory: ".\data"
Data file: data_matrix_100.txt
Gen 5: best_fit = 55.8008
```

## Command-Line Options

```
Options:
  --data_dir <path>    Data directory (default: ./data)
  --data_file <name>   Data file name (default: data_matrix_100.txt)
  --generations <n>    Number of generations (default: 500)
  --popsize <n>        Population size (default: 40)
  --seed <n>           Random seed (default: 42)
  --pini <f>           Heuristic init probability 0-1 (default: 0.4)
  --solver <name>      Solver: GA, DE, GDE, CCHIHH (default: GA)
  --migration          Enable rotated-ring subpopulation migration
  --nsubpop <n>        Number of subpopulations for migration (default: 8)
  --stable             Enable CCHIHH-Stable mode
  --resample_gate <n>  Stagnation gate for block resample (default: 15)
  --reward_clip <f>    Stable reward clip (default: 0.2)
  --eps0 <f>           Stable epsilon start (default: 0.2)
  --eps_min <f>        Stable epsilon min (default: 0.02)
  --eps_k <f>          Stable epsilon decay k (default: 0.01)
  --lr0 <f>            Stable learning rate start (default: 0.05)
  --lr_k <f>           Stable learning rate decay k (default: 0.002)
  --bench_eval <n>     Run evaluation benchmark with N iterations
  --init_only          Only run initialization and print Pini comparisons
  --synthetic          Run synthetic phi encoding/decoding self-check
  --help               Show this help message
```

### Examples

```powershell
# Basic GA run
.\build\Release\CED_Schedule.exe --data_dir .\data --data_file data_matrix_100.txt --seed 42

# GDE solver with migration
.\build\Release\CED_Schedule.exe --solver GDE --migration --nsubpop 8 --seed 42

# Benchmark
.\build\Release\CED_Schedule.exe --data_dir .\data --bench_eval 1000 --seed 42

# Pini init comparison (3 seeds)
.\build\Release\CED_Schedule.exe --data_dir .\data --init_only --seed 42

# Synthetic phi encoding check
.\build\Release\CED_Schedule.exe --synthetic --seed 42
```

## Algorithms

| Solver | Description |
|--------|-------------|
| GA | Genetic Algorithm (default) |
| DE | Differential Evolution |
| GDE | Gbest-centric DE with rollback crossover |

## Migration (Step 7)

Rotated-ring subpopulation migration:
- Population divided into `nsubpop` groups
- Every 5 generations, subpops exchange individuals in a ring topology
- Elitist-in-probability: 80% chance of sending best, 20% random
- Ring rotates via `Dispara = (Dispara + 1) mod (nsubpop - 1)`

## Data Files

Requires `data_matrix_100.txt` in data directory.

This repository includes a small, deterministic sample data file (synthetic values for self-check) at:

- `data/data_matrix_100.txt`

You can use it directly via `--data_dir .\\data --data_file data_matrix_100.txt`.

Additional datasets shipped in `data/`:
- `Machines_3000.txt`
- `Power_Consumption.txt`
- `Tasks_in_3000.txt`

## Reproducibility

To reproduce a small summary over seeds 1-10 (example: GA, 100 generations):

```powershell
1..10 | ForEach-Object {
  .\build\Release\CED_Schedule.exe --solver GA --seed $_ --generations 100 --data_dir .\data --data_file data_matrix_100.txt
}
```

## Debugging nearest-device

To enable nearest-device debug output:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DDEBUG_NEAREST=ON
cmake --build build --config Release
```
