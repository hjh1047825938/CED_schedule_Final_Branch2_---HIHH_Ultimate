# Eval Supplement Runs

This supplement only fills missing results for the three new solvers.

- New solvers: RDE, L-SRTDE, NL-SHADE-LBC
- Seed: 1
- Eval budget: 400000
- Max parallel jobs: 2
- Main alpha=0.5 results are reused for alpha sensitivity at alpha=0.5.
- Main nominal results are reused as degradation baselines.

## Generated Files

- `results/eval/main/`: per-run main comparison JSON files
- `results/eval/degradation/`: per-run degradation JSON files
- `results/eval/alpha/`: per-run alpha sensitivity JSON files
- `results/eval/traces/`: convergence traces with eval count and elapsed seconds
- `results/eval/logs/`: raw run logs
- `results/eval/summaries/`: merged CSV/JSON summaries

## Counts

- Main runs: 9
- Degradation runs: 72
- Alpha records: 27

## Re-run

```powershell
python scripts/run_eval_new_solver_supplement.py --max_parallel 2
```