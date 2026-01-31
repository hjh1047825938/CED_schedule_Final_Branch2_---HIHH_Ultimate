import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Config
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "gen10000_seed0_20_avg_every50.csv"
OUT_PATH = ROOT / "results" / "gen10000_seed0_20_avg_every50_smooth_w5.png"
WINDOW = 5

# Load CSV
with DATA_PATH.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise SystemExit(f"No data in {DATA_PATH}")

# Parse columns
gens = [int(r["gen"]) for r in rows]
series_names = [
    "GA",
    "DE",
    "GDE",
    "CCHIHH (non-stable)",
    "CCHIHH (stable)",
]

values = {name: [float(r[name]) for r in rows] for name in series_names}

# Centered rolling mean with edge shrink (min_periods=1 behavior)

def rolling_mean(vals, window):
    out = []
    half = window // 2
    n = len(vals)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        subset = vals[start:end]
        out.append(sum(subset) / len(subset))
    return out

smoothed = {name: rolling_mean(vals, WINDOW) for name, vals in values.items()}

# Plot
plt.figure(figsize=(9, 5.5))
for name in series_names:
    plt.plot(gens, smoothed[name], label=name, linewidth=2)

plt.title("Seed 0-20 Avg Convergence (every 50 gens)\nSmoothed with window=5")
plt.xlabel("Generation")
plt.ylabel("Best fitness (avg)")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=200)
print(f"Saved: {OUT_PATH}")
