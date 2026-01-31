import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Config
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "gen10000_seed0_20_stats_every50.csv"
OUT_MEAN = ROOT / "results" / "gen10000_seed0_20_mean_curve.png"
OUT_VAR = ROOT / "results" / "gen10000_seed0_20_var_curve.png"

SERIES = [
    "GA",
    "DE",
    "GDE",
    "CCHIHH (non-stable)",
    "CCHIHH (stable)",
]


def load_rows():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No data in {DATA_PATH}")
    return rows


def plot_series(rows, kind, out_path, title, ylabel):
    gens = [int(r["gen"]) for r in rows]
    plt.figure(figsize=(9, 5.5))
    for name in SERIES:
        col = f"{name} {kind}"
        vals = [float(r[col]) for r in rows]
        plt.plot(gens, vals, label=name, linewidth=2)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


def main():
    rows = load_rows()
    plot_series(
        rows,
        "mean",
        OUT_MEAN,
        "Seed 0-20 Avg Convergence (every 50 gens)",
        "Best fitness (mean)",
    )
    plot_series(
        rows,
        "var",
        OUT_VAR,
        "Seed 0-20 Variance (every 50 gens)",
        "Best fitness (variance)",
    )


if __name__ == "__main__":
    main()
