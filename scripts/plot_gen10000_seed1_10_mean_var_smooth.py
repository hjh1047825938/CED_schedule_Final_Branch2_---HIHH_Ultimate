import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Config
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "gen10000_seed1_10_stats_every50.csv"
OUT_MEAN = ROOT / "results" / "gen10000_seed1_10_mean_curve_smooth_w5.png"
OUT_VAR = ROOT / "results" / "gen10000_seed1_10_var_curve_smooth_w5.png"

SERIES = [
    "GA",
    "DE",
    "GDE",
    "CCHIHH (non-stable)",
    "CCHIHH (stable)",
]

WINDOW = 5


def load_rows():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No data in {DATA_PATH}")
    return rows


def moving_average(vals, window):
    if window <= 1:
        return vals
    out = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_series(rows, kind, out_path, title, ylabel):
    gens = [int(r["gen"]) for r in rows]
    plt.figure(figsize=(9, 5.5))
    for name in SERIES:
        col = f"{name} {kind}"
        vals = [float(r[col]) for r in rows]
        smooth = moving_average(vals, WINDOW)
        plt.plot(gens, smooth, label=name, linewidth=2)

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
        f"Seed 1-10 Avg Convergence (every 50 gens, MA w={WINDOW})",
        "Best fitness (mean, smoothed)",
    )
    plot_series(
        rows,
        "var",
        OUT_VAR,
        f"Seed 1-10 Variance (every 50 gens, MA w={WINDOW})",
        "Best fitness (variance, smoothed)",
    )


if __name__ == "__main__":
    main()
