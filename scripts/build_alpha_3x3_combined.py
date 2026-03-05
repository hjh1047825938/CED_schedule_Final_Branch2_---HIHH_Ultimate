import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ALPHAS = ["0.2", "0.5", "0.8"]
SCALES = ["T100", "T200", "T500"]
SOLVERS = ["CCHIHH_Full", "CGA", "IMOMA", "DSAC_DE"]
SOLVER_COLORS = {
    "CCHIHH_Full": "#1f77b4",
    "CGA": "#d62728",
    "IMOMA": "#2ca02c",
    "DSAC_DE": "#ff7f0e",
}
SOLVER_LABELS = {
    "CCHIHH_Full": "CCHIHH-full",
    "CGA": "CGA",
    "IMOMA": "IMOMA",
    "DSAC_DE": "DSAC-DE",
}
Z95 = 1.96


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    return float(x.strip())


def main():
    ap = argparse.ArgumentParser(description="Build a 3x3 combined alpha-scale figure.")
    ap.add_argument(
        "--aggregated_root",
        default="outputs/results/alpha_sensitivity_cga_imoma_dsac/aggregated",
    )
    ap.add_argument(
        "--out_dir",
        default="outputs/results/alpha_sensitivity_cga_imoma_dsac/tables",
    )
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    agg_root = root / args.aggregated_root
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
            "figure.figsize": (14, 11),
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 3, constrained_layout=True)

    for r, alpha in enumerate(ALPHAS):
        for c, scale in enumerate(SCALES):
            ax = axes[r][c]
            fp = agg_root / f"alpha_{alpha}" / scale / f"{scale}_baseline_mean_var.csv"
            rows = read_csv_rows(fp)
            gens = [int(float(row["gen"])) for row in rows]

            for s in SOLVERS:
                mean_key = f"{s} mean"
                var_key = f"{s} var"
                means = [to_float(row[mean_key]) for row in rows]
                vars_ = [to_float(row[var_key]) for row in rows]
                cis = [Z95 * math.sqrt(max(v, 0.0) / max(1, args.n)) for v in vars_]
                color = SOLVER_COLORS[s]
                ax.plot(gens, means, color=color, label=SOLVER_LABELS[s])
                ax.fill_between(
                    gens,
                    [m - ci for m, ci in zip(means, cis)],
                    [m + ci for m, ci in zip(means, cis)],
                    color=color,
                    alpha=0.12,
                )

            ax.set_title(f"{scale} | alpha={alpha}")
            if r == 2:
                ax.set_xlabel("Generation")
            if c == 0:
                ax.set_ylabel("Best fitness")
            ax.grid(True, linestyle="--", alpha=0.6)
            if r == 0 and c == 0:
                ax.legend()

    pdf_path = out_dir / "alpha_3x3_combined_mean_ci.pdf"
    png_path = out_dir / "alpha_3x3_combined_mean_ci.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
