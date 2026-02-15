import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


SCALES = ["T100", "T200", "T500"]
SOLVERS = ["CCHIHH_Full", "CGA", "IMOMA"]
Z95 = 1.96


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    return float(x.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_root", default="results/cchihh_ablation_suite/baseline")
    parser.add_argument("--out_dir", default="results/baseline_full_cga_imoma_gde")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    baseline_root = root / args.baseline_root
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Table: per-scale summary for all four solvers
    table_rows = []
    for scale in SCALES:
        perf_rows = read_csv_rows(baseline_root / scale / "final_performance_summary.csv")
        by_cfg = {r["config"].strip(): r for r in perf_rows}
        row = [scale]
        for s in SOLVERS:
            rr = by_cfg[s]
            row.append(f"{to_float(rr['mean']):.6f} ± {to_float(rr['std']):.6f}")
        table_rows.append(row)

    table_path = out_dir / "table_baseline_full_cga_imoma_gde.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Problem", "CCHIHH-full (Mean±Std)", "CGA (Mean±Std)", "IMOMA (Mean±Std)"])
        w.writerows(table_rows)

    # Figure: 1x3 subplots, each with 4 curves and 95% CI
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "figure.figsize": (12, 4),
            "pdf.fonttype": 42,
        }
    )

    colors = {
        "CCHIHH_Full": "#1f77b4",
        "CGA": "#d62728",
        "IMOMA": "#2ca02c",
    }
    labels = {
        "CCHIHH_Full": "CCHIHH-full",
        "CGA": "CGA",
        "IMOMA": "IMOMA",
    }

    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    for i, scale in enumerate(SCALES):
        rows = read_csv_rows(baseline_root / scale / f"{scale}_baseline_mean_var.csv")
        gens = [int(float(r["gen"])) for r in rows]
        ax = axes[i]
        for s in SOLVERS:
            mean_key = f"{s} mean"
            var_key = f"{s} var"
            means = [to_float(r[mean_key]) for r in rows]
            vars_ = [to_float(r[var_key]) for r in rows]
            cis = [Z95 * math.sqrt(max(v, 0.0) / args.n) for v in vars_]
            c = colors[s]
            ax.plot(gens, means, label=labels[s], color=c)
            ax.fill_between(gens, [m - ci for m, ci in zip(means, cis)], [m + ci for m, ci in zip(means, cis)], color=c, alpha=0.12)

        ax.set_title(scale, fontweight="bold")
        ax.set_xlabel("Generation")
        if i == 0:
            ax.set_ylabel("Best fitness")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    pdf_path = out_dir / "baseline_full_cga_imoma_gde_mean_ci.pdf"
    png_path = out_dir / "baseline_full_cga_imoma_gde_mean_ci.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)

    print(f"Saved: {table_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
