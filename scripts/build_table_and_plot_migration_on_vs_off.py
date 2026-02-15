import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


SCALES = ["T100", "T200", "T500"]
Z95 = 1.96


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    return float(x.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="results/cchihh_ablation_suite/migration")
    parser.add_argument("--out_dir", default="results/migration_on_vs_off")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_root = root / args.root_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    table_rows = []
    for scale in SCALES:
        perf = read_csv_rows(data_root / scale / "final_performance_summary.csv")
        row_on = next(r for r in perf if r["config"].strip() == "migration_true")
        row_off = next(r for r in perf if r["config"].strip() == "migration_false")

        mean_on = to_float(row_on["mean"])
        std_on = to_float(row_on["std"])
        mean_off = to_float(row_off["mean"])
        std_off = to_float(row_off["std"])
        improvement = (mean_off - mean_on) / mean_off * 100.0

        wil = read_csv_rows(data_root / scale / "wilcoxon_pairwise.csv")
        wrow = next(r for r in wil if r["A"].strip() == "migration_true" and r["B"].strip() == "migration_false")
        p_value = to_float(wrow["p_value"])

        table_rows.append(
            [
                scale,
                f"{mean_on:.6f} ± {std_on:.6f}",
                f"{mean_off:.6f} ± {std_off:.6f}",
                f"{improvement:.4f}",
                f"{p_value:.9g}",
            ]
        )

    with (out_dir / "table_migration_on_vs_off.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Problem", "migration_on (Mean±Std)", "migration_off (Mean±Std)", "Improvement (%)", "p-value"])
        w.writerows(table_rows)

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

    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    for i, scale in enumerate(SCALES):
        rows = read_csv_rows(data_root / scale / f"{scale}_migration_mean_var.csv")
        gens = [int(float(r["gen"])) for r in rows]

        mean_on = [to_float(r["migration_true mean"]) for r in rows]
        var_on = [to_float(r["migration_true var"]) for r in rows]
        mean_off = [to_float(r["migration_false mean"]) for r in rows]
        var_off = [to_float(r["migration_false var"]) for r in rows]

        ci_on = [Z95 * math.sqrt(max(v, 0.0) / args.n) for v in var_on]
        ci_off = [Z95 * math.sqrt(max(v, 0.0) / args.n) for v in var_off]

        ax = axes[i]
        ax.plot(gens, mean_on, label="migration on", color="#1f77b4")
        ax.plot(gens, mean_off, label="migration off", color="#d62728")
        ax.fill_between(gens, [m - c for m, c in zip(mean_on, ci_on)], [m + c for m, c in zip(mean_on, ci_on)], color="#1f77b4", alpha=0.12)
        ax.fill_between(gens, [m - c for m, c in zip(mean_off, ci_off)], [m + c for m, c in zip(mean_off, ci_off)], color="#d62728", alpha=0.15)
        ax.set_title(scale, fontweight="bold")
        ax.set_xlabel("Generation")
        if i == 0:
            ax.set_ylabel("Best fitness")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    fig.savefig(out_dir / "migration_on_vs_off_mean_ci.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "migration_on_vs_off_mean_ci.png", bbox_inches="tight", dpi=300)

    print(f"Saved: {out_dir / 'table_migration_on_vs_off.csv'}")
    print(f"Saved: {out_dir / 'migration_on_vs_off_mean_ci.pdf'}")
    print(f"Saved: {out_dir / 'migration_on_vs_off_mean_ci.png'}")


if __name__ == "__main__":
    main()
