import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


SCALES = ["T100", "T200", "T500"]
SOLVERS = ["CCHIHH_Full", "CGA", "IMOMA"]
Z95 = 1.96
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


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    return float(x.strip())


def parse_csv_list(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_root", default="results/cchihh_ablation_suite/baseline")
    parser.add_argument("--out_dir", default="results/baseline_full_cga_imoma_gde")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--scales", default="T100,T200,T500")
    parser.add_argument("--solvers", default="CCHIHH_Full,CGA,IMOMA")
    parser.add_argument("--tag", default="baseline_full_cga_imoma_gde")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    baseline_root = root / args.baseline_root
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_csv_list(args.scales)
    solvers = parse_csv_list(args.solvers)
    if not scales:
        raise SystemExit("--scales cannot be empty")
    if not solvers:
        raise SystemExit("--solvers cannot be empty")

    table_rows = []
    for scale in scales:
        perf_rows = read_csv_rows(baseline_root / scale / "final_performance_summary.csv")
        by_cfg = {r["config"].strip(): r for r in perf_rows}
        row = [scale]
        for s in solvers:
            if s not in by_cfg:
                raise SystemExit(f"Solver {s} missing in {baseline_root / scale / 'final_performance_summary.csv'}")
            rr = by_cfg[s]
            row.append(f"{to_float(rr['mean']):.6f} +- {to_float(rr['std']):.6f}")
        table_rows.append(row)

    table_path = out_dir / f"table_{args.tag}.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["Problem"] + [f"{SOLVER_LABELS.get(s, s)} (Mean+-Std)" for s in solvers]
        w.writerow(header)
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

    fig, axes = plt.subplots(1, len(scales), constrained_layout=True)
    if len(scales) == 1:
        axes = [axes]

    for i, scale in enumerate(scales):
        rows = read_csv_rows(baseline_root / scale / f"{scale}_baseline_mean_var.csv")
        if not rows:
            raise SystemExit(f"No rows in {baseline_root / scale / f'{scale}_baseline_mean_var.csv'}")
        gens = [int(float(r["gen"])) for r in rows]
        ax = axes[i]
        for s in solvers:
            mean_key = f"{s} mean"
            var_key = f"{s} var"
            if mean_key not in rows[0] or var_key not in rows[0]:
                raise SystemExit(f"Missing mean/var columns for {s} in {baseline_root / scale / f'{scale}_baseline_mean_var.csv'}")
            means = [to_float(r[mean_key]) for r in rows]
            vars_ = [to_float(r[var_key]) for r in rows]
            cis = [Z95 * math.sqrt(max(v, 0.0) / args.n) for v in vars_]
            c = SOLVER_COLORS.get(s, "#1f77b4")
            ax.plot(gens, means, label=SOLVER_LABELS.get(s, s), color=c)
            ax.fill_between(gens, [m - ci for m, ci in zip(means, cis)], [m + ci for m, ci in zip(means, cis)], color=c, alpha=0.12)

        ax.set_title(scale, fontweight="bold")
        ax.set_xlabel("Generation")
        if i == 0:
            ax.set_ylabel("Best fitness")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    pdf_path = out_dir / f"{args.tag}_mean_ci.pdf"
    png_path = out_dir / f"{args.tag}_mean_ci.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)

    print(f"Saved: {table_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
