import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No data in {path}")
    return rows, reader.fieldnames


def collect_series(files, columns):
    gens = None
    series = {col: [] for col in columns}
    for fp in files:
        rows, fields = load_csv(fp)
        if gens is None:
            gens = [int(r["gen"]) for r in rows]
        else:
            gens_fp = [int(r["gen"]) for r in rows]
            if gens_fp != gens:
                raise SystemExit(f"Generation mismatch in {fp}")
        for col in columns:
            if col not in fields:
                raise SystemExit(f"Missing column {col} in {fp}")
            series[col].append([float(r[col]) for r in rows])
    return gens, series


def mean(vals):
    return sum(vals) / len(vals)


def plot_group_mean_only(gens, series_map, colors, out_prefix: Path, title: str):
    plt.figure(figsize=(9, 5.5))
    for label, data in series_map.items():
        means = []
        for i in range(len(gens)):
            vals = [seed_series[i] for seed_series in data]
            means.append(mean(vals))
        color = colors.get(label)
        plt.plot(gens, means, label=label, linewidth=2, color=color)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Selection frequency (mean)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".png"), dpi=220)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results/ablation_cchihh_bandit_fixed_gen10000_seed1_10",
        help="Directory containing CCHIHH_cfg1_bandit_on_opstats_seed*.csv",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    stats_dir = root / args.results_dir
    files = sorted(stats_dir.glob("CCHIHH_cfg1_bandit_on_opstats_seed*.csv"))
    if len(files) != 10:
        raise SystemExit(f"Expected 10 op-stats files, got {len(files)} in {stats_dir}")

    colors = {
        "GA": "#1f77b4",
        "DE": "#ff7f0e",
        "GDE": "#2ca02c",
        "BITFLIP": "#d62728",
        "SWAP": "#9467bd",
        "VNS": "#8c564b",
        "LEVY": "#e377c2",
        "RESAMPLE": "#7f7f7f",
    }

    groups = {
        "offload": [
            ("offload_GA", "GA"),
            ("offload_DE", "DE"),
            ("offload_BITFLIP", "BITFLIP"),
            ("offload_RESAMPLE", "RESAMPLE"),
        ],
        "seq": [
            ("seq_GA", "GA"),
            ("seq_SWAP", "SWAP"),
            ("seq_VNS", "VNS"),
            ("seq_RESAMPLE", "RESAMPLE"),
        ],
        "dev": [
            ("dev_DE", "DE"),
            ("dev_GDE", "GDE"),
            ("dev_LEVY", "LEVY"),
            ("dev_RESAMPLE", "RESAMPLE"),
        ],
        "overall": [
            ("overall_GA", "GA"),
            ("overall_DE", "DE"),
            ("overall_GDE", "GDE"),
            ("overall_BITFLIP", "BITFLIP"),
            ("overall_SWAP", "SWAP"),
            ("overall_VNS", "VNS"),
            ("overall_LEVY", "LEVY"),
            ("overall_RESAMPLE", "RESAMPLE"),
        ],
    }

    out_dir = stats_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, cols in groups.items():
        col_names = [c for c, _ in cols]
        gens, series = collect_series(files, col_names)
        series_map = {}
        for col, label in cols:
            series_map[label] = series[col]
        plot_group_mean_only(
            gens,
            series_map,
            colors,
            out_dir / f"cchihh_cfg1_bandit_on_op_freq_{key}_mean",
            f"CCHIHH cfg1 Operator Frequency - {key} (mean)",
        )


if __name__ == "__main__":
    main()
