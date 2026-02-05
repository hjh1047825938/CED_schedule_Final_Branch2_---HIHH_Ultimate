import csv
import math
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


def variance(vals):
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def plot_group(gens, series_map, colors, out_prefix: Path, title: str, n: int):
    z = 1.96
    plt.figure(figsize=(9, 5.5))

    for label, data in series_map.items():
        # data: list of seed series, each is list over gens
        means = []
        vars_ = []
        for i in range(len(gens)):
            vals = [seed_series[i] for seed_series in data]
            means.append(mean(vals))
            vars_.append(variance(vals))
        ci = [z * math.sqrt(max(v, 0.0) / n) for v in vars_]
        lower = [m - c for m, c in zip(means, ci)]
        upper = [m + c for m, c in zip(means, ci)]
        color = colors.get(label)
        plt.plot(gens, means, label=label, linewidth=2, color=color)
        plt.fill_between(gens, lower, upper, alpha=0.18, color=color)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Selection frequency")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".png"), dpi=200)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")


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
    plt.savefig(out_prefix.with_suffix(".png"), dpi=200)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")


def moving_average(vals, window):
    if window <= 1:
        return vals
    out = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_group_smoothed(gens, series_map, colors, out_prefix: Path, title: str, n: int, window: int):
    z = 1.96
    plt.figure(figsize=(9, 5.5))

    for label, data in series_map.items():
        means = []
        vars_ = []
        for i in range(len(gens)):
            vals = [seed_series[i] for seed_series in data]
            means.append(mean(vals))
            vars_.append(variance(vals))
        means = moving_average(means, window)
        vars_ = moving_average(vars_, window)
        ci = [z * math.sqrt(max(v, 0.0) / n) for v in vars_]
        lower = [m - c for m, c in zip(means, ci)]
        upper = [m + c for m, c in zip(means, ci)]
        color = colors.get(label)
        plt.plot(gens, means, label=label, linewidth=2, color=color)
        plt.fill_between(gens, lower, upper, alpha=0.18, color=color)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Selection frequency")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".png"), dpi=200)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")


def plot_group_mean_only_smoothed(gens, series_map, colors, out_prefix: Path, title: str, window: int):
    plt.figure(figsize=(9, 5.5))
    for label, data in series_map.items():
        means = []
        for i in range(len(gens)):
            vals = [seed_series[i] for seed_series in data]
            means.append(mean(vals))
        means = moving_average(means, window)
        color = colors.get(label)
        plt.plot(gens, means, label=label, linewidth=2, color=color)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Selection frequency (mean)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".png"), dpi=200)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")


def main():
    root = Path(__file__).resolve().parents[1]
    stats_dir = root / "past_results" / "ablation_cchihh_stable"
    files = sorted(stats_dir.glob("CCHIHH_stable_opstats_seed*.csv"))
    if not files:
        raise SystemExit(f"No op stats files found in {stats_dir}")

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

    out_dir = root / "past_results" / "ablation_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, cols in groups.items():
        col_names = [c for c, _ in cols]
        gens, series = collect_series(files, col_names)
        series_map = {}
        for col, label in cols:
            series_map[label] = series[col]
        plot_group(
            gens,
            series_map,
            colors,
            out_dir / f"cchihh_op_freq_{key}_mean_ci",
            f"CCHIHH (stable) Operator Frequency - {key}",
            n=len(files),
        )
        plot_group_mean_only(
            gens,
            series_map,
            colors,
            out_dir / f"cchihh_op_freq_{key}_mean_only",
            f"CCHIHH (stable) Operator Frequency - {key} (mean)",
        )
        plot_group_smoothed(
            gens,
            series_map,
            colors,
            out_dir / f"cchihh_op_freq_{key}_mean_ci_smooth_w5",
            f"CCHIHH (stable) Operator Frequency - {key} (MA w=5)",
            n=len(files),
            window=5,
        )
        plot_group_mean_only_smoothed(
            gens,
            series_map,
            colors,
            out_dir / f"cchihh_op_freq_{key}_mean_only_smooth_w5",
            f"CCHIHH (stable) Operator Frequency - {key} (mean, MA w=5)",
            window=5,
        )


if __name__ == "__main__":
    main()
