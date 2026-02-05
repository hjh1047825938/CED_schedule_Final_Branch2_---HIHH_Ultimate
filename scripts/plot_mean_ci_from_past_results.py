import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+-eE]+)")


def parse_file(path: Path):
    gens = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            gen = int(m.group(1))
            val = float(m.group(2))
            gens[gen] = val
    return gens


def mean(vals):
    return sum(vals) / len(vals)


def variance(vals):
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def build_series(results_dir: Path, pattern: str):
    files = sorted(results_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No files found for pattern: {pattern} in {results_dir}")
    series_vals = {}
    for fp in files:
        data = parse_file(fp)
        if not data:
            raise SystemExit(f"No Gen lines found in {fp}")
        for gen, val in data.items():
            series_vals.setdefault(gen, []).append(val)
    return series_vals


def compute_stats(series_vals):
    gens_sorted = sorted(series_vals.keys())
    means = []
    vars_ = []
    ns = []
    for gen in gens_sorted:
        vals = series_vals[gen]
        means.append(mean(vals))
        vars_.append(variance(vals))
        ns.append(len(vals))
    return gens_sorted, means, vars_, ns


def plot_mean_ci(series_map, colors, out_prefix: Path, title: str):
    z = 1.96
    plt.figure(figsize=(9, 5.5))
    for label, (gens, means, vars_, ns) in series_map.items():
        n_eff = [max(1, n) for n in ns]
        ci = [z * math.sqrt(max(v, 0.0) / n) for v, n in zip(vars_, n_eff)]
        lower = [m - c for m, c in zip(means, ci)]
        upper = [m + c for m, c in zip(means, ci)]
        color = colors.get(label)
        plt.plot(gens, means, label=label, linewidth=2, color=color)
        plt.fill_between(gens, lower, upper, alpha=0.18, color=color)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_variance(series_map, colors, out_prefix: Path, title: str):
    plt.figure(figsize=(9, 5.5))
    for label, (gens, _means, vars_, _ns) in series_map.items():
        color = colors.get(label)
        plt.plot(gens, vars_, label=label, linewidth=2, color=color)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (variance)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    past_results = root / "past_results"

    dir_main = past_results / "gen10000_seed1_10"
    main_series = {
        "GA": build_series(dir_main, "GA_seed*.txt"),
        "DE": build_series(dir_main, "DE_seed*.txt"),
        "GDE": build_series(dir_main, "GDE_seed*.txt"),
        "CCHIHH (non-stable)": build_series(dir_main, "CCHIHH_base_seed*.txt"),
        "CCHIHH (stable)": build_series(dir_main, "CCHIHH_stable_seed*.txt"),
    }

    colors_main = {
        "GA": "#1f77b4",
        "DE": "#ff7f0e",
        "GDE": "#2ca02c",
        "CCHIHH (non-stable)": "#d62728",
        "CCHIHH (stable)": "#9467bd",
    }

    stats_main = {
        label: compute_stats(series_vals)
        for label, series_vals in main_series.items()
    }

    plot_mean_ci(
        stats_main,
        colors_main,
        past_results / "gen10000_seed1_10_mean_ci_compare",
        "Seed 1-10 Mean (every 50 gens)",
    )
    plot_variance(
        stats_main,
        colors_main,
        past_results / "gen10000_seed1_10_var_compare",
        "Seed 1-10 Variance (every 50 gens)",
    )

    dir_compare = past_results / "gen10000_seed1_10_ga_slhh_vs_cchihh"
    compare_series = {
        "GA-SLHH": build_series(dir_compare, "GA-SLHH_seed*.txt"),
        "CCHIHH (stable)": build_series(dir_compare, "CCHIHH_stable_seed*.txt"),
    }
    colors_compare = {
        "GA-SLHH": "#1f77b4",
        "CCHIHH (stable)": "#9467bd",
    }
    stats_compare = {
        label: compute_stats(series_vals)
        for label, series_vals in compare_series.items()
    }

    plot_mean_ci(
        stats_compare,
        colors_compare,
        past_results / "gen10000_ga_slhh_vs_cchihh_mean_ci_compare",
        "GA-SLHH vs CCHIHH (stable) Mean (every 50 gens)",
    )
    plot_variance(
        stats_compare,
        colors_compare,
        past_results / "gen10000_ga_slhh_vs_cchihh_var_compare",
        "GA-SLHH vs CCHIHH (stable) Variance (every 50 gens)",
    )


if __name__ == "__main__":
    main()
