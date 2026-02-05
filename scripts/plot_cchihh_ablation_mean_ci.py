import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


LINE_RE = r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+-eE]+)"


def parse_file(path: Path):
    gens = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "Gen" and parts[2] == "best_fit" and parts[3] == "=":
                gen = int(parts[1].rstrip(":"))
                val = float(parts[4])
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
    for gen in gens_sorted:
        vals = series_vals[gen]
        means.append(mean(vals))
        vars_.append(variance(vals))
    return gens_sorted, means, vars_


def moving_average(vals, window):
    if window <= 1:
        return vals
    out = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def write_stats_csv(out_csv: Path, gens, mean_a, var_a, mean_b, var_b, label_a, label_b):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gen", f"{label_a} mean", f"{label_a} var", f"{label_b} mean", f"{label_b} var"])
        for i, gen in enumerate(gens):
            writer.writerow([gen, mean_a[i], var_a[i], mean_b[i], var_b[i]])


def plot_mean_ci(gens, mean_a, var_a, mean_b, var_b, label_a, label_b, out_prefix: Path, title: str, n: int):
    z = 1.96
    ci_a = [z * math.sqrt(max(v, 0.0) / n) for v in var_a]
    ci_b = [z * math.sqrt(max(v, 0.0) / n) for v in var_b]

    plt.figure(figsize=(9, 5.5))
    plt.plot(gens, mean_a, label=label_a, linewidth=2, color="#9467bd")
    plt.fill_between(gens, [m - c for m, c in zip(mean_a, ci_a)],
                     [m + c for m, c in zip(mean_a, ci_a)], alpha=0.18, color="#9467bd")

    plt.plot(gens, mean_b, label=label_b, linewidth=2, color="#d62728")
    plt.fill_between(gens, [m - c for m, c in zip(mean_b, ci_b)],
                     [m + c for m, c in zip(mean_b, ci_b)], alpha=0.18, color="#d62728")

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=9)
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".png"), dpi=200)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")


def plot_mean_only(gens, mean_a, mean_b, label_a, label_b, out_prefix: Path, title: str):
    plt.figure(figsize=(9, 5.5))
    plt.plot(gens, mean_a, label=label_a, linewidth=2, color="#9467bd")
    plt.plot(gens, mean_b, label=label_b, linewidth=2, color="#d62728")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (mean)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=9)
    plt.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".png"), dpi=200)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")


def main():
    root = Path(__file__).resolve().parents[1]
    base_dir = root / "past_results"
    out_dir = base_dir / "ablation_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = base_dir / "ablation_cchihh_stable"
    ablations = [
        ("nomig", base_dir / "ablation_cchihh_nomig", "CCHIHH (stable, no migration)"),
        ("randop", base_dir / "ablation_cchihh_random_ops", "CCHIHH (stable, random ops)"),
        ("noblock", base_dir / "ablation_cchihh_no_blocks", "CCHIHH (stable, no blocks)"),
    ]

    baseline_series = build_series(baseline_dir, "CCHIHH_stable_seed*.txt")
    gens, mean_base, var_base = compute_stats(baseline_series)

    smooth_w = 5
    for key, ab_dir, ab_label in ablations:
        ab_series = build_series(ab_dir, f"CCHIHH_stable_{key}_seed*.txt")
        gens_b, mean_ab, var_ab = compute_stats(ab_series)
        if gens_b != gens:
            raise SystemExit(f"Generation mismatch for {key}")

        label_base = "CCHIHH (stable)"
        csv_path = out_dir / f"cchihh_{key}_stats_every50.csv"
        write_stats_csv(csv_path, gens, mean_base, var_base, mean_ab, var_ab, label_base, ab_label)

        plot_mean_ci(
            gens,
            mean_base,
            var_base,
            mean_ab,
            var_ab,
            label_base,
            ab_label,
            out_dir / f"cchihh_{key}_mean_ci",
            f"{ab_label} vs CCHIHH (stable)",
            n=10,
        )
        plot_mean_only(
            gens,
            mean_base,
            mean_ab,
            label_base,
            ab_label,
            out_dir / f"cchihh_{key}_mean_only",
            f"{ab_label} vs CCHIHH (stable) Mean",
        )

        mean_base_s = moving_average(mean_base, smooth_w)
        var_base_s = moving_average(var_base, smooth_w)
        mean_ab_s = moving_average(mean_ab, smooth_w)
        var_ab_s = moving_average(var_ab, smooth_w)
        plot_mean_ci(
            gens,
            mean_base_s,
            var_base_s,
            mean_ab_s,
            var_ab_s,
            label_base,
            ab_label,
            out_dir / f"cchihh_{key}_mean_ci_smooth_w{smooth_w}",
            f"{ab_label} vs CCHIHH (stable) (MA w={smooth_w})",
            n=10,
        )
        plot_mean_only(
            gens,
            mean_base_s,
            mean_ab_s,
            label_base,
            ab_label,
            out_dir / f"cchihh_{key}_mean_only_smooth_w{smooth_w}",
            f"{ab_label} vs CCHIHH (stable) Mean (MA w={smooth_w})",
        )


if __name__ == "__main__":
    main()
