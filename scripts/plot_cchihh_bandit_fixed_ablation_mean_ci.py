import math
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


CONFIGS = [
    ("cfg1_bandit_on", "1) use_bandit=true", "#1f77b4"),
    ("cfg2_bandit_off", "2) use_bandit=false", "#ff7f0e"),
    ("cfg3_fixed_ops", "3) fixed ops per block", "#2ca02c"),
]


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


def build_cfg_stats(results_dir: Path, cfg_key: str):
    files = sorted(results_dir.glob(f"{cfg_key}_seed*.txt"))
    if len(files) != 10:
        raise SystemExit(f"Expected 10 files for {cfg_key}, got {len(files)}")

    per_gen = {}
    for fp in files:
        parsed = parse_file(fp)
        if not parsed:
            raise SystemExit(f"No Gen lines in {fp}")
        for g, v in parsed.items():
            per_gen.setdefault(g, []).append(v)

    common_gens = sorted(g for g, vals in per_gen.items() if len(vals) == 10)
    if not common_gens:
        raise SystemExit(f"No common generation points for {cfg_key}")

    means = []
    vars_ = []
    for g in common_gens:
        vals = per_gen[g]
        means.append(mean(vals))
        vars_.append(variance(vals))

    return common_gens, means, vars_


def plot_series(out_prefix: Path, title: str, series_list):
    z = 1.96
    n = 10

    plt.figure(figsize=(9, 5.5))
    for item in series_list:
        gens = item["gens"]
        means = item["means"]
        vars_ = item["vars"]
        label = item["label"]
        color = item["color"]

        ci = [z * math.sqrt(max(v, 0.0) / n) for v in vars_]
        lo = [m - c for m, c in zip(means, ci)]
        hi = [m + c for m, c in zip(means, ci)]

        plt.plot(gens, means, label=label, color=color, linewidth=2)
        plt.fill_between(gens, lo, hi, color=color, alpha=0.18)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".png"), dpi=220)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results/ablation_cchihh_bandit_fixed_gen10000_seed1_10",
        help="Directory containing cfg*_seed*.txt logs.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_dir = root / args.results_dir
    out_dir = results_dir / "plots"

    stats = {}
    for key, label, color in CONFIGS:
        gens, means, vars_ = build_cfg_stats(results_dir, key)
        stats[key] = {
            "gens": gens,
            "means": means,
            "vars": vars_,
            "label": label,
            "color": color,
        }

    g_ref = stats["cfg1_bandit_on"]["gens"]
    for key, _, _ in CONFIGS[1:]:
        if stats[key]["gens"] != g_ref:
            raise SystemExit(f"Generation mismatch: cfg1 vs {key}")

    plot_series(
        out_dir / "cchihh_ablation_1_vs_2_mean_ci",
        "CCHIHH Ablation: 1 vs 2 (mean ± 95% CI)",
        [stats["cfg1_bandit_on"], stats["cfg2_bandit_off"]],
    )
    plot_series(
        out_dir / "cchihh_ablation_1_vs_3_mean_ci",
        "CCHIHH Ablation: 1 vs 3 (mean ± 95% CI)",
        [stats["cfg1_bandit_on"], stats["cfg3_fixed_ops"]],
    )
    plot_series(
        out_dir / "cchihh_ablation_1_2_3_mean_ci",
        "CCHIHH Ablation: 1,2,3 (mean ± 95% CI)",
        [stats["cfg1_bandit_on"], stats["cfg2_bandit_off"], stats["cfg3_fixed_ops"]],
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
