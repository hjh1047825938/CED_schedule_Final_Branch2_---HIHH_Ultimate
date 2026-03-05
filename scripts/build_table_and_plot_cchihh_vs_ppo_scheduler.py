import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

SCALES = ["T100", "T200", "T500"]
Z95 = 1.96
GEN_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([0-9.+-eE]+)")


def parse_curve(path: Path):
    curve = {}
    idx = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            m = GEN_RE.match(s)
            if m:
                curve[int(m.group(1))] = float(m.group(2))
                continue
            idx += 1
            try:
                curve[idx] = float(s)
            except ValueError:
                continue
    return curve


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def load_group(files):
    runs = []
    for fp in sorted(files):
        c = parse_curve(fp)
        if c:
            runs.append((fp, c))
    return runs


def main():
    root = Path.cwd()
    cchihh_root = root / "outputs/results/cchihh_ablation_suite/baseline"
    ppo_root = root / "outputs/results/PPO"
    out_dir = root / "outputs/results/baseline_cchihh_vs_ppo_scheduler"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    table_rows = []

    for scale in SCALES:
        c_runs = load_group((cchihh_root / scale).glob("CCHIHH_Full_seed*.txt"))
        p_runs = load_group(ppo_root.glob(f"{scale}_seed*.txt"))
        if len(c_runs) < 1 or len(p_runs) < 1:
            raise SystemExit(f"Missing runs for {scale}: CCHIHH={len(c_runs)} PPO={len(p_runs)}")

        common = set(c_runs[0][1].keys()) & set(p_runs[0][1].keys())
        for _, c in c_runs[1:]:
            common &= set(c.keys())
        for _, c in p_runs[1:]:
            common &= set(c.keys())
        if not common:
            raise SystemExit(f"No common generations in {scale}")

        gens = sorted(common)
        c_mean, c_var, p_mean, p_var = [], [], [], []
        for g in gens:
            c_vals = [c[g] for _, c in c_runs]
            p_vals = [c[g] for _, c in p_runs]
            c_mean.append(mean(c_vals))
            c_var.append(variance(c_vals))
            p_mean.append(mean(p_vals))
            p_var.append(variance(p_vals))

        c_finals = [c[gens[-1]] for _, c in c_runs]
        p_finals = [c[gens[-1]] for _, c in p_runs]
        c_m, c_s = mean(c_finals), math.sqrt(variance(c_finals))
        p_m, p_s = mean(p_finals), math.sqrt(variance(p_finals))
        improve = ((p_m - c_m) / p_m * 100.0) if abs(p_m) > 1e-12 else 0.0
        table_rows.append([scale, f"{c_m:.6f} +- {c_s:.6f}", f"{p_m:.6f} +- {p_s:.6f}", f"{improve:.4f}"])

        all_stats[scale] = {
            "gens": gens,
            "c_mean": c_mean,
            "c_var": c_var,
            "p_mean": p_mean,
            "p_var": p_var,
            "n_c": len(c_runs),
            "n_p": len(p_runs),
        }

    with (out_dir / "table_baseline_cchihh_vs_ppo_scheduler.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Problem", "CCHIHH-full (Mean+-Std)", "PPO-scheduler (Mean+-Std)", "Improvement (%)"])
        w.writerows(table_rows)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "figure.figsize": (12, 4),
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    for i, scale in enumerate(SCALES):
        ax = axes[i]
        st = all_stats[scale]
        gens = st["gens"]
        c_ci = [Z95 * math.sqrt(max(v, 0.0) / st["n_c"]) for v in st["c_var"]]
        p_ci = [Z95 * math.sqrt(max(v, 0.0) / st["n_p"]) for v in st["p_var"]]

        ax.plot(gens, st["c_mean"], label="CCHIHH-full", color="#1f77b4")
        ax.fill_between(
            gens,
            [m - c for m, c in zip(st["c_mean"], c_ci)],
            [m + c for m, c in zip(st["c_mean"], c_ci)],
            color="#1f77b4",
            alpha=0.14,
        )
        ax.plot(gens, st["p_mean"], label="PPO-scheduler", color="#ff7f0e")
        ax.fill_between(
            gens,
            [m - c for m, c in zip(st["p_mean"], p_ci)],
            [m + c for m, c in zip(st["p_mean"], p_ci)],
            color="#ff7f0e",
            alpha=0.14,
        )

        ax.set_title(scale, fontweight="bold")
        ax.set_xlabel("Generation")
        if i == 0:
            ax.set_ylabel("Best fitness")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    fig.savefig(out_dir / "baseline_cchihh_vs_ppo_scheduler_mean_ci.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "baseline_cchihh_vs_ppo_scheduler_mean_ci.pdf", dpi=300, bbox_inches="tight")

    print("Saved:", out_dir / "table_baseline_cchihh_vs_ppo_scheduler.csv")
    print("Saved:", out_dir / "baseline_cchihh_vs_ppo_scheduler_mean_ci.png")
    print("Saved:", out_dir / "baseline_cchihh_vs_ppo_scheduler_mean_ci.pdf")


if __name__ == "__main__":
    main()
