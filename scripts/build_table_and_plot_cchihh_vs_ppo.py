import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


SCALES = ["T100", "T200", "T500"]
SOLVERS = ["CCHIHH_Full", "PPO"]
SOLVER_LABELS = {
    "CCHIHH_Full": "CCHIHH-full",
    "PPO": "PPO",
}
SOLVER_COLORS = {
    "CCHIHH_Full": "#1f77b4",
    "PPO": "#ff7f0e",
}
LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+-eE]+)")
Z95 = 1.96


def parse_log(path: Path):
    out = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            out[int(m.group(1))] = float(m.group(2))
    return out


def mean(vals):
    return sum(vals) / len(vals)


def variance(vals):
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def load_solver_runs(scale_dir: Path, solver: str):
    runs = []
    for fp in sorted(scale_dir.glob(f"{solver}_seed*.txt")):
        series = parse_log(fp)
        if series:
            runs.append((fp, series))
    return runs


def load_sb3_summary(path: Path):
    data = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            scale = row.get("problem_scale", "").strip()
            if scale:
                data[scale] = row
    return data


def build_scale_stats(scale_dir: Path):
    per_solver = {}
    for solver in SOLVERS:
        runs = load_solver_runs(scale_dir, solver)
        if not runs:
            raise SystemExit(f"No valid logs for {solver} in {scale_dir}")

        common_gens = set(runs[0][1].keys())
        for _, series in runs[1:]:
            common_gens &= set(series.keys())
        if not common_gens:
            raise SystemExit(f"No common generations for {solver} in {scale_dir}")

        gens = sorted(common_gens)
        stats = {}
        finals = []
        for g in gens:
            vals = [series[g] for _, series in runs]
            stats[g] = {
                "mean": mean(vals),
                "var": variance(vals),
            }
        for _, series in runs:
            finals.append(series[gens[-1]])

        per_solver[solver] = {
            "n": len(runs),
            "gens": gens,
            "stats": stats,
            "finals": finals,
            "final_mean": mean(finals),
            "final_std": math.sqrt(variance(finals)),
        }

    common_all = set(per_solver[SOLVERS[0]]["gens"])
    for s in SOLVERS[1:]:
        common_all &= set(per_solver[s]["gens"])
    if not common_all:
        raise SystemExit(f"No common generations between solvers in {scale_dir}")

    gens_all = sorted(common_all)
    return per_solver, gens_all


def build_scale_stats_sb3(scale_dir: Path, sb3_row: dict):
    cchihh_runs = load_solver_runs(scale_dir, "CCHIHH_Full")
    if not cchihh_runs:
        raise SystemExit(f"No valid logs for CCHIHH_Full in {scale_dir}")

    common_gens = set(cchihh_runs[0][1].keys())
    for _, series in cchihh_runs[1:]:
        common_gens &= set(series.keys())
    if not common_gens:
        raise SystemExit(f"No common generations for CCHIHH_Full in {scale_dir}")

    gens = sorted(common_gens)
    c_stats = {}
    c_finals = []
    for g in gens:
        vals = [series[g] for _, series in cchihh_runs]
        c_stats[g] = {
            "mean": mean(vals),
            "var": variance(vals),
        }
    for _, series in cchihh_runs:
        c_finals.append(series[gens[-1]])

    try:
        ppo_final = float(sb3_row["final_policy_fitness"])
    except Exception as exc:
        raise SystemExit(f"Invalid final_policy_fitness for {scale_dir.name} in SB3 summary") from exc

    per_solver = {
        "CCHIHH_Full": {
            "n": len(cchihh_runs),
            "gens": gens,
            "stats": c_stats,
            "finals": c_finals,
            "final_mean": mean(c_finals),
            "final_std": math.sqrt(variance(c_finals)),
        },
        "PPO": {
            "n": 1,
            "gens": gens,
            "stats": {g: {"mean": ppo_final, "var": 0.0} for g in gens},
            "finals": [ppo_final],
            "final_mean": ppo_final,
            "final_std": 0.0,
            "sb3_training_steps": int(float(sb3_row.get("actual_training_steps", "0") or 0)),
            "sb3_training_min": float(sb3_row.get("training_wall_clock_min", "0") or 0.0),
            "sb3_budget_min": float(sb3_row.get("time_budget_min", "0") or 0.0),
            "sb3_curve_png": sb3_row.get("convergence_curve_png", ""),
        },
    }
    return per_solver, gens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_root", default="outputs/results/cchihh_ablation_suite/baseline")
    parser.add_argument("--out_dir", default="outputs/results/baseline_cchihh_vs_ppo")
    parser.add_argument(
        "--ppo_sb3_summary",
        default="",
        help="Optional CSV from run_ppo_sb3_subprocess_baseline.py summary to integrate SB3 PPO results",
    )
    args = parser.parse_args()

    root = Path.cwd()
    baseline_root = root / args.baseline_root
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sb3_mode = False
    sb3_summary = {}
    if args.ppo_sb3_summary:
        sb3_path = Path(args.ppo_sb3_summary)
        if not sb3_path.is_absolute():
            sb3_path = root / sb3_path
        if not sb3_path.exists():
            raise SystemExit(f"SB3 summary CSV not found: {sb3_path}")
        sb3_summary = load_sb3_summary(sb3_path)
        sb3_mode = True

    scale_data = {}
    table_rows = []
    for scale in SCALES:
        scale_dir = baseline_root / scale
        if sb3_mode:
            if scale not in sb3_summary:
                raise SystemExit(f"Scale {scale} missing in SB3 summary CSV")
            per_solver, gens_all = build_scale_stats_sb3(scale_dir, sb3_summary[scale])
        else:
            per_solver, gens_all = build_scale_stats(scale_dir)

        scale_data[scale] = (per_solver, gens_all)

        cchihh_mean = per_solver["CCHIHH_Full"]["final_mean"]
        cchihh_std = per_solver["CCHIHH_Full"]["final_std"]
        ppo_mean = per_solver["PPO"]["final_mean"]
        ppo_std = per_solver["PPO"]["final_std"]
        improve_pct = (ppo_mean - cchihh_mean) / ppo_mean * 100.0 if ppo_mean != 0 else 0.0

        if sb3_mode:
            table_rows.append(
                [
                    scale,
                    f"{cchihh_mean:.6f} +- {cchihh_std:.6f}",
                    f"{ppo_mean:.6f} +- {ppo_std:.6f}",
                    f"{improve_pct:.4f}",
                    str(per_solver["PPO"]["sb3_training_steps"]),
                    f"{per_solver['PPO']['sb3_training_min']:.3f}",
                    f"{per_solver['PPO']['sb3_budget_min']:.2f}",
                    per_solver["PPO"]["sb3_curve_png"],
                ]
            )
        else:
            p_val = mannwhitneyu(
                per_solver["CCHIHH_Full"]["finals"],
                per_solver["PPO"]["finals"],
                alternative="two-sided",
                method="exact",
            ).pvalue
            table_rows.append(
                [
                    scale,
                    f"{cchihh_mean:.6f} +- {cchihh_std:.6f}",
                    f"{ppo_mean:.6f} +- {ppo_std:.6f}",
                    f"{improve_pct:.4f}",
                    f"{p_val:.9f}",
                ]
            )

    table_path = out_dir / "table_baseline_cchihh_vs_ppo.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if sb3_mode:
            w.writerow(
                [
                    "Problem",
                    "CCHIHH-full (Mean+-Std)",
                    "PPO-SB3 (Mean+-Std)",
                    "Improvement (%)",
                    "PPO train steps",
                    "PPO wall-clock min",
                    "PPO budget min",
                    "PPO convergence png",
                ]
            )
        else:
            w.writerow(["Problem", "CCHIHH-full (Mean+-Std)", "PPO (Mean+-Std)", "Improvement (%)", "p-value"])
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
        per_solver, gens_all = scale_data[scale]
        ax = axes[i]

        for s in SOLVERS:
            means = [per_solver[s]["stats"][g]["mean"] for g in gens_all]
            vars_ = [per_solver[s]["stats"][g]["var"] for g in gens_all]
            n = max(1, per_solver[s]["n"])
            cis = [Z95 * math.sqrt(max(v, 0.0) / n) for v in vars_]
            c = SOLVER_COLORS[s]

            ax.plot(gens_all, means, label=SOLVER_LABELS[s], color=c)
            if not (sb3_mode and s == "PPO"):
                ax.fill_between(
                    gens_all,
                    [m - ci for m, ci in zip(means, cis)],
                    [m + ci for m, ci in zip(means, cis)],
                    color=c,
                    alpha=0.14,
                )

        ax.set_title(scale, fontweight="bold")
        ax.set_xlabel("Generation")
        if i == 0:
            ax.set_ylabel("Best fitness")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    pdf_path = out_dir / "baseline_cchihh_vs_ppo_mean_ci.pdf"
    png_path = out_dir / "baseline_cchihh_vs_ppo_mean_ci.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)

    print(f"Saved: {table_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
