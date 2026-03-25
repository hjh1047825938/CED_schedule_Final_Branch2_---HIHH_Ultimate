from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import friedmanchisquare, wilcoxon
except ImportError:  # pragma: no cover
    friedmanchisquare = None
    wilcoxon = None

try:
    from scripts.wallclock_common import aggregate_curves, load_curve_csv
except ModuleNotFoundError:  # pragma: no cover
    from wallclock_common import aggregate_curves, load_curve_csv


ALPHAS = [0.2, 0.5, 0.8]
SCALES = [
    ("T100", 30.0),
    ("T200", 60.0),
    ("T500", 300.0),
]
ALGO_FILE_KEYS = {
    "CCHIHH": "CCHIHH",
    "DSAC-DE": "DSAC_DE",
    "CGA": "CGA",
    "IMOMA": "IMOMA",
    "PPO": "PPO",
}
COLORS = {
    "CCHIHH": "#E74C3C",
    "DSAC-DE": "#E67E22",
    "CGA": "#3498DB",
    "IMOMA": "#2ECC71",
    "PPO": "#9B59B6",
}
LINEWIDTHS = {"CCHIHH": 2.0, "others": 1.5}
SHADED_ALPHA = 0.15
MARKERS = {"CCHIHH": "*", "DSAC-DE": "o", "CGA": "^", "IMOMA": "s", "PPO": "D"}
Q_ALPHA = {4: 2.569, 5: 2.728}
PPO_TXT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class FinalPoint:
    fitness: float
    makespan: float
    energy: float
    generation: int


def configure_style() -> None:
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 12
    matplotlib.rcParams["axes.linewidth"] = 1.0
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["axes.grid"] = False


def alpha_dir(alpha: float) -> str:
    return f"alpha{alpha:.1f}"


def run_path(root: Path, alpha: float, scale: str, algo: str, seed: int) -> Path:
    return root / alpha_dir(alpha) / scale / f"{ALGO_FILE_KEYS[algo]}_seed{seed}.csv"


def load_final_point(path: Path) -> FinalPoint:
    curve = load_curve_csv(path)
    return FinalPoint(
        fitness=float(curve.best_fitness[-1]),
        makespan=float(curve.best_f1[-1]),
        energy=float(curve.best_f2[-1]),
        generation=int(curve.generation[-1]),
    )


def existing_paths(root: Path, alpha: float, scale: str, algo: str, seeds: int = 10) -> list[Path]:
    return [path for seed in range(1, seeds + 1) if (path := run_path(root, alpha, scale, algo, seed)).exists()]


def load_final_series(root: Path, alpha: float, scale: str, algo: str, seeds: int = 10) -> list[FinalPoint]:
    return [load_final_point(path) for path in existing_paths(root, alpha, scale, algo, seeds)]


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    return float(arr.mean()), float(arr.std(ddof=1 if arr.size > 1 else 0))


def coeff_var(values: list[float]) -> float:
    mean, std = mean_std(values)
    if not np.isfinite(mean) or abs(mean) < 1e-12:
        return math.nan
    return float(std / mean)


def paired_pvalue(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) == 0 or wilcoxon is None:
        return math.nan
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return math.nan


def improvement_pct(cchihh_mean: float, baseline_mean: float) -> float:
    return (baseline_mean - cchihh_mean) / baseline_mean * 100.0


def safe_lighten(color: str, factor: float = 0.28) -> tuple[float, float, float]:
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    return tuple(np.clip(rgb + (1.0 - rgb) * factor, 0.0, 1.0))


def read_ppo_iteration_fallback(root: Path, scale: str, seeds: int = 10) -> list[float]:
    candidates = [
        root / "outputs" / "results" / "cchihh_ablation_suite" / "baseline" / scale,
        root / "outputs" / "results" / "PPO",
        root / "results" / "PPO_runs",
        root / "results" / "PPO",
    ]
    values: list[float] = []
    for seed in range(1, seeds + 1):
        found = None
        for base in candidates:
            options = [
                base / f"PPO_seed{seed}.txt",
                base / f"{scale}_seed{seed}.txt",
                base / f"{scale}_seed{seed}" / f"{scale}_seed{seed}.txt",
            ]
            for option in options:
                if option.exists():
                    found = option
                    break
            if found is not None:
                break
        if found is None:
            return []
        numbers = []
        for line in found.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("Gen ") and "best_fit" in s:
                match = PPO_TXT_RE.findall(s)
                if match:
                    numbers.append(float(match[-1]))
                    continue
            if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s):
                numbers.append(float(s))
        if not numbers:
            return []
        values.append(numbers[-1])
    return values


def ppo_mode(root: Path, repo_root: Path) -> str:
    all_present = all(len(existing_paths(root, alpha, scale, "PPO")) == 10 for alpha in ALPHAS for scale, _ in SCALES)
    if all_present:
        return "wallclock"
    alpha05_present = all(len(existing_paths(root, 0.5, scale, "PPO")) == 10 for scale, _ in SCALES)
    fallback_present = all(len(read_ppo_iteration_fallback(repo_root, scale)) == 10 for scale, _ in SCALES)
    if alpha05_present or fallback_present:
        return "fallback"
    return "absent"


def plot_mean_ci(ax, root: Path, alpha: float, scale: str, budget: float, algos: list[str]) -> None:
    for algo in algos:
        paths = existing_paths(root, alpha, scale, algo)
        if not paths:
            continue
        summary = aggregate_curves(paths, time_budget=budget)
        color = COLORS[algo]
        lw = LINEWIDTHS["CCHIHH"] if algo == "CCHIHH" else LINEWIDTHS["others"]
        ax.plot(summary.grid, summary.mean, color=color, linewidth=lw, label=algo)
        ax.fill_between(summary.grid, summary.mean - summary.ci95, summary.mean + summary.ci95, color=color, alpha=SHADED_ALPHA)


def save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_comprehensive_baseline(root: Path, output_dir: Path, repo_root: Path, ppo_status: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    algo_order = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA", "PPO"]

    for ax, (scale, _) in zip(axes, SCALES):
        data = {}
        for algo in algo_order[:-1]:
            finals = [p.fitness for p in load_final_series(root, 0.5, scale, algo)]
            data[algo] = finals
        if ppo_status == "wallclock":
            data["PPO"] = [p.fitness for p in load_final_series(root, 0.5, scale, "PPO")]
        elif ppo_status == "fallback":
            data["PPO"] = read_ppo_iteration_fallback(repo_root, scale)

        vals = [data[a] for a in algo_order if data.get(a)]
        labels = [a for a in algo_order if data.get(a)]
        bp = ax.boxplot(vals, patch_artist=True, labels=labels)
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(COLORS[label] if label == "CCHIHH" else safe_lighten(COLORS[label]))
            patch.set_edgecolor(COLORS[label])
        ax.set_title(scale)
        ax.set_ylabel("Final fitness")

        c_mean = float(np.mean(data["CCHIHH"]))
        y_top = max(max(v) for v in vals)
        lift = (max(max(v) for v in vals) - min(min(v) for v in vals) + 1e-9) * 0.08
        for idx, label in enumerate(labels[1:], start=2):
            b_mean = float(np.mean(data[label]))
            impr = improvement_pct(c_mean, b_mean)
            p = paired_pvalue(data["CCHIHH"], data[label])
            arrow = "↓" if impr >= 0 else "↑"
            mark = " **" if np.isfinite(p) and p < 0.05 else ""
            ax.text(idx, y_top + lift, f"{arrow}{abs(impr):.1f}%{mark}", ha="center", va="bottom", fontsize=10)

    save_pdf(fig, output_dir / "fig_comprehensive_baseline_comparison.pdf")


def fig_convergence_group(root: Path, output_dir: Path, filename: str, algos: list[str], ppo_status: str | None = None, repo_root: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, (ax, (scale, budget)) in enumerate(zip(axes, SCALES)):
        plot_mean_ci(ax, root, 0.5, scale, budget, algos)
        if ppo_status == "fallback" and repo_root is not None and "PPO" in algos:
            finals = read_ppo_iteration_fallback(repo_root, scale)
            if finals:
                ax.axhline(float(np.mean(finals)), color=COLORS["PPO"], linestyle="--", linewidth=1.5)
                ax.text(budget * 0.52, float(np.mean(finals)), "PPO (10k gen)", color=COLORS["PPO"], fontsize=10, va="bottom")
        ax.set_title(scale)
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_xlim(0.0, budget)
        ax.set_ylabel("Best fitness")
        if idx == 0:
            ax.legend(loc="upper right", frameon=False)
    save_pdf(fig, output_dir / filename)


def fig_boxplot_four(root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    algos = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"]
    for ax, (scale, _) in zip(axes, SCALES):
        vals = [[p.fitness for p in load_final_series(root, 0.5, scale, algo)] for algo in algos]
        bp = ax.boxplot(vals, patch_artist=True, labels=algos)
        for patch, label in zip(bp["boxes"], algos):
            patch.set_facecolor(COLORS[label] if label == "CCHIHH" else safe_lighten(COLORS[label]))
            patch.set_edgecolor(COLORS[label])
        ax.set_title(scale)
        ax.set_ylabel("Final fitness")
    save_pdf(fig, output_dir / "boxplot_fitness.pdf")


def fig_scatter(root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    algos = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"]
    for ax, (scale, _) in zip(axes, SCALES):
        for algo in algos:
            points = load_final_series(root, 0.5, scale, algo)
            if not points:
                continue
            ax.scatter(
                [p.makespan for p in points],
                [p.energy for p in points],
                c=COLORS[algo],
                marker=MARKERS[algo],
                s=150 if algo == "CCHIHH" else 70,
                label=algo,
                alpha=0.85,
            )
        ax.set_title(scale)
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Energy")
    axes[0].legend(frameon=False)
    save_pdf(fig, output_dir / "makespan_energy_scatter.pdf")


def _radar_polygon(ax, angles: np.ndarray, values: list[float], color: str, label: str) -> None:
    pts = np.column_stack([angles, values + values[:1]])
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5 if label != "CCHIHH" else 2.0, label=label)
    ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.12)


def fig_radar(root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), subplot_kw={"projection": "polar"})
    algos = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"]
    labels = ["Solution Quality", "Stability", "Convergence Speed", "Budget Utilization"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    for ax, (scale, budget) in zip(axes, SCALES):
        final_fitness = {algo: [p.fitness for p in load_final_series(root, 0.5, scale, algo)] for algo in algos}
        generations = {algo: [p.generation for p in load_final_series(root, 0.5, scale, algo)] for algo in algos}
        speed = {}
        for algo in algos:
            paths = existing_paths(root, 0.5, scale, algo)
            if not paths:
                speed[algo] = math.nan
                continue
            summary = aggregate_curves(paths, time_budget=budget)
            index = int(round(0.2 * (len(summary.grid) - 1)))
            speed[algo] = float(summary.mean[index])

        max_gen = max(float(np.mean(generations[a])) for a in algos if generations[a])
        quality = {a: 1.0 / np.mean(final_fitness[a]) for a in algos}
        stability = {a: 1.0 / coeff_var(final_fitness[a]) for a in algos}
        conv_speed = {a: 1.0 / speed[a] for a in algos}
        budget_use = {a: float(np.mean(generations[a])) / max_gen for a in algos}

        metric_maps = [quality, stability, conv_speed, budget_use]
        normed = []
        for metric in metric_maps:
            vals = np.asarray([metric[a] for a in algos], dtype=float)
            lo, hi = float(vals.min()), float(vals.max())
            if abs(hi - lo) < 1e-12:
                normed.append({a: 1.0 for a in algos})
            else:
                normed.append({a: (metric[a] - lo) / (hi - lo) for a in algos})

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(scale)
        for algo in algos:
            values = [normed[i][algo] for i in range(len(labels))]
            _radar_polygon(ax, np.asarray(angles), values, COLORS[algo], algo)
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.2, 1.15), frameon=False)
    save_pdf(fig, output_dir / "fig_radar.pdf")


def fig_generations(root: Path, output_dir: Path, include_ppo: bool) -> None:
    algos = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"] + (["PPO"] if include_ppo else [])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (scale, _) in zip(axes, SCALES):
        means, errs = [], []
        for algo in algos:
            gens = [p.generation for p in load_final_series(root, 0.5, scale, algo)]
            if not gens:
                continue
            arr = np.asarray(gens, dtype=float)
            means.append(float(arr.mean()))
            errs.append(float(arr.std(ddof=1 if arr.size > 1 else 0) / math.sqrt(arr.size)))
        labels = [algo for algo in algos if load_final_series(root, 0.5, scale, algo)]
        xpos = np.arange(len(labels))
        ax.bar(xpos, means, yerr=errs, color=[COLORS[a] for a in labels], capsize=4)
        for x, mean in zip(xpos, means):
            ax.text(x, mean, f"{int(round(mean))}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_title(scale)
        ax.set_ylabel("Mean generations completed")
    save_pdf(fig, output_dir / "fig_wallclock_generations.pdf")


def average_rank(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and math.isclose(arr[order[j]], arr[order[i]], abs_tol=1e-12):
            j += 1
        rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def build_cd_cliques(ordered: list[tuple[str, float]], cd: float) -> list[list[str]]:
    cliques = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if ordered[j][1] - ordered[i][1] < cd:
                clique = [name for name, _ in ordered[i : j + 1]]
                if len(clique) >= 2:
                    cliques.append(clique)
    maximal = []
    for clique in cliques:
        if clique not in maximal and not any(set(clique) < set(other) for other in cliques):
            maximal.append(clique)
    return maximal


def draw_cd_diagram(avg_ranks: dict[str, float], cd: float, title: str, out_path: Path) -> None:
    ordered = sorted(avg_ranks.items(), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(8, 3))
    k = len(ordered)
    ax.set_xlim(0.7, k + 0.3)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    y_axis = 0.55
    ax.hlines(y_axis, 1, k, color="black")
    for tick in range(1, k + 1):
        ax.vlines(tick, y_axis - 0.03, y_axis + 0.03, color="black")
        ax.text(tick, y_axis - 0.08, str(tick), ha="center", va="top")
    top_y = [0.86, 0.76, 0.86]
    bot_y = [0.22, 0.12]
    ti = bi = 0
    for idx, (name, rank) in enumerate(ordered):
        top = idx % 2 == 0
        y = top_y[min(ti, len(top_y) - 1)] if top else bot_y[min(bi, len(bot_y) - 1)]
        if top:
            ti += 1
        else:
            bi += 1
        ax.vlines(rank, y_axis, y - 0.03 if top else y + 0.03, color="gray")
        ax.text(rank, y, f"{name} ({rank:.3f})", ha="center", va="bottom" if top else "top")
    for idx, clique in enumerate(build_cd_cliques(ordered, cd)):
        ranks = [avg_ranks[name] for name in clique]
        y = 0.94 - idx * 0.05
        ax.hlines(y, min(ranks), max(ranks), color="black", linewidth=2.0)
        ax.vlines([min(ranks), max(ranks)], y - 0.014, y + 0.014, color="black")
    ax.text(k, 0.94, f"CD = {cd:.3f}", ha="right", va="center")
    ax.set_title(title)
    save_pdf(fig, out_path)


def fig_cd(root: Path, output_dir: Path, repo_root: Path, ppo_status: str) -> None:
    base_algos = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"]
    results = {}
    for alpha in ALPHAS:
        for scale, _ in SCALES:
            results[(alpha, scale)] = [float(np.mean([p.fitness for p in load_final_series(root, alpha, scale, algo)])) for algo in base_algos]
    rank_matrix = np.asarray([average_rank(v) for v in results.values()], dtype=float)
    avg_ranks = {algo: float(rank_matrix[:, idx].mean()) for idx, algo in enumerate(base_algos)}
    cd = Q_ALPHA[4] * math.sqrt(4 * 5 / (6 * len(results)))
    draw_cd_diagram(avg_ranks, cd, "Critical Difference Diagram (N=9, 4 algorithms)", output_dir / "fig_cd_diagram_N9.pdf")

    if ppo_status != "absent":
        ppo_results = {}
        for scale, _ in SCALES:
            base = [float(np.mean([p.fitness for p in load_final_series(root, 0.5, scale, algo)])) for algo in base_algos]
            if ppo_status == "wallclock":
                ppo_vals = [p.fitness for p in load_final_series(root, 0.5, scale, "PPO")]
            else:
                ppo_vals = read_ppo_iteration_fallback(repo_root, scale)
            ppo_results[scale] = base + [float(np.mean(ppo_vals))]
        rank_matrix = np.asarray([average_rank(v) for v in ppo_results.values()], dtype=float)
        avg_ranks = {algo: float(rank_matrix[:, idx].mean()) for idx, algo in enumerate(base_algos + ["PPO"])}
        cd = Q_ALPHA[5] * math.sqrt(5 * 6 / (6 * len(ppo_results)))
        draw_cd_diagram(avg_ranks, cd, "Critical Difference Diagram (N=3, with PPO)", output_dir / "fig_cd_diagram_N3_with_PPO.pdf")


def fig_alpha_sensitivity(root: Path, output_dir: Path, include_ppo: bool) -> None:
    algos = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"] + (["PPO"] if include_ppo else [])
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for row, alpha in enumerate(ALPHAS):
        for col, (scale, budget) in enumerate(SCALES):
            ax = axes[row, col]
            plot_mean_ci(ax, root, alpha, scale, budget, algos)
            ax.set_xlim(0.0, budget)
            ax.set_xlabel("Wall-clock time (s)")
            ax.set_ylabel("Best fitness")
            ax.text(0.03, 0.95, f"α={alpha:.1f}, {scale}", transform=ax.transAxes, ha="left", va="top")
    axes[0, 0].legend(frameon=False, loc="upper right")
    save_pdf(fig, output_dir / "alpha_sensitivity.pdf")


def write_table_vs(root: Path, tables_dir: Path, algo: str, repo_root: Path, ppo_status: str) -> None:
    rows = []
    for scale, _ in SCALES:
        c_vals = [p.fitness for p in load_final_series(root, 0.5, scale, "CCHIHH")]
        if algo == "PPO" and ppo_status != "wallclock":
            b_vals = read_ppo_iteration_fallback(repo_root, scale)
        else:
            b_vals = [p.fitness for p in load_final_series(root, 0.5, scale, algo)]
        c_mean, c_std = mean_std(c_vals)
        b_mean, b_std = mean_std(b_vals)
        rows.append(
            {
                "scale": scale,
                "cchihh_mean": c_mean,
                "cchihh_std": c_std,
                f"{ALGO_FILE_KEYS[algo].lower()}_mean": b_mean,
                f"{ALGO_FILE_KEYS[algo].lower()}_std": b_std,
                "improvement_pct": improvement_pct(c_mean, b_mean),
                "p_value": paired_pvalue(c_vals, b_vals),
            }
        )
    out_name = {
        "CGA": "table_vs_CGA.csv",
        "IMOMA": "table_vs_IMOMA.csv",
        "PPO": "table_vs_PPO.csv",
        "DSAC-DE": "table_vs_DSAC_DE.csv",
    }[algo]
    out_path = tables_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_table_cv(root: Path, tables_dir: Path, repo_root: Path, ppo_status: str) -> None:
    algos = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"] + (["PPO"] if ppo_status != "absent" else [])
    rows = []
    for scale, _ in SCALES:
        row = {"scale": scale}
        for algo in algos:
            if algo == "PPO" and ppo_status != "wallclock":
                vals = read_ppo_iteration_fallback(repo_root, scale)
            else:
                vals = [p.fitness for p in load_final_series(root, 0.5, scale, algo)]
            if vals:
                row[f"{ALGO_FILE_KEYS[algo].lower()}_cv"] = coeff_var(vals)
        rows.append(row)
    out_path = tables_dir / "table_cv.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}, key=lambda x: (x != "scale", x)))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wall-clock matched figures and tables.")
    parser.add_argument("--input_root", default="results/wallclock")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--tables_dir", default="results/wallclock/tables")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = repo_root / input_root
    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = repo_root / figures_dir
    tables_dir = Path(args.tables_dir)
    if not tables_dir.is_absolute():
        tables_dir = repo_root / tables_dir

    configure_style()
    status = ppo_mode(input_root, repo_root)

    fig_comprehensive_baseline(input_root, figures_dir, repo_root, status)
    fig_convergence_group(input_root, figures_dir, "CGA_IMOMA_convergence.pdf", ["CCHIHH", "CGA", "IMOMA"])
    fig_convergence_group(input_root, figures_dir, "PPO_convergence.pdf", ["CCHIHH", "PPO"], status, repo_root)
    fig_convergence_group(input_root, figures_dir, "DSAC_convergence.pdf", ["CCHIHH", "DSAC-DE"])
    fig_boxplot_four(input_root, figures_dir)
    fig_scatter(input_root, figures_dir)
    fig_radar(input_root, figures_dir)
    fig_generations(input_root, figures_dir, include_ppo=(status == "wallclock"))
    fig_cd(input_root, figures_dir, repo_root, status)
    fig_alpha_sensitivity(input_root, figures_dir, include_ppo=(status == "wallclock"))

    for algo in ["CGA", "IMOMA", "PPO", "DSAC-DE"]:
        if algo == "PPO" and status == "absent":
            continue
        write_table_vs(input_root, tables_dir, algo, repo_root, status)
    write_table_cv(input_root, tables_dir, repo_root, status)


if __name__ == "__main__":
    main()
