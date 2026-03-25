from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required to generate the updated baseline artifacts.") from exc


ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "results" / "eval"
FIG_DIR = ROOT / "figures_updated"
TABLE_DIR = ROOT / "tables_updated"

ORDER = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA", "RDE", "NL-SHADE-LBC", "L-SRTDE", "PPO"]
EVOLUTIONARY = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA", "RDE", "NL-SHADE-LBC", "L-SRTDE"]
ALPHA_ORDER = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA", "RDE", "NL-SHADE-LBC", "L-SRTDE"]
SCALES = ["T100", "T200", "T500"]
ALPHAS = [0.2, 0.5, 0.8]
CI_Z = 1.96
EVAL_GRID = np.linspace(0.0, 400000.0, 401)
MAX_CURVE_POINTS = 200
CD_Q_ALPHA = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

COLORS = {
    "CCHIHH": "#E63946",
    "DSAC-DE": "#457B9D",
    "CGA": "#2A9D8F",
    "IMOMA": "#E9C46A",
    "PPO": "#8D99AE",
    "RDE": "#F4A261",
    "NL-SHADE-LBC": "#6A0572",
    "L-SRTDE": "#264653",
}

LINESTYLES = {
    "CCHIHH": "-",
    "DSAC-DE": "-",
    "CGA": "-",
    "IMOMA": "-",
    "PPO": "--",
    "RDE": "-",
    "NL-SHADE-LBC": "-",
    "L-SRTDE": "-",
}

MARKERS = {
    "CCHIHH": "o",
    "DSAC-DE": "s",
    "CGA": "^",
    "IMOMA": "D",
    "PPO": "x",
    "RDE": "v",
    "NL-SHADE-LBC": "P",
    "L-SRTDE": "*",
}

DISPLAY = {
    "CCHIHH": "CCHIHH",
    "DSAC-DE": "DSAC-DE",
    "CGA": "CGA",
    "IMOMA": "IMOMA",
    "RDE": "RDE",
    "NL-SHADE-LBC": "NL-SHADE-LBC",
    "L-SRTDE": "L-SRTDE",
    "PPO": "PPO (ref.)",
}

NEW_SOLVER_KEYS = {"RDE": "rde", "NL-SHADE-LBC": "nl_shade_lbc", "L-SRTDE": "l_srtde"}
RUNTIME_KEYS = {"CCHIHH": "CCHIHH", "DSAC-DE": "DSAC_DE", "CGA": "CGA", "IMOMA": "IMOMA", "PPO": "PPO"}
OLD_DEGRADATION_KEYS = {"CCHIHH": "CCHIHH", "DSAC-DE": "DSAC-DE", "CGA": "CGA", "IMOMA": "IMOMA"}

DEGRADATION_SCENARIOS = [
    {"key": "cloud_reduction", "title": "(a) Cloud Reduction", "levels": [("r10", "10%"), ("r20", "20%"), ("r30", "30%")]},
    {"key": "edge_reduction", "title": "(b) Edge Reduction", "levels": [("r10", "10%"), ("r20", "20%"), ("r30", "30%")]},
    {"key": "device_reduction", "title": "(c) Device Reduction", "levels": [("r10", "10%"), ("r20", "20%"), ("r30", "30%")]},
    {"key": "communication_inflation", "title": "(d) Comm. Inflation", "levels": [("p20", "20%"), ("p40", "40%"), ("p60", "60%")]},
]

TIME_RE = re.compile(r"Time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*s")


@dataclass
class Curve:
    evaluations: np.ndarray
    fitness: np.ndarray


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 7.5,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "0.8",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png")
    plt.close(fig)


def alpha_tag(alpha: float) -> str:
    return f"alpha{alpha:.1f}"


def fmt_mean_std(values: list[float], digits: int = 6) -> str:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1 if len(values) > 1 else 0))
    return rf"${mean:.{digits}f} \pm {std:.{digits}f}$"


def coeff_var(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1 if arr.size > 1 else 0))
    return math.nan if math.isclose(mean, 0.0, abs_tol=1e-12) else std / mean


def improvement_pct(cchihh_values: list[float], baseline_values: list[float]) -> float:
    c_mean = float(np.mean(cchihh_values))
    b_mean = float(np.mean(baseline_values))
    return (b_mean - c_mean) / b_mean * 100.0


def wilcoxon_pvalue(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return math.nan
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return math.nan


def rank_array(values: list[float]) -> np.ndarray:
    return rankdata(values, method="average")


def downsample_for_plot(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    n = len(arrays[0])
    if n <= MAX_CURVE_POINTS:
        return arrays
    indices = np.linspace(0, n - 1, MAX_CURVE_POINTS, dtype=int)
    return tuple(arr[indices] for arr in arrays)


def read_curve_csv(path: Path) -> Curve:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty curve file: {path}")
    if "eval_count" in rows[0]:
        xs = np.asarray([float(row["eval_count"]) for row in rows], dtype=float)
        ys = np.asarray([float(row["best_fitness"]) for row in rows], dtype=float)
    else:
        generations = np.asarray([float(row["generation"]) for row in rows], dtype=float)
        xs = generations * (400000.0 / max(generations[-1], 1.0))
        ys = np.asarray([float(row["best_fitness"]) for row in rows], dtype=float)
    if xs[0] > 0.0:
        xs = np.concatenate([[0.0], xs])
        ys = np.concatenate([[ys[0]], ys])
    if xs[-1] < 400000.0:
        xs = np.concatenate([xs, [400000.0]])
        ys = np.concatenate([ys, [ys[-1]]])
    return Curve(evaluations=xs, fitness=ys)


def interpolate_curve(curve: Curve, grid: np.ndarray = EVAL_GRID) -> np.ndarray:
    return np.interp(grid, curve.evaluations, curve.fitness)


def summarize_curves(paths: Iterable[Path], grid: np.ndarray = EVAL_GRID) -> tuple[np.ndarray, np.ndarray]:
    mats = np.vstack([interpolate_curve(read_curve_csv(path), grid) for path in paths])
    mean = mats.mean(axis=0)
    std = mats.std(axis=0, ddof=1 if mats.shape[0] > 1 else 0)
    ci = CI_Z * std / math.sqrt(mats.shape[0])
    return mean, ci


def trace_paths_main(algo: str, scale: str) -> list[Path]:
    if algo == "CCHIHH":
        base = EVAL_ROOT / "rerun_full" / "alpha0.5" / scale
        return [base / f"cchihh_full_{scale}_s{seed}_eval.csv" for seed in range(1, 11)]
    if algo == "DSAC-DE":
        base = EVAL_ROOT / "dsac_de_multiscale" / "alpha0.5" / scale
        return [base / f"DSAC_DE_{scale}_s{seed}_eval.csv" for seed in range(1, 11)]
    if algo == "CGA":
        base = EVAL_ROOT / "cga_baseline" / "alpha0.5" / scale
        return [base / f"CGA_{scale}_s{seed}_eval.csv" for seed in range(1, 11)]
    if algo == "IMOMA":
        base = EVAL_ROOT / "imoma_baseline" / "alpha0.5" / scale
        return [base / f"IMOMA_{scale}_s{seed}_eval.csv" for seed in range(1, 11)]
    if algo == "PPO":
        base = EVAL_ROOT / "ppo_baseline" / "alpha0.5" / scale
        return [base / f"PPO_{scale}_s{seed}_eval.csv" for seed in range(1, 11)]
    key = NEW_SOLVER_KEYS[algo]
    return [EVAL_ROOT / "traces" / f"{key}_{scale}_main_alpha0.5_seed{seed}.csv" for seed in range(1, 11)]


def trace_paths_alpha(algo: str, scale: str, alpha: float) -> list[Path]:
    if algo == "CCHIHH":
        if math.isclose(alpha, 0.5):
            return trace_paths_main(algo, scale)
        base = EVAL_ROOT / "rerun_full" / "rerun_alpha" / alpha_tag(alpha) / scale
        return [base / f"cchihh_full_{scale}_s{seed}_eval.csv" for seed in range(1, 11)]
    if algo == "DSAC-DE":
        if math.isclose(alpha, 0.5):
            return trace_paths_main(algo, scale)
        base = EVAL_ROOT / "dsac_alpha" / alpha_tag(alpha) / scale
        return [base / f"DSAC_DE_{scale}_s{seed}_eval.csv" for seed in range(1, 11)]
    if algo == "CGA":
        if math.isclose(alpha, 0.5):
            return trace_paths_main(algo, scale)
        base = EVAL_ROOT / "cga_alpha" / alpha_tag(alpha) / scale
        return [base / f"CGA_{scale}_s{seed}_eval.csv" for seed in range(1, 11)]
    if algo == "IMOMA":
        if math.isclose(alpha, 0.5):
            return trace_paths_main(algo, scale)
        base = EVAL_ROOT / "imoma_alpha" / alpha_tag(alpha) / scale
        return [base / f"IMOMA_{scale}_s{seed}_eval.csv" for seed in range(1, 11)]
    key = NEW_SOLVER_KEYS[algo]
    if math.isclose(alpha, 0.5):
        return [EVAL_ROOT / "traces" / f"{key}_{scale}_main_alpha0.5_seed{seed}.csv" for seed in range(1, 11)]
    return [EVAL_ROOT / "traces" / f"{key}_{scale}_alpha{alpha:.1f}_seed{seed}.csv" for seed in range(1, 11)]


def runtime_paths_old(algo: str, scale: str) -> list[Path]:
    key = RUNTIME_KEYS[algo]
    base = ROOT / "results" / "wallclock" / "alpha0.5" / scale
    return [base / f"{key}_seed{seed}.csv" for seed in range(1, 11)]


def runtime_log_path_old(algo: str, scale: str, seed: int) -> Path | None:
    if algo == "DSAC-DE":
        return ROOT / "outputs" / "results" / "dsac_de_multiscale" / scale / f"DSAC_DE_seed{seed}.txt"
    if algo in {"CGA", "IMOMA", "PPO"}:
        return ROOT / "outputs" / "results" / "cchihh_ablation_suite" / "baseline" / scale / f"{algo}_seed{seed}.txt"
    if algo == "CCHIHH":
        return ROOT / "outputs" / "results" / "cchihh_full_canonical_multiscale" / scale / f"CCHIHH_full_seed{seed}.txt"
    return None


def parse_runtime_from_log(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = TIME_RE.findall(text)
    if not matches:
        raise ValueError(f"Could not parse runtime from {path}")
    return float(matches[-1])


def load_runtime_old(algo: str, scale: str) -> list[float]:
    runtimes: list[float] = []
    for seed, path in enumerate(runtime_paths_old(algo, scale), start=1):
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise ValueError(f"Empty runtime file: {path}")
            runtimes.append(float(rows[-1]["time_seconds"]))
            continue
        log_path = runtime_log_path_old(algo, scale, seed)
        if log_path is None or not log_path.exists():
            raise FileNotFoundError(f"Missing runtime sources for {algo} {scale} seed{seed}")
        runtimes.append(parse_runtime_from_log(log_path))
    return runtimes


def load_runtime_new(algo: str, scale: str) -> list[float]:
    key = NEW_SOLVER_KEYS[algo]
    runtimes: list[float] = []
    for seed in range(1, 11):
        path = EVAL_ROOT / "main" / f"{key}_{scale}_alpha0.5_seed{seed}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        runtimes.append(float(data["wall_clock_seconds"]))
    return runtimes


def load_finals_from_paths(paths: list[Path]) -> list[float]:
    return [float(read_curve_csv(path).fitness[-1]) for path in paths]


def load_nominal_finals(algo: str, scale: str) -> list[float]:
    return load_finals_from_paths(trace_paths_main(algo, scale))


def load_alpha_finals(algo: str, scale: str, alpha: float) -> list[float]:
    return load_finals_from_paths(trace_paths_alpha(algo, scale, alpha))


def mean_runtime(algo: str, scale: str) -> float:
    runs = load_runtime_old(algo, scale) if algo in RUNTIME_KEYS else load_runtime_new(algo, scale)
    return float(np.mean(runs))


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1 if arr.size > 1 else 0))
    ci = CI_Z * std / math.sqrt(arr.size) if arr.size else math.nan
    return mean, ci


def load_old_degradation_rows() -> list[dict[str, str]]:
    path = EVAL_ROOT / "stress_robustness" / "index.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_new_degradation_rows() -> list[dict[str, str]]:
    path = EVAL_ROOT / "summaries" / "degradation_new_solvers.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def degradation_algorithms_available(scale: str = "T500") -> list[str]:
    old_rows = load_old_degradation_rows()
    new_rows = load_new_degradation_rows()
    available: list[str] = []
    for algo in ORDER:
        if algo in OLD_DEGRADATION_KEYS:
            exists = any(row["instance"] == scale and row["algorithm"] == OLD_DEGRADATION_KEYS[algo] for row in old_rows)
        elif algo in NEW_SOLVER_KEYS:
            exists = any(row["scale"] == scale and row["solver"] == algo for row in new_rows)
        else:
            exists = False
        if exists:
            available.append(algo)
    return available


def degradation_series(algo: str, scenario_key: str, severity: str, metric: str, scale: str = "T500") -> list[float]:
    if algo in OLD_DEGRADATION_KEYS:
        rows = [
            row
            for row in load_old_degradation_rows()
            if row["instance"] == scale
            and row["algorithm"] == OLD_DEGRADATION_KEYS[algo]
            and row["scenario_family"] == scenario_key
            and row["severity"] == severity
        ]
        rows.sort(key=lambda row: int(row["seed"]))
        if metric == "fitness":
            return [float(row["best_fitness"]) for row in rows]
        nominal = load_nominal_finals(algo, scale)
        return [float(row["best_fitness"]) / nominal[int(row["seed"]) - 1] for row in rows]

    rows = [
        row
        for row in load_new_degradation_rows()
        if row["scale"] == scale
        and row["solver"] == algo
        and row["scenario_family"] == scenario_key
        and row["severity"] == severity
    ]
    rows.sort(key=lambda row: int(row["seed"]))
    if metric == "fitness":
        return [float(row["final_fitness"]) for row in rows]
    return [float(row["prr"]) for row in rows]


def write_table5() -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CCHIHH vs. baselines on nominal settings (10 runs). Positive improvement indicates lower fitness than the baseline.}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Scale & Baseline & CCHIHH & Baseline & Improvement (\%) & Wilcoxon $p$ \\",
        r"\midrule",
    ]
    for scale in SCALES:
        c_vals = load_nominal_finals("CCHIHH", scale)
        for algo in ORDER[1:]:
            b_vals = load_nominal_finals(algo, scale)
            improvement = improvement_pct(c_vals, b_vals)
            p_value = wilcoxon_pvalue(c_vals, b_vals) if algo != "PPO" else math.nan
            sig = "**" if algo != "PPO" and np.isfinite(p_value) and p_value < 0.05 else ""
            p_text = "--" if algo == "PPO" or not np.isfinite(p_value) else f"{p_value:.4g}"
            baseline_name = "PPO$^\\dagger$" if algo == "PPO" else algo
            lines.append(
                f"{scale} & {baseline_name} & {fmt_mean_std(c_vals)} & {fmt_mean_std(b_vals)} & {improvement:.2f}{sig} & {p_text} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\\",
            r"\footnotesize{$^\dagger$ PPO is descriptive only; no inferential claim is made for its $p$-value.}",
            r"\end{table}",
            "",
        ]
    )
    (TABLE_DIR / "table5_baseline_improvement.tex").write_text("\n".join(lines), encoding="utf-8")


def write_table6() -> None:
    header = "Scale & " + " & ".join(["CCHIHH", "DSAC-DE", "CGA", "IMOMA", "RDE", "NL-SHADE-LBC", "L-SRTDE", "PPO$^\\dagger$"]) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Run-to-run dispersion measured by coefficient of variation (CV = Std / Mean) over 10 runs.}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for scale in SCALES:
        row = [scale] + [f"{coeff_var(load_nominal_finals(algo, scale)):.4f}" for algo in ORDER]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLE_DIR / "table6_cv_dispersion.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_curve_panel(ax: plt.Axes, algorithms: list[str], path_getter, scale: str, alpha: float | None = None) -> None:
    for algo in algorithms:
        paths = path_getter(algo, scale) if alpha is None else path_getter(algo, scale, alpha)
        mean, ci = summarize_curves(paths)
        evals_plot, mean_plot, low_plot, high_plot = downsample_for_plot(EVAL_GRID, mean, mean - ci, mean + ci)
        ax.plot(
            evals_plot,
            mean_plot,
            color=COLORS[algo],
            linewidth=1.5 if algo == "CCHIHH" else 1.0,
            linestyle=LINESTYLES[algo],
            label=DISPLAY[algo],
        )
        ax.fill_between(evals_plot, low_plot, high_plot, color=COLORS[algo], alpha=0.15)
    ax.set_xlim(0, 400000)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best fitness")


def figure24() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 9.2))
    for row, alpha in enumerate(ALPHAS):
        for col, scale in enumerate(SCALES):
            ax = axes[row, col]
            plot_curve_panel(ax, ALPHA_ORDER, trace_paths_alpha, scale, alpha)
            if row == 0:
                ax.set_title(scale)
            if row < 2:
                ax.set_xlabel("")
            if col > 0:
                ax.set_ylabel("")
            if col == 0:
                ax.text(-0.26, 0.5, rf"$\alpha={alpha:.1f}$", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(ALPHA_ORDER),
        fontsize=7.5,
        frameon=True,
        columnspacing=1.0,
        handlelength=1.5,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    save_figure(fig, "fig24_alpha_sensitivity")


def figure25() -> None:
    algorithms = degradation_algorithms_available("T500")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8))
    axes = axes.flatten()
    for ax, scenario in zip(axes, DEGRADATION_SCENARIOS):
        x = np.arange(len(scenario["levels"]))
        for algo in algorithms:
            means = []
            cis = []
            for severity, _ in scenario["levels"]:
                values = degradation_series(algo, scenario["key"], severity, "fitness", "T500")
                mean, ci = mean_ci(values)
                means.append(mean)
                cis.append(ci)
            ax.errorbar(
                x,
                means,
                yerr=cis,
                color=COLORS[algo],
                linestyle=LINESTYLES[algo],
                linewidth=1.5 if algo == "CCHIHH" else 1.0,
                marker=MARKERS[algo],
                markersize=4,
                capsize=2,
                label=DISPLAY[algo],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in scenario["levels"]])
        ax.set_title(scenario["title"])
        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("Best Fitness")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(len(algorithms), 4), frameon=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    save_figure(fig, "fig25_degradation_abs")


def figure26() -> None:
    algorithms = degradation_algorithms_available("T500")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8))
    axes = axes.flatten()
    for ax, scenario in zip(axes, DEGRADATION_SCENARIOS):
        x = np.arange(len(scenario["levels"]))
        for algo in algorithms:
            means = []
            cis = []
            for severity, _ in scenario["levels"]:
                values = degradation_series(algo, scenario["key"], severity, "prr", "T500")
                mean, ci = mean_ci(values)
                means.append(mean)
                cis.append(ci)
            ax.errorbar(
                x,
                means,
                yerr=cis,
                color=COLORS[algo],
                linestyle=LINESTYLES[algo],
                linewidth=1.5 if algo == "CCHIHH" else 1.0,
                marker=MARKERS[algo],
                markersize=4,
                capsize=2,
                label=DISPLAY[algo],
            )
        ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in scenario["levels"]])
        ax.set_title(scenario["title"])
        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("PRR")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(len(algorithms), 4), frameon=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    save_figure(fig, "fig26_degradation_prr")


def figure27() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for ax, scale in zip(axes, SCALES):
        data = [load_nominal_finals(algo, scale) for algo in ORDER]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=[DISPLAY[algo] for algo in ORDER], widths=0.68)
        for patch, algo in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COLORS[algo])
            patch.set_alpha(0.18 if algo != "CCHIHH" else 0.28)
            patch.set_edgecolor(COLORS[algo])
            patch.set_linewidth(1.0)
        for median, algo in zip(bp["medians"], ORDER):
            median.set_color(COLORS[algo])
            median.set_linewidth(1.2)
        c_vals = data[0]
        top = max(float(np.max(vals)) for vals in data)
        bottom = min(float(np.min(vals)) for vals in data)
        pad = (top - bottom) * 0.24 if not math.isclose(top, bottom) else 0.1
        ax.set_ylim(bottom - pad * 0.10, top + pad)
        for idx, algo in enumerate(ORDER[1:], start=2):
            vals = load_nominal_finals(algo, scale)
            improvement = improvement_pct(c_vals, vals)
            p_value = wilcoxon_pvalue(c_vals, vals) if algo != "PPO" else math.nan
            suffix = " **" if algo != "PPO" and np.isfinite(p_value) and p_value < 0.05 else ""
            ax.text(idx, float(np.max(vals)) + pad * 0.08, f"{improvement:.1f}%{suffix}", ha="center", va="bottom", fontsize=8)
        ax.set_title(scale)
        ax.set_ylabel("Final fitness")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_figure(fig, "fig27_baseline_overview")


def figure28() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for ax, scale in zip(axes, SCALES):
        plot_curve_panel(ax, ORDER, trace_paths_main, scale)
        ax.set_title(scale)
        if ax is not axes[0]:
            ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
    save_figure(fig, "fig28_convergence_all_baselines")


def figure29() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for ax, scale in zip(axes, SCALES):
        means = [mean_runtime(algo, scale) for algo in ORDER]
        bars = ax.bar(range(len(ORDER)), means, color=[COLORS[algo] for algo in ORDER], alpha=0.85, width=0.72)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([DISPLAY[algo] for algo in ORDER], rotation=20)
        ax.set_ylabel("Runtime (s)")
        ax.set_title(scale)
    fig.tight_layout()
    save_figure(fig, "fig29_runtime_comparison")


def contiguous_cliques(ordered: list[tuple[str, float]], cd: float) -> list[list[str]]:
    cliques: list[list[str]] = []
    for start in range(len(ordered)):
        for end in range(start + 1, len(ordered)):
            if ordered[end][1] - ordered[start][1] < cd:
                clique = [name for name, _ in ordered[start : end + 1]]
                if clique not in cliques:
                    cliques.append(clique)
    maximal: list[list[str]] = []
    for clique in cliques:
        if not any(set(clique).issubset(set(other)) and clique != other for other in cliques):
            maximal.append(clique)
    return maximal


def draw_cd_diagram(avg_ranks: dict[str, float], cd: float, friedman_p: float) -> None:
    ordered = sorted(avg_ranks.items(), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    ax.set_xlim(0.7, len(ordered) + 0.3)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    y_axis = 0.55
    ax.hlines(y_axis, 1, len(ordered), color="black", linewidth=1.0)
    for tick in range(1, len(ordered) + 1):
        ax.vlines(tick, y_axis - 0.03, y_axis + 0.03, color="black", linewidth=1.0)
        ax.text(tick, y_axis - 0.08, str(tick), ha="center", va="top")
    ax.text(1.0, y_axis + 0.08, "Better", ha="left", va="bottom")
    ax.text(len(ordered), y_axis + 0.08, "Worse", ha="right", va="bottom")
    cd_left = 1.0
    cd_right = min(len(ordered), cd_left + cd)
    ax.hlines(0.88, cd_left, cd_right, color="black", linewidth=1.2)
    ax.vlines([cd_left, cd_right], 0.86, 0.90, color="black", linewidth=1.2)
    ax.text((cd_left + cd_right) / 2.0, 0.92, f"CD = {cd:.3f}", ha="center", va="bottom")
    split = math.ceil(len(ordered) / 2)
    for idx, (algo, rank) in enumerate(ordered):
        left_side = idx < split
        y_text = 0.36 - (idx if left_side else idx - split) * 0.055
        elbow = 0.95 if left_side else len(ordered) + 0.05
        text_x = 0.72 if left_side else len(ordered) + 0.18
        ha = "right" if left_side else "left"
        ax.plot([rank, rank], [y_axis, y_text], color=COLORS[algo], linewidth=1.0)
        ax.plot([rank, elbow], [y_text, y_text], color=COLORS[algo], linewidth=1.0)
        ax.text(text_x, y_text, algo, ha=ha, va="center", color=COLORS[algo], fontsize=9)
    for idx, clique in enumerate(contiguous_cliques(ordered, cd)):
        left = min(avg_ranks[name] for name in clique)
        right = max(avg_ranks[name] for name in clique)
        ax.hlines(0.15 + idx * 0.05, left, right, color="black", linewidth=1.8)
    ax.set_title(f"Critical Difference Diagram (Friedman-Nemenyi, p={friedman_p:.4g})", pad=6)
    fig.tight_layout()
    save_figure(fig, "fig30_cd_diagram")


def figure30() -> None:
    rows = []
    for scale in SCALES:
        for alpha in ALPHAS:
            rows.append([float(np.mean(load_alpha_finals(algo, scale, alpha))) for algo in EVOLUTIONARY])
    rank_matrix = np.vstack([rank_array(row) for row in rows])
    avg_ranks = {algo: float(rank_matrix[:, idx].mean()) for idx, algo in enumerate(EVOLUTIONARY)}
    _, friedman_p = friedmanchisquare(*[rank_matrix[:, idx] for idx in range(rank_matrix.shape[1])])
    cd = CD_Q_ALPHA[len(EVOLUTIONARY)] * math.sqrt(len(EVOLUTIONARY) * (len(EVOLUTIONARY) + 1) / (6.0 * len(rows)))
    draw_cd_diagram(avg_ranks, cd, float(friedman_p))


def normalize_lower_better(values: dict[str, float]) -> dict[str, float]:
    arr = np.asarray(list(values.values()), dtype=float)
    lo = float(arr.min())
    hi = float(arr.max())
    if math.isclose(lo, hi):
        return {key: 1.0 for key in values}
    return {key: (hi - value) / (hi - lo) for key, value in values.items()}


def figure31() -> None:
    metrics = ["Solution Quality", "Stability", "Efficiency", "Convergence Speed"]
    theta = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    theta = np.concatenate([theta, theta[:1]])
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), subplot_kw={"projection": "polar"})
    for ax, scale in zip(axes, SCALES):
        quality = {algo: float(np.mean(load_nominal_finals(algo, scale))) for algo in ORDER}
        stability = {algo: coeff_var(load_nominal_finals(algo, scale)) for algo in ORDER}
        efficiency = {algo: mean_runtime(algo, scale) for algo in ORDER}
        speed = {algo: float(summarize_curves(trace_paths_main(algo, scale), np.array([80000.0]))[0][0]) for algo in ORDER}
        normalized = {
            "Solution Quality": normalize_lower_better(quality),
            "Stability": normalize_lower_better(stability),
            "Efficiency": normalize_lower_better(efficiency),
            "Convergence Speed": normalize_lower_better(speed),
        }
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels(metrics)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
        for algo in ORDER:
            values = [normalized[metric][algo] for metric in metrics]
            values.append(values[0])
            ax.plot(theta, values, color=COLORS[algo], linestyle=LINESTYLES[algo], linewidth=1.5 if algo == "CCHIHH" else 1.0, label=DISPLAY[algo])
            ax.fill(theta, values, color=COLORS[algo], alpha=0.08)
        ax.set_title(scale, pad=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))
    save_figure(fig, "fig31_radar_comparison")


def write_notes() -> None:
    notes = [
        "Figure 24 excludes PPO per user instruction.",
        "PPO alpha=0.2/0.8 traces are not present in the repository, so PPO is omitted from alpha sensitivity.",
        "Figures 25 and 26 use T500 degradation data.",
    ]
    degradation_algos = degradation_algorithms_available("T500")
    missing_degradation = [algo for algo in ORDER if algo not in degradation_algos]
    if missing_degradation:
        notes.append("Missing degradation data for: " + ", ".join(missing_degradation) + ".")
    (TABLE_DIR / "generation_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def validate_inputs() -> None:
    missing: list[str] = []
    for scale in SCALES:
        for algo in ORDER:
            for path in trace_paths_main(algo, scale):
                if not path.exists():
                    missing.append(str(path))
        for algo in ALPHA_ORDER:
            for alpha in ALPHAS:
                for path in trace_paths_alpha(algo, scale, alpha):
                    if not path.exists():
                        missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing[:20]))


def main() -> None:
    configure_style()
    ensure_dirs()
    validate_inputs()
    write_table5()
    write_table6()
    figure24()
    figure25()
    figure26()
    figure27()
    figure28()
    figure29()
    figure30()
    figure31()
    write_notes()


if __name__ == "__main__":
    main()
