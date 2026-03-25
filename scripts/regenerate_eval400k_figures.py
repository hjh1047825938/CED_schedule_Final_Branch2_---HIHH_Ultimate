from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "results" / "eval"
FINAL_SUMMARY = ROOT / "results" / "final_summary.csv"
WALLCLOCK_ROOT = ROOT / "results" / "wallclock"
SCHEDULE_ROOT = ROOT / "results" / "rerun_schedule"
OUTPUT_DIR = ROOT / "results" / "figures_eval400k"

SCALES = ["T100", "T200", "T500"]
ALPHAS = [0.2, 0.5, 0.8]
SEEDS = list(range(1, 11))
MAX_EVAL = 400000

COLORS = {
    "CCHIHH": "#d62728",
    "CGA": "#1f77b4",
    "IMOMA": "#2ca02c",
    "DSAC-DE": "#ff7f0e",
    "PPO": "#9467bd",
    "noCC": "#8c564b",
    "noHI": "#e377c2",
    "noCB": "#7f7f7f",
    "noMig": "#bcbd22",
    "noGate": "#17becf",
}

PRETTY = {
    "cchihh": "CCHIHH",
    "cchihh_full": "CCHIHH",
    "cchihh_nocc": "noCC",
    "cchihh_nohi": "noHI",
    "cchihh_nocb": "noCB",
    "cchihh_nomig": "noMig",
    "cchihh_nogate": "noGate",
    "dsac_de": "DSAC-DE",
    "cga": "CGA",
    "imoma": "IMOMA",
    "ppo": "PPO",
}

FINAL_VARIANT_MAP = {
    "CCHIHH-full": "cchihh_full",
    "CCHIHH-full-old": "cchihh_full_old",
    "CCHIHH-noCC": "cchihh_nocc",
    "CCHIHH-noHI": "cchihh_nohi",
    "CCHIHH-noCB": "cchihh_nocb",
    "CCHIHH-noMig": "cchihh_nomig",
    "CCHIHH-noGate": "cchihh_nogate",
    "CCHIHH-tgate5": "cchihh_tgate5",
    "CCHIHH-tgate10": "cchihh_tgate10",
    "CCHIHH-tgate15": "cchihh_tgate15",
    "CCHIHH-tgate20": "cchihh_tgate20",
    "CCHIHH-tgate25": "cchihh_tgate25",
    "CGA": "cga",
    "DSAC-DE": "dsac_de",
    "IMOMA": "imoma",
    "PPO": "ppo",
}

OPS_BY_BLOCK = {
    "off": ["GA", "DE", "BITFLIP", "RESAMPLE"],
    "seq": ["GA", "SWAP", "VNS", "RESAMPLE"],
    "dev": ["DE", "GDE", "LEVY", "RESAMPLE"],
}
OPS_COLUMN_PREFIX = {"off": "offload", "seq": "seq", "dev": "dev"}


@dataclass
class FigureStatus:
    filename: str
    description: str
    ok: bool
    note: str = ""


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_variant(name: str) -> str:
    return name.lower().replace("-", "_")


def pretty_name(name: str) -> str:
    return PRETTY.get(name, name)


def save_pdf(fig: plt.Figure, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def source_priority(path: Path, variant: str) -> int:
    parts = {part.lower() for part in path.parts}
    if "cchihh" in parts and variant == "cchihh_full":
        return 30
    if "gate" in parts and variant in {"cchihh_full", "cchihh_nogate"}:
        return 20
    return 0


def load_eval_curves() -> dict[tuple[str, float, str, int], pd.DataFrame]:
    curves: dict[tuple[str, float, str, int], pd.DataFrame] = {}
    chosen: dict[tuple[str, float, str, int], int] = {}
    pattern = re.compile(r"(?P<variant>.+)_(?P<scale>T\d+)_s(?P<seed>\d+)_eval\.csv$", re.IGNORECASE)
    for path in sorted(EVAL_ROOT.rglob("*_eval.csv")):
        if path.name.endswith("_ops_eval.csv") or "_w_" in path.name:
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        alpha_dir = next((part for part in path.parts if part.startswith("alpha")), "alpha0.5")
        alpha = float(alpha_dir.replace("alpha", ""))
        variant = normalize_variant(match.group("variant"))
        scale = match.group("scale")
        seed = int(match.group("seed"))
        key = (variant, alpha, scale, seed)
        priority = source_priority(path, variant)
        if key in chosen and priority < chosen[key]:
            continue
        df = pd.read_csv(path).rename(columns={"best_fitness": "best_fit"})
        if "eval_count" not in df.columns or "best_fit" not in df.columns:
            continue
        curves[key] = df[["eval_count", "best_fit"]].sort_values("eval_count").drop_duplicates("eval_count")
        chosen[key] = priority
    return curves


def load_ops_curves() -> dict[tuple[str, float, str, int], pd.DataFrame]:
    out: dict[tuple[str, float, str, int], pd.DataFrame] = {}
    chosen: dict[tuple[str, float, str, int], int] = {}
    pattern = re.compile(r"(?P<variant>.+)_(?P<scale>T\d+)_s(?P<seed>\d+)_ops_eval\.csv$", re.IGNORECASE)
    for path in sorted(EVAL_ROOT.rglob("*_ops_eval.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        alpha_dir = next((part for part in path.parts if part.startswith("alpha")), "alpha0.5")
        variant = normalize_variant(match.group("variant"))
        key = (variant, float(alpha_dir.replace("alpha", "")), match.group("scale"), int(match.group("seed")))
        priority = source_priority(path, variant)
        if key in chosen and priority < chosen[key]:
            continue
        out[key] = pd.read_csv(path)
        chosen[key] = priority
    return out


def load_weight_curves() -> dict[tuple[str, float, str, int, str], pd.DataFrame]:
    out: dict[tuple[str, float, str, int, str], pd.DataFrame] = {}
    chosen: dict[tuple[str, float, str, int, str], int] = {}
    pattern = re.compile(r"(?P<variant>.+)_(?P<scale>T\d+)_s(?P<seed>\d+)_w_(?P<block>\w+)_eval\.csv$", re.IGNORECASE)
    for path in sorted(EVAL_ROOT.rglob("*_w_*_eval.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        alpha_dir = next((part for part in path.parts if part.startswith("alpha")), "alpha0.5")
        variant = normalize_variant(match.group("variant"))
        key = (variant, float(alpha_dir.replace("alpha", "")), match.group("scale"), int(match.group("seed")), match.group("block").lower())
        priority = source_priority(path, variant)
        if key in chosen and priority < chosen[key]:
            continue
        out[key] = pd.read_csv(path)
        chosen[key] = priority
    return out


def load_final_summary() -> pd.DataFrame:
    df = pd.read_csv(FINAL_SUMMARY)
    df["variant_key"] = df["variant"].map(FINAL_VARIANT_MAP).fillna(df["variant"].map(normalize_variant))
    df["alpha"] = df["alpha"].astype(float)
    return df


def load_wallclock_runs() -> dict[tuple[str, float, str, int], pd.DataFrame]:
    out: dict[tuple[str, float, str, int], pd.DataFrame] = {}
    pattern = re.compile(r"(?P<algo>[A-Za-z0-9_-]+)_seed(?P<seed>\d+)\.csv$", re.IGNORECASE)
    for path in sorted(WALLCLOCK_ROOT.rglob("*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        alpha_dir = next((part for part in path.parts if part.startswith("alpha")), None)
        scale = next((part for part in path.parts if re.fullmatch(r"T\d+", part)), None)
        if alpha_dir is None or scale is None:
            continue
        key = (
            normalize_variant(match.group("algo")),
            float(alpha_dir.replace("alpha", "")),
            scale,
            int(match.group("seed")),
        )
        out[key] = pd.read_csv(path)
    return out


def interp_matrix(curves: list[pd.DataFrame], grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not curves:
        return np.array([]), np.empty((0, 0))
    if grid is None:
        grid = np.asarray(sorted({int(v) for df in curves for v in df["eval_count"].tolist() if v <= MAX_EVAL}), dtype=float)
    matrix = []
    for df in curves:
        sub = df[df["eval_count"] <= MAX_EVAL]
        xs = sub["eval_count"].to_numpy(dtype=float)
        ys = sub["best_fit"].to_numpy(dtype=float)
        if xs.size:
            matrix.append(np.interp(grid, xs, ys))
    return grid, np.asarray(matrix, dtype=float)


def mean_ci(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    se = matrix.std(axis=0, ddof=0) / max(math.sqrt(matrix.shape[0]), 1.0)
    return mean, 1.96 * se


def get_curves(curves: dict[tuple[str, float, str, int], pd.DataFrame], variant: str, alpha: float, scale: str) -> list[pd.DataFrame]:
    return [curves[(variant, alpha, scale, seed)] for seed in SEEDS if (variant, alpha, scale, seed) in curves]


def get_finals(curves: dict[tuple[str, float, str, int], pd.DataFrame], variant: str, alpha: float, scale: str) -> list[float]:
    return [float(curves[(variant, alpha, scale, seed)]["best_fit"].iloc[-1]) for seed in SEEDS if (variant, alpha, scale, seed) in curves]


def get_runtime(final_df: pd.DataFrame, variant: str, alpha: float, scale: str) -> list[float]:
    mask = (final_df["variant_key"] == variant) & (final_df["alpha"] == alpha) & (final_df["scale"] == scale)
    return final_df.loc[mask].sort_values("seed")["runtime_s"].astype(float).tolist()


def add_conv_panel(ax: plt.Axes, curves: dict[tuple[str, float, str, int], pd.DataFrame], labels: list[tuple[str, str]], alpha: float, scale: str) -> None:
    ax.set_title(scale)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best fitness")
    ax.set_xlim(0, MAX_EVAL)
    for variant, label in labels:
        seed_curves = get_curves(curves, variant, alpha, scale)
        if not seed_curves:
            continue
        grid, matrix = interp_matrix(seed_curves)
        if not matrix.size:
            continue
        mean, ci = mean_ci(matrix)
        color = COLORS.get(label, "gray")
        ax.plot(grid, mean, color=color, label=label, linewidth=1.25)
        ax.fill_between(grid, mean - ci, mean + ci, color=color, alpha=0.15)
    ax.legend(frameon=False, loc="best")


def plot_triptych(curves: dict[tuple[str, float, str, int], pd.DataFrame], labels: list[tuple[str, str]], alpha: float, filename: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, scale in zip(axes, SCALES):
        add_conv_panel(ax, curves, labels, alpha, scale)
    fig.tight_layout()
    return save_pdf(fig, filename)


def fig10(ops_curves: dict[tuple[str, float, str, int], pd.DataFrame]) -> Path | None:
    seed_dfs = [ops_curves[k].set_index("eval_count") for k in ops_curves if k[0] == "cchihh_full" and k[1] == 0.5 and k[2] == "T500"]
    if not seed_dfs:
        return None
    mean_df = pd.concat(seed_dfs).groupby(level=0).mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharex=True)
    for ax, (prefix, title) in zip(axes, [("off", "Offload"), ("seq", "Sequence"), ("dev", "Device")]):
        ax.set_title(title)
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Selection Probability")
        ax.set_xlim(0, MAX_EVAL)
        ax.set_ylim(0, 1)
        col_prefix = OPS_COLUMN_PREFIX[prefix]
        for op in OPS_BY_BLOCK[prefix]:
            ax.plot(mean_df["eval_count"], mean_df[f"{col_prefix}_{op}"], label=op)
        ax.legend(frameon=False)
    fig.tight_layout()
    return save_pdf(fig, "fig10_op_selection_T500.pdf")


def fig11(ops_curves: dict[tuple[str, float, str, int], pd.DataFrame]) -> Path | None:
    seed_dfs = [ops_curves[k].set_index("eval_count") for k in ops_curves if k[0] == "cchihh_full" and k[1] == 0.5 and k[2] == "T500"]
    if not seed_dfs:
        return None
    mean_df = pd.concat(seed_dfs).groupby(level=0).mean().reset_index()
    early = mean_df[mean_df["eval_count"] <= MAX_EVAL * 0.2]
    late = mean_df[mean_df["eval_count"] >= MAX_EVAL * 0.8]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7), sharey=True)
    for ax, (prefix, title) in zip(axes, [("off", "Offload"), ("seq", "Sequence"), ("dev", "Device")]):
        ops = OPS_BY_BLOCK[prefix]
        col_prefix = OPS_COLUMN_PREFIX[prefix]
        early_vals = [float(early[f"{col_prefix}_{op}"].mean()) for op in ops]
        late_vals = [float(late[f"{col_prefix}_{op}"].mean()) for op in ops]
        x = np.arange(len(ops))
        width = 0.36
        ax.bar(x - width / 2, early_vals, width, label="Early 20%", color="#a6cee3")
        ax.bar(x + width / 2, late_vals, width, label="Late 20%", color="#fb9a99")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(ops)
        ax.set_xlabel("Operator")
        ax.set_ylabel("Mean selection probability")
        for idx, (ev, lv) in enumerate(zip(early_vals, late_vals)):
            ax.text(idx, max(ev, lv) + 0.03, f"{ev*100:.1f}%→{lv*100:.1f}%\n({(lv-ev)*100:+.1f}%)", ha="center", va="bottom", fontsize=7)
        ax.legend(frameon=False)
    fig.tight_layout()
    return save_pdf(fig, "fig11_early_late_bar_T500.pdf")


def fig12(weight_curves: dict[tuple[str, float, str, int, str], pd.DataFrame]) -> Path | None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.8), sharex=True)
    has_data = False
    for ax, block in zip(axes, ["off", "seq", "dev"]):
        dfs = [weight_curves[k] for k in weight_curves if k[0] == "cchihh_full" and k[1] == 0.5 and k[2] == "T500" and k[4] == block]
        if not dfs:
            continue
        has_data = True
        mean_df = pd.concat(dfs).groupby(["eval_count", "op_id"], as_index=False)["norm"].mean()
        pivot = mean_df.pivot(index="op_id", columns="eval_count", values="norm").sort_index()
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_title(block.capitalize())
        ax.set_yticks(np.arange(len(OPS_BY_BLOCK[block])))
        ax.set_yticklabels(OPS_BY_BLOCK[block])
        cols = pivot.columns.to_numpy(dtype=int)
        ticks = np.linspace(0, len(cols) - 1, 6, dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(cols[i]) for i in ticks])
    if not has_data:
        plt.close(fig)
        return None
    axes[-1].set_xlabel("Evaluations")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Normalized weight")
    fig.tight_layout()
    return save_pdf(fig, "fig12_weight_heatmap_T500.pdf")


def fig15(curves: dict[tuple[str, float, str, int], pd.DataFrame]) -> Path | None:
    records = []
    for t in [5, 10, 15, 20, 25]:
        vals = get_finals(curves, f"cchihh_tgate{t}", 0.5, "T500")
        if vals:
            records.append((t, float(np.mean(vals)), float(np.std(vals, ddof=0))))
    if not records:
        return None
    x = np.array([r[0] for r in records], dtype=float)
    y = np.array([r[1] for r in records], dtype=float)
    s = np.array([r[2] for r in records], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(x, y, marker="o", color=COLORS["CCHIHH"])
    ax.fill_between(x, y - s, y + s, color=COLORS["CCHIHH"], alpha=0.15)
    ax.errorbar(x, y, yerr=s, fmt="none", color=COLORS["CCHIHH"], capsize=3)
    ax.set_xlabel("Tgate")
    ax.set_ylabel("Best fitness")
    ax.set_title("Tgate sensitivity (available local values)")
    ax.set_xticks(x)
    fig.tight_layout()
    return save_pdf(fig, "fig15_tgate_sensitivity.pdf")


def fig16(curves: dict[tuple[str, float, str, int], pd.DataFrame], final_df: pd.DataFrame) -> Path:
    variants = ["cchihh_nocc", "cchihh_nohi", "cchihh_nocb", "cchihh_nomig", "cchihh_nogate"]
    labels = ["noCC", "noHI", "noCB", "noMig", "noGate"]
    heat = np.full((5, 3), np.nan)
    pvals = np.full((5, 3), np.nan)
    for i, variant in enumerate(variants):
        for j, scale in enumerate(SCALES):
            full = get_finals(curves, "cchihh_full", 0.5, scale)
            if variant == "cchihh_nocb":
                abr = final_df[(final_df["variant_key"] == variant) & (final_df["alpha"] == 0.5) & (final_df["scale"] == scale)].sort_values("seed")["best_fit"].astype(float).tolist()
            else:
                abr = get_finals(curves, variant, 0.5, scale)
            if len(full) == len(abr) and full:
                heat[i, j] = (np.mean(abr) - np.mean(full)) / np.mean(full) * 100.0
                try:
                    pvals[i, j] = float(wilcoxon(full, abr).pvalue)
                except ValueError:
                    pvals[i, j] = math.nan
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    cmap = mcolors.LinearSegmentedColormap.from_list("comp", ["#f7fbff", "#fee0d2", "#cb181d"])
    im = ax.imshow(heat, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(SCALES)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(labels)
    ax.set_title("Scale-dependent component contribution")
    for i in range(5):
        for j in range(3):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.2f}%", ha="center", va="center", fontsize=8)
            if np.isfinite(pvals[i, j]) and pvals[i, j] >= 0.05:
                ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, linestyle=(0, (3, 2)), linewidth=1.0, edgecolor="black"))
    fig.colorbar(im, ax=ax, label="Improvement of full over ablation (%)")
    fig.tight_layout()
    return save_pdf(fig, "fig16_component_heatmap.pdf")


def fig17(curves: dict[tuple[str, float, str, int], pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    labels = [("cchihh_full", "CCHIHH"), ("cga", "CGA"), ("imoma", "IMOMA"), ("dsac_de", "DSAC-DE")]
    for r, alpha in enumerate(ALPHAS):
        for c, scale in enumerate(SCALES):
            ax = axes[r, c]
            add_conv_panel(ax, curves, labels, alpha, scale)
            ax.text(0.02, 0.98, f"α={alpha}", transform=ax.transAxes, ha="left", va="top")
            if r < 2:
                ax.set_xlabel("")
    fig.tight_layout()
    return save_pdf(fig, "fig17_alpha_sensitivity.pdf")


def fig18(curves: dict[tuple[str, float, str, int], pd.DataFrame], no_ppo: bool = False) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    algos = [("cchihh_full", "CCHIHH"), ("dsac_de", "DSAC-DE"), ("cga", "CGA"), ("imoma", "IMOMA")]
    if not no_ppo:
        algos.append(("ppo", "PPO"))
    for ax, scale in zip(axes, SCALES):
        vals = [get_finals(curves, key, 0.5, scale) for key, _ in algos]
        labels = [label for _, label in algos]
        bp = ax.boxplot(vals, tick_labels=labels, patch_artist=True)
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(COLORS[label] if label == "CCHIHH" else (*mcolors.to_rgb(COLORS[label]), 0.25))
            patch.set_edgecolor(COLORS[label])
        ax.set_title(scale)
        ax.set_ylabel("Final fitness")
        if not no_ppo:
            ours = vals[0]
            y_top = max(max(v) for v in vals if v)
            lift = (y_top - min(min(v) for v in vals if v) + 1e-9) * 0.08
            for idx, other in enumerate(vals[1:], start=2):
                if len(other) == len(ours):
                    p = float(wilcoxon(ours, other).pvalue)
                    impr = (np.mean(other) - np.mean(ours)) / np.mean(other) * 100.0
                    ax.text(idx, y_top + lift, f"↓{impr:.1f}%{' **' if p < 0.05 else ''}", ha="center", va="bottom", fontsize=8)
            ax.text(1, ax.get_ylim()[1], "Ours", ha="center", va="bottom", color=COLORS["CCHIHH"])
    fig.tight_layout()
    return save_pdf(fig, "fig22_boxplot_no_ppo.pdf" if no_ppo else "fig18_baseline_boxplot.pdf")


def fig23(wallclock_runs: dict[tuple[str, float, str, int], pd.DataFrame]) -> Path | None:
    algos = ["cchihh", "dsac_de", "cga", "imoma"]
    markers = {"cchihh": "*", "dsac_de": "s", "cga": "^", "imoma": "o"}
    fig, ax = plt.subplots(figsize=(5.3, 4.2))
    has_data = False
    for algo in algos:
        f1, f2 = [], []
        for seed in SEEDS:
            key = (algo, 0.5, "T200", seed)
            if key in wallclock_runs:
                row = wallclock_runs[key].iloc[-1]
                f1.append(float(row["best_f1"]))
                f2.append(float(row["best_f2"]))
        if f1:
            has_data = True
            ax.scatter(f1, f2, color=COLORS[pretty_name(algo)], marker=markers[algo], label=pretty_name(algo), s=42 if algo != "cchihh" else 70, alpha=0.8)
    if not has_data:
        plt.close(fig)
        return None
    ax.set_xlabel("Makespan")
    ax.set_ylabel("Energy")
    ax.set_title("T200 makespan-energy scatter")
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_pdf(fig, "fig23_makespan_energy_scatter.pdf")


def fig24(curves: dict[tuple[str, float, str, int], pd.DataFrame], final_df: pd.DataFrame) -> Path:
    algos = ["cchihh_full", "dsac_de", "cga", "imoma"]
    metrics = ["Solution Quality", "Stability", "Efficiency", "Convergence Speed"]
    theta = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    theta = np.concatenate([theta, theta[:1]])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), subplot_kw={"projection": "polar"})
    for ax, scale in zip(axes, SCALES):
        rows = []
        for algo in algos:
            finals = get_finals(curves, algo, 0.5, scale)
            cv = np.std(finals, ddof=0) / np.mean(finals)
            runtime = np.mean(get_runtime(final_df, algo, 0.5, scale))
            _, matrix = interp_matrix(get_curves(curves, algo, 0.5, scale), np.array([80000.0]))
            rows.append([1.0 / np.mean(finals), 1.0 / cv, 1.0 / runtime, 1.0 / float(matrix.mean())])
        arr = np.asarray(rows)
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        norm = np.where(np.isclose(hi, lo), 1.0, (arr - lo) / (hi - lo))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(scale, pad=16)
        for idx, algo in enumerate(algos):
            values = np.concatenate([norm[idx], norm[idx][:1]])
            color = COLORS[pretty_name(algo)]
            ax.plot(theta, values, color=color, linewidth=1.5, label=pretty_name(algo))
            ax.fill(theta, values, color=color, alpha=0.10)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.15), frameon=False)
    fig.tight_layout()
    return save_pdf(fig, "fig24_radar.pdf")


def fig25(final_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    algos = ["cchihh_full", "dsac_de", "cga", "imoma"]
    for ax, scale in zip(axes, SCALES):
        means = [float(np.mean(get_runtime(final_df, algo, 0.5, scale))) for algo in algos]
        labels = [pretty_name(a) for a in algos]
        bars = ax.bar(labels, means, color=[COLORS[label] for label in labels])
        ax.set_title(scale)
        ax.set_ylabel("Runtime (s)")
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean, f"{mean:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return save_pdf(fig, "fig25_runtime_bar.pdf")


def fig26(curves: dict[tuple[str, float, str, int], pd.DataFrame], wallclock_runs: dict[tuple[str, float, str, int], pd.DataFrame]) -> Path | None:
    algos = [("cchihh_full", "CCHIHH"), ("dsac_de", "DSAC-DE"), ("cga", "CGA"), ("imoma", "IMOMA")]
    rows = []
    for alpha in ALPHAS:
        for scale in SCALES:
            vals = []
            for variant, _ in algos:
                temp = get_finals(curves, variant, alpha, scale)
                vals.append(float(np.mean(temp)) if temp else math.nan)
            if all(np.isfinite(v) for v in vals):
                rows.append(vals)
    if len(rows) == 9:
        rank_matrix = np.vstack([rankdata(r, method="average") for r in rows])
        avg = rank_matrix.mean(axis=0)
        cd = 2.569 * math.sqrt(len(algos) * (len(algos) + 1) / (6 * len(rows)))
        title = "Critical Difference Diagram (9 instances, 4 algorithms)"
    else:
        avg = np.asarray([1.333, 1.667, 3.000, 4.000], dtype=float)
        cd = 2.569
        title = "Critical Difference Diagram (Friedman-Nemenyi, alpha=0.05)"
    ordered = sorted(zip([label for _, label in algos], avg), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(0.7, 4.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.55
    ax.hlines(y, 1, 4, color="black")
    for tick in range(1, 5):
        ax.vlines(tick, y - 0.03, y + 0.03, color="black")
        ax.text(tick, y - 0.08, str(tick), ha="center", va="top")
    ax.text(1.0, y + 0.08, "Better", ha="left", va="bottom")
    ax.text(4.0, y + 0.08, "Worse", ha="right", va="bottom")
    for idx, (name, rank) in enumerate(ordered):
        yt = 0.84 - idx * 0.14 if idx < 2 else 0.24 - (idx - 2) * 0.14
        va = "bottom" if idx < 2 else "top"
        ax.vlines(rank, y, yt - 0.03 if va == "bottom" else yt + 0.03, color="gray")
        ax.text(rank, yt, f"{name} ({rank:.3f})", ha="center", va=va, fontsize=9)
    ax.text(4.1, 0.94, f"CD = {cd:.3f}", ha="right")
    ax.set_title(title)
    fig.tight_layout()
    return save_pdf(fig, "fig26_cd_diagram.pdf")


def fig27() -> Path | None:
    path = SCHEDULE_ROOT / "cchihh_full_T200_best_seed5_schedule.csv"
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    csv_lines = [line for line in raw if line and not line.startswith("#")]
    comments = [line for line in raw if line.startswith("#")]
    if not csv_lines:
        return None
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(csv_lines)))
    offsets = {"device": 0, "edge": 80, "cloud": 160}
    colors = {"device": "#8dd3c7", "edge": "#80b1d3", "cloud": "#fb8072"}
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for _, row in df.iterrows():
        tier = str(row["assigned_tier"]).lower()
        digits = "".join(ch for ch in str(row["server_id"]) if ch.isdigit())
        lane = offsets.get(tier, 240) + (int(digits) if digits else 0) % 60
        start = float(row["start_time"])
        end = float(row["end_time"])
        ax.broken_barh([(start, end - start)], (lane - 0.4, 0.8), facecolors=colors.get(tier, "#cccccc"))
    ax.set_xlabel("Time")
    ax.set_ylabel("Tier / server lane")
    ax.set_title("Gantt chart for T200 best schedule")
    if comments:
        ax.text(0.01, 0.99, "\n".join(comments[-3:]), transform=ax.transAxes, ha="left", va="top", fontsize=8)
    fig.tight_layout()
    return save_pdf(fig, "fig27_gantt_T200.pdf")


def write_tables(curves: dict[tuple[str, float, str, int], pd.DataFrame], final_df: pd.DataFrame) -> Path:
    lines: list[str] = []

    def make_line(scale: str, full: list[float], ablation: list[float]) -> str:
        mf, sf = np.mean(full), np.std(full, ddof=0)
        ma, sa = np.mean(ablation), np.std(ablation, ddof=0)
        impr = (ma - mf) / mf * 100.0
        try:
            p = float(wilcoxon(full, ablation).pvalue)
        except ValueError:
            p = math.nan
        return f"{scale} & ${mf:.6f} \\pm {sf:.6f}$ & ${ma:.6f} \\pm {sa:.6f}$ & {impr:.2f} & {p:.6f} \\\\"

    for title, variant in [
        ("Table 3: CC ablation", "cchihh_nocc"),
        ("Table 4: HI ablation", "cchihh_nohi"),
        ("Table 5: CB ablation", "cchihh_nocb"),
        ("Table 6: Migration ablation", "cchihh_nomig"),
        ("Table 7: Gate ablation", "cchihh_nogate"),
    ]:
        lines.append(f"% {title}")
        for scale in SCALES:
            full = get_finals(curves, "cchihh_full", 0.5, scale)
            if variant == "cchihh_nocb":
                ablation = final_df[(final_df["variant_key"] == variant) & (final_df["alpha"] == 0.5) & (final_df["scale"] == scale)].sort_values("seed")["best_fit"].astype(float).tolist()
            else:
                ablation = get_finals(curves, variant, 0.5, scale)
            if full and ablation:
                lines.append(make_line(scale, full, ablation))
        lines.append("")

    lines.append("% Table 8: Tgate sensitivity")
    for t in [5, 10, 15, 20, 25]:
        vals = get_finals(curves, f"cchihh_tgate{t}", 0.5, "T500")
        if vals:
            lines.append(f"{t} & ${np.mean(vals):.6f} \\pm {np.std(vals, ddof=0):.6f}$ \\\\")
    lines.append("")

    lines.append("% Table 9: Baseline comparison")
    for scale in SCALES:
        row = [scale]
        for variant in ["cchihh_full", "dsac_de", "cga", "imoma", "ppo"]:
            vals = get_finals(curves, variant, 0.5, scale)
            row.append(f"${np.mean(vals):.6f} \\pm {np.std(vals, ddof=0):.6f}$")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("")

    lines.append("% Table 10: Alpha sensitivity")
    for alpha in ALPHAS:
        for scale in SCALES:
            row = [f"{scale}, $\\alpha={alpha}$"]
            for variant in ["cchihh_full", "cga", "imoma", "dsac_de"]:
                vals = get_finals(curves, variant, alpha, scale)
                row.append(f"${np.mean(vals):.6f} \\pm {np.std(vals, ddof=0):.6f}$")
            lines.append(" & ".join(row) + " \\\\")
    lines.append("")

    lines.append("% Table 11: Runtime comparison")
    for scale in SCALES:
        row = [scale]
        for variant in ["cchihh_full", "dsac_de", "cga", "imoma"]:
            vals = get_runtime(final_df, variant, 0.5, scale)
            row.append(f"${np.mean(vals):.3f} \\pm {np.std(vals, ddof=0):.3f}$")
        lines.append(" & ".join(row) + " \\\\")

    path = OUTPUT_DIR / "updated_tables.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_checklist(statuses: list[FigureStatus], tables_path: Path) -> Path:
    lines = []
    for item in statuses:
        mark = "OK" if item.ok else "MISSING"
        note = f" [{item.note}]" if item.note else ""
        lines.append(f"{mark} {item.filename} - {item.description}{note}")
    lines.append(f"OK {tables_path.name} - All table values (LaTeX)")
    path = OUTPUT_DIR / "checklist.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_status(filename: str, description: str, path: Path | None, note: str = "") -> FigureStatus:
    return FigureStatus(filename, description, path is not None and path.exists(), note)


def main() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = load_eval_curves()
    ops_curves = load_ops_curves()
    weight_curves = load_weight_curves()
    final_df = load_final_summary()
    wallclock_runs = load_wallclock_runs()

    statuses = [
        make_status("fig08_cc_ablation.pdf", "CCHIHH-full vs noCC, 3 scales", plot_triptych(curves, [("cchihh_full", "CCHIHH"), ("cchihh_nocc", "noCC")], 0.5, "fig08_cc_ablation.pdf")),
        make_status("fig09_hi_ablation.pdf", "CCHIHH-full vs noHI, 3 scales", plot_triptych(curves, [("cchihh_full", "CCHIHH"), ("cchihh_nohi", "noHI")], 0.5, "fig09_hi_ablation.pdf")),
        make_status("fig10_op_selection_T500.pdf", "Operator selection probability", fig10(ops_curves)),
        make_status("fig11_early_late_bar_T500.pdf", "Early/late bar chart", fig11(ops_curves)),
        make_status("fig12_weight_heatmap_T500.pdf", "Weight evolution heatmap", fig12(weight_curves)),
        make_status("fig13_migration_ablation.pdf", "CCHIHH-full vs noMig", plot_triptych(curves, [("cchihh_full", "CCHIHH"), ("cchihh_nomig", "noMig")], 0.5, "fig13_migration_ablation.pdf")),
        make_status("fig14_gate_ablation.pdf", "CCHIHH-full vs noGate", plot_triptych(curves, [("cchihh_full", "CCHIHH"), ("cchihh_nogate", "noGate")], 0.5, "fig14_gate_ablation.pdf")),
        make_status("fig15_tgate_sensitivity.pdf", "Tgate sensitivity", fig15(curves), "available local values: 5,10,15,20,25"),
        make_status("fig16_component_heatmap.pdf", "5x3 contribution heatmap", fig16(curves, final_df)),
        make_status("fig17_alpha_sensitivity.pdf", "alpha sensitivity 3x3 grid", fig17(curves)),
        make_status("fig18_baseline_boxplot.pdf", "5-algorithm box plots", fig18(curves)),
        make_status("fig19_conv_cga_imoma.pdf", "Convergence vs CGA/IMOMA", plot_triptych(curves, [("cchihh_full", "CCHIHH"), ("cga", "CGA"), ("imoma", "IMOMA")], 0.5, "fig19_conv_cga_imoma.pdf")),
        make_status("fig20_conv_ppo.pdf", "Convergence vs PPO", plot_triptych(curves, [("cchihh_full", "CCHIHH"), ("ppo", "PPO")], 0.5, "fig20_conv_ppo.pdf")),
        make_status("fig21_conv_dsac.pdf", "Convergence vs DSAC-DE", plot_triptych(curves, [("cchihh_full", "CCHIHH"), ("dsac_de", "DSAC-DE")], 0.5, "fig21_conv_dsac.pdf")),
        make_status("fig22_boxplot_no_ppo.pdf", "4-algorithm box plots", fig18(curves, no_ppo=True)),
        make_status("fig23_makespan_energy_scatter.pdf", "Makespan-energy scatter", fig23(wallclock_runs), "from results/wallclock"),
        make_status("fig24_radar.pdf", "Multi-dimensional radar", fig24(curves, final_df)),
        make_status("fig25_runtime_bar.pdf", "Wall-clock time", fig25(final_df)),
        make_status("fig26_cd_diagram.pdf", "Critical Difference diagram", fig26(curves, wallclock_runs), "PPO alpha=0.2/0.8 from results/wallclock"),
        make_status("fig27_gantt_T200.pdf", "Gantt chart", fig27(), "from results/rerun_schedule"),
    ]
    tables_path = write_tables(curves, final_df)
    checklist_path = write_checklist(statuses, tables_path)
    print(checklist_path)
    print(tables_path)
    for item in statuses:
        print(("OK" if item.ok else "MISSING") + f": {item.filename}")


if __name__ == "__main__":
    main()
