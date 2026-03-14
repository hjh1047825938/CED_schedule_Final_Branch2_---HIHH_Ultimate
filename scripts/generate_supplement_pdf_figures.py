from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
SUPPLEMENT_SHARED_DIR = ROOT / "results" / "supplement" / "shared_v2"
SUPPLEMENT_DSAC_DIR = ROOT / "results" / "supplement" / "dsac"
SUPPLEMENT_ANALYSIS_DIR = ROOT / "results" / "supplement" / "analysis"
FULL_REWARD_ROOT = ROOT / "outputs" / "results" / "operator_ablation"
FULL_LOG_ROOT = ROOT / "outputs" / "results" / "cchihh_full_canonical_multiscale"
BASELINE_ROOT = ROOT / "outputs" / "results" / "cchihh_ablation_suite" / "baseline"

SCALES = [100, 200, 500]
GEN_CUTOFF = 1000
BLOCK_IDS = {"offload": 0, "seq": 1, "dev": 2}
BLOCK_SHORT = {"offload": "off", "seq": "seq", "dev": "dev"}
OP_NAMES = {
    "offload": ["GA", "DE", "BitFlip", "Resample"],
    "seq": ["GA", "Swap", "VNS", "Resample"],
    "dev": ["DE", "GDE", "Levy", "Resample"],
}
MAIN_COLORS = {
    "full-offload": "#1f77b4",
    "full-seq": "#4c9ad4",
    "full-dev": "#86bce8",
    "sharedBandit": "#d62728",
    "CCHIHH-full": "#1f77b4",
    "CCHIHH-shared": "#d62728",
    "DSAC-DE": "#2ca02c",
    "CGA": "#7f7f7f",
    "IMOMA": "#98a2b3",
}

GEN_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([-+0-9.eE]+)")
FINAL_RE = re.compile(r"The best (?:scalar )?solution\s*=\s*([-+0-9.eE]+)")
TIME_RE = re.compile(r"Time\s*=\s*([-+0-9.eE]+)\s*s")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw:
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                pass
    for enc in ("utf-8", "gb18030", "cp936", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_log(path: Path) -> dict:
    text = read_text_auto(path)
    curve = {}
    for line in text.splitlines():
        m = GEN_RE.match(line.strip())
        if m:
            curve[int(m.group(1))] = float(m.group(2))

    final_match = FINAL_RE.findall(text)
    runtime_match = TIME_RE.findall(text)
    final_best = float(final_match[-1]) if final_match else (curve[max(curve)] if curve else math.nan)
    runtime_s = float(runtime_match[-1]) if runtime_match else math.nan
    return {"curve": curve, "final_best": final_best, "runtime_s": runtime_s}


def load_full_reward_variance(scale: int, seed: int) -> pd.DataFrame:
    path = FULL_REWARD_ROOT / f"T{scale}" / "full" / f"op_rewards_T{scale}_run{seed - 1}.csv"
    df = pd.read_csv(path)
    agg = (
        df.groupby(["gen", "block_id"])["reward"]
        .agg(["mean", "var", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "gen": "generation",
                "mean": "reward_mean",
                "var": "reward_variance",
                "min": "reward_min",
                "max": "reward_max",
            }
        )
    )
    agg["reward_variance"] = agg["reward_variance"].fillna(0.0)
    return agg


def load_shared_reward_variance(scale: int, seed: int) -> pd.DataFrame:
    path = SUPPLEMENT_SHARED_DIR / f"reward_var_shared_T{scale}_seed{seed}.csv"
    return pd.read_csv(path)


def align_seed_series(series_list: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = sorted(set.intersection(*[set(g.tolist()) for g, _ in series_list]))
    gens = np.array(common, dtype=int)
    aligned = []
    for g, v in series_list:
        mapping = dict(zip(g.tolist(), v.tolist()))
        aligned.append([mapping[x] for x in common])
    arr = np.asarray(aligned, dtype=float)
    return gens, arr.mean(axis=0), arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1], dtype=float)


def build_reward_variance_curves(scale: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    curves = {}
    for block_name, block_id in BLOCK_IDS.items():
        seed_series = []
        for seed in range(1, 11):
            df = load_full_reward_variance(scale, seed)
            sub = df[df["block_id"] == block_id].sort_values("generation")
            seed_series.append((sub["generation"].to_numpy(dtype=int), sub["reward_variance"].to_numpy(dtype=float)))
        curves[f"full-{block_name}"] = align_seed_series(seed_series)

    shared_series = []
    for seed in range(1, 11):
        df = load_shared_reward_variance(scale, seed)
        shared = (
            df.groupby("generation", as_index=False)["reward_variance"]
            .mean()
            .sort_values("generation")
        )
        shared_series.append(
            (shared["generation"].to_numpy(dtype=int), shared["reward_variance"].to_numpy(dtype=float))
        )
    curves["sharedBandit"] = align_seed_series(shared_series)
    return curves


def save_reward_variance_figures(report_lines: list[str]) -> None:
    labels = ["full-offload", "full-seq", "full-dev", "sharedBandit"]
    for scale in SCALES:
        curves = build_reward_variance_curves(scale)
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            gridspec_kw={"height_ratios": [1, 1]},
        )
        for label in labels:
            gens, mean, std = curves[label]
            color = MAIN_COLORS[label]
            linestyle = "--" if label == "sharedBandit" else "-"
            linewidth = 2.5 if label == "sharedBandit" else 1.8
            mean_safe = np.maximum(mean, 1e-15)
            lower = np.maximum(mean - std, 1e-15)
            upper = np.maximum(mean + std, 1e-15)
            ax_top.plot(gens, mean_safe, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            ax_top.fill_between(gens, lower, upper, color=color, alpha=0.10)
        ax_top.set_yscale("log")
        ax_top.set_xlim(0, 10000)
        ax_top.set_ylabel("Reward Variance (log scale)")
        ax_top.set_title(f"Reward Signal Variance - T{scale}")
        ax_top.legend(loc="upper right")

        for label in labels:
            gens, mean, std = curves[label]
            mask = gens >= 1000
            if not np.any(mask):
                continue
            color = MAIN_COLORS[label]
            linestyle = "--" if label == "sharedBandit" else "-"
            linewidth = 2.5 if label == "sharedBandit" else 1.8
            ax_bot.plot(gens[mask], mean[mask], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            ax_bot.fill_between(
                gens[mask],
                np.maximum(mean[mask] - std[mask], 0.0),
                mean[mask] + std[mask],
                color=color,
                alpha=0.10,
            )
        ax_bot.set_xlim(1000, 10000)
        ax_bot.set_xlabel("Generation")
        ax_bot.set_ylabel("Reward Variance (linear scale)")
        ax_bot.set_title("Late-stage detail (Generation 1000-10000)")
        ax_bot.legend(loc="upper right")

        out_path = FIG_DIR / f"reward_variance_T{scale}.pdf"
        plt.tight_layout()
        fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        report_lines.append(f"Saved: {out_path}")

    scales = [100, 200, 500]
    plot_labels = ["full-offload", "full-seq", "full-dev", "sharedBandit"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scales))
    width = 0.18
    offsets = (np.arange(len(plot_labels)) - (len(plot_labels) - 1) / 2.0) * width
    for idx, label in enumerate(plot_labels):
        means = []
        stds = []
        for scale in scales:
            seed_avgs = []
            for seed in range(1, 11):
                if label == "sharedBandit":
                    df = load_shared_reward_variance(scale, seed)
                    per_gen = (
                        df.groupby("generation", as_index=False)["reward_variance"]
                        .mean()
                    )
                    per_gen = per_gen[per_gen["generation"] >= GEN_CUTOFF]
                    seed_avgs.append(float(per_gen["reward_variance"].mean()))
                else:
                    block_id = BLOCK_IDS[label.split("-", 1)[1]]
                    df = load_full_reward_variance(scale, seed)
                    block_df = df[(df["block_id"] == block_id) & (df["generation"] >= GEN_CUTOFF)]
                    seed_avgs.append(float(block_df["reward_variance"].mean()))
            means.append(float(np.mean(seed_avgs)))
            stds.append(float(np.std(seed_avgs, ddof=1)))
        hatch = "//" if label == "sharedBandit" else None
        ax.bar(
            x + offsets[idx],
            means,
            width,
            yerr=stds,
            label=label,
            color=MAIN_COLORS[label],
            alpha=0.75,
            hatch=hatch,
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{s}" for s in scales])
    ax.set_ylabel(f"Average Reward Variance (Generation {GEN_CUTOFF}-10000)")
    ax.set_title("Late-stage Reward Signal Variance: Per-block vs Shared Bandit")
    ax.legend()
    ax.text(
        0.98,
        0.95,
        f"Computed over generations {GEN_CUTOFF}-10000\n(initial transient excluded)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
    )
    out_path = FIG_DIR / "reward_variance_barplot.pdf"
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    report_lines.append(f"Saved: {out_path}")


def load_weight_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path).sort_values(["gen", "op_id"])
    pivot = df.pivot(index="op_id", columns="gen", values="w6").sort_index().sort_index(axis=1)
    return pivot.columns.to_numpy(dtype=int), pivot.to_numpy(dtype=float)


def aggregate_weight_matrix(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    gens_ref = None
    for path in paths:
        gens, matrix = load_weight_matrix(path)
        if gens_ref is None:
            gens_ref = gens
        else:
            common = np.intersect1d(gens_ref, gens)
            if common.size == 0:
                raise ValueError(f"No common generations in {path}")
            keep_ref = np.isin(gens_ref, common)
            keep_cur = np.isin(gens, common)
            matrices = [m[:, keep_ref] for m in matrices]
            matrix = matrix[:, keep_cur]
            gens_ref = common
        matrices.append(matrix)
    stack = np.stack(matrices, axis=0)
    return gens_ref, stack.mean(axis=0)


def save_heatmap_comparison(report_lines: list[str]) -> None:
    scale = 500
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    full_mats = []
    shared_mats = []
    gens_store = {}
    for block_name, short in BLOCK_SHORT.items():
        full_paths = [
            FULL_REWARD_ROOT / f"T{scale}" / "full" / f"op_weights_{'offload' if short == 'off' else short}_T{scale}_run{seed - 1}.csv"
            for seed in range(1, 11)
        ]
        shared_paths = [
            SUPPLEMENT_SHARED_DIR / f"weights_shared_{short}_T{scale}_seed{seed}.csv"
            for seed in range(1, 11)
        ]
        gens_full, mat_full = aggregate_weight_matrix(full_paths)
        gens_shared, mat_shared = aggregate_weight_matrix(shared_paths)
        full_mats.append(mat_full)
        shared_mats.append(mat_shared)
        gens_store[f"full-{block_name}"] = gens_full
        gens_store[f"shared-{block_name}"] = gens_shared

    max_abs = max(
        float(np.max(np.abs(mat)))
        for mat in full_mats + shared_mats
    )
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    for col, (block_name, short) in enumerate(BLOCK_SHORT.items()):
        full_paths = [
            FULL_REWARD_ROOT / f"T{scale}" / "full" / f"op_weights_{'offload' if short == 'off' else short}_T{scale}_run{seed - 1}.csv"
            for seed in range(1, 11)
        ]
        shared_paths = [
            SUPPLEMENT_SHARED_DIR / f"weights_shared_{short}_T{scale}_seed{seed}.csv"
            for seed in range(1, 11)
        ]
        gens_full, mat_full = aggregate_weight_matrix(full_paths)
        gens_shared, mat_shared = aggregate_weight_matrix(shared_paths)

        ax0 = axes[0, col]
        im0 = ax0.imshow(
            mat_full,
            aspect="auto",
            cmap="RdYlBu_r",
            norm=norm,
            extent=[gens_full[0], gens_full[-1], len(OP_NAMES[block_name]) - 0.5, -0.5],
            interpolation="bilinear",
        )
        ax0.set_yticks(range(len(OP_NAMES[block_name])))
        ax0.set_yticklabels(OP_NAMES[block_name])
        ax0.set_title(f"Per-block - {block_name.capitalize()}")
        if col == 0:
            ax0.set_ylabel("CCHIHH-full")
        fig.colorbar(im0, ax=ax0, shrink=0.8)

        ax1 = axes[1, col]
        im1 = ax1.imshow(
            mat_shared,
            aspect="auto",
            cmap="RdYlBu_r",
            norm=norm,
            extent=[gens_shared[0], gens_shared[-1], len(OP_NAMES[block_name]) - 0.5, -0.5],
            interpolation="bilinear",
        )
        ax1.set_yticks(range(len(OP_NAMES[block_name])))
        ax1.set_yticklabels(OP_NAMES[block_name])
        ax1.set_xlabel("Generation")
        ax1.set_title(f"Shared - {block_name.capitalize()}")
        if col == 0:
            ax1.set_ylabel("CCHIHH-shared")
        fig.colorbar(im1, ax=ax1, shrink=0.8)

    plt.suptitle("Bandit Weight Evolution: Per-block vs Shared (T500, mean over 10 seeds)", fontsize=14, y=1.02)
    out_path = FIG_DIR / "heatmap_comparison_T500.pdf"
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    report_lines.append(f"Saved: {out_path}")


def load_full_shared_finals() -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    full_vals = {}
    shared_vals = {}
    for scale in SCALES:
        full_vals[scale] = []
        shared_vals[scale] = []
        for seed in range(1, 11):
            full_log = parse_log(FULL_LOG_ROOT / f"T{scale}" / f"CCHIHH_full_seed{seed}.txt")
            shared_log = parse_log(SUPPLEMENT_SHARED_DIR / f"cchihh_shared_T{scale}_seed{seed}.log")
            full_vals[scale].append(full_log["final_best"])
            shared_vals[scale].append(shared_log["final_best"])
    return full_vals, shared_vals


def save_shared_bandit_boxplot(report_lines: list[str]) -> None:
    full_vals, shared_vals = load_full_shared_finals()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for idx, scale in enumerate(SCALES):
        ax = axes[idx]
        bp = ax.boxplot(
            [full_vals[scale], shared_vals[scale]],
            tick_labels=["Per-Block\nBandit", "Shared\nBandit"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor(MAIN_COLORS["CCHIHH-full"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(MAIN_COLORS["CCHIHH-shared"])
        bp["boxes"][1].set_alpha(0.6)
        ax.set_title(f"T{scale}")
        if idx == 0:
            ax.set_ylabel("Best Fitness")
    plt.suptitle("Solution Quality: Per-block vs Shared Bandit", fontsize=14)
    out_path = FIG_DIR / "shared_bandit_boxplot.pdf"
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    report_lines.append(f"Saved: {out_path}")


def align_curves(curve_dicts: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(set.intersection(*[set(curve.keys()) for curve in curve_dicts]))
    gens = np.array(common, dtype=int)
    arr = np.array([[curve[g] for g in common] for curve in curve_dicts], dtype=float)
    return gens, arr


def save_shared_convergence(report_lines: list[str]) -> None:
    scale = 500
    full_curves = []
    shared_curves = []
    for seed in range(1, 11):
        full_curves.append(parse_log(FULL_LOG_ROOT / f"T{scale}" / f"CCHIHH_full_seed{seed}.txt")["curve"])
        shared_curves.append(parse_log(SUPPLEMENT_SHARED_DIR / f"cchihh_shared_T{scale}_seed{seed}.log")["curve"])
    gens_full, arr_full = align_curves(full_curves)
    gens_shared, arr_shared = align_curves(shared_curves)
    common = np.intersect1d(gens_full, gens_shared)
    full_map = {g: arr_full[:, idx] for idx, g in enumerate(gens_full.tolist())}
    shared_map = {g: arr_shared[:, idx] for idx, g in enumerate(gens_shared.tolist())}
    full_arr = np.array([full_map[g] for g in common], dtype=float).T
    shared_arr = np.array([shared_map[g] for g in common], dtype=float).T

    fig, ax = plt.subplots(figsize=(10, 6))
    full_mean = full_arr.mean(axis=0)
    full_ci = 1.96 * full_arr.std(axis=0, ddof=1) / math.sqrt(full_arr.shape[0])
    shared_mean = shared_arr.mean(axis=0)
    shared_ci = 1.96 * shared_arr.std(axis=0, ddof=1) / math.sqrt(shared_arr.shape[0])
    ax.plot(common, full_mean, color=MAIN_COLORS["CCHIHH-full"], label="CCHIHH-full (per-block)", linewidth=2)
    ax.fill_between(common, full_mean - full_ci, full_mean + full_ci, color=MAIN_COLORS["CCHIHH-full"], alpha=0.15)
    ax.plot(common, shared_mean, color=MAIN_COLORS["CCHIHH-shared"], linestyle="--", label="CCHIHH-shared (shared bandit)", linewidth=2)
    ax.fill_between(common, shared_mean - shared_ci, shared_mean + shared_ci, color=MAIN_COLORS["CCHIHH-shared"], alpha=0.15)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Convergence Comparison on T500: Per-block vs Shared Bandit")
    ax.legend()
    out_path = FIG_DIR / "convergence_shared_vs_full_T500.pdf"
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    report_lines.append(f"Saved: {out_path}")


def load_baseline_algorithm_logs(scale: int, algo_stem: str) -> list[dict]:
    paths = sorted((BASELINE_ROOT / f"T{scale}").glob(f"{algo_stem}_seed*.txt"))
    return [parse_log(path) for path in paths if re.match(rf"{re.escape(algo_stem)}_seed([1-9]|10)\.txt$", path.name)]


def load_new_dsac_logs(scale: int) -> list[dict]:
    paths = sorted(SUPPLEMENT_DSAC_DIR.glob(f"dsac_de_T{scale}_seed*.log"))
    return [parse_log(path) for path in paths]


def save_baseline_boxplot(report_lines: list[str]) -> None:
    algorithms = [
        ("CCHIHH-full", "CCHIHH_Full", MAIN_COLORS["CCHIHH-full"]),
        ("DSAC-DE", None, MAIN_COLORS["DSAC-DE"]),
        ("CGA", "CGA", MAIN_COLORS["CGA"]),
        ("IMOMA", "IMOMA", MAIN_COLORS["IMOMA"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, scale in enumerate(SCALES):
        ax = axes[idx]
        data = []
        labels = []
        colors = []
        for label, stem, color in algorithms:
            if label == "DSAC-DE":
                runs = load_new_dsac_logs(scale)
            else:
                runs = load_baseline_algorithm_logs(scale, stem)
            if not runs:
                continue
            data.append([run["final_best"] for run in runs])
            labels.append(label.replace("-", "-\n") if label == "DSAC-DE" else label)
            colors.append(color)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(f"T{scale}")
        if idx == 0:
            ax.set_ylabel("Best Fitness")
    out_path = FIG_DIR / "boxplot_fitness.pdf"
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    report_lines.append(f"Saved: {out_path}")


def save_dsac_convergence(report_lines: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, scale in enumerate(SCALES):
        ax = axes[idx]
        full_runs = [parse_log(FULL_LOG_ROOT / f"T{scale}" / f"CCHIHH_full_seed{seed}.txt")["curve"] for seed in range(1, 11)]
        dsac_runs = [parse_log(SUPPLEMENT_DSAC_DIR / f"dsac_de_T{scale}_seed{seed}.log")["curve"] for seed in range(1, 11)]
        gens_full, arr_full = align_curves(full_runs)
        gens_dsac, arr_dsac = align_curves(dsac_runs)
        common = np.intersect1d(gens_full, gens_dsac)
        full_map = {g: arr_full[:, idx2] for idx2, g in enumerate(gens_full.tolist())}
        dsac_map = {g: arr_dsac[:, idx2] for idx2, g in enumerate(gens_dsac.tolist())}
        full_arr = np.array([full_map[g] for g in common], dtype=float).T
        dsac_arr = np.array([dsac_map[g] for g in common], dtype=float).T
        full_mean = full_arr.mean(axis=0)
        full_ci = 1.96 * full_arr.std(axis=0, ddof=1) / math.sqrt(full_arr.shape[0])
        dsac_mean = dsac_arr.mean(axis=0)
        dsac_ci = 1.96 * dsac_arr.std(axis=0, ddof=1) / math.sqrt(dsac_arr.shape[0])
        ax.plot(common, full_mean, color=MAIN_COLORS["CCHIHH-full"], label="CCHIHH-full", linewidth=2)
        ax.fill_between(common, full_mean - full_ci, full_mean + full_ci, color=MAIN_COLORS["CCHIHH-full"], alpha=0.15)
        ax.plot(common, dsac_mean, color=MAIN_COLORS["DSAC-DE"], label="DSAC-DE", linewidth=2)
        ax.fill_between(common, dsac_mean - dsac_ci, dsac_mean + dsac_ci, color=MAIN_COLORS["DSAC-DE"], alpha=0.15)
        ax.set_title(f"T{scale}")
        ax.set_xlabel("Generation")
        if idx == 0:
            ax.set_ylabel("Best Fitness")
        ax.legend()
    out_path = FIG_DIR / "DSAC.pdf"
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    report_lines.append(f"Saved: {out_path}")


def normalize_lower_better(metric: dict[str, float]) -> dict[str, float]:
    values = pd.Series(metric, dtype=float)
    best = values.min()
    worst = values.max()
    if math.isclose(best, worst):
        return {k: 1.0 for k in values.index}
    scaled = (worst - values) / (worst - best)
    return scaled.clip(0.0, 1.0).to_dict()


def first_reach_generation(curve: dict[int, float], threshold: float) -> int:
    for gen in sorted(curve):
        if curve[gen] <= threshold:
            return gen
    return max(curve) if curve else 10000


def save_radar_figures(report_lines: list[str]) -> None:
    algo_specs = [
        ("CCHIHH-full", "CCHIHH_Full", MAIN_COLORS["CCHIHH-full"]),
        ("DSAC-DE", None, MAIN_COLORS["DSAC-DE"]),
        ("CGA", "CGA", MAIN_COLORS["CGA"]),
        ("IMOMA", "IMOMA", MAIN_COLORS["IMOMA"]),
    ]
    labels = ["Solution Quality", "Stability", "Efficiency", "Convergence Speed"]
    for scale in SCALES:
        stats = {}
        final_means = {}
        for display, stem, color in algo_specs:
            if display == "DSAC-DE":
                runs = load_new_dsac_logs(scale)
            else:
                runs = load_baseline_algorithm_logs(scale, stem)
            if not runs:
                continue
            finals = np.array([run["final_best"] for run in runs], dtype=float)
            runtimes = np.array([run["runtime_s"] for run in runs if math.isfinite(run["runtime_s"])], dtype=float)
            final_means[display] = float(finals.mean())
            stats[display] = {
                "fitness": float(finals.mean()),
                "cv": float(finals.std(ddof=1) / max(finals.mean(), 1e-12)),
                "runtime": float(runtimes.mean()) if runtimes.size else math.nan,
                "color": color,
                "curves": [run["curve"] for run in runs],
            }
        if not stats:
            continue

        threshold = min(final_means.values()) * 1.05
        conv_metric = {}
        for display, payload in stats.items():
            conv_vals = [first_reach_generation(curve, threshold) for curve in payload["curves"]]
            conv_metric[display] = float(np.mean(conv_vals))

        normalized = {
            "Solution Quality": normalize_lower_better({k: v["fitness"] for k, v in stats.items()}),
            "Stability": normalize_lower_better({k: v["cv"] for k, v in stats.items()}),
            "Efficiency": normalize_lower_better({k: v["runtime"] for k, v in stats.items()}),
            "Convergence Speed": normalize_lower_better(conv_metric),
        }

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw={"projection": "polar"})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        for display, _, color in algo_specs:
            if display not in stats:
                continue
            values = [normalized[label][display] for label in labels]
            values += values[:1]
            ax.plot(angles, values, color=color, linewidth=2, label=display)
            ax.fill(angles, values, color=color, alpha=0.12)
        ax.set_title(f"T{scale}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))
        out_path = FIG_DIR / f"fig_radar_T{scale}.pdf"
        plt.tight_layout()
        fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        report_lines.append(f"Saved: {out_path}")


def maybe_save_cd_diagram(report_lines: list[str]) -> None:
    alpha_root = ROOT / "results" / "alpha_sensitivity_dsac_new"
    if not alpha_root.exists():
        report_lines.append("Skipped: results/figures/fig_cd_diagram_N9.pdf (missing DSAC-DE alpha=0.2/0.8 rerun data)")
        return
    report_lines.append("Skipped: results/figures/fig_cd_diagram_N9.pdf (script does not yet support new alpha-sensitivity DSAC aggregation)")


def write_report(report_lines: list[str]) -> None:
    report_path = FIG_DIR / "figure_generation_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved: {report_path}")


def main() -> None:
    setup_style()
    ensure_dir(FIG_DIR)
    report_lines: list[str] = []
    save_reward_variance_figures(report_lines)
    save_heatmap_comparison(report_lines)
    save_shared_bandit_boxplot(report_lines)
    save_shared_convergence(report_lines)
    save_baseline_boxplot(report_lines)
    save_dsac_convergence(report_lines)
    save_radar_figures(report_lines)
    maybe_save_cd_diagram(report_lines)
    for line in report_lines:
        print(line)
    write_report(report_lines)


if __name__ == "__main__":
    main()
