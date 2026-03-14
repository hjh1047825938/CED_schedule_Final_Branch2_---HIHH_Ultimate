from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

COLORS = {
    "CCHIHH": "#D62728",
    "CCHIHH-full": "#D62728",
    "CGA": "#2CA02C",
    "IMOMA": "#1F77B4",
    "DSAC-DE": "#FF7F0E",
    "PPO": "#9467BD",
    "noCC": "#1F77B4",
    "noHI": "#1F77B4",
    "noMig": "#1F77B4",
    "noGate": "#1F77B4",
}

SCALES = ["T100", "T200", "T500"]
ALPHAS = [0.2, 0.5, 0.8]
Z95 = 1.96


def setup_style():
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 300,
        }
    )
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)


def save_pdf(fig: plt.Figure, filename: str):
    fig.savefig(FIGURES / filename, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_pdf_png(fig: plt.Figure, filename_stem: str):
    fig.savefig(FIGURES / f"{filename_stem}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES / f"{filename_stem}.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_data():
    conv = pd.read_csv(RESULTS / "convergence_summary.csv")
    final = pd.read_csv(RESULTS / "final_summary.csv")
    return conv, final


def ci_from_series(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return Z95 * values.std(ddof=0) / math.sqrt(len(values))


def aggregate_curve(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("gen")["best_fit"]
        .agg(mean="mean", var=lambda s: float(np.var(s, ddof=0)), n="count")
        .reset_index()
    )
    agg["ci"] = Z95 * np.sqrt(agg["var"] / agg["n"].clip(lower=1))
    return agg


def get_runs(final: pd.DataFrame, variant: str, scale: str, alpha: float = 0.5, source: str | None = None):
    out = final[(final["variant"] == variant) & (final["scale"] == scale) & (final["alpha"] == alpha)]
    if source is not None:
        out = out[out["source"] == source]
    return out.copy()


def get_curves(conv: pd.DataFrame, variant: str, scale: str, alpha: float = 0.5, source: str | None = None):
    out = conv[(conv["variant"] == variant) & (conv["scale"] == scale) & (conv["alpha"] == alpha)]
    if source is not None:
        out = out[out["source"] == source]
    return out.copy()


def pvalue_paired(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def latex_table(path: Path, header: list[str], rows: list[list[str]], note: str | None = None):
    lines = [
        "\\begin{tabular}{" + "l" * len(header) + "}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    if note:
        lines.append(f"% NOTE: {note}")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_two_variant_grid(conv: pd.DataFrame, left_variant: str, right_variant: str, filename: str,
                          left_label: str, right_label: str, right_color: str, title_prefix: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for ax, scale in zip(axes, SCALES):
        left = aggregate_curve(get_curves(conv, left_variant, scale, 0.5, "new_rerun"))
        right = aggregate_curve(get_curves(conv, right_variant, scale, 0.5))
        ax.plot(left["gen"], left["mean"], color=COLORS["CCHIHH-full"], label=left_label)
        ax.fill_between(left["gen"], left["mean"] - left["ci"], left["mean"] + left["ci"], color=COLORS["CCHIHH-full"], alpha=0.2)
        ax.plot(right["gen"], right["mean"], color=right_color, label=right_label)
        ax.fill_between(right["gen"], right["mean"] - right["ci"], right["mean"] + right["ci"], color=right_color, alpha=0.2)
        ax.set_title(f"{title_prefix}{scale}")
        ax.set_xlabel("Generation")
        if ax is axes[0]:
            ax.set_ylabel("Best fitness")
        ax.set_xlim(0, 10000)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    save_pdf(fig, filename)


def build_ablation_tables(final: pd.DataFrame):
    specs = [
        ("table3_ablation_cc.tex", "CCHIHH-noCC", "Table 3", None),
        ("table4_ablation_hi.tex", "CCHIHH-noHI", "Table 4", None),
        ("table5_ablation_cb.tex", "CCHIHH-noCB", "Table 5", "Mixed NEW CCHIHH-full vs OLD noCB/fixed_ops data."),
        ("table7_ablation_migration.tex", "CCHIHH-noMig", "Table 7", None),
        ("table8_ablation_gating.tex", "CCHIHH-noGate", "Table 8", None),
    ]
    for filename, variant, _, note in specs:
        rows = []
        csv_rows = []
        for scale in SCALES:
            full = get_runs(final, "CCHIHH-full", scale, 0.5, "new_rerun").sort_values("seed")
            other_source = "old_noCB" if variant == "CCHIHH-noCB" else "new_rerun"
            other = get_runs(final, variant, scale, 0.5, other_source).sort_values("seed")
            mf, sf = full["best_fit"].mean(), full["best_fit"].std(ddof=1)
            mo, so = other["best_fit"].mean(), other["best_fit"].std(ddof=1)
            impr = (mo - mf) / mo * 100.0
            p = pvalue_paired(full["best_fit"].to_numpy(), other["best_fit"].to_numpy())
            rows.append([
                scale,
                f"{mf:.6f}$\\pm${sf:.6f}",
                f"{mo:.6f}$\\pm${so:.6f}",
                f"{impr:.2f}",
                f"{p:.6f}",
            ])
            csv_rows.append(
                {
                    "scale": scale,
                    "full_mean": round(float(mf), 6),
                    "full_std": round(float(sf), 6),
                    "other_mean": round(float(mo), 6),
                    "other_std": round(float(so), 6),
                    "improvement_pct": round(float(impr), 4),
                    "wilcoxon_p": round(float(p), 6),
                }
            )
        latex_table(TABLES / filename, ["Scale", "Full", variant.replace("CCHIHH-", ""), "Impr.(\\%)", "Wilcoxon $p$"], rows, note=note)
        pd.DataFrame(csv_rows).to_csv(TABLES / filename.replace(".tex", ".csv"), index=False)


def build_baseline_tables(final: pd.DataFrame):
    pairs = [
        ("table9_vs_cga.tex", "CGA"),
        ("table10_vs_imoma.tex", "IMOMA"),
        ("table11_vs_ppo.tex", "PPO"),
        ("table12_vs_dsac_de.tex", "DSAC-DE"),
    ]
    for filename, baseline in pairs:
        rows = []
        for scale in SCALES:
            full = get_runs(final, "CCHIHH-full", scale, 0.5, "new_rerun").sort_values("seed")
            base = get_runs(final, baseline, scale, 0.5).sort_values("seed")
            mf, sf = full["best_fit"].mean(), full["best_fit"].std(ddof=1)
            mb, sb = base["best_fit"].mean(), base["best_fit"].std(ddof=1)
            impr = (mb - mf) / mb * 100.0
            p = pvalue_paired(full["best_fit"].to_numpy(), base["best_fit"].to_numpy())
            rows.append([scale, f"{mf:.6f}$\\pm${sf:.6f}", f"{mb:.6f}$\\pm${sb:.6f}", f"{impr:.2f}", f"{p:.6f}"])
        latex_table(TABLES / filename, ["Scale", "CCHIHH", baseline, "Impr.(\\%)", "Wilcoxon $p$"], rows)


def build_cv_table(final: pd.DataFrame):
    rows = []
    for scale in SCALES:
        vals = []
        for variant in ["CCHIHH-full", "DSAC-DE", "CGA", "IMOMA"]:
            source = "new_rerun" if variant == "CCHIHH-full" else None
            runs = get_runs(final, variant, scale, 0.5, source)
            cv = runs["best_fit"].std(ddof=1) / runs["best_fit"].mean()
            vals.append(f"{cv:.4f}")
        rows.append([scale] + vals)
    latex_table(TABLES / "table13_cv_dispersion.tex", ["Scale", "CCHIHH", "DSAC-DE", "CGA", "IMOMA"], rows)


def build_operator_table():
    rows = []
    op_dir = RESULTS / "rerun_full"
    for block, cols in {
        "offload": ["offload_GA", "offload_DE", "offload_BITFLIP", "offload_RESAMPLE"],
        "seq": ["seq_GA", "seq_SWAP", "seq_VNS", "seq_RESAMPLE"],
        "dev": ["dev_DE", "dev_GDE", "dev_LEVY", "dev_RESAMPLE"],
    }.items():
        frames = [pd.read_csv(op_dir / f"cchihh_full_T500_s{seed}_ops.csv") for seed in range(1, 11)]
        df = pd.concat(frames, ignore_index=True)
        early = df[df["gen"] <= 2000][cols].mean()
        late = df[df["gen"] >= 8000][cols].mean()
        for col in cols:
            rows.append([block, col.replace(block + "_", ""), f"{early[col]:.4f}", f"{late[col]:.4f}", f"{(late[col]-early[col]):+.4f}"])
    latex_table(TABLES / "table6_operator_early_late.tex", ["Block", "Operator", "Early", "Late", "Δ"], rows)


def fig_operator_selection():
    op_dir = RESULTS / "rerun_full"
    frames = [pd.read_csv(op_dir / f"cchihh_full_T500_s{seed}_ops.csv") for seed in range(1, 11)]
    df = pd.concat(frames).groupby("gen", as_index=False).mean(numeric_only=True)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    specs = [
        ("offload", ["offload_GA", "offload_DE", "offload_BITFLIP", "offload_RESAMPLE"], ["GA", "DE", "BITFLIP", "RESAMPLE"]),
        ("seq", ["seq_GA", "seq_SWAP", "seq_VNS", "seq_RESAMPLE"], ["GA", "SWAP", "VNS", "RESAMPLE"]),
        ("dev", ["dev_DE", "dev_GDE", "dev_LEVY", "dev_RESAMPLE"], ["DE", "GDE", "LEVY", "RESAMPLE"]),
    ]
    palette = ["#D62728", "#1F77B4", "#2CA02C", "#7F7F7F"]
    for ax, (title, cols, labels) in zip(axes, specs):
        ax.stackplot(df["gen"], *[df[c] for c in cols], labels=labels, colors=palette, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        if ax is axes[0]:
            ax.set_ylabel("Selection Probability")
        ax.set_ylim(0, 1)
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save_pdf(fig, "fig04_op_selection_T500.pdf")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for ax, (title, cols, labels) in zip(axes, specs):
        early = df[df["gen"] <= 2000][cols].mean()
        late = df[df["gen"] >= 8000][cols].mean()
        x = np.arange(len(cols))
        ax.bar(x - 0.18, early.to_numpy(), width=0.36, color=[mcolors.to_rgba(c, 0.45) for c in palette])
        ax.bar(x + 0.18, late.to_numpy(), width=0.36, color=palette)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25)
        ax.set_title(title)
        ax.set_ylim(0, max(early.max(), late.max()) * 1.25)
        biggest = (late - early).abs().idxmax()
        idx = cols.index(biggest)
        text = f"{labels[idx]} {early[biggest]*100:.1f}%→{late[biggest]*100:.1f}%"
        ax.annotate(text, xy=(idx + 0.18, late[biggest]), xytext=(idx, ax.get_ylim()[1] * 0.92),
                    arrowprops=dict(arrowstyle="->", lw=0.8), ha="center", fontsize=8)
    save_pdf(fig, "fig05_early_vs_late_T500.pdf")


def fig_bandit_weights():
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.4), constrained_layout=True)
    specs = [
        ("Offload", "w_off", ["GA", "DE", "BITFLIP", "RESAMPLE"]),
        ("Seq", "w_seq", ["GA", "SWAP", "VNS", "RESAMPLE"]),
        ("Dev", "w_dev", ["DE", "GDE", "LEVY", "RESAMPLE"]),
    ]
    for ax, (title, suffix, labels) in zip(axes, specs):
        mats = []
        for seed in range(1, 11):
            df = pd.read_csv(RESULTS / "rerun_full" / f"cchihh_full_T500_s{seed}_{suffix}.csv")
            pivot = df.pivot(index="op_id", columns="gen", values="norm").sort_index()
            mat = pivot.to_numpy()
            denom = np.maximum(mat.sum(axis=0, keepdims=True), 1e-12)
            mats.append(mat / denom)
        avg = np.mean(mats, axis=0)
        im = ax.imshow(avg, aspect="auto", cmap="Reds", origin="lower")
        ax.set_title(title)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
    axes[-1].set_xlabel("Generation index")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Normalized weight")
    save_pdf(fig, "fig06_bandit_weights_T500.pdf")


def fig_tgate(final: pd.DataFrame):
    rows = []
    for gate in (5, 10, 15, 20, 25):
        runs = final[(final["variant"] == f"CCHIHH-tgate{gate}") & (final["scale"] == "T500")]
        rows.append((gate, runs["best_fit"].mean(), runs["best_fit"].std(ddof=1)))
    df = pd.DataFrame(rows, columns=["gate", "mean", "std"])
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(df["gate"], df["mean"], color=COLORS["CCHIHH-full"], marker="o")
    ax.fill_between(df["gate"], df["mean"] - df["std"], df["mean"] + df["std"], color=COLORS["CCHIHH-full"], alpha=0.2)
    ax.errorbar(df["gate"], df["mean"], yerr=df["std"], fmt="none", ecolor=COLORS["CCHIHH-full"], capsize=3)
    ax.set_xlabel("Tgate")
    ax.set_ylabel("Best fitness")
    ax.grid(True, alpha=0.25)
    save_pdf(fig, "fig09_tgate_sensitivity.pdf")


def fig_component_heatmap(final: pd.DataFrame):
    rows = [("w/o CC", "CCHIHH-noCC"), ("w/o Islands", "CCHIHH-noHI"), ("w/o Bandit", "CCHIHH-noCB"), ("w/o Gating", "CCHIHH-noGate"), ("w/o Migration", "CCHIHH-noMig")]
    data = np.zeros((5, 3))
    pvals = np.ones((5, 3))
    for i, (_, variant) in enumerate(rows):
        for j, scale in enumerate(SCALES):
            full = get_runs(final, "CCHIHH-full", scale, 0.5, "new_rerun").sort_values("seed")
            other = get_runs(final, variant, scale, 0.5, "old_noCB" if variant == "CCHIHH-noCB" else "new_rerun").sort_values("seed")
            data[i, j] = (other["best_fit"].mean() - full["best_fit"].mean()) / other["best_fit"].mean() * 100.0
            pvals[i, j] = pvalue_paired(full["best_fit"].to_numpy(), other["best_fit"].to_numpy())
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(data, cmap="Reds", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(SCALES)
    ax.set_yticks(range(5))
    ax.set_yticklabels([r[0] for r in rows])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = f"{data[i,j]:.1f}%\n$p$={pvals[i,j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
            if pvals[i, j] >= 0.05:
                ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linestyle="--", linewidth=1.0))
    ax.set_title("Scale-dependent component contribution")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Improvement (%)")
    fig.text(0.5, -0.02, "w/o Bandit uses OLD noCB/fixed_ops data.", ha="center", fontsize=8)
    save_pdf(fig, "fig10_component_contribution.pdf")


def fig_alpha_sensitivity(conv: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.4), constrained_layout=True)
    algos = [("CCHIHH-full", "new_rerun", COLORS["CCHIHH-full"]), ("CGA", None, COLORS["CGA"]), ("IMOMA", None, COLORS["IMOMA"]), ("DSAC-DE", None, COLORS["DSAC-DE"])]
    for i, alpha in enumerate(ALPHAS):
        for j, scale in enumerate(SCALES):
            ax = axes[i, j]
            for variant, source, color in algos:
                src = source
                if variant != "CCHIHH-full":
                    src = "old_alpha_runs" if alpha in (0.2, 0.8) else None
                data = get_curves(conv, variant, scale, alpha, src)
                if data.empty and alpha == 0.5 and variant != "CCHIHH-full":
                    data = get_curves(conv, variant, scale, alpha)
                agg = aggregate_curve(data)
                label = "CCHIHH" if variant == "CCHIHH-full" else variant
                ax.plot(agg["gen"], agg["mean"], color=color, label=label)
                ax.fill_between(agg["gen"], agg["mean"] - agg["ci"], agg["mean"] + agg["ci"], color=color, alpha=0.18)
            ax.set_title(f"$\\alpha$={alpha}, {scale}")
            if i == 2:
                ax.set_xlabel("Generation")
            if j == 0:
                ax.set_ylabel("Best fitness")
            ax.grid(True, alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=2)
    save_pdf(fig, "fig11_alpha_sensitivity.pdf")


def fig_vs_baselines(conv: pd.DataFrame):
    plot_two_variant_grid(conv, "CCHIHH-full", "CGA", "fig12_vs_CGA_IMOMA.pdf", "CCHIHH", "CGA", COLORS["CGA"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for ax, scale in zip(axes, SCALES):
        cch = aggregate_curve(get_curves(conv, "CCHIHH-full", scale, 0.5, "new_rerun"))
        cga = aggregate_curve(get_curves(conv, "CGA", scale, 0.5))
        imo = aggregate_curve(get_curves(conv, "IMOMA", scale, 0.5))
        ax.plot(cch["gen"], cch["mean"], color=COLORS["CCHIHH-full"], label="CCHIHH")
        ax.fill_between(cch["gen"], cch["mean"] - cch["ci"], cch["mean"] + cch["ci"], color=COLORS["CCHIHH-full"], alpha=0.2)
        ax.plot(cga["gen"], cga["mean"], color=COLORS["CGA"], label="CGA")
        ax.fill_between(cga["gen"], cga["mean"] - cga["ci"], cga["mean"] + cga["ci"], color=COLORS["CGA"], alpha=0.2)
        ax.plot(imo["gen"], imo["mean"], color=COLORS["IMOMA"], label="IMOMA")
        ax.fill_between(imo["gen"], imo["mean"] - imo["ci"], imo["mean"] + imo["ci"], color=COLORS["IMOMA"], alpha=0.2)
        ax.set_title(scale)
        ax.set_xlabel("Generation")
        if ax is axes[0]:
            ax.set_ylabel("Best fitness")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    save_pdf(fig, "fig12_vs_CGA_IMOMA.pdf")
    plot_two_variant_grid(conv, "CCHIHH-full", "PPO", "fig13_vs_PPO.pdf", "CCHIHH", "PPO", COLORS["PPO"])
    plot_two_variant_grid(conv, "CCHIHH-full", "DSAC-DE", "fig14_vs_DSAC_DE.pdf", "CCHIHH", "DSAC-DE", COLORS["DSAC-DE"])


def fig_boxplots(final: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    order = ["CCHIHH-full", "DSAC-DE", "CGA", "IMOMA"]
    labels = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"]
    box_colors = [COLORS["CCHIHH-full"], COLORS["DSAC-DE"], COLORS["CGA"], COLORS["IMOMA"]]
    for ax, scale in zip(axes, SCALES):
        data = []
        for variant in order:
            src = "new_rerun" if variant == "CCHIHH-full" else None
            data.append(get_runs(final, variant, scale, 0.5, src)["best_fit"].to_numpy())
        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(mcolors.to_rgba(color, 0.45))
        ax.set_title(scale)
        ax.set_ylabel("Final objective")
        ax.grid(True, axis="y", alpha=0.2)
    save_pdf(fig, "fig15_boxplots.pdf")


def fig_placeholder_makespan_energy():
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "Fig 16 requires per-seed makespan/energy decomposition for old baseline best solutions.\n"
        "Those values are not present in the archived baseline logs, and baseline reruns were excluded.",
        ha="center",
        va="center",
        fontsize=10,
    )
    save_pdf(fig, "fig16_makespan_energy.pdf")


def fig_radar(final: pd.DataFrame, conv: pd.DataFrame):
    labels = ["Solution Quality", "Stability", "Efficiency", "Convergence Speed"]
    algos = ["CCHIHH-full", "CGA", "IMOMA", "DSAC-DE"]
    display = {"CCHIHH-full": "CCHIHH", "CGA": "CGA", "IMOMA": "IMOMA", "DSAC-DE": "DSAC-DE"}
    colors = [COLORS["CCHIHH-full"], COLORS["CGA"], COLORS["IMOMA"], COLORS["DSAC-DE"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), subplot_kw={"projection": "polar"}, constrained_layout=True)
    for ax, scale in zip(axes, SCALES):
        metrics = {}
        for algo in algos:
            src = "new_rerun" if algo == "CCHIHH-full" else None
            runs = get_runs(final, algo, scale, 0.5, src)
            curve = get_curves(conv, algo, scale, 0.5, src)
            gen2000 = curve[curve["gen"] == 2000].groupby("seed")["best_fit"].last().mean()
            metrics[algo] = {
                "fitness": runs["best_fit"].mean(),
                "cv": runs["best_fit"].std(ddof=1) / runs["best_fit"].mean(),
                "runtime": runs["runtime_s"].replace("", np.nan).astype(float).mean(),
                "conv2000": gen2000,
            }
        values_by_metric = {
            "fitness": np.array([metrics[a]["fitness"] for a in algos]),
            "cv": np.array([metrics[a]["cv"] for a in algos]),
            "runtime": np.array([metrics[a]["runtime"] for a in algos]),
            "conv2000": np.array([metrics[a]["conv2000"] for a in algos]),
        }
        def normalize_lower(arr):
            best, worst = np.nanmin(arr), np.nanmax(arr)
            if math.isclose(best, worst):
                return np.ones_like(arr)
            return (worst - arr) / (worst - best)
        radar_vals = np.column_stack([
            normalize_lower(values_by_metric["fitness"]),
            normalize_lower(values_by_metric["cv"]),
            normalize_lower(values_by_metric["runtime"]),
            normalize_lower(values_by_metric["conv2000"]),
        ])
        theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        theta = np.r_[theta, theta[0]]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        for vals, color, algo in zip(radar_vals, colors, algos):
            vals = np.r_[vals, vals[0]]
            ax.plot(theta, vals, color=color, label=display[algo])
            ax.fill(theta, vals, color=color, alpha=0.12)
        ax.set_title(scale)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.1, 1.12), frameon=False)
    save_pdf(fig, "fig17_radar.pdf")


def fig_runtime(final: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    algos = ["CCHIHH-full", "DSAC-DE", "CGA", "IMOMA"]
    labels = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"]
    colors = [COLORS["CCHIHH-full"], COLORS["DSAC-DE"], COLORS["CGA"], COLORS["IMOMA"]]
    for ax, scale in zip(axes, SCALES):
        vals = []
        for algo in algos:
            src = "new_rerun" if algo == "CCHIHH-full" else None
            vals.append(get_runs(final, algo, scale, 0.5, src)["runtime_s"].astype(float).mean())
        bars = ax.bar(labels, vals, color=colors)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(scale)
        ax.set_ylabel("Runtime (s)")
    save_pdf(fig, "fig18_runtime.pdf")


def fig_cd(final: pd.DataFrame):
    algos = ["CCHIHH-full", "DSAC-DE", "CGA", "IMOMA", "PPO"]
    instances = []
    for alpha in ALPHAS:
        for scale in SCALES:
            values = {}
            for algo in algos:
                if algo == "CCHIHH-full":
                    src = "new_rerun"
                elif algo in ("CGA", "IMOMA", "DSAC-DE") and alpha in (0.2, 0.8):
                    src = "old_alpha_runs"
                else:
                    src = None
                runs = get_runs(final, algo, scale, alpha, src)
                if runs.empty and alpha == 0.5:
                    runs = get_runs(final, algo, scale, alpha)
                values[algo] = runs["best_fit"].mean()
            instances.append(values)
        #
    ranks = []
    for inst in instances:
        s = pd.Series(inst).rank(method="average", ascending=True)
        ranks.append(s)
    rank_df = pd.DataFrame(ranks)
    avg_ranks = rank_df.mean(axis=0).sort_values()
    stat, p = friedmanchisquare(*[rank_df[c].to_numpy() for c in algos])
    cd = 2.728 * math.sqrt(len(algos) * (len(algos) + 1) / (6 * len(instances)))

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")
    y = 0.56
    ax.hlines(y, 1, len(algos), color="black")
    for i in range(1, len(algos) + 1):
        ax.vlines(i, y - 0.04, y + 0.04, color="black")
        ax.text(i, y - 0.1, str(i), ha="center")
    ypos = [0.82, 0.22, 0.82, 0.22, 0.82]
    for (name, rank), yp in zip(avg_ranks.items(), ypos):
        ax.vlines(rank, y, yp - 0.03 if yp > y else yp + 0.03, color="black", linewidth=0.9)
        ax.text(rank, yp, name.replace("CCHIHH-full", "CCHIHH"), ha="center", va="center")
    ax.plot([avg_ranks.iloc[0], avg_ranks.iloc[0] + cd], [0.93, 0.93], color="black", linewidth=2)
    ax.vlines([avg_ranks.iloc[0], avg_ranks.iloc[0] + cd], 0.91, 0.95, color="black")
    ax.text(avg_ranks.iloc[0] + cd / 2, 0.97, f"CD={cd:.3f}", ha="center", fontsize=9)
    ax.text(1.0, 0.04, f"Friedman p={p:.6f}", ha="left", fontsize=9)
    save_pdf(fig, "fig19_critical_difference.pdf")


def fig_gantt():
    path = RESULTS / "rerun_schedule" / "cchihh_full_T200_best_seed5_schedule.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = {}
    for line in lines:
        if line.startswith("# "):
            key, value = line[2:].split("=", 1)
            meta[key.strip()] = value.strip()

    df = pd.read_csv(path, comment="#")
    ce = df[df["type"] == "CE"].copy()
    ce["task_num"] = ce["task_id"].str.extract(r"(\d+)").astype(int)
    ce["duration"] = ce["end_time"] - ce["start_time"]

    cloud = ce[ce["assigned_tier"] == "cloud"].copy()
    edge = ce[ce["assigned_tier"] == "edge"].copy()

    def order_resources(frame: pd.DataFrame) -> list[str]:
        stats = (
            frame.groupby("server_id")
            .agg(task_count=("task_id", "count"), first_start=("start_time", "min"), last_end=("end_time", "max"))
            .reset_index()
        )
        stats["server_num"] = (
            stats["server_id"].str.extract(r"(\d+)")[0].fillna("999999").astype(int)
        )
        stats = stats.sort_values(
            by=["task_count", "first_start", "last_end", "server_num"],
            ascending=[False, True, True, True],
        )
        return stats["server_id"].tolist()

    def collapse_single_use(frame: pd.DataFrame, tier: str) -> tuple[pd.DataFrame, list[str], set[str]]:
        counts = frame["server_id"].value_counts()
        keep = counts[counts >= 2].index.tolist()
        top_singletons = counts[counts == 1].index.tolist()[:8]
        keep_set = set(keep) | set(top_singletons)
        collapsed = frame.copy()
        collapsed["plot_server_id"] = collapsed["server_id"]
        aggregate_label = f"{tier}_single_use"
        collapsed.loc[~collapsed["server_id"].isin(keep_set), "plot_server_id"] = aggregate_label
        order_frame = collapsed.loc[:, ["plot_server_id", "task_id", "start_time", "end_time"]].rename(
            columns={"plot_server_id": "server_id"}
        )
        ordered = order_resources(order_frame)
        if aggregate_label in collapsed["plot_server_id"].values and aggregate_label not in ordered:
            ordered.append(aggregate_label)
        return collapsed, ordered, keep_set

    cloud_plot, cloud_resources, cloud_kept = collapse_single_use(cloud, "cloud")
    edge_plot, edge_resources, edge_kept = collapse_single_use(edge, "edge")

    row_gap = 0.38
    bar_height = 0.88

    def tier_color(task_num: int, tier: str) -> tuple:
        cycle = [0.42, 0.5, 0.58, 0.66]
        light = cycle[task_num % len(cycle)]
        hue = 0.59 if tier == "cloud" else 0.07
        sat = 0.65 if tier == "cloud" else 0.75
        return mcolors.hsv_to_rgb((hue, sat, light))

    fig, (ax_cloud, ax_edge) = plt.subplots(
        2,
        1,
        figsize=(18, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [max(len(cloud_resources), 1), max(len(edge_resources), 1)], "hspace": 0.08},
    )

    def build_axis(
        ax: plt.Axes,
        frame: pd.DataFrame,
        resources: list[str],
        tier: str,
        bg: str,
        edgecolor: str,
        title: str,
        kept_resources: set[str],
    ):
        y_positions = {res: idx * (1.0 + row_gap) for idx, res in enumerate(resources)}
        ax.set_facecolor(bg)
        for _, row in frame.iterrows():
            ax.barh(
                y_positions[row["plot_server_id"]],
                row["duration"],
                left=row["start_time"],
                height=bar_height,
                color=tier_color(int(row["task_num"]), tier),
                edgecolor=edgecolor,
                linewidth=0.4,
                zorder=3,
            )

        tick_positions = [y_positions[res] for res in resources]
        counts = frame["plot_server_id"].value_counts()
        labels = []
        for idx, res in enumerate(resources):
            if res.endswith("_single_use"):
                labels.append("single-use")
            elif res in kept_resources and (counts[res] >= 2 or idx < 12 or idx % 5 == 0):
                labels.append(res)
            else:
                labels.append("")

        ax.set_yticks(tick_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#C9D2DA", linewidth=0.8, alpha=0.6, zorder=0)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Server ID")
        ax.set_title(title, loc="left", fontsize=12, pad=8)
        return y_positions

    cloud_singletons = int((cloud["server_id"].value_counts() == 1).sum())
    edge_singletons = int((edge["server_id"].value_counts() == 1).sum())

    build_axis(
        ax_cloud,
        cloud_plot,
        cloud_resources,
        "cloud",
        "#F4F8FC",
        "#E6EEF7",
        f"Cloud Servers ({len(cloud_resources)} plotted, {cloud_singletons} single-use servers merged)",
        cloud_kept,
    )
    build_axis(
        ax_edge,
        edge_plot,
        edge_resources,
        "edge",
        "#FFF5EF",
        "#FFF1E8",
        f"Edge Servers ({len(edge_resources)} plotted, {edge_singletons} single-use servers merged)",
        edge_kept,
    )

    makespan = float(meta.get("makespan", ce["end_time"].max()))
    energy = float(meta.get("energy", 0.0))
    seed = meta.get("seed", "5")

    xmax = float(np.ceil(makespan / 100.0) * 100.0)
    xticks = np.arange(0.0, xmax + 1.0, 100.0)

    ax_cloud.set_xlim(0, xmax)
    ax_edge.set_xlim(0, xmax)
    ax_edge.set_xticks(xticks)
    ax_edge.set_xlabel("Time")

    legend_handles = [
        patches.Patch(facecolor=tier_color(0, "cloud"), edgecolor="none", label="Cloud-assigned CE tasks"),
        patches.Patch(facecolor=tier_color(1, "edge"), edgecolor="none", label="Edge-assigned CE tasks"),
    ]
    ax_cloud.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.suptitle("T200 Best CCHIHH Schedule", fontsize=17, y=0.985)
    fig.text(
        0.5,
        0.958,
        f"Seed {seed} | makespan = {makespan:.2f} | total energy = {energy:.2f}",
        ha="center",
        va="center",
        fontsize=12,
    )

    save_pdf_png(fig, "fig20_gantt_T200_v2")


def figs_and_tables(conv: pd.DataFrame, final: pd.DataFrame):
    plot_two_variant_grid(conv, "CCHIHH-full", "CCHIHH-noCC", "fig02_ablation_CC.pdf", "CCHIHH-full", "CCHIHH-noCC", COLORS["noCC"])
    plot_two_variant_grid(conv, "CCHIHH-full", "CCHIHH-noHI", "fig03_ablation_HI.pdf", "CCHIHH-full", "CCHIHH-noHI", COLORS["noHI"])
    fig_operator_selection()
    fig_bandit_weights()
    plot_two_variant_grid(conv, "CCHIHH-full", "CCHIHH-noMig", "fig07_ablation_migration.pdf", "migration on", "migration off", COLORS["noMig"])
    plot_two_variant_grid(conv, "CCHIHH-full", "CCHIHH-noGate", "fig08_ablation_gate.pdf", "gate on", "gate off", COLORS["noGate"])
    fig_tgate(final)
    fig_component_heatmap(final)
    fig_alpha_sensitivity(conv)
    fig_vs_baselines(conv)
    fig_boxplots(final)
    fig_placeholder_makespan_energy()
    fig_radar(final, conv)
    fig_runtime(final)
    fig_cd(final)
    fig_gantt()
    build_ablation_tables(final)
    build_operator_table()
    build_baseline_tables(final)
    build_cv_table(final)


def main():
    setup_style()
    conv, final = load_data()
    figs_and_tables(conv, final)
    print(f"Saved figures to {FIGURES}")
    print(f"Saved tables to {TABLES}")


if __name__ == "__main__":
    main()
