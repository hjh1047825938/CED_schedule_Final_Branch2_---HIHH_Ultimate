from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
OUTPUTS_DIR = ROOT / "outputs"

COLORS = {
    "CCHIHH": "#E63946",
    "CGA": "#457B9D",
    "IMOMA": "#2A9D8F",
    "DSAC-DE": "#E9C46A",
    "PPO": "#6A4C93",
    "full": "#E63946",
    "noCC": "#457B9D",
    "noHI": "#2A9D8F",
    "noCB": "#E9C46A",
    "noGate": "#6A4C93",
    "noMig": "#8D6E63",
}

ABBREV_TO_CANONICAL = {
    "full": "CCHIHH-full",
    "noCC": "CCHIHH-noCC",
    "noHI": "CCHIHH-noHI",
    "noCB": "CCHIHH-noCB",
    "noGate": "CCHIHH-noGate",
    "noMig": "CCHIHH-noMig",
}


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "grid.color": "#BDBDBD",
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "savefig.dpi": 300,
            "figure.dpi": 300,
        }
    )
    sns.set_theme(style="whitegrid", context="paper")
    FIG_DIR.mkdir(exist_ok=True)


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def savefig_multi(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_heatmap() -> None:
    data = pd.DataFrame(
        {
            "T100": [6.28, 2.97, 3.74, -0.10, 1.86],
            "T200": [15.76, 33.66, 19.92, -2.04, 15.42],
            "T500": [26.25, 40.05, 22.88, 3.01, 30.65],
        },
        index=["w/o CC", "w/o Islands", "w/o Bandit", "w/o Gating", "w/o Migration"],
    )

    annot = data.copy().astype(object)
    for row in data.index:
        for col in data.columns:
            txt = f"{data.loc[row, col]:.2f}%"
            if row == "w/o Gating" and col in {"T100", "T200"}:
                txt += " (n.s.)"
            annot.loc[row, col] = txt

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ablation_diverging",
        [(0.0, "#DCEAF7"), (0.08, "#F7FBFF"), (0.09, "#FFFFFF"), (1.0, "#8B0000")],
    )
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=40)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        norm=norm,
        annot=annot.values,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Improvement (%)"},
    )
    ax.set_title("Contribution of Each Component Across Problem Scales", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    for (row, col), value in np.ndenumerate(data.values):
        if row == 3 and col in (0, 1):
            rect = patches.Rectangle(
                (col, row),
                1,
                1,
                fill=False,
                edgecolor="#4F6D7A",
                linewidth=1.2,
                linestyle=(0, (3, 2)),
            )
            ax.add_patch(rect)

    savefig(fig, "fig_ablation_heatmap.pdf")


def minmax_best_one(values: dict[str, float], lower_is_better: bool) -> dict[str, float]:
    arr = pd.Series(values, dtype=float)
    if lower_is_better:
        best = arr.min()
        worst = arr.max()
        if math.isclose(best, worst):
            return {k: 1.0 for k in arr.index}
        scaled = (worst - arr) / (worst - best)
    else:
        best = arr.max()
        worst = arr.min()
        if math.isclose(best, worst):
            return {k: 1.0 for k in arr.index}
        scaled = (arr - worst) / (best - worst)
    return scaled.clip(0.0, 1.0).to_dict()


def plot_radar_charts() -> None:
    datasets = {
        "T100": {
            "fitness": {"CCHIHH": 0.3423, "CGA": 0.3532, "IMOMA": 0.3775, "DSAC-DE": 0.3248},
            "cv": {"CCHIHH": 0.0027, "CGA": 0.0043, "IMOMA": 0.0039, "DSAC-DE": 0.0030},
            "runtime": {"CCHIHH": 33.1, "CGA": 76.8, "IMOMA": 99.6, "DSAC-DE": 42.9},
            "conv": {"CCHIHH": 1000, "CGA": 3000, "IMOMA": 5000, "DSAC-DE": 800},
        },
        "T200": {
            "fitness": {"CCHIHH": 0.0383, "CGA": 0.0495, "IMOMA": 0.0576, "DSAC-DE": 0.0472},
            "cv": {"CCHIHH": 0.0505, "CGA": 0.0221, "IMOMA": 0.0280, "DSAC-DE": 0.0350},
            "runtime": {"CCHIHH": 85.3, "CGA": 211.7, "IMOMA": 341.3, "DSAC-DE": 139.3},
            "conv": {"CCHIHH": 1800, "CGA": 4200, "IMOMA": 6500, "DSAC-DE": 2600},
        },
        "T500": {
            "fitness": {"CCHIHH": 0.0260, "CGA": 0.0392, "IMOMA": 0.0465, "DSAC-DE": 0.0372},
            "cv": {"CCHIHH": 0.0359, "CGA": 0.0224, "IMOMA": 0.0295, "DSAC-DE": 0.0300},
            "runtime": {"CCHIHH": 289.1, "CGA": 720.3, "IMOMA": 1207.8, "DSAC-DE": 289.4},
            "conv": {"CCHIHH": 2600, "CGA": 5600, "IMOMA": 8400, "DSAC-DE": 3200},
        },
    }
    labels = ["Solution Quality", "Stability", "Efficiency", "Convergence Speed"]
    algorithms = ["CCHIHH", "CGA", "IMOMA", "DSAC-DE"]

    for problem, raw in datasets.items():
        normalized = {
            "Solution Quality": minmax_best_one(raw["fitness"], lower_is_better=True),
            "Stability": minmax_best_one(raw["cv"], lower_is_better=True),
            "Efficiency": minmax_best_one(raw["runtime"], lower_is_better=True),
            "Convergence Speed": minmax_best_one(raw["conv"], lower_is_better=True),
        }

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={"projection": "polar"})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        for algo in algorithms:
            values = [normalized[label][algo] for label in labels]
            values += values[:1]
            color = COLORS[algo]
            ax.plot(angles, values, color=color, marker="o", label=algo)
            ax.fill(angles, values, color=color, alpha=0.15)

        ax.set_title(problem, pad=14)
        ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.08), frameon=False)
        savefig(fig, f"fig_radar_{problem}.pdf")


def plot_cd_diagram() -> None:
    avg_ranks = {
        "CCHIHH": 1.333,
        "DSAC-DE": 1.667,
        "CGA": 3.000,
        "IMOMA": 4.000,
        "PPO": 5.000,
    }
    cd = 2.728
    ordered = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    cliques = [
        ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"],
        ["CGA", "IMOMA", "PPO"],
    ]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_axis = 0.55
    ax.hlines(y_axis, 1, 5, color="black", linewidth=1.0)
    for tick in range(1, 6):
        ax.vlines(tick, y_axis - 0.03, y_axis + 0.03, color="black", linewidth=1.0)
        ax.text(tick, y_axis - 0.08, f"{tick}", ha="center", va="top")
    ax.text(1, y_axis + 0.08, "Better", ha="left", va="bottom")
    ax.text(5, y_axis + 0.08, "Worse", ha="right", va="bottom")

    top_y = [0.84, 0.72, 0.84]
    bot_y = [0.26, 0.14]
    top_idx = 0
    bot_idx = 0
    for idx, (name, rank) in enumerate(ordered):
        if idx % 2 == 0:
            y_text = top_y[top_idx]
            top_idx += 1
            va = "bottom"
        else:
            y_text = bot_y[bot_idx]
            bot_idx += 1
            va = "top"
        ax.vlines(rank, y_axis, y_text - 0.03 if va == "bottom" else y_text + 0.03, color="black", linewidth=0.9)
        ax.text(rank, y_text, f"{name} ({rank:.3f})", ha="center", va=va)

    bar_y = [0.95, 0.08]
    for idx, clique in enumerate(cliques):
        ranks = [avg_ranks[name] for name in clique]
        ax.hlines(bar_y[idx], min(ranks), max(ranks), color="black", linewidth=3.0)
        ax.vlines([min(ranks), max(ranks)], bar_y[idx] - 0.015, bar_y[idx] + 0.015, color="black", linewidth=1.0)

    ax.text(4.85, 0.95, f"CD = {cd:.3f}", ha="right", va="center")
    ax.set_title("Critical Difference Diagram (Friedman-Nemenyi, α=0.05)", pad=6)
    savefig(fig, "fig_cd_diagram.pdf")


def plot_operator_selection_early_vs_late() -> None:
    early_color = "#F4A582"
    late_color = "#B2182B"
    up_color = "#B2182B"
    down_color = "#457B9D"
    width = 0.35

    blocks = {
        "Offload": {
            "operators": ["GA", "DE", "BITFLIP", "RESAMPLE"],
            "early": [0.399, 0.505, 0.081, 0.015],
            "late": [0.411, 0.389, 0.125, 0.076],
            "annotations": [
                {"idx": 1, "text": "DE\n50.5% \u2192 38.9%\n(\u219311.6%)", "xytext": (0.75, 0.57), "color": down_color},
            ],
        },
        "Sequence": {
            "operators": ["GA", "SWAP", "VNS", "RESAMPLE"],
            "early": [0.292, 0.436, 0.259, 0.013],
            "late": [0.344, 0.158, 0.306, 0.193],
            "annotations": [
                {"idx": 1, "text": "SWAP\n43.6% \u2192 15.8%\n(\u219327.8%)", "xytext": (0.55, 0.57), "color": down_color},
                {"idx": 3, "text": "RESAMPLE\n1.3% \u2192 19.3%\n(\u219118.0%)", "xytext": (2.35, 0.44), "color": up_color},
            ],
        },
        "Device": {
            "operators": ["DE", "GDE", "LEVY", "RESAMPLE"],
            "early": [0.367, 0.495, 0.128, 0.011],
            "late": [0.377, 0.250, 0.216, 0.158],
            "annotations": [
                {"idx": 1, "text": "GDE\n49.5% \u2192 25.0%\n(\u219324.5%)", "xytext": (0.72, 0.57), "color": down_color},
            ],
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=True)
    fig.subplots_adjust(wspace=0.22)

    legend_handles = None
    for ax, (block_name, block) in zip(axes, blocks.items()):
        x = np.arange(len(block["operators"]))
        early = np.array(block["early"])
        late = np.array(block["late"])

        bars_early = ax.bar(
            x - width / 2,
            early,
            width=width,
            color=early_color,
            edgecolor=late_color,
            linewidth=1.0,
            label="Early Stage (gen 0–2k)",
            zorder=3,
        )
        bars_late = ax.bar(
            x + width / 2,
            late,
            width=width,
            color=late_color,
            edgecolor=late_color,
            linewidth=1.0,
            label="Late Stage (gen 8k–10k)",
            zorder=3,
        )
        if legend_handles is None:
            legend_handles = (bars_early[0], bars_late[0])

        ax.set_title(f"{block_name} Block")
        ax.set_xticks(x)
        ax.set_xticklabels(block["operators"])
        ax.set_ylim(0, 0.6)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for ann in block["annotations"]:
            idx = ann["idx"]
            target_y = max(early[idx], late[idx])
            target_x = x[idx] + (width / 2 if late[idx] >= early[idx] else -width / 2)
            ax.annotate(
                ann["text"],
                xy=(target_x, target_y),
                xytext=ann["xytext"],
                textcoords="data",
                ha="left",
                va="top",
                fontsize=7,
                color=ann["color"],
                arrowprops={
                    "arrowstyle": "->",
                    "color": ann["color"],
                    "lw": 1.0,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
            )

    axes[0].set_ylabel("Selection Probability")
    axes[0].legend(
        legend_handles,
        ["Early Stage (gen 0–2k)", "Late Stage (gen 8k–10k)"],
        loc="lower left",
        bbox_to_anchor=(-0.03, 1.03),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    savefig_multi(fig, "fig_operator_selection_early_vs_late")


def plot_diversity_curves() -> None:
    files = {
        "full": OUTPUTS_DIR / "results" / "diversity_logs" / "T500" / "diversity_log_full_seed1.csv",
        "noHI": OUTPUTS_DIR / "results" / "diversity_logs" / "T500" / "diversity_log_noHI_seed1.csv",
        "noMig": OUTPUTS_DIR / "results" / "diversity_logs" / "T500" / "diversity_log_noMig_seed1.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for variant, path in files.items():
        if path.exists():
            frames[variant] = pd.read_csv(path)

    if not frames:
        return

    fig, axes = plt.subplots(2, 3, figsize=(7, 6), sharex=True, sharey="row")
    block_order = [("offload", "Offload"), ("seq", "Sequence"), ("dev", "Device")]
    row_specs = [
        ("global_diversity", "Global Diversity"),
        ("inter_diversity", "Inter-island Diversity"),
    ]
    variant_labels = {"full": "CCHIHH-full", "noHI": "CCHIHH-noHI", "noMig": "CCHIHH-noMig"}

    for row_idx, (metric, row_label) in enumerate(row_specs):
        for col_idx, (block_key, block_title) in enumerate(block_order):
            ax = axes[row_idx, col_idx]
            for variant, df in frames.items():
                block_df = df[df["block"] == block_key].copy()
                if block_df.empty or metric not in block_df.columns:
                    continue
                markevery = max(int(round(500 / 50)), 1)
                ax.plot(
                    block_df["generation"],
                    block_df[metric],
                    color=COLORS[variant],
                    label=variant_labels[variant],
                    marker="o",
                    markevery=markevery,
                )
            ax.set_title(f"{block_title} - {'Global' if metric == 'global_diversity' else 'Inter'}")
            ax.set_xlim(0, 10000)
            ax.set_ylim(0, 0.12 if metric == "global_diversity" else 0.08)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax.grid(False, axis="x")
            if row_idx == 1:
                ax.set_xlabel("Generation")
            if col_idx == 0:
                ax.set_ylabel(row_label)

    axes[0, 0].legend(loc="upper right", frameon=False)
    fig.suptitle("Population Diversity Decomposition on T500", y=0.98, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig_multi(fig, "fig_diversity_curves_v2")


def plot_gantt_t200() -> None:
    path = OUTPUTS_DIR / "results" / "schedule_exports" / "schedule_export_T200_seed1.csv"
    if not path.exists():
        return

    rows = []
    meta = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, value = line[1:].split("=", 1)
                meta[key.strip()] = value.strip()
                continue
            rows.append(line)
    df = pd.read_csv(pd.io.common.StringIO("\n".join(rows)))
    df["start_time"] = pd.to_numeric(df["start_time"])
    df["end_time"] = pd.to_numeric(df["end_time"])
    df["duration"] = df["end_time"] - df["start_time"]
    df = df[df["duration"] > 0].copy()

    ce_df = df[df["type"] == "CE"].copy()
    mfg_df = df[df["type"] == "MFG"].copy()
    mfg_df = mfg_df.sort_values(["start_time", "end_time"]).head(50).copy()
    selected_df = pd.concat([ce_df, mfg_df], ignore_index=True)

    cloud_ids = list(ce_df[ce_df["assigned_tier"] == "cloud"]["server_id"].value_counts().head(10).index)
    edge_ids = list(ce_df[ce_df["assigned_tier"] == "edge"]["server_id"].value_counts().head(10).index)
    device_ids = list(mfg_df["server_id"].value_counts().head(15).index)

    lane_labels = cloud_ids + edge_ids + device_ids
    lane_y = {label: idx for idx, label in enumerate(lane_labels[::-1])}

    fig, ax = plt.subplots(figsize=(7, 6))

    cloud_count = len(cloud_ids)
    edge_count = len(edge_ids)
    total_count = len(lane_labels)
    device_count = len(device_ids)

    def add_band(start_row: int, count: int, color: str) -> None:
        if count <= 0:
            return
        ymin = total_count - (start_row + count) - 0.5
        ymax = total_count - start_row - 0.5
        ax.axhspan(ymin, ymax, color=color, alpha=0.18, zorder=0)

    add_band(0, cloud_count, "#DCEAF7")
    add_band(cloud_count, edge_count, "#E3F4E8")
    add_band(cloud_count + edge_count, device_count, "#FFF4D6")

    task_colors = {"CE": "#457B9D", "MFG": "#F4A261"}
    for _, row in selected_df.iterrows():
        server_id = row["server_id"]
        if server_id not in lane_y:
            continue
        y = lane_y[server_id]
        ax.barh(
            y,
            row["duration"],
            left=row["start_time"],
            height=0.7,
            color=task_colors[row["type"]],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    makespan = float(meta.get("makespan", selected_df["end_time"].max()))
    ax.axvline(makespan, color=COLORS["CCHIHH"], linestyle="--", linewidth=1.2, zorder=2)
    ax.text(makespan, total_count + 0.3, f"Makespan = {makespan:.1f}", color=COLORS["CCHIHH"], ha="right", va="bottom")

    if cloud_count > 0:
        ax.text(0, total_count - cloud_count / 2 - 0.5, "Cloud Servers", va="center", ha="left", fontsize=8)
    if edge_count > 0:
        ax.text(0, total_count - cloud_count - edge_count / 2 - 0.5, "Edge Servers", va="center", ha="left", fontsize=8)
    if device_count > 0:
        ax.text(0, device_count / 2 - 0.5, "Local Devices", va="center", ha="left", fontsize=8)

    ax.set_yticks(range(total_count))
    ax.set_yticklabels(lane_labels[::-1])
    ax.set_xlabel("Time")
    ax.set_ylabel("Resources")
    ax.set_title("Schedule Visualization of Best Solution on T200")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.grid(False, axis="y")

    legend_handles = [
        Line2D([0], [0], color=task_colors["CE"], lw=6, label="CE Tasks"),
        Line2D([0], [0], color=task_colors["MFG"], lw=6, label="Manufacturing Operations"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.text(0.5, 0.01, "Showing all CE tasks and the earliest 50 manufacturing operations for readability.", ha="center", fontsize=8)
    savefig(fig, "fig_gantt_T200.pdf")


def write_missing_data_note() -> None:
    lines = [
        "Skipped figures and missing data requirements",
        "",
        "All requested source data are now available locally.",
        "fig_diversity_curves_v2.pdf uses real decomposed diversity logs from outputs/results/diversity_logs/T500/.",
        "fig_gantt_T200.pdf uses real schedule export from outputs/results/schedule_exports/.",
    ]
    (FIG_DIR / "missing_figure_requirements.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    plot_ablation_heatmap()
    plot_radar_charts()
    plot_cd_diagram()
    plot_operator_selection_early_vs_late()
    plot_diversity_curves()
    plot_gantt_t200()
    write_missing_data_note()


if __name__ == "__main__":
    main()
