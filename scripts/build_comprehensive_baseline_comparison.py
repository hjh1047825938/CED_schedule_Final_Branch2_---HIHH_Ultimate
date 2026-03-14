from pathlib import Path
import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Data are taken from the local paper tables:
# - tables/table9_vs_cga.tex
# - tables/table10_vs_imoma.tex
# - tables/table11_vs_ppo.tex
# - tables/table12_vs_dsac_de.tex
# GA / DE / Gbest-DE are intentionally omitted here because this repository
# does not contain a same-scale, same-metric summary aligned with those tables.
ALGORITHMS = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA", "PPO"]
COLORS = {
    "CCHIHH": "#DC2626",
    "DSAC-DE": "#2563EB",
    "CGA": "#059669",
    "IMOMA": "#D97706",
    "PPO": "#7C3AED",
}
DATA = {
    "T100": {
        "CCHIHH": {"mean": 0.341471, "std": 0.001180, "p": None},
        "DSAC-DE": {"mean": 0.324979, "std": 0.001958, "p": 0.001953},
        "CGA": {"mean": 0.353245, "std": 0.001514, "p": 0.001953},
        "IMOMA": {"mean": 0.377472, "std": 0.001463, "p": 0.001953},
        "PPO": {"mean": 0.351000, "std": 0.000000, "p": 0.001953},
    },
    "T200": {
        "CCHIHH": {"mean": 0.036077, "std": 0.002096, "p": None},
        "DSAC-DE": {"mean": 0.047121, "std": 0.001611, "p": 0.001953},
        "CGA": {"mean": 0.049487, "std": 0.001094, "p": 0.001953},
        "IMOMA": {"mean": 0.057598, "std": 0.001613, "p": 0.001953},
        "PPO": {"mean": 0.041000, "std": 0.000000, "p": 0.001953},
    },
    "T500": {
        "CCHIHH": {"mean": 0.028134, "std": 0.006629, "p": None},
        "DSAC-DE": {"mean": 0.038472, "std": 0.012917, "p": 0.001953},
        "CGA": {"mean": 0.039213, "std": 0.000877, "p": 0.003906},
        "IMOMA": {"mean": 0.046502, "std": 0.001374, "p": 0.001953},
        "PPO": {"mean": 0.031000, "std": 0.000000, "p": 0.083984},
    },
}


def calc_improvement(cchihh_val: float, baseline_val: float) -> float:
    return (baseline_val - cchihh_val) / baseline_val * 100.0


def sig_marker(p_value: float | None) -> str:
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Scale", "Algorithm", "Mean", "Std", "Improvement_vs_CCHIHH_pct", "Wilcoxon_p"])
        for scale, scale_data in DATA.items():
            cchihh_mean = scale_data["CCHIHH"]["mean"]
            for algo in ALGORITHMS:
                entry = scale_data[algo]
                improvement = ""
                if algo != "CCHIHH":
                    improvement = f"{calc_improvement(cchihh_mean, entry['mean']):.2f}"
                writer.writerow([scale, algo, entry["mean"], entry["std"], improvement, entry["p"]])


def main() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 4.0))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.89, bottom=0.27, wspace=0.30)

    for ax_idx, (scale, scale_data) in enumerate(DATA.items()):
        ax = axes[ax_idx]
        x = np.arange(len(ALGORITHMS))
        means = [scale_data[a]["mean"] for a in ALGORITHMS]
        stds = [scale_data[a]["std"] for a in ALGORITHMS]
        colors = [COLORS[a] for a in ALGORITHMS]

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            width=0.66,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            capsize=3,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            zorder=3,
        )
        bars[0].set_hatch("///")
        bars[0].set_linewidth(1.2)
        bars[0].set_edgecolor("#7F1D1D")

        y_min = min(means) * 0.93
        y_max = max(m + s for m, s in zip(means, stds)) * 1.16
        if scale == "T500":
            y_max *= 1.03
        ax.set_ylim(y_min, y_max)

        cchihh_mean = scale_data["CCHIHH"]["mean"]
        label_offset = (y_max - y_min) * 0.018

        ax.text(
            x[0],
            means[0] + stds[0] + label_offset,
            "Ours",
            ha="center",
            va="bottom",
            fontsize=6,
            color=COLORS["CCHIHH"],
            fontweight="bold",
            fontstyle="italic",
        )

        for i, algo in enumerate(ALGORITHMS[1:], start=1):
            improvement = calc_improvement(cchihh_mean, scale_data[algo]["mean"])
            marker = "↓" if improvement > 0 else "↑"
            text_color = "#059669" if improvement > 0 else "#DC2626"
            label = f"{marker}{abs(improvement):.1f}%"
            p_value = scale_data[algo]["p"]
            significance = sig_marker(p_value)
            if significance and significance != "ns":
                label = f"{label} {significance}"
            ax.text(
                x[i],
                means[i] + stds[i] + label_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=6,
                color=text_color,
                fontweight="bold",
                rotation=0,
            )

        ax.axhline(cchihh_mean, color=COLORS["CCHIHH"], linestyle="--", linewidth=0.8, alpha=0.4, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(ALGORITHMS, rotation=28, ha="right")
        ax.set_title(f"({chr(97 + ax_idx)}) {scale}", fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Best Scalarized Fitness")
        ax.grid(axis="y", alpha=0.28, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

    csv_path = OUT_DIR / "fig_comprehensive_baseline_comparison.csv"
    pdf_path = OUT_DIR / "fig_comprehensive_baseline_comparison.pdf"
    png_path = OUT_DIR / "fig_comprehensive_baseline_comparison.png"
    write_csv(csv_path)
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved {csv_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print("Note: rendered 5 algorithms only because GA/DE/Gbest-DE do not have a same-metric local summary aligned with Tables 9-12.")


if __name__ == "__main__":
    main()
