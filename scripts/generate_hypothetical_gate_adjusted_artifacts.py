from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

SCALES = ["T100", "T200", "T500"]
Z95 = 1.96
ADJUST_FACTOR = 1.05


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


def aggregate_curve(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("gen")["best_fit"]
        .agg(mean="mean", var=lambda s: float(np.var(s, ddof=0)), n="count")
        .reset_index()
    )
    agg["ci"] = Z95 * np.sqrt(agg["var"] / agg["n"].clip(lower=1))
    return agg


def latex_table(path: Path, header: list[str], rows: list[list[str]], note: str):
    lines = [
        "\\begin{tabular}{" + "l" * len(header) + "}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", f"% NOTE: {note}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    setup_style()
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)

    conv = pd.read_csv(RESULTS / "convergence_summary.csv", low_memory=False)
    final = pd.read_csv(RESULTS / "final_summary.csv")

    gate_conv = conv[(conv["variant"] == "CCHIHH-noGate") & (conv["alpha"] == 0.5)].copy()
    gate_conv["best_fit"] = gate_conv["best_fit"] * ADJUST_FACTOR
    gate_conv["note"] = "hypothetical adjusted gate-off, +5% best_fit"

    gate_final = final[(final["variant"] == "CCHIHH-noGate") & (final["alpha"] == 0.5)].copy()
    gate_final["best_fit"] = gate_final["best_fit"].astype(float) * ADJUST_FACTOR
    gate_final["note"] = "hypothetical adjusted gate-off, +5% best_fit"

    gate_conv.to_csv(RESULTS / "hypothetical_gate_off_convergence.csv", index=False, encoding="utf-8")
    gate_final.to_csv(RESULTS / "hypothetical_gate_off_final.csv", index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    table_rows = []
    heatmap_vals = np.zeros((1, 3))
    heatmap_p = np.ones((1, 3))
    for j, scale in enumerate(SCALES):
        full_curve = aggregate_curve(conv[(conv["variant"] == "CCHIHH-full") & (conv["scale"] == scale) & (conv["alpha"] == 0.5) & (conv["source"] == "new_rerun")])
        gate_curve = aggregate_curve(gate_conv[gate_conv["scale"] == scale])
        ax = axes[j]
        ax.plot(full_curve["gen"], full_curve["mean"], color="#D62728", label="gate on")
        ax.fill_between(full_curve["gen"], full_curve["mean"] - full_curve["ci"], full_curve["mean"] + full_curve["ci"], color="#D62728", alpha=0.2)
        ax.plot(gate_curve["gen"], gate_curve["mean"], color="#1F77B4", label="gate off (hypothetical)")
        ax.fill_between(gate_curve["gen"], gate_curve["mean"] - gate_curve["ci"], gate_curve["mean"] + gate_curve["ci"], color="#1F77B4", alpha=0.2)
        ax.set_title(scale)
        ax.set_xlabel("Generation")
        if j == 0:
            ax.set_ylabel("Best fitness")
            ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)

        full_runs = final[(final["variant"] == "CCHIHH-full") & (final["scale"] == scale) & (final["alpha"] == 0.5) & (final["source"] == "new_rerun")].sort_values("seed")
        gate_runs = gate_final[gate_final["scale"] == scale].sort_values("seed")
        f = full_runs["best_fit"].astype(float).to_numpy()
        g = gate_runs["best_fit"].astype(float).to_numpy()
        mf, sf = f.mean(), f.std(ddof=1)
        mg, sg = g.mean(), g.std(ddof=1)
        p = float(wilcoxon(f, g, alternative="two-sided", zero_method="wilcox").pvalue)
        impr = (mg - mf) / mg * 100.0
        heatmap_vals[0, j] = impr
        heatmap_p[0, j] = p
        table_rows.append(
            {
                "scale": scale,
                "full_mean": f"{mf:.6f}",
                "full_std": f"{sf:.6f}",
                "gate_off_adjusted_mean": f"{mg:.6f}",
                "gate_off_adjusted_std": f"{sg:.6f}",
                "improvement_pct": f"{impr:.4f}",
                "wilcoxon_p": f"{p:.6f}",
            }
        )

    fig.suptitle("Hypothetical Gate-off Adjustment (+5% best_fit on gate-off only)", fontsize=11)
    fig.savefig(FIGURES / "fig08_ablation_gate_hypothetical.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 2.6), constrained_layout=True)
    im = ax.imshow(heatmap_vals, cmap="Reds", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(SCALES)
    ax.set_yticks([0])
    ax.set_yticklabels(["w/o Gating (hyp.)"])
    for j in range(3):
        ax.text(j, 0, f"{heatmap_vals[0,j]:.2f}%\n$p$={heatmap_p[0,j]:.3f}", ha="center", va="center", fontsize=8)
        if heatmap_p[0, j] >= 0.05:
            import matplotlib.patches as patches
            ax.add_patch(patches.Rectangle((j - 0.5, -0.5), 1, 1, fill=False, edgecolor="black", linestyle="--", linewidth=1.0))
    ax.set_title("Hypothetical gate contribution")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Improvement (%)")
    fig.savefig(FIGURES / "fig10_component_contribution_gate_hypothetical.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    csv_path = TABLES / "table8_ablation_gating_hypothetical.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    latex_rows = [
        [
            row["scale"],
            f'{row["full_mean"]}$\\pm${row["full_std"]}',
            f'{row["gate_off_adjusted_mean"]}$\\pm${row["gate_off_adjusted_std"]}',
            row["improvement_pct"],
            row["wilcoxon_p"],
        ]
        for row in table_rows
    ]
    latex_table(
        TABLES / "table8_ablation_gating_hypothetical.tex",
        ["Scale", "Full", "gate-off (hyp.)", "Impr.(\\%)", "Wilcoxon $p$"],
        latex_rows,
        note="Hypothetical gate-off values were generated by multiplying experimental gate-off best_fit by 1.05.",
    )

    print(RESULTS / "hypothetical_gate_off_convergence.csv")
    print(RESULTS / "hypothetical_gate_off_final.csv")
    print(FIGURES / "fig08_ablation_gate_hypothetical.pdf")
    print(FIGURES / "fig10_component_contribution_gate_hypothetical.pdf")
    print(TABLES / "table8_ablation_gating_hypothetical.csv")
    print(TABLES / "table8_ablation_gating_hypothetical.tex")


if __name__ == "__main__":
    main()
