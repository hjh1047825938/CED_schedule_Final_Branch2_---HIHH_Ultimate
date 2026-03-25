from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
LINE_FIG_OUT = ROOT / "figures" / "cchihh_over_shared_bandit_by_size.pdf"
BAR_FIG_OUT = ROOT / "figures" / "cchihh_over_shared_bandit_by_size_bar.pdf"
TABLE_OUT = ROOT / "tables" / "table_cchihh_over_shared_bandit_by_size.tex"
CSV_OUT = ROOT / "tables" / "table_cchihh_over_shared_bandit_by_size.csv"
SCALES = ["T50", "T100", "T200", "T300", "T400", "T500"]
SEEDS = range(1, 11)


def load_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["eval_count", "best_fitness"]].sort_values("eval_count")


def cchihh_path(scale: str, seed: int) -> Path:
    if scale in {"T50", "T300", "T400"}:
        path = ROOT / "results" / "eval" / "cchihh" / "alpha0.5" / scale / f"cchihh_full_{scale}_s{seed}_eval.csv"
        if path.exists():
            return path
    if scale == "T500":
        path = ROOT / "results" / "eval" / "cchihh" / scale / "alpha0.5" / f"cchihh_full_{scale}_s{seed}_eval.csv"
        if path.exists():
            return path
    path = ROOT / "results" / "eval" / "gate" / scale / f"cchihh_full_{scale}_s{seed}_eval.csv"
    if path.exists():
        return path
    return ROOT / "results" / "eval" / "rerun_full" / "alpha0.5" / scale / f"cchihh_full_{scale}_s{seed}_eval.csv"


def shared_path(scale: str, seed: int) -> Path:
    return ROOT / "results" / "eval" / "sharedbandit" / "alpha0.5" / scale / f"cchihh_shared_bandit_{scale}_s{seed}_eval.csv"


def final_values(paths: list[Path]) -> np.ndarray:
    vals = []
    for path in paths:
        df = load_curve(path)
        sub = df[df["eval_count"] <= 400000]
        vals.append(float(sub["best_fitness"].iloc[-1]))
    return np.asarray(vals, dtype=float)


def collect_stats() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for scale in SCALES:
        c_vals = final_values([cchihh_path(scale, seed) for seed in SEEDS])
        s_vals = final_values([shared_path(scale, seed) for seed in SEEDS])
        c_mean = float(c_vals.mean())
        c_std = float(c_vals.std(ddof=0))
        s_mean = float(s_vals.mean())
        s_std = float(s_vals.std(ddof=0))
        improvement = (s_mean - c_mean) / c_mean * 100.0
        paired_improvement = (s_vals - c_vals) / c_vals * 100.0
        improvement_ci95 = float(
            1.96 * paired_improvement.std(ddof=0) / np.sqrt(len(paired_improvement))
        )
        try:
            p_value = float(wilcoxon(c_vals, s_vals).pvalue)
        except ValueError:
            p_value = float("nan")
        rows.append(
            {
                "scale": scale,
                "problem_size": int(scale[1:]),
                "cchihh_mean": c_mean,
                "cchihh_std": c_std,
                "shared_mean": s_mean,
                "shared_std": s_std,
                "relative_improvement_pct": improvement,
                "relative_improvement_ci95": improvement_ci95,
                "p_value": p_value,
            }
        )
    return rows


def build_line_plot(rows: list[dict[str, float]]) -> None:
    LINE_FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    sizes = [row["problem_size"] for row in rows]
    improvements = [row["relative_improvement_pct"] for row in rows]
    ci95 = [row["relative_improvement_ci95"] for row in rows]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(
        sizes,
        improvements,
        yerr=ci95,
        color="#d62728",
        linewidth=1.8,
        marker="o",
        markersize=5.5,
        capsize=3.5,
        elinewidth=1.0,
        label="CCHIHH over Shared Bandit",
    )
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Relative Improvement (%)")
    ax.set_xticks(sizes)
    ax.set_xticklabels([row["scale"] for row in rows])
    pad = max(1.0, 0.08 * (max(improvements) - min(improvements) if len(improvements) > 1 else 1.0))
    ax.set_ylim(min(i - c for i, c in zip(improvements, ci95)) - pad, max(i + c for i, c in zip(improvements, ci95)) + pad)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(LINE_FIG_OUT, format="pdf")
    plt.close(fig)


def build_bar_plot(rows: list[dict[str, float]]) -> None:
    BAR_FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["scale"] for row in rows]
    improvements = [row["relative_improvement_pct"] for row in rows]
    ci95 = [row["relative_improvement_ci95"] for row in rows]
    x = np.arange(len(labels))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(
        x,
        improvements,
        width=0.62,
        color="#d62728",
        alpha=0.85,
        edgecolor="#8c1d18",
        linewidth=0.8,
        label="CCHIHH over Shared Bandit",
    )
    ax.errorbar(
        x,
        improvements,
        yerr=ci95,
        fmt="none",
        ecolor="#8c1d18",
        elinewidth=1.0,
        capsize=3.5,
        capthick=1.0,
        zorder=3,
    )
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Relative Improvement (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    pad = max(1.0, 0.08 * (max(improvements) - min(improvements) if len(improvements) > 1 else 1.0))
    ax.set_ylim(min(0.0, min(i - c for i, c in zip(improvements, ci95))) - pad, max(i + c for i, c in zip(improvements, ci95)) + pad)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.legend(frameon=True)
    for bar, val in zip(bars, improvements):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.25 if y >= 0 else -0.25
        ax.text(bar.get_x() + bar.get_width() / 2.0, y + offset, f"{val:.2f}", ha="center", va=va, fontsize=9)
    fig.tight_layout()
    fig.savefig(BAR_FIG_OUT, format="pdf")
    plt.close(fig)


def write_tables(rows: list[dict[str, float]]) -> None:
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    tex_lines = [
        "% CCHIHH over Shared Bandit by problem size",
        "% Scale & CCHIHH & Shared Bandit & Relative Improvement(%) & p-value \\\\",
    ]
    csv_lines = [
        "scale,problem_size,cchihh_mean,cchihh_std,shared_mean,shared_std,relative_improvement_pct,relative_improvement_ci95,p_value"
    ]
    for row in rows:
        tex_lines.append(
            f"{row['scale']} & ${row['cchihh_mean']:.6f} \\pm {row['cchihh_std']:.6f}$ & "
            f"${row['shared_mean']:.6f} \\pm {row['shared_std']:.6f}$ & "
            f"{row['relative_improvement_pct']:.2f} $\\pm$ {row['relative_improvement_ci95']:.2f} & {row['p_value']:.6f} \\\\"
        )
        csv_lines.append(
            f"{row['scale']},{row['problem_size']},{row['cchihh_mean']:.6f},{row['cchihh_std']:.6f},"
            f"{row['shared_mean']:.6f},{row['shared_std']:.6f},{row['relative_improvement_pct']:.2f},{row['relative_improvement_ci95']:.2f},{row['p_value']:.6f}"
        )
    TABLE_OUT.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    CSV_OUT.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_stats()
    build_line_plot(rows)
    build_bar_plot(rows)
    write_tables(rows)
    print(LINE_FIG_OUT)
    print(BAR_FIG_OUT)
    print(TABLE_OUT)
    print(CSV_OUT)


if __name__ == "__main__":
    main()
