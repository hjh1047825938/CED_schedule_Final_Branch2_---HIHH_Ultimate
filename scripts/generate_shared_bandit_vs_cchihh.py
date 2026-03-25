from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = ROOT / "figures" / "conv_shared_bandit_vs_cchihh.pdf"
TABLE_OUT = ROOT / "tables" / "table_shared_bandit_vs_cchihh.tex"
CSV_OUT = ROOT / "tables" / "table_shared_bandit_vs_cchihh.csv"
SCALES = ["T100", "T200", "T500"]
SEEDS = range(1, 11)

COLORS = {
    "CCHIHH": "#d62728",
    "Shared Bandit": "#6a3d9a",
}


def load_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["eval_count", "best_fitness"]].sort_values("eval_count")


def cchihh_path(scale: str, seed: int) -> Path:
    if scale == "T500":
        preferred = ROOT / "results" / "eval" / "cchihh" / scale / "alpha0.5" / f"cchihh_full_{scale}_s{seed}_eval.csv"
        if preferred.exists():
            return preferred
    preferred = ROOT / "results" / "eval" / "gate" / scale / f"cchihh_full_{scale}_s{seed}_eval.csv"
    if preferred.exists():
        return preferred
    return ROOT / "results" / "eval" / "rerun_full" / "alpha0.5" / scale / f"cchihh_full_{scale}_s{seed}_eval.csv"


def shared_path(scale: str, seed: int) -> Path:
    return ROOT / "results" / "eval" / "sharedbandit" / "alpha0.5" / scale / f"cchihh_shared_bandit_{scale}_s{seed}_eval.csv"


def aggregate(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curves = [load_curve(path) for path in paths]
    grid = np.asarray(sorted({int(v) for df in curves for v in df["eval_count"].tolist() if v <= 400000}), dtype=float)
    matrix = []
    finals = []
    for df in curves:
        sub = df[df["eval_count"] <= 400000]
        xs = sub["eval_count"].to_numpy(dtype=float)
        ys = sub["best_fitness"].to_numpy(dtype=float)
        matrix.append(np.interp(grid, xs, ys))
        finals.append(float(ys[-1]))
    arr = np.asarray(matrix)
    mean = arr.mean(axis=0)
    ci = 1.96 * arr.std(axis=0, ddof=0) / np.sqrt(arr.shape[0])
    return grid, mean, ci, np.asarray(finals, dtype=float)


def plain_ticks() -> tuple[list[int], list[str]]:
    return [0, 100000, 200000, 300000, 400000], ["0", "100000", "200000", "300000", "400000"]


def build_plot() -> dict[str, dict[str, np.ndarray]]:
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
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xticks, xlabels = plain_ticks()
    stats: dict[str, dict[str, np.ndarray]] = {}
    for idx, (ax, scale) in enumerate(zip(axes, SCALES)):
        c_grid, c_mean, c_ci, c_finals = aggregate([cchihh_path(scale, seed) for seed in SEEDS])
        s_grid, s_mean, s_ci, s_finals = aggregate([shared_path(scale, seed) for seed in SEEDS])
        stats[scale] = {"cchihh": c_finals, "shared": s_finals}

        ax.plot(c_grid, c_mean, color=COLORS["CCHIHH"], linewidth=1.5, label="CCHIHH")
        ax.fill_between(c_grid, c_mean - c_ci, c_mean + c_ci, color=COLORS["CCHIHH"], alpha=0.15)
        ax.plot(s_grid, s_mean, color=COLORS["Shared Bandit"], linewidth=1.5, label="Shared Bandit")
        ax.fill_between(s_grid, s_mean - s_ci, s_mean + s_ci, color=COLORS["Shared Bandit"], alpha=0.15)

        ax.set_title(scale)
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Weighted Objective")
        ax.set_xlim(0, 400000)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.grid(True, alpha=0.3, linewidth=0.6)
        if idx == 0:
            ax.legend(frameon=True)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG_OUT, format="pdf")
    plt.close(fig)
    return stats


def write_tables(stats: dict[str, dict[str, np.ndarray]]) -> None:
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    tex_lines = ["% Shared bandit vs CCHIHH", "% Scale & CCHIHH & Shared Bandit & Improvement(%) & p-value \\\\"]
    csv_lines = ["scale,cchihh_mean,cchihh_std,shared_mean,shared_std,improvement_pct,p_value"]
    for scale in SCALES:
        c = stats[scale]["cchihh"]
        s = stats[scale]["shared"]
        c_mean, c_std = float(c.mean()), float(c.std(ddof=0))
        s_mean, s_std = float(s.mean()), float(s.std(ddof=0))
        improvement = (s_mean - c_mean) / c_mean * 100.0
        try:
            p_value = float(wilcoxon(c, s).pvalue)
        except ValueError:
            p_value = float("nan")
        tex_lines.append(f"{scale} & ${c_mean:.6f} \\pm {c_std:.6f}$ & ${s_mean:.6f} \\pm {s_std:.6f}$ & {improvement:.2f} & {p_value:.6f} \\\\")
        csv_lines.append(f"{scale},{c_mean:.6f},{c_std:.6f},{s_mean:.6f},{s_std:.6f},{improvement:.2f},{p_value:.6f}")
    TABLE_OUT.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    CSV_OUT.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def main() -> None:
    stats = build_plot()
    write_tables(stats)
    print(FIG_OUT)
    print(TABLE_OUT)
    print(CSV_OUT)


if __name__ == "__main__":
    main()
