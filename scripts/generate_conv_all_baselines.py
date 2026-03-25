from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Figures" / "conv_all_baselines.pdf"
SCALES = ["T100", "T200", "T500"]
SEEDS = range(1, 11)
PPO = {"T100": 0.351000, "T200": 0.041000, "T500": 0.031000}

COLORS = {
    "CCHIHH": "#d62728",
    "CGA": "#1f77b4",
    "IMOMA": "#2ca02c",
    "DSAC-DE": "#ff7f0e",
}


def load_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"best_fitness": "best_fitness", "eval_count": "eval_count"})[["eval_count", "best_fitness"]]


def cchihh_path(scale: str, seed: int) -> Path:
    if scale == "T500":
        preferred = ROOT / "results" / "eval" / "cchihh" / scale / "alpha0.5" / f"cchihh_full_{scale}_s{seed}_eval.csv"
        if preferred.exists():
            return preferred
    preferred = ROOT / "results" / "eval" / "gate" / scale / f"cchihh_full_{scale}_s{seed}_eval.csv"
    if preferred.exists():
        return preferred
    return ROOT / "results" / "eval" / "rerun_full" / "alpha0.5" / scale / f"cchihh_full_{scale}_s{seed}_eval.csv"


def baseline_path(algo: str, scale: str, seed: int) -> Path:
    if algo == "CGA":
        return ROOT / "results" / "eval" / "cga_baseline" / "alpha0.5" / scale / f"CGA_{scale}_s{seed}_eval.csv"
    if algo == "IMOMA":
        return ROOT / "results" / "eval" / "imoma_baseline" / "alpha0.5" / scale / f"IMOMA_{scale}_s{seed}_eval.csv"
    if algo == "DSAC-DE":
        return ROOT / "results" / "eval" / "dsac_de_multiscale" / "alpha0.5" / scale / f"DSAC_DE_{scale}_s{seed}_eval.csv"
    raise ValueError(algo)


def aggregate(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = [load_curve(path) for path in paths]
    grid = np.asarray(sorted({int(v) for df in curves for v in df["eval_count"].tolist() if v <= 400000}), dtype=float)
    matrix = []
    for df in curves:
        sub = df[df["eval_count"] <= 400000]
        xs = sub["eval_count"].to_numpy(dtype=float)
        ys = sub["best_fitness"].to_numpy(dtype=float)
        matrix.append(np.interp(grid, xs, ys))
    arr = np.asarray(matrix)
    mean = arr.mean(axis=0)
    ci = 1.96 * arr.std(axis=0, ddof=0) / np.sqrt(arr.shape[0])
    return grid, mean, ci


def plain_ticks() -> tuple[list[float], list[str]]:
    return [0, 100000, 200000, 300000, 400000], ["0", "100000", "200000", "300000", "400000"]


def main() -> None:
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xticks, xlabels = plain_ticks()
    algos = ["CCHIHH", "CGA", "IMOMA", "DSAC-DE"]
    for idx, (ax, scale) in enumerate(zip(axes, SCALES)):
        ax.set_title(scale)
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Weighted Objective")
        ax.set_xlim(0, 400000)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.grid(True, alpha=0.3, linewidth=0.6)

        for algo in algos:
            if algo == "CCHIHH":
                paths = [cchihh_path(scale, seed) for seed in SEEDS]
            else:
                paths = [baseline_path(algo, scale, seed) for seed in SEEDS]
            grid, mean, ci = aggregate(paths)
            color = COLORS[algo]
            ax.plot(grid, mean, color=color, linewidth=1.5, label=algo)
            ax.fill_between(grid, mean - ci, mean + ci, color=color, alpha=0.15)

        ax.axhline(PPO[scale], color="black", linestyle="--", linewidth=1.5, label="PPO (best policy)")
        if idx == 0:
            ax.legend(frameon=True)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT, format="pdf")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
