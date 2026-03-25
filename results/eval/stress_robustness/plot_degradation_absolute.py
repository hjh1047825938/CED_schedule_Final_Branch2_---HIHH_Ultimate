#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "results" / "eval" / "stress_robustness"
OUTPUT_PATH = Path(__file__).resolve().parent / "figures" / "degradation_absolute.pdf"

ALGORITHMS = {
    "CCHIHH": {"dir": "cchihh", "file": "CCHIHH", "color": "#E74C3C"},
    "DSAC-DE": {"dir": "dsac_de", "file": "DSAC_DE", "color": "#E67E22"},
    "CGA": {"dir": "cga", "file": "CGA", "color": "#3498DB"},
    "IMOMA": {"dir": "imoma", "file": "IMOMA", "color": "#2ECC71"},
}

PANELS = [
    {
        "key": "cloud_reduction",
        "title": "(a) Cloud Reduction",
        "levels": [("r10", 0.1, "10%"), ("r20", 0.2, "20%"), ("r30", 0.3, "30%")],
    },
    {
        "key": "edge_reduction",
        "title": "(b) Edge Reduction",
        "levels": [("r10", 0.1, "10%"), ("r20", 0.2, "20%"), ("r30", 0.3, "30%")],
    },
    {
        "key": "device_reduction",
        "title": "(c) Device Reduction",
        "levels": [("r10", 0.1, "10%"), ("r20", 0.2, "20%"), ("r30", 0.3, "30%")],
    },
    {
        "key": "communication_inflation",
        "title": "(d) Comm. Inflation",
        "levels": [("p20", 0.2, "20%"), ("p40", 0.4, "40%"), ("p60", 0.6, "60%")],
    },
]


def load_final_best_fitness(csv_path: Path) -> float:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")
    if "best_fitness" not in rows[-1]:
        raise KeyError(f"Column 'best_fitness' not found in {csv_path}")
    return float(rows[-1]["best_fitness"])


def build_path(algo_dir: str, algo_file: str, panel_key: str, level_key: str) -> Path:
    return (
        DATA_ROOT
        / algo_dir
        / "alpha0.5"
        / "T500"
        / panel_key
        / level_key
        / f"{algo_file}_T500_{panel_key}_{level_key}_s1_eval.csv"
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    legend_handles = None

    for ax, panel in zip(axes, PANELS):
        x_positions = list(range(len(panel["levels"])))
        x_labels = [label for _, _, label in panel["levels"]]

        for algo_name, meta in ALGORITHMS.items():
            y_values = []
            for level_key, _, _ in panel["levels"]:
                csv_path = build_path(meta["dir"], meta["file"], panel["key"], level_key)
                y_values.append(load_final_best_fitness(csv_path))

            line, = ax.plot(
                x_positions,
                y_values,
                color=meta["color"],
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=6,
                label=algo_name,
            )
            if legend_handles is None:
                legend_handles = []
            if len(legend_handles) < len(ALGORITHMS):
                legend_handles.append(line)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_title(panel["title"])
        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("Best Fitness")
        ax.grid(True, color="lightgray", alpha=0.3)

    fig.legend(
        handles=legend_handles,
        labels=list(ALGORITHMS.keys()),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
