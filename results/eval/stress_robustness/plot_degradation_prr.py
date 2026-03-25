from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIGURE_PATH = ROOT / "figures" / "degradation_prr.pdf"

ALGORITHMS = {
    "CCHIHH": {
        "dir": "cchihh",
        "file_prefix": "CCHIHH",
        "color": "#E74C3C",
    },
    "DSAC-DE": {
        "dir": "dsac_de",
        "file_prefix": "DSAC_DE",
        "color": "#E67E22",
    },
    "CGA": {
        "dir": "cga",
        "file_prefix": "CGA",
        "color": "#3498DB",
    },
    "IMOMA": {
        "dir": "imoma",
        "file_prefix": "IMOMA",
        "color": "#2ECC71",
    },
}

DEGRADATIONS = [
    {
        "title": "(a) Cloud Reduction",
        "folder": "cloud_reduction",
        "levels": [0.1, 0.2, 0.3],
        "suffixes": ["r10", "r20", "r30"],
    },
    {
        "title": "(b) Edge Reduction",
        "folder": "edge_reduction",
        "levels": [0.1, 0.2, 0.3],
        "suffixes": ["r10", "r20", "r30"],
    },
    {
        "title": "(c) Device Reduction",
        "folder": "device_reduction",
        "levels": [0.1, 0.2, 0.3],
        "suffixes": ["r10", "r20", "r30"],
    },
    {
        "title": "(d) Comm. Inflation",
        "folder": "communication_inflation",
        "levels": [0.2, 0.4, 0.6],
        "suffixes": ["p20", "p40", "p60"],
    },
]


def load_last_best_fitness(csv_path: Path) -> float:
    frame = pd.read_csv(csv_path)
    if "best_fitness" not in frame.columns or frame.empty:
        raise ValueError(f"Invalid input file: {csv_path}")
    return float(frame["best_fitness"].iloc[-1])


def build_path(algo_meta: dict[str, str], folder: str, suffix: str | None) -> Path:
    base = ROOT / algo_meta["dir"] / "alpha0.5" / "T500"
    prefix = algo_meta["file_prefix"]
    if folder == "nominal":
        return base / "nominal" / "nominal" / f"{prefix}_T500_nominal_nominal_s1_eval.csv"
    return base / folder / suffix / f"{prefix}_T500_{folder}_{suffix}_s1_eval.csv"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    baseline_values = {
        algo: load_last_best_fitness(build_path(meta, "nominal", None))
        for algo, meta in ALGORITHMS.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for ax, degradation in zip(axes, DEGRADATIONS):
        for algo, meta in ALGORITHMS.items():
            prr_values = []
            baseline = baseline_values[algo]
            for suffix in degradation["suffixes"]:
                degraded = load_last_best_fitness(build_path(meta, degradation["folder"], suffix))
                prr_values.append(degraded / baseline)

            ax.plot(
                degradation["levels"],
                prr_values,
                color=meta["color"],
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=6,
                label=algo,
            )

        ax.axhline(
            y=1.0,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label="baseline",
        )
        ax.set_title(degradation["title"])
        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("Performance Retention Rate (closer to 1.0 = more robust)")
        ax.set_xticks(degradation["levels"], [f"{int(level * 100)}%" for level in degradation["levels"]])
        ax.grid(True, color="lightgray", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if legend_handles is None:
            legend_handles = handles
            legend_labels = labels

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
