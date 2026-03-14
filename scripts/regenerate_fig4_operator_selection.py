from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "results" / "cchihh_ablation_suite" / "bandit" / "T500"
FIG_DIR = ROOT / "figures"

FILE_GLOB = "bandit_adaptive_opstats_seed*.csv"
OUTPUT_STEM = "fig4_operator_selection_T500"

BLOCK_SPECS = [
    (
        "Offload Block",
        [
            ("offload_GA", "GA"),
            ("offload_DE", "DE"),
            ("offload_BITFLIP", "BITFLIP"),
            ("offload_RESAMPLE", "RESAMPLE"),
        ],
    ),
    (
        "Sequence Block",
        [
            ("seq_GA", "GA"),
            ("seq_SWAP", "SWAP"),
            ("seq_VNS", "VNS"),
            ("seq_RESAMPLE", "RESAMPLE"),
        ],
    ),
    (
        "Device Block",
        [
            ("dev_DE", "DE"),
            ("dev_GDE", "GDE"),
            ("dev_LEVY", "LEVY"),
            ("dev_RESAMPLE", "RESAMPLE"),
        ],
    ),
]

LINE_STYLES = {
    "GA": "-",
    "DE": "--",
    "BITFLIP": "-.",
    "RESAMPLE": ":",
    "SWAP": "--",
    "VNS": "-.",
    "GDE": "--",
    "LEVY": "-.",
}

COLORS = {
    "GA": "#1f77b4",
    "DE": "#2ca02c",
    "BITFLIP": "#808000",
    "RESAMPLE": "#d62728",
    "SWAP": "#2ca02c",
    "VNS": "#808000",
    "GDE": "#2ca02c",
    "LEVY": "#808000",
}

SMOOTH_WINDOW = 2  # Logged every 50 generations; this yields a 100-generation rolling mean.


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_mean_curve() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob(FILE_GLOB))
    if not files:
        raise SystemExit(f"No input files matched {INPUT_DIR / FILE_GLOB}")

    frames = [pd.read_csv(path) for path in files]
    gens_ref = frames[0]["gen"].tolist()
    for path, frame in zip(files[1:], frames[1:]):
        gens = frame["gen"].tolist()
        if gens != gens_ref:
            raise SystemExit(f"Generation mismatch in {path}")

    df = pd.concat(frames, ignore_index=True).groupby("gen", as_index=False).mean(numeric_only=True)
    value_columns = [col for col in df.columns if col != "gen"]
    df[value_columns] = df[value_columns].rolling(window=SMOOTH_WINDOW, min_periods=1, center=True).mean()
    return df


def plot(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (title, series_specs) in zip(axes, BLOCK_SPECS):
        for column, label in series_specs:
            ax.plot(
                df["gen"],
                df[column].clip(lower=0.0, upper=1.0),
                label=label,
                color=COLORS[label],
                linestyle=LINE_STYLES[label],
            )
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_xlim(df["gen"].min(), df["gen"].max())
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="0.75")
        ax.grid(False)

    axes[0].set_ylabel("Selection Probability")
    fig.tight_layout()

    pdf_path = FIG_DIR / f"{OUTPUT_STEM}.pdf"
    png_path = FIG_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    setup_style()
    df = load_mean_curve()
    plot(df)


if __name__ == "__main__":
    main()
