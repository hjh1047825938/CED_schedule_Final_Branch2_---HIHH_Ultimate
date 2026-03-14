import csv
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import friedmanchisquare
except ImportError:  # pragma: no cover
    friedmanchisquare = None


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
OUTPUT_PDF = FIGURES_DIR / "fig_cd_diagram_N9.pdf"
OUTPUT_TXT = FIGURES_DIR / "fig_cd_diagram_N9_summary.txt"

ALGORITHMS = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA", "PPO"]
ALPHA_VALUES = ["0.2", "0.5", "0.8"]
SCALES = ["T100", "T200", "T500"]
ALGO_MAP = {
    "CCHIHH": "CCHIHH_Full",
    "DSAC-DE": "DSAC_DE",
    "CGA": "CGA",
    "IMOMA": "IMOMA",
}
PPO_FALLBACK = {
    "T100_a02": 0.210,
    "T200_a02": 0.065,
    "T500_a02": 0.055,
    "T100_a05": 0.3889,
    "T200_a05": 0.0718,
    "T500_a05": 0.0575,
    "T100_a08": 0.496,
    "T200_a08": 0.068,
    "T500_a08": 0.048,
}
Q_ALPHA = 2.728
OLD_CD = 2.728


def read_summary_csv(alpha: str, scale: str) -> dict[str, float]:
    path = (
        ROOT
        / "outputs"
        / "results"
        / "alpha_sensitivity_cga_imoma_dsac"
        / "aggregated"
        / f"alpha_{alpha}"
        / scale
        / "final_performance_summary.csv"
    )
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["config"]: float(row["mean"]) for row in rows}


def build_results() -> tuple[dict[str, list[float]], list[str]]:
    results: dict[str, list[float]] = {}
    notes: list[str] = []
    for alpha in ALPHA_VALUES:
        for scale in SCALES:
            instance = f"{scale}_a{alpha.replace('.', '')}"
            exact_rows = read_summary_csv(alpha, scale)
            fitnesses = []
            for algo in ALGORITHMS:
                if algo == "PPO":
                    fitnesses.append(PPO_FALLBACK[instance])
                    if alpha in {"0.2", "0.8"}:
                        notes.append(
                            f"{instance}: PPO uses Figure 6 approximate endpoint value ({PPO_FALLBACK[instance]:.4f})"
                        )
                else:
                    fitnesses.append(exact_rows[ALGO_MAP[algo]])
            results[instance] = fitnesses
    return results, notes


def average_rank(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and math.isclose(arr[order[j]], arr[order[i]], rel_tol=0.0, abs_tol=1e-12):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def compute_rank_matrix(results: dict[str, list[float]]) -> np.ndarray:
    return np.array([average_rank(vals) for vals in results.values()], dtype=float)


def compute_friedman(rank_matrix: np.ndarray) -> tuple[float, float]:
    if friedmanchisquare is not None:
        stat, p_value = friedmanchisquare(*[rank_matrix[:, i] for i in range(rank_matrix.shape[1])])
        return float(stat), float(p_value)

    n, k = rank_matrix.shape
    r_bar = rank_matrix.mean(axis=0)
    stat = 12 * n / (k * (k + 1)) * float(np.sum(r_bar**2)) - 3 * n * (k + 1)
    return stat, float("nan")


def build_nonsig_pairs(sorted_items: list[tuple[str, float]], cd: float) -> list[tuple[str, str]]:
    out = []
    for (name_a, rank_a), (name_b, rank_b) in combinations(sorted_items, 2):
        if abs(rank_a - rank_b) < cd:
            out.append((name_a, name_b))
    return out


def build_cliques(sorted_items: list[tuple[str, float]], cd: float) -> list[list[str]]:
    cliques: list[list[str]] = []
    n = len(sorted_items)
    for i in range(n):
        for j in range(i + 1, n):
            if sorted_items[j][1] - sorted_items[i][1] < cd:
                names = [name for name, _ in sorted_items[i : j + 1]]
                if len(names) >= 2:
                    cliques.append(names)
    maximal: list[list[str]] = []
    for clique in cliques:
        if not any(set(clique) < set(other) for other in cliques):
            if clique not in maximal:
                maximal.append(clique)
    return maximal


def plot_cd_diagram(avg_ranks: dict[str, float], cd: float) -> list[list[str]]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    ordered = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    cliques = build_cliques(ordered, cd)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    y_axis = 0.55
    ax.hlines(y_axis, 1, 5, color="black", linewidth=1.0)
    for tick in range(1, 6):
        ax.vlines(tick, y_axis - 0.03, y_axis + 0.03, color="black", linewidth=1.0)
        ax.text(tick, y_axis - 0.08, f"{tick}", ha="center", va="top", fontsize=9)

    ax.text(1.0, y_axis + 0.10, "Better \u2190", ha="left", va="bottom", fontsize=9)
    ax.text(5.0, y_axis + 0.10, "\u2192 Worse", ha="right", va="bottom", fontsize=9)

    top_y = [0.86, 0.76, 0.86]
    bot_y = [0.22, 0.12]
    top_index = 0
    bot_index = 0
    for idx, (name, rank) in enumerate(ordered):
        is_top = idx % 2 == 0
        y_text = top_y[top_index] if is_top else bot_y[bot_index]
        if is_top:
            top_index += 1
        else:
            bot_index += 1
        ax.vlines(
            rank,
            y_axis,
            y_text - 0.03 if is_top else y_text + 0.03,
            color="gray",
            linewidth=1.0,
        )
        ax.text(
            rank,
            y_text,
            f"{name} ({rank:.3f})",
            ha="center",
            va="bottom" if is_top else "top",
            fontsize=10,
            fontweight="bold" if idx == 0 else "normal",
            color="black",
        )

    for idx, clique in enumerate(cliques):
        y_bar = 0.94 - idx * 0.05
        clique_ranks = [avg_ranks[name] for name in clique]
        x0, x1 = min(clique_ranks), max(clique_ranks)
        ax.hlines(y_bar, x0, x1, color="black", linewidth=2.0)
        ax.vlines([x0, x1], y_bar - 0.013, y_bar + 0.013, color="black", linewidth=1.0)

    ax.text(5.05, 0.94, f"CD = {cd:.3f}", ha="right", va="center", fontsize=9)
    ax.set_title("Critical Difference Diagram (Friedman-Nemenyi, α=0.05, N=9)", fontsize=11, pad=8)
    fig.text(
        0.5,
        0.03,
        "9 instances: 3 scales (T100/T200/T500) × 3 objective weights (α=0.2/0.5/0.8)",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return cliques


def main() -> None:
    results, notes = build_results()
    rank_matrix = compute_rank_matrix(results)
    avg_rank_values = rank_matrix.mean(axis=0)
    avg_ranks = {algo: float(avg_rank_values[i]) for i, algo in enumerate(ALGORITHMS)}
    friedman_stat, p_value = compute_friedman(rank_matrix)

    k = len(ALGORITHMS)
    n = len(results)
    cd = Q_ALPHA * math.sqrt(k * (k + 1) / (6 * n))

    ordered = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    nonsig_pairs = build_nonsig_pairs(ordered, cd)
    cliques = plot_cd_diagram(avg_ranks, cd)

    lines = [
        "CD Diagram N=9 Summary",
        "",
        "Data source:",
        "- CCHIHH, DSAC-DE, CGA, IMOMA: exact mean fitness from outputs/results/alpha_sensitivity_cga_imoma_dsac/aggregated/alpha_*/T*/final_performance_summary.csv",
        "- PPO: exact α=0.5 values from paper main results; α=0.2 and α=0.8 values from Figure 6 endpoint approximations supplied in prompt",
        "",
        "Per-instance mean fitness used:",
    ]
    for instance, values in results.items():
        joined = ", ".join(f"{algo}={value:.6f}" for algo, value in zip(ALGORITHMS, values))
        lines.append(f"- {instance}: {joined}")
    lines.extend(
        [
            "",
            "Average ranks:",
        ]
    )
    for name, rank in ordered:
        lines.append(f"- {name}: {rank:.3f}")

    lines.extend(
        [
            "",
            f"Friedman chi-square = {friedman_stat:.4f}",
            f"p-value = {p_value:.6f}" if not math.isnan(p_value) else "p-value = unavailable (SciPy missing)",
            f"CD = {cd:.4f}",
            "",
            "Algorithm pairs without significant difference (|Δrank| < CD):",
        ]
    )
    for name_a, name_b in nonsig_pairs:
        diff = abs(avg_ranks[name_a] - avg_ranks[name_b])
        lines.append(f"- {name_a} vs {name_b}: Δrank = {diff:.3f}")

    lines.extend(
        [
            "",
            "Non-significant connected groups drawn on the CD diagram:",
        ]
    )
    for clique in cliques:
        lines.append(f"- {' | '.join(clique)}")

    lines.extend(
        [
            "",
            "Notes:",
        ]
    )
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "Old vs new CD comparison:",
            f"- Old CD diagram (N=3): CD={OLD_CD:.3f}, 只能区分 top-2 vs PPO",
            f"- New CD diagram (N=9): CD={cd:.3f}, 预期能区分 CCHIHH/DSAC-DE vs IMOMA/PPO，且 CCHIHH vs CGA 仍不显著",
        ]
    )

    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved PDF: {OUTPUT_PDF}")
    print(f"Saved summary: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
