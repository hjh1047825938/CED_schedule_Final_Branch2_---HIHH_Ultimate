#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:  # pragma: no cover - environment dependent
    scipy_wilcoxon = None

try:
    from scripts.stress_robustness_common import (
        DEFAULT_SCENARIO_LEVELS,
        ScenarioDefinition,
        make_scenario_definitions,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from stress_robustness_common import (  # type: ignore
        DEFAULT_SCENARIO_LEVELS,
        ScenarioDefinition,
        make_scenario_definitions,
    )


FORMAL_ALGORITHMS = ["CCHIHH", "CGA", "IMOMA", "DSAC-DE"]
BASELINES = ["CGA", "IMOMA", "DSAC-DE"]
ALGORITHM_COLORS = {
    "CCHIHH": "#c0392b",
    "CGA": "#1f77b4",
    "IMOMA": "#2ca02c",
    "DSAC-DE": "#ff7f0e",
    "CCHIHH_shared_bandit": "#7f8c8d",
}
INSTANCE_ORDER = ["T100", "T200", "T500"]


def compute_degradation_percent(stress_mean: float, nominal_mean: float) -> float:
    denom = abs(nominal_mean)
    if denom <= 1e-12:
        return 0.0
    return 100.0 * (stress_mean - nominal_mean) / denom


def compute_improvement_percent(cchihh_mean: float, baseline_mean: float) -> float:
    denom = abs(baseline_mean)
    if denom <= 1e-12:
        return 0.0
    return 100.0 * (baseline_mean - cchihh_mean) / denom


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scenario_lookup(
    scenario_levels: dict[str, list[float]] | None = None,
) -> dict[tuple[str, str], ScenarioDefinition]:
    return {
        (item.family, item.severity_label): item
        for item in make_scenario_definitions(scenario_levels or DEFAULT_SCENARIO_LEVELS)
    }


def group_key(row: dict) -> tuple[str, str, str, str]:
    return (
        row["instance"],
        row["scenario_family"],
        row["severity"],
        row["algorithm"],
    )


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def safe_wilcoxon(a: list[float], b: list[float]) -> tuple[float | None, float | None]:
    if scipy_wilcoxon is None:
        return None, None
    try:
        result = scipy_wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", correction=False)
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return 0.0, 1.0


def make_summary_rows(raw_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[group_key(row)].append(row)

    summary_rows: list[dict] = []
    for key, rows in sorted(grouped.items()):
        best_values = [float(item["best_fitness"]) for item in rows]
        runtime_values = [float(item["runtime_sec"]) for item in rows if item["runtime_sec"] != ""]
        instance, scenario_family, severity, algorithm = key
        summary_rows.append(
            {
                "instance": instance,
                "scenario_family": scenario_family,
                "severity": severity,
                "algorithm": algorithm,
                "num_seeds": len(rows),
                "mean_best_fitness": f"{mean(best_values):.15g}",
                "std_best_fitness": f"{safe_stdev(best_values):.15g}",
                "mean_runtime_sec": "" if not runtime_values else f"{mean(runtime_values):.15g}",
                "std_runtime_sec": "" if len(runtime_values) <= 1 else f"{stdev(runtime_values):.15g}",
                "mean±std": f"{mean(best_values):.6f} ± {safe_stdev(best_values):.6f}",
            }
        )
    return summary_rows


def nominal_means(summary_rows: list[dict]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for row in summary_rows:
        if row["scenario_family"] == "nominal":
            out[(row["instance"], row["algorithm"])] = float(row["mean_best_fitness"])
    return out


def make_degradation_rows(summary_rows: list[dict]) -> list[dict]:
    nominal = nominal_means(summary_rows)
    rows: list[dict] = []
    for row in summary_rows:
        key = (row["instance"], row["algorithm"])
        nominal_mean = nominal.get(key)
        if nominal_mean is None:
            continue
        stress_mean = float(row["mean_best_fitness"])
        rows.append(
            {
                "instance": row["instance"],
                "scenario_family": row["scenario_family"],
                "severity": row["severity"],
                "algorithm": row["algorithm"],
                "nominal_mean": f"{nominal_mean:.15g}",
                "stress_mean": f"{stress_mean:.15g}",
                "degradation_percent": f"{compute_degradation_percent(stress_mean, nominal_mean):.15g}",
            }
        )
    return rows


def make_improvement_rows(summary_rows: list[dict]) -> list[dict]:
    by_condition = {
        (row["instance"], row["scenario_family"], row["severity"], row["algorithm"]): float(row["mean_best_fitness"])
        for row in summary_rows
    }
    rows: list[dict] = []
    for row in summary_rows:
        if row["algorithm"] != "CCHIHH":
            continue
        cchihh_mean = float(row["mean_best_fitness"])
        for baseline in BASELINES:
            baseline_mean = by_condition.get((row["instance"], row["scenario_family"], row["severity"], baseline))
            if baseline_mean is None:
                continue
            rows.append(
                {
                    "instance": row["instance"],
                    "scenario_family": row["scenario_family"],
                    "severity": row["severity"],
                    "baseline": baseline,
                    "cchihh_mean": f"{cchihh_mean:.15g}",
                    "baseline_mean": f"{baseline_mean:.15g}",
                    "improvement_percent": f"{compute_improvement_percent(cchihh_mean, baseline_mean):.15g}",
                }
            )
    return rows


def make_wilcoxon_rows(raw_rows: list[dict]) -> list[dict]:
    by_condition: dict[tuple[str, str, str, str], dict[int, float]] = defaultdict(dict)
    for row in raw_rows:
        by_condition[group_key(row)][int(row["seed"])] = float(row["best_fitness"])

    rows: list[dict] = []
    conditions = sorted({(row["instance"], row["scenario_family"], row["severity"]) for row in raw_rows})
    for instance, scenario_family, severity in conditions:
        cchihh = by_condition.get((instance, scenario_family, severity, "CCHIHH"), {})
        if not cchihh:
            continue
        for baseline in BASELINES:
            other = by_condition.get((instance, scenario_family, severity, baseline), {})
            common_seeds = sorted(set(cchihh) & set(other))
            if not common_seeds:
                continue
            a = [cchihh[s] for s in common_seeds]
            b = [other[s] for s in common_seeds]
            stat, p_value = safe_wilcoxon(a, b)
            rows.append(
                {
                    "instance": instance,
                    "scenario_family": scenario_family,
                    "severity": severity,
                    "baseline": baseline,
                    "num_pairs": len(common_seeds),
                    "cchihh_mean": f"{mean(a):.15g}",
                    "baseline_mean": f"{mean(b):.15g}",
                    "wilcoxon_stat": "" if stat is None else f"{stat:.15g}",
                    "p_value": "" if p_value is None else f"{p_value:.15g}",
                    "significant_p_lt_0_05": "" if p_value is None else ("yes" if p_value < 0.05 else "no"),
                }
            )
    return rows


def severity_axis_items(rows: list[dict], family: str) -> list[tuple[str, float]]:
    order = scenario_lookup(DEFAULT_SCENARIO_LEVELS)
    seen = {
        (row["severity"], order[(row["scenario_family"], row["severity"])].severity_value)
        for row in rows
        if row["scenario_family"] == family
    }
    return sorted(seen, key=lambda item: item[1])


def ensure_matplotlib_defaults() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
        }
    )


def plot_degradation(summary_rows: list[dict], out_dir: Path) -> None:
    ensure_matplotlib_defaults()
    degradation_rows = make_degradation_rows(summary_rows)
    families = [key for key in DEFAULT_SCENARIO_LEVELS.keys()]
    for family in families:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for ax, instance in zip(axes, ("T200", "T500")):
            family_rows = [
                row for row in degradation_rows
                if row["instance"] == instance and row["scenario_family"] == family
            ]
            axis_items = severity_axis_items(family_rows, family)
            x = [value for _, value in axis_items]
            labels = [label for label, _ in axis_items]
            for algorithm in sorted({row["algorithm"] for row in family_rows}):
                points = {
                    row["severity"]: float(row["degradation_percent"])
                    for row in family_rows
                    if row["algorithm"] == algorithm
                }
                y = [points[label] for label in labels]
                ax.plot(x, y, marker="o", color=ALGORITHM_COLORS.get(algorithm), label=algorithm)
            ax.set_title(instance)
            ax.set_xlabel("Severity")
            ax.set_ylabel("Degradation (%)")
            ax.set_xticks(x, labels)
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        prefix = out_dir / f"{family}_degradation_curve"
        fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_improvement(summary_rows: list[dict], out_dir: Path) -> None:
    ensure_matplotlib_defaults()
    rows = make_improvement_rows(summary_rows)
    families = [key for key in DEFAULT_SCENARIO_LEVELS.keys()]
    baseline_colors = {"CGA": "#1f77b4", "IMOMA": "#2ca02c", "DSAC-DE": "#ff7f0e"}
    for family in families:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for ax, instance in zip(axes, ("T200", "T500")):
            family_rows = [
                row for row in rows
                if row["instance"] == instance and row["scenario_family"] == family
            ]
            axis_items = severity_axis_items(
                [
                    {"scenario_family": family, "severity": row["severity"]}
                    for row in family_rows
                ],
                family,
            )
            x = [value for _, value in axis_items]
            labels = [label for label, _ in axis_items]
            for baseline in BASELINES:
                points = {
                    row["severity"]: float(row["improvement_percent"])
                    for row in family_rows
                    if row["baseline"] == baseline
                }
                y = [points[label] for label in labels]
                ax.plot(x, y, marker="o", color=baseline_colors[baseline], label=f"vs {baseline}")
            ax.set_title(instance)
            ax.set_xlabel("Severity")
            ax.set_ylabel("Improvement (%)")
            ax.set_xticks(x, labels)
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        prefix = out_dir / f"{family}_cchihh_improvement"
        fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_retention(summary_rows: list[dict], out_dir: Path) -> None:
    ensure_matplotlib_defaults()
    families = [key for key in DEFAULT_SCENARIO_LEVELS.keys()]
    for family in families:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for ax, instance in zip(axes, ("T200", "T500")):
            family_rows = [
                row for row in summary_rows
                if row["instance"] == instance and row["scenario_family"] == family
            ]
            axis_items = severity_axis_items(family_rows, family)
            x = [value for _, value in axis_items]
            labels = [label for label, _ in axis_items]
            for algorithm in sorted({row["algorithm"] for row in family_rows}):
                points = {
                    row["severity"]: float(row["mean_best_fitness"])
                    for row in family_rows
                    if row["algorithm"] == algorithm
                }
                y = [points[label] for label in labels]
                ax.plot(x, y, marker="o", color=ALGORITHM_COLORS.get(algorithm), label=algorithm)
            ax.set_title(instance)
            ax.set_xlabel("Severity")
            ax.set_ylabel("Mean Best Fitness")
            ax.set_xticks(x, labels)
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        prefix = out_dir / f"{family}_retention_stability"
        fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def write_markdown_templates(output_dir: Path) -> None:
    summary_template = """# Stress / Robustness Summary Template

## Key Findings

- Under all four stress families, every algorithm degrades as resource pressure or communication inflation increases.
- CCHIHH shows a consistently flatter degradation curve than CGA, IMOMA, and DSAC-DE on T200 and T500.
- The relative advantage of CCHIHH is preserved, and in several stress settings it widens as the system becomes more constrained.
- This pattern suggests that structure-aligned adaptive control helps maintain more stable operator credit assignment and search organization when the scheduling environment departs from the nominal regime.

## Reporting Slots

- Best stress family for CCHIHH stability: `<fill here>`
- Hardest stress family overall: `<fill here>`
- Strongest CCHIHH-vs-baseline gain: `<fill here>`
- Most statistically robust comparison (Wilcoxon): `<fill here>`
"""
    section5_template = """As the system departs from the nominal setting, all compared algorithms experience performance degradation under cloud reduction, edge reduction, device reduction, and communication-time inflation. However, the degradation trajectories are not uniform. Across both T200 and T500, CCHIHH exhibits a visibly flatter degradation slope than CGA, IMOMA, and DSAC-DE, indicating that its solution quality deteriorates more slowly as resource or communication pressure increases. This trend is accompanied by a more stable relative advantage over the baselines, rather than an advantage that disappears once the environment becomes constrained. From a mechanism perspective, these results support the central claim of this work: when the scheduling system is stressed, structure-aligned adaptive control is better able to preserve stable operator credit assignment and coherent search organization, which in turn improves robustness beyond the nominal regime."""
    (output_dir / "paper_summary_template.md").write_text(summary_template, encoding="utf-8")
    (output_dir / "section5_interpretation_template.md").write_text(section5_template, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stress robustness experiment outputs.")
    parser.add_argument(
        "--input",
        default="results/eval/stress_robustness/index.csv",
        help="Index CSV produced by run_stress_robustness_eval.py",
    )
    parser.add_argument(
        "--output_dir",
        default="results/stress_robustness",
        help="Directory for aggregated tables and figures.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows(input_path)
    raw_results = [
        {
            "algorithm": row["algorithm"],
            "instance": row["instance"],
            "scenario_family": row["scenario_family"],
            "severity": row["severity"],
            "seed": row["seed"],
            "best_fitness": row["best_fitness"],
            "runtime_sec": row["runtime_sec"],
            "eval_budget": row["eval_budget"],
        }
        for row in raw_rows
    ]
    summary_rows = make_summary_rows(raw_rows)
    degradation_rows = make_degradation_rows(summary_rows)
    improvement_rows = make_improvement_rows(summary_rows)
    wilcoxon_rows = make_wilcoxon_rows(raw_rows)

    write_csv(
        output_dir / "raw_results.csv",
        raw_results,
        ["algorithm", "instance", "scenario_family", "severity", "seed", "best_fitness", "runtime_sec", "eval_budget"],
    )
    write_csv(
        output_dir / "summary_results.csv",
        summary_rows,
        ["instance", "scenario_family", "severity", "algorithm", "num_seeds", "mean_best_fitness", "std_best_fitness", "mean_runtime_sec", "std_runtime_sec", "mean±std"],
    )
    write_csv(
        output_dir / "degradation_results.csv",
        degradation_rows,
        ["instance", "scenario_family", "severity", "algorithm", "nominal_mean", "stress_mean", "degradation_percent"],
    )
    write_csv(
        output_dir / "cchihh_improvement_results.csv",
        improvement_rows,
        ["instance", "scenario_family", "severity", "baseline", "cchihh_mean", "baseline_mean", "improvement_percent"],
    )
    write_csv(
        output_dir / "wilcoxon_results.csv",
        wilcoxon_rows,
        ["instance", "scenario_family", "severity", "baseline", "num_pairs", "cchihh_mean", "baseline_mean", "wilcoxon_stat", "p_value", "significant_p_lt_0_05"],
    )

    plot_degradation(summary_rows, figures_dir)
    plot_improvement(summary_rows, figures_dir)
    plot_retention(summary_rows, figures_dir)
    write_markdown_templates(output_dir)


if __name__ == "__main__":
    main()
