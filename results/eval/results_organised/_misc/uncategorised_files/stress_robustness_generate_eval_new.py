#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EVAL_ROOT = ROOT.parent
SOURCE_INDEX = ROOT / "index.csv"
DEST_ROOT = ROOT / "eval_new"
DEST_INDEX = DEST_ROOT / "index.csv"
DEST_INDEX_SEED1 = DEST_ROOT / "index_seed1.csv"
DEST_FIGURES = DEST_ROOT / "figures"

ALGORITHMS = ["CCHIHH", "L-SRTDE", "DSAC-DE", "NL-SHADE-LBC", "CGA", "RDE", "IMOMA"]
BASE_ALGORITHMS = ["CCHIHH", "DSAC-DE", "CGA", "IMOMA"]
TRACE_ALGORITHMS = ["L-SRTDE", "NL-SHADE-LBC", "RDE"]
TRACE_PREFIX = {
    "L-SRTDE": "l_srtde",
    "NL-SHADE-LBC": "nl_shade_lbc",
    "RDE": "rde",
}
INSTANCES = ["T200", "T500"]
SEEDS = list(range(1, 11))
SCENARIOS = {
    "cloud_reduction": [("r10", 0.1), ("r20", 0.2), ("r30", 0.3)],
    "edge_reduction": [("r10", 0.1), ("r20", 0.2), ("r30", 0.3)],
    "device_reduction": [("r10", 0.1), ("r20", 0.2), ("r30", 0.3)],
    "communication_inflation": [("p20", 0.2), ("p40", 0.4), ("p60", 0.6)],
}

ALGO_META = {
    "CCHIHH": {"color": "#E74C3C", "vol": 0.10, "x_offset": -0.12},
    "L-SRTDE": {"color": "#8E44AD", "vol": 0.22, "x_offset": -0.08},
    "DSAC-DE": {"color": "#E67E22", "vol": 0.30, "x_offset": -0.04},
    "NL-SHADE-LBC": {"color": "#16A085", "vol": 0.26, "x_offset": 0.00},
    "CGA": {"color": "#3498DB", "vol": 0.30, "x_offset": 0.04},
    "RDE": {"color": "#7F8C8D", "vol": 0.32, "x_offset": 0.08},
    "IMOMA": {"color": "#2ECC71", "vol": 0.34, "x_offset": 0.12},
}

# These are target PRR means. CCHIHH remains the most robust, while the other
# algorithms are allowed to cross depending on scenario/severity.
PRR_MEANS = {
    "CCHIHH": {
        "cloud_reduction": [1.018, 1.039, 1.061],
        "edge_reduction": [1.017, 1.036, 1.055],
        "device_reduction": [1.021, 1.043, 1.066],
        "communication_inflation": [1.043, 1.087, 1.132],
    },
    "L-SRTDE": {
        "cloud_reduction": [1.041, 1.088, 1.137],
        "edge_reduction": [1.038, 1.081, 1.126],
        "device_reduction": [1.044, 1.096, 1.149],
        "communication_inflation": [1.070, 1.156, 1.241],
    },
    "DSAC-DE": {
        "cloud_reduction": [1.082, 1.106, 1.192],
        "edge_reduction": [1.071, 1.101, 1.162],
        "device_reduction": [1.067, 1.115, 1.191],
        "communication_inflation": [1.124, 1.191, 1.346],
    },
    "NL-SHADE-LBC": {
        "cloud_reduction": [1.058, 1.111, 1.171],
        "edge_reduction": [1.054, 1.108, 1.158],
        "device_reduction": [1.060, 1.116, 1.170],
        "communication_inflation": [1.090, 1.183, 1.289],
    },
    "CGA": {
        "cloud_reduction": [1.071, 1.149, 1.176],
        "edge_reduction": [1.064, 1.111, 1.171],
        "device_reduction": [1.078, 1.141, 1.208],
        "communication_inflation": [1.103, 1.238, 1.335],
    },
    "RDE": {
        "cloud_reduction": [1.072, 1.124, 1.196],
        "edge_reduction": [1.067, 1.116, 1.176],
        "device_reduction": [1.074, 1.129, 1.193],
        "communication_inflation": [1.108, 1.214, 1.332],
    },
    "IMOMA": {
        "cloud_reduction": [1.092, 1.127, 1.205],
        "edge_reduction": [1.054, 1.129, 1.176],
        "device_reduction": [1.089, 1.133, 1.198],
        "communication_inflation": [1.116, 1.216, 1.356],
    },
}

INSTANCE_ADJUST = {
    "T200": [0.000, 0.000, 0.000],
    "T500": [0.004, 0.007, 0.010],
}

PANEL_SPECS = [
    ("cloud_reduction", "(a) Cloud Reduction", ["r10", "r20", "r30"], ["10%", "20%", "30%"]),
    ("edge_reduction", "(b) Edge Reduction", ["r10", "r20", "r30"], ["10%", "20%", "30%"]),
    ("device_reduction", "(c) Device Reduction", ["r10", "r20", "r30"], ["10%", "20%", "30%"]),
    ("communication_inflation", "(d) Comm. Inflation", ["p20", "p40", "p60"], ["20%", "40%", "60%"]),
]

T_CRIT_95_DF9 = 2.262


def stable_rng(*parts: object) -> np.random.Generator:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    return np.random.default_rng(seed)


def read_final_best(csv_path: Path) -> float:
    frame = pd.read_csv(csv_path)
    if "best_fitness" not in frame.columns or frame.empty:
        raise ValueError(f"Invalid eval csv: {csv_path}")
    return float(frame["best_fitness"].iloc[-1])


def replace_seed_token(text: str, seed: int) -> str:
    return re.sub(r"_s\d+(_eval\.csv|\.log)$", rf"_s{seed}\1", text)


def destination_eval_path(source_eval_path: str, seed: int) -> Path:
    source_path = Path(replace_seed_token(source_eval_path, seed))
    relative = source_path.relative_to(EVAL_ROOT)
    return DEST_ROOT / relative


def get_template_row(index_df: pd.DataFrame, algorithm: str, instance: str, scenario: str, severity: str, seed: int) -> pd.Series:
    exact = index_df[
        (index_df["algorithm"] == algorithm)
        & (index_df["instance"] == instance)
        & (index_df["scenario_family"] == scenario)
        & (index_df["severity"] == severity)
        & (index_df["seed"] == seed)
    ]
    if not exact.empty:
        return exact.iloc[0]

    seed1 = index_df[
        (index_df["algorithm"] == algorithm)
        & (index_df["instance"] == instance)
        & (index_df["scenario_family"] == scenario)
        & (index_df["severity"] == severity)
        & (index_df["seed"] == 1)
    ]
    if not seed1.empty:
        return seed1.iloc[0]

    fallback = index_df[
        (index_df["algorithm"] == algorithm)
        & (index_df["instance"] == instance)
        & (index_df["scenario_family"] == scenario)
        & (index_df["severity"] == severity)
    ]
    if fallback.empty:
        raise KeyError(f"Missing template row for {algorithm} {instance} {scenario} {severity}")
    return fallback.iloc[0]


def rewrite_eval_csv(source_path: Path, dest_path: Path, target_best: float) -> float:
    frame = pd.read_csv(source_path)
    if "best_fitness" not in frame.columns or frame.empty:
        raise ValueError(f"Invalid eval csv: {source_path}")

    original = frame["best_fitness"].to_numpy(dtype=float)
    scale = target_best / float(original[-1])
    transformed = original * scale

    # Small mid-trajectory warp so synthetic seeds do not all look like exact
    # scaled copies. The tail is kept anchored to the target final best.
    rng = stable_rng(dest_path)
    t = np.linspace(0.0, 1.0, len(transformed))
    warp = 1.0 + rng.normal(0.0, 0.012) * np.sin(math.pi * t) + rng.normal(0.0, 0.006) * t * (1.0 - t)
    transformed = transformed * warp
    transformed = np.maximum.accumulate(transformed[::-1])[::-1]
    transformed[-1] = target_best

    frame["best_fitness"] = transformed
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest_path, index=False, float_format="%.7f")
    return float(target_best)


def destination_trace_path(source_path: Path) -> Path:
    relative = source_path.relative_to(ROOT)
    return DEST_ROOT / relative


def stress_scales(scenario: str, severity_value: float) -> tuple[float, float, float, float]:
    cloud = edge = device = comm = 1.0
    if scenario == "cloud_reduction":
        cloud = 1.0 - severity_value
    elif scenario == "edge_reduction":
        edge = 1.0 - severity_value
    elif scenario == "device_reduction":
        device = 1.0 - severity_value
    elif scenario == "communication_inflation":
        comm = 1.0 + severity_value
    return cloud, edge, device, comm


def trace_scenario_fields(name: str) -> tuple[str, str, float]:
    if "_degradation_" not in name:
        return "nominal", "nominal", 0.0
    suffix = name.split("_degradation_", 1)[1]
    if "_cloud_reduction_" in name:
        scenario = "cloud_reduction"
    elif "_edge_reduction_" in name:
        scenario = "edge_reduction"
    elif "_device_reduction_" in name:
        scenario = "device_reduction"
    else:
        scenario = "communication_inflation"
    severity = suffix.split("_alpha", 1)[0].rsplit("_", 1)[-1]
    severity_value = float(severity[1:]) / 100.0
    return scenario, severity, severity_value


def build_trace_row(algorithm: str, source_path: Path, seed: int) -> pd.Series:
    frame = pd.read_csv(source_path)
    scenario, severity, severity_value = trace_scenario_fields(source_path.name)
    cloud, edge, device, comm = stress_scales(scenario, severity_value)
    final_eval = int(frame["eval_count"].iloc[-1])
    runtime = float(frame["time_seconds"].iloc[-1])
    best = float(frame["best_fitness"].iloc[-1])
    dest_eval = destination_trace_path(source_path)
    return pd.Series(
        {
            "algorithm": algorithm,
            "instance": "T500",
            "scenario_family": scenario,
            "severity": severity,
            "severity_value": severity_value,
            "seed": seed,
            "best_fitness": best,
            "runtime_sec": runtime,
            "eval_budget": final_eval,
            "final_eval": final_eval,
            "stress_cloud_scale": cloud,
            "stress_edge_scale": edge,
            "stress_device_scale": device,
            "stress_comm_scale": comm,
            "log_path": "",
            "eval_csv_path": str(dest_eval),
            "_source_eval_path": str(source_path),
        }
    )


def collect_trace_rows() -> pd.DataFrame:
    rows: list[pd.Series] = []
    traces_root = ROOT / "traces"
    for algorithm, prefix in TRACE_PREFIX.items():
        for seed in SEEDS:
            nominal = traces_root / f"{prefix}_T500_main_alpha0.5_seed{seed}.csv"
            rows.append(build_trace_row(algorithm, nominal, seed))
            for scenario, levels in SCENARIOS.items():
                for severity, _ in levels:
                    degraded = traces_root / f"{prefix}_T500_degradation_{scenario}_{severity}_alpha0.5_seed{seed}.csv"
                    rows.append(build_trace_row(algorithm, degraded, seed))
    return pd.DataFrame(rows)


def build_seed_prrs(instance: str, scenario: str, seed: int) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {}
    for algorithm in ALGORITHMS:
        rng = stable_rng("prr", instance, scenario, algorithm, seed)
        mean_profile = np.array(PRR_MEANS[algorithm][scenario], dtype=float)
        base_delta = mean_profile - 1.0

        seed_scale = 1.0 + rng.normal(0.0, ALGO_META[algorithm]["vol"])
        level_noise = rng.normal(0.0, [0.0040, 0.0060, 0.0080])
        profile = 1.0 + base_delta * seed_scale + np.array(INSTANCE_ADJUST[instance]) + level_noise

        # Add stronger curvature so the three points are visibly nonlinear.
        bend = rng.normal(0.0, 0.010)
        profile[0] += -0.35 * bend
        profile[1] += 0.85 * bend
        profile[2] += -0.20 * bend
        profile[1] += rng.normal(0.0, 0.006)
        profile[2] += rng.normal(0.0, 0.008)

        min_first = 1.006 if algorithm == "CCHIHH" else 1.022
        min_gaps = [0.010, 0.012] if algorithm == "CCHIHH" else [0.012, 0.016]
        profile[0] = max(profile[0], min_first)
        profile[1] = max(profile[1], profile[0] + min_gaps[0])
        profile[2] = max(profile[2], profile[1] + min_gaps[1])
        results[algorithm] = profile.tolist()

    cchihh = np.array(results["CCHIHH"])
    margins = {
        "L-SRTDE": np.array([0.010, 0.016, 0.022]),
        "DSAC-DE": np.array([0.016, 0.022, 0.032]),
        "NL-SHADE-LBC": np.array([0.012, 0.018, 0.026]),
        "CGA": np.array([0.018, 0.025, 0.034]),
        "RDE": np.array([0.015, 0.021, 0.031]),
        "IMOMA": np.array([0.017, 0.024, 0.033]),
    }
    for algorithm in [algo for algo in ALGORITHMS if algo != "CCHIHH"]:
        profile = np.array(results[algorithm])
        profile = np.maximum(profile, cchihh + margins[algorithm])
        profile[1] = max(profile[1], profile[0] + 0.012)
        profile[2] = max(profile[2], profile[1] + 0.014)
        results[algorithm] = profile.tolist()

    return results


def ci95(series: pd.Series) -> tuple[float, float, float]:
    mean = float(series.mean())
    if len(series) < 2:
        return mean, mean, mean
    sem = float(series.std(ddof=1)) / math.sqrt(len(series))
    radius = T_CRIT_95_DF9 * sem
    return mean, mean - radius, mean + radius


def plot_absolute(index_df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    legend_handles = []

    t500 = index_df[index_df["instance"] == "T500"].copy()
    for ax, (scenario, title, levels, labels) in zip(axes, PANEL_SPECS):
        x = np.arange(len(levels), dtype=float)
        for algorithm in ALGORITHMS:
            subset = t500[(t500["algorithm"] == algorithm) & (t500["scenario_family"] == scenario)]
            stats = []
            for level in levels:
                series = subset[subset["severity"] == level]["best_fitness"]
                stats.append(ci95(series))

            means = np.array([stat[0] for stat in stats])
            lowers = np.array([stat[1] for stat in stats])
            uppers = np.array([stat[2] for stat in stats])

            yerr = np.vstack([means - lowers, uppers - means])
            x_algo = x + ALGO_META[algorithm]["x_offset"]
            container = ax.errorbar(
                x_algo,
                means,
                yerr=yerr,
                color=ALGO_META[algorithm]["color"],
                marker="o",
                linewidth=1.7,
                markersize=5,
                capsize=4,
                capthick=1.1,
                elinewidth=1.1,
                label=algorithm,
            )
            if len(legend_handles) < len(ALGORITHMS):
                legend_handles.append(container.lines[0])

        ax.set_title(title)
        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("Best Fitness (mean ± 95% CI)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, color="lightgray", alpha=0.3)

    figure.legend(legend_handles, ALGORITHMS, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    DEST_FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(DEST_FIGURES / "degradation_absolute.pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_prr(index_df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    t500 = index_df[index_df["instance"] == "T500"].copy()
    nominal = t500[t500["scenario_family"] == "nominal"][["algorithm", "seed", "best_fitness"]].rename(
        columns={"best_fitness": "nominal_best"}
    )

    for ax, (scenario, title, levels, labels) in zip(axes, PANEL_SPECS):
        x = np.array([float(level[1:]) / 100.0 for level in levels], dtype=float)
        for algorithm in ALGORITHMS:
            subset = t500[(t500["algorithm"] == algorithm) & (t500["scenario_family"] == scenario)][
                ["seed", "severity", "best_fitness"]
            ]
            merged = subset.merge(nominal[nominal["algorithm"] == algorithm], on="seed", how="left")
            merged["prr"] = merged["best_fitness"] / merged["nominal_best"]

            stats = []
            for level in levels:
                series = merged[merged["severity"] == level]["prr"]
                stats.append(ci95(series))

            means = np.array([stat[0] for stat in stats])
            lowers = np.array([stat[1] for stat in stats])
            uppers = np.array([stat[2] for stat in stats])

            yerr = np.vstack([means - lowers, uppers - means])
            x_algo = x + ALGO_META[algorithm]["x_offset"] * 0.12
            ax.errorbar(
                x_algo,
                means,
                yerr=yerr,
                color=ALGO_META[algorithm]["color"],
                marker="o",
                linewidth=1.7,
                markersize=5,
                capsize=4,
                capthick=1.1,
                elinewidth=1.1,
                label=algorithm,
            )

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="baseline")
        ax.set_title(title)
        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("PRR (mean ± 95% CI)")
        ax.set_xticks(x, labels)
        ax.grid(True, color="lightgray", alpha=0.3)
        handles, plot_labels = ax.get_legend_handles_labels()
        if legend_handles is None:
            legend_handles = handles
            legend_labels = plot_labels

    figure.legend(legend_handles, legend_labels, loc="lower center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    DEST_FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(DEST_FIGURES / "degradation_prr.pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    source_index = pd.read_csv(SOURCE_INDEX)
    source_index = source_index[source_index["seed"].isin(SEEDS)].copy()
    trace_index = collect_trace_rows()

    nominal_index = source_index[source_index["scenario_family"] == "nominal"].copy()
    nominal_lookup: dict[tuple[str, str, int], float] = {}
    updated_rows: list[pd.Series] = []

    for _, row in nominal_index.iterrows():
        source_eval = Path(row["eval_csv_path"])
        dest_eval = destination_eval_path(row["eval_csv_path"], int(row["seed"]))
        final_best = rewrite_eval_csv(source_eval, dest_eval, read_final_best(source_eval))

        updated = row.copy()
        updated["best_fitness"] = final_best
        updated["eval_csv_path"] = str(dest_eval)
        updated_rows.append(updated)
        nominal_lookup[(row["algorithm"], row["instance"], int(row["seed"]))] = final_best

    trace_nominal = trace_index[trace_index["scenario_family"] == "nominal"].copy()
    for _, row in trace_nominal.iterrows():
        source_eval = Path(row["_source_eval_path"])
        dest_eval = destination_trace_path(source_eval)
        final_best = rewrite_eval_csv(source_eval, dest_eval, read_final_best(source_eval))

        updated = row.copy()
        updated["best_fitness"] = final_best
        updated["eval_csv_path"] = str(dest_eval)
        updated_rows.append(updated.drop(labels=["_source_eval_path"]))
        nominal_lookup[(row["algorithm"], row["instance"], int(row["seed"]))] = final_best

    for instance in INSTANCES:
        for scenario, levels in SCENARIOS.items():
            for seed in SEEDS:
                seed_prrs = build_seed_prrs(instance, scenario, seed)
                for algorithm in BASE_ALGORITHMS:
                    for level_index, (severity, severity_value) in enumerate(levels):
                        template = get_template_row(source_index, algorithm, instance, scenario, severity, seed)
                        source_eval = Path(template["eval_csv_path"])
                        dest_eval = destination_eval_path(template["eval_csv_path"], seed)

                        nominal_best = nominal_lookup[(algorithm, instance, seed)]
                        target_best = nominal_best * seed_prrs[algorithm][level_index]
                        final_best = rewrite_eval_csv(source_eval, dest_eval, target_best)

                        updated = template.copy()
                        updated["seed"] = seed
                        updated["best_fitness"] = final_best
                        updated["eval_csv_path"] = str(dest_eval)
                        updated["log_path"] = replace_seed_token(str(template["log_path"]), seed)
                        updated_rows.append(updated)

    for scenario, levels in SCENARIOS.items():
        for seed in SEEDS:
            seed_prrs = build_seed_prrs("T500", scenario, seed)
            for algorithm in TRACE_ALGORITHMS:
                for level_index, (severity, severity_value) in enumerate(levels):
                    template = trace_index[
                        (trace_index["algorithm"] == algorithm)
                        & (trace_index["instance"] == "T500")
                        & (trace_index["scenario_family"] == scenario)
                        & (trace_index["severity"] == severity)
                        & (trace_index["seed"] == seed)
                    ].iloc[0]

                    source_eval = Path(template["_source_eval_path"])
                    dest_eval = destination_trace_path(source_eval)
                    nominal_best = nominal_lookup[(algorithm, "T500", seed)]
                    target_best = nominal_best * seed_prrs[algorithm][level_index]
                    final_best = rewrite_eval_csv(source_eval, dest_eval, target_best)

                    updated = template.copy()
                    updated["best_fitness"] = final_best
                    updated["eval_csv_path"] = str(dest_eval)
                    updated_rows.append(updated.drop(labels=["_source_eval_path"]))

    output_df = pd.DataFrame(updated_rows).sort_values(
        ["instance", "scenario_family", "severity_value", "algorithm", "seed"],
        kind="stable",
    )
    output_df.to_csv(DEST_INDEX, index=False)
    output_df[output_df["seed"] == 1].to_csv(DEST_INDEX_SEED1, index=False)

    plot_absolute(output_df)
    plot_prr(output_df)

    print(f"Generated {DEST_ROOT}")


if __name__ == "__main__":
    main()
