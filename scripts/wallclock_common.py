from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class CurveData:
    time_seconds: np.ndarray
    generation: np.ndarray
    best_fitness: np.ndarray
    best_f1: np.ndarray
    best_f2: np.ndarray


@dataclass
class CurveSummary:
    grid: np.ndarray
    per_seed: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    ci95: np.ndarray


def load_curve_csv(path: Path) -> CurveData:
    times: list[float] = []
    generations: list[int] = []
    fitnesses: list[float] = []
    f1_values: list[float] = []
    f2_values: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"time_seconds", "generation", "best_fitness", "best_f1", "best_f2"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing required wall-clock columns")
        for row in reader:
            times.append(float(row["time_seconds"]))
            generations.append(int(float(row["generation"])))
            fitnesses.append(float(row["best_fitness"]))
            f1_values.append(float(row["best_f1"]))
            f2_values.append(float(row["best_f2"]))

    if not times:
        raise ValueError(f"{path} does not contain any wall-clock rows")
    if abs(times[0]) > 1e-12:
        raise ValueError(f"{path} must start with a t=0 row")

    return CurveData(
        time_seconds=np.asarray(times, dtype=float),
        generation=np.asarray(generations, dtype=int),
        best_fitness=np.asarray(fitnesses, dtype=float),
        best_f1=np.asarray(f1_values, dtype=float),
        best_f2=np.asarray(f2_values, dtype=float),
    )


def interpolate_series(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if times.ndim != 1 or values.ndim != 1 or len(times) != len(values):
        raise ValueError("times and values must be aligned 1D arrays")
    if len(times) == 0:
        raise ValueError("cannot interpolate an empty series")
    if abs(times[0]) > 1e-12:
        raise ValueError("series must include the initial t=0 row")

    uniq_times, uniq_idx = np.unique(times, return_index=True)
    uniq_values = values[uniq_idx]
    return np.interp(grid, uniq_times, uniq_values, left=uniq_values[0], right=uniq_values[-1])


def aggregate_curves(paths: Iterable[Path], time_budget: float, n_grid: int = 500) -> CurveSummary:
    if time_budget <= 0.0:
        raise ValueError("time_budget must be positive")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")

    curves = [load_curve_csv(Path(path)) for path in paths]
    if not curves:
        raise ValueError("at least one curve is required")

    grid = np.linspace(0.0, float(time_budget), int(n_grid))
    per_seed = np.vstack([interpolate_series(curve.time_seconds, curve.best_fitness, grid) for curve in curves])
    mean = per_seed.mean(axis=0)
    if per_seed.shape[0] > 1:
        std = per_seed.std(axis=0, ddof=1)
        ci95 = 1.96 * std / math.sqrt(per_seed.shape[0])
    else:
        std = np.zeros_like(mean)
        ci95 = np.zeros_like(mean)

    return CurveSummary(grid=grid, per_seed=per_seed, mean=mean, std=std, ci95=ci95)
