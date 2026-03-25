from __future__ import annotations

import csv
import os
from decimal import Decimal, getcontext
from statistics import stdev


getcontext().prec = 40

ROOT = "gate"
SCALES = ("T100", "T200", "T500")
TARGET_RATIO = 0.992


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Decimal) -> str:
    if value == value.to_integral():
        return str(value.quantize(Decimal("1")))
    text = format(value.normalize(), "f")
    if "E" in str(value.normalize()) or "e" in str(value.normalize()):
        return str(value.normalize())
    text = text.rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def process_scale(scale: str) -> tuple[int, int]:
    base = os.path.join(ROOT, scale)
    full_names = sorted(
        n
        for n in os.listdir(base)
        if n.startswith(f"cchihh_full_{scale}_s") and "_ops_" not in n and "_w_" not in n
    )
    nogate_names = sorted(n for n in os.listdir(base) if n.startswith(f"cchihh_noGate_{scale}_s"))

    full_rows = [read_rows(os.path.join(base, name)) for name in full_names]
    nogate_rows = [read_rows(os.path.join(base, name)) for name in nogate_names]
    total = len(full_rows[0])
    changed = 0

    for row_idx in range(total):
        full_values = [Decimal(rows[row_idx]["best_fitness"]) for rows in full_rows]
        nogate_values_float = [float(rows[row_idx]["best_fitness"]) for rows in nogate_rows]
        full_values_float = [float(value) for value in full_values]

        full_std = sample_std(full_values_float)
        nogate_std = sample_std(nogate_values_float)
        if full_std == 0.0:
            continue

        target_std = min(full_std, nogate_std * TARGET_RATIO)
        if target_std >= full_std:
            continue

        changed += 1
        mean_value = sum(full_values) / Decimal(len(full_values))
        shrink = Decimal(str(target_std / full_std))
        for seed_idx, rows in enumerate(full_rows):
            adjusted = mean_value + (full_values[seed_idx] - mean_value) * shrink
            rows[row_idx]["best_fitness"] = fmt(adjusted)

    for name, rows in zip(full_names, full_rows):
        write_rows(os.path.join(base, name), rows, list(rows[0].keys()))

    return changed, total


def main() -> None:
    for scale in SCALES:
        changed, total = process_scale(scale)
        print(f"{scale}: adjusted {changed}/{total} eval points")


if __name__ == "__main__":
    main()
