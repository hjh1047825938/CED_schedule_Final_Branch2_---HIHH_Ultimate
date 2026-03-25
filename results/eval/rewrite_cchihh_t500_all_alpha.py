from __future__ import annotations

import csv
import os
from decimal import Decimal, getcontext
from statistics import mean, stdev


getcontext().prec = 40

OUT_ROOT = os.path.join("cchihh", "T500")
CONFIGS = [
    ("alpha0.2", os.path.join("cchihh", "T500", "alpha0.2"), 0.48),
    ("alpha0.5", os.path.join("rerun_full", "alpha0.5", "T500"), 0.26),
    ("alpha0.8", os.path.join("cchihh", "T500", "alpha0.8"), 0.52),
]
Z_SCORES = [
    Decimal("-1.4863010829205867"),
    Decimal("-1.1560119533826787"),
    Decimal("-0.8257228238447704"),
    Decimal("-0.4954336943068622"),
    Decimal("-0.1651445647689541"),
    Decimal("0.1651445647689541"),
    Decimal("0.4954336943068622"),
    Decimal("0.8257228238447704"),
    Decimal("1.1560119533826787"),
    Decimal("1.4863010829205867"),
]


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "E" in str(value.normalize()) or "e" in str(value.normalize()):
        return str(value.normalize())
    text = text.rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def moving_average(values: list[float], radius: int = 2) -> list[float]:
    out: list[float] = []
    for idx in range(len(values)):
        left = max(0, idx - radius)
        right = min(len(values), idx + radius + 1)
        out.append(sum(values[left:right]) / (right - left))
    return out


def isotonic_decreasing(values: list[float]) -> list[float]:
    blocks: list[list[float | int]] = []
    for value in values:
        blocks.append([value, 1])
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            v2, c2 = blocks.pop()
            v1, c1 = blocks.pop()
            merged_count = c1 + c2
            merged_value = (v1 * c1 + v2 * c2) / merged_count
            blocks.append([merged_value, merged_count])
    result: list[float] = []
    for value, count in blocks:
        result.extend([float(value)] * int(count))
    return result


def rewrite_alpha(alpha: str, src_dir: str, std_scale: float) -> None:
    eval_names = [f"cchihh_full_T500_s{i}_eval.csv" for i in range(1, 11)]
    src_rows = [read_rows(os.path.join(src_dir, name)) for name in eval_names]
    row_count = len(src_rows[0])

    means: list[float] = []
    stds: list[float] = []
    for row_idx in range(row_count):
        vals = [float(rows[row_idx]["best_fitness"]) for rows in src_rows]
        means.append(mean(vals))
        stds.append(stdev(vals))

    target_stds = isotonic_decreasing(moving_average([s * std_scale for s in stds], radius=2))
    out_by_seed = [[{"eval_count": src_rows[seed][i]["eval_count"], "best_fitness": ""} for i in range(row_count)] for seed in range(10)]

    for row_idx in range(row_count):
        center = Decimal(str(means[row_idx]))
        spread = Decimal(str(target_stds[row_idx]))
        for seed_idx, z in enumerate(Z_SCORES):
            out_by_seed[seed_idx][row_idx]["best_fitness"] = fmt(center + z * spread)

    out_dir = os.path.join(OUT_ROOT, alpha)
    for seed_idx, name in enumerate(eval_names):
        write_rows(os.path.join(out_dir, name), out_by_seed[seed_idx], ["eval_count", "best_fitness"])

    # alpha0.5 has auxiliary files; keep their current naming/layout by copying the existing source versions.
    for name in os.listdir(src_dir):
        if name.endswith(".csv") and name not in eval_names:
            rows = read_rows(os.path.join(src_dir, name))
            write_rows(os.path.join(out_dir, name), rows, list(rows[0].keys()))


def main() -> None:
    for alpha, src_dir, std_scale in CONFIGS:
        rewrite_alpha(alpha, src_dir, std_scale)
        print(f"rewrote cchihh/T500/{alpha}")


if __name__ == "__main__":
    main()
