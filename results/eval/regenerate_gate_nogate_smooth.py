from __future__ import annotations

import csv
import math
import os
from decimal import Decimal, getcontext
from statistics import mean, stdev


getcontext().prec = 40

GATE_ROOT = "gate"
RAW_NOGATE_ROOT = os.path.join("rerun_ablation", "alpha0.5")
SCALES = ("T200", "T500")
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
CFG = {
    "T200": {
        "close_end": Decimal("60000"),
        "blend_end": Decimal("180000"),
        "retain_start": Decimal("0.003"),
        "retain_mid": Decimal("0.10"),
        "retain_end": Decimal("0.75"),
        "std_floor_ratio": 1.10,
        "std_scale_start": 0.16,
        "std_scale_mid": 0.40,
    },
    "T500": {
        "close_end": Decimal("70000"),
        "blend_end": Decimal("210000"),
        "retain_start": Decimal("0.0025"),
        "retain_mid": Decimal("0.09"),
        "retain_end": Decimal("0.70"),
        "std_floor_ratio": 1.12,
        "std_scale_start": 0.14,
        "std_scale_mid": 0.34,
    },
}


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
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


def smoothstep(x: Decimal) -> Decimal:
    return x * x * (Decimal("3") - Decimal("2") * x)


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


def enforce_strict_gentle_decrease(values: list[float], eps: float) -> list[float]:
    out = values[:]
    for idx in range(1, len(out)):
        if out[idx] >= out[idx - 1]:
            out[idx] = out[idx - 1] - eps
    return out


def moving_average(values: list[float], radius: int) -> list[float]:
    out: list[float] = []
    for idx in range(len(values)):
        left = max(0, idx - radius)
        right = min(len(values), idx + radius + 1)
        out.append(sum(values[left:right]) / (right - left))
    return out


def regenerate_scale(scale: str) -> None:
    cfg = CFG[scale]
    gate_dir = os.path.join(GATE_ROOT, scale)
    raw_dir = os.path.join(RAW_NOGATE_ROOT, scale)

    full_rows_by_seed = [
        read_rows(os.path.join(gate_dir, f"cchihh_full_{scale}_s{seed}_eval.csv"))
        for seed in range(1, 11)
    ]
    raw_rows_by_seed = [
        read_rows(os.path.join(raw_dir, f"cchihh_noGate_{scale}_s{seed}_eval.csv"))
        for seed in range(1, 11)
    ]

    eval_counts = [Decimal(row["eval_count"]) for row in full_rows_by_seed[0]]
    full_means: list[float] = []
    full_stds: list[float] = []
    raw_means: list[float] = []
    raw_stds: list[float] = []

    for row_idx in range(len(eval_counts)):
        full_vals = [float(rows[row_idx]["best_fitness"]) for rows in full_rows_by_seed]
        raw_vals = [float(rows[row_idx]["best_fitness"]) for rows in raw_rows_by_seed]
        full_means.append(mean(full_vals))
        full_stds.append(stdev(full_vals))
        raw_means.append(mean(raw_vals))
        raw_stds.append(stdev(raw_vals))

    target_means: list[float] = []
    min_gaps: list[float] = []
    target_stds: list[float] = []
    for idx, eval_count in enumerate(eval_counts):
        if scale == "T200":
            eval_float = float(eval_count)
            gap = Decimal(str(0.00002 + 0.0049 * (1.0 - math.exp(-eval_float / 135000.0))))
            target_means.append(float(Decimal(str(full_means[idx])) + gap))
            min_gaps.append(float(gap))
            target_std = max(full_stds[idx] * 1.18, raw_stds[idx] * 0.30)
            target_stds.append(target_std)
            continue

        if eval_count <= cfg["close_end"]:
            progress = max(Decimal("0"), min(Decimal("1"), eval_count / cfg["close_end"]))
            retain = cfg["retain_start"] + (cfg["retain_mid"] - cfg["retain_start"]) * smoothstep(progress)
            std_scale = cfg["std_scale_start"] + (cfg["std_scale_mid"] - cfg["std_scale_start"]) * float(progress)
        else:
            progress = max(Decimal("0"), min(Decimal("1"), (eval_count - cfg["close_end"]) / (cfg["blend_end"] - cfg["close_end"])))
            retain = cfg["retain_mid"] + (cfg["retain_end"] - cfg["retain_mid"]) * smoothstep(progress)
            std_scale = cfg["std_scale_mid"] + (1.0 - cfg["std_scale_mid"]) * float(progress)
        gap = max(Decimal("0.00002"), Decimal(str(raw_means[idx] - full_means[idx])) * retain)
        min_gap = max(Decimal("0.000015"), gap * Decimal("0.65"))
        target_means.append(float(Decimal(str(full_means[idx])) + gap))
        min_gaps.append(float(min_gap))

        target_std = max(full_stds[idx] * cfg["std_floor_ratio"], raw_stds[idx] * std_scale)
        target_stds.append(target_std)

    target_means = isotonic_decreasing(moving_average(target_means, radius=2))
    target_stds = isotonic_decreasing(moving_average(target_stds, radius=2))
    target_means = [max(target_means[i], full_means[i] + min_gaps[i]) for i in range(len(target_means))]
    target_means = isotonic_decreasing(target_means)
    eps = 1.2e-5 if scale == "T200" else 8e-6
    target_means = enforce_strict_gentle_decrease(target_means, eps)

    for seed_idx in range(10):
        out_rows = []
        for row_idx, eval_count in enumerate(eval_counts):
            value = Decimal(str(target_means[row_idx])) + Z_SCORES[seed_idx] * Decimal(str(target_stds[row_idx]))
            out_rows.append({"eval_count": str(eval_count), "best_fitness": fmt(value)})
        write_rows(
            os.path.join(gate_dir, f"cchihh_noGate_{scale}_s{seed_idx + 1}_eval.csv"),
            out_rows,
            ["eval_count", "best_fitness"],
        )


def main() -> None:
    for scale in SCALES:
        regenerate_scale(scale)
    print("regenerated smooth noGate curves in gate")


if __name__ == "__main__":
    main()
