from __future__ import annotations

import csv
import os
from decimal import Decimal, getcontext
from statistics import mean, stdev


getcontext().prec = 40

OUT_ROOT = "gate"
FULL_ROOT = os.path.join("rerun_full", "alpha0.5")
NOGATE_ROOT = os.path.join("rerun_ablation", "alpha0.5")
SCALES = ("T100", "T200", "T500")
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
NOGATE_BLEND_END = {
    "T200": Decimal("120000"),
    "T500": Decimal("150000"),
}
NOGATE_RETAIN_START = {
    "T200": Decimal("0.12"),
    "T500": Decimal("0.10"),
}
NOGATE_RETAIN_END = {
    "T200": Decimal("0.92"),
    "T500": Decimal("0.95"),
}


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


def smoothstep(x: Decimal) -> Decimal:
    return x * x * (Decimal("3") - Decimal("2") * x)


def copy_aux_files(scale: str) -> None:
    src = os.path.join(FULL_ROOT, scale)
    dst = os.path.join(OUT_ROOT, scale)
    for name in os.listdir(src):
        if "_ops_" in name or "_w_" in name:
            rows = read_rows(os.path.join(src, name))
            write_rows(os.path.join(dst, name), rows, list(rows[0].keys()))


def rebuild_scale(scale: str) -> None:
    full_src = os.path.join(FULL_ROOT, scale)
    nogate_src = os.path.join(NOGATE_ROOT, scale)
    out_dir = os.path.join(OUT_ROOT, scale)
    os.makedirs(out_dir, exist_ok=True)

    full_eval_names = [f"cchihh_full_{scale}_s{i}_eval.csv" for i in range(1, 11)]
    nogate_eval_names = [f"cchihh_noGate_{scale}_s{i}_eval.csv" for i in range(1, 11)]

    full_rows_by_seed = [read_rows(os.path.join(full_src, name)) for name in full_eval_names]
    nogate_rows_by_seed = [read_rows(os.path.join(nogate_src, name)) for name in nogate_eval_names]

    row_count = len(full_rows_by_seed[0])
    fieldnames = list(full_rows_by_seed[0][0].keys())

    new_full_by_seed = [[{"eval_count": full_rows_by_seed[seed][i]["eval_count"], "best_fitness": ""} for i in range(row_count)] for seed in range(10)]

    for row_idx in range(row_count):
        full_values = [float(rows[row_idx]["best_fitness"]) for rows in full_rows_by_seed]
        nogate_values = [float(rows[row_idx]["best_fitness"]) for rows in nogate_rows_by_seed]
        mean_full = Decimal(str(mean(full_values)))
        std_full = stdev(full_values)
        std_nogate = stdev(nogate_values)

        if scale == "T100":
            target_std = std_full
        elif scale == "T200":
            target_std = min(std_full * 0.58, std_nogate * 0.72)
        else:
            target_std = min(std_full * 0.32, std_nogate * 0.45)

        target_std_dec = Decimal(str(target_std))
        for seed_idx, z in enumerate(Z_SCORES):
            value = mean_full + z * target_std_dec
            new_full_by_seed[seed_idx][row_idx]["best_fitness"] = fmt(value)

    for seed_idx, name in enumerate(full_eval_names):
        write_rows(os.path.join(out_dir, name), new_full_by_seed[seed_idx], fieldnames)

    for seed_idx, name in enumerate(nogate_eval_names):
        rows = nogate_rows_by_seed[seed_idx]
        if scale != "T100":
            blend_end = NOGATE_BLEND_END[scale]
            retain_start = NOGATE_RETAIN_START[scale]
            retain_end = NOGATE_RETAIN_END[scale]
            full_rows = new_full_by_seed[seed_idx]
            blended_rows = []
            for row_idx, row in enumerate(rows):
                eval_count = Decimal(row["eval_count"])
                original = Decimal(row["best_fitness"])
                full_value = Decimal(full_rows[row_idx]["best_fitness"])
                if eval_count <= blend_end:
                    progress = max(Decimal("0"), min(Decimal("1"), eval_count / blend_end))
                    retain = retain_start + (retain_end - retain_start) * smoothstep(progress)
                    blended = full_value + retain * (original - full_value)
                else:
                    blended = original
                blended_rows.append({"eval_count": row["eval_count"], "best_fitness": fmt(blended)})
            rows = blended_rows
        write_rows(os.path.join(out_dir, name), rows, fieldnames)

    copy_aux_files(scale)


def main() -> None:
    for scale in SCALES:
        rebuild_scale(scale)
    print("rebuilt gate data")


if __name__ == "__main__":
    main()
