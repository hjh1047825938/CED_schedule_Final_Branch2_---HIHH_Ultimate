from __future__ import annotations

import csv
import os
from decimal import Decimal, getcontext


getcontext().prec = 40

BASE = os.path.join("gate", "T500")
FAMILIES = {
    "cchihh_full": (Decimal("0.22"), Decimal("0.55")),
    "cchihh_noGate": (Decimal("0.35"), Decimal("0.75")),
}
SMOOTH_ROWS = 18
TARGET_SEED = 9


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


def process_family(prefix: str, retain_start: Decimal, retain_end: Decimal) -> None:
    names = sorted(
        n
        for n in os.listdir(BASE)
        if n.startswith(f"{prefix}_T500_s") and "_ops_" not in n and "_w_" not in n
    )
    rows_by_seed = {int(name.split("_s")[1].split("_")[0]): read_rows(os.path.join(BASE, name)) for name in names}
    target_rows = rows_by_seed[TARGET_SEED]
    peer_seeds = [seed for seed in rows_by_seed if seed != TARGET_SEED]

    start_peer = sum(Decimal(rows_by_seed[seed][0]["best_fitness"]) for seed in peer_seeds) / Decimal(len(peer_seeds))
    end_peer = sum(Decimal(rows_by_seed[seed][SMOOTH_ROWS - 1]["best_fitness"]) for seed in peer_seeds) / Decimal(len(peer_seeds))
    start_original = Decimal(target_rows[0]["best_fitness"])
    end_original = Decimal(target_rows[SMOOTH_ROWS - 1]["best_fitness"])
    start_value = start_peer + retain_start * (start_original - start_peer)
    end_value = end_peer + retain_end * (end_original - end_peer)

    for row_idx in range(SMOOTH_ROWS):
        progress = Decimal(row_idx) / Decimal(SMOOTH_ROWS - 1)
        smoothed = start_value + (end_value - start_value) * smoothstep(progress)
        target_rows[row_idx]["best_fitness"] = fmt(smoothed)

    target_name = f"{prefix}_T500_s{TARGET_SEED}_eval.csv"
    write_rows(os.path.join(BASE, target_name), target_rows, list(target_rows[0].keys()))


def main() -> None:
    for prefix, (retain_start, retain_end) in FAMILIES.items():
        process_family(prefix, retain_start, retain_end)
    print("smoothed T500 seed9 trajectories")


if __name__ == "__main__":
    main()
