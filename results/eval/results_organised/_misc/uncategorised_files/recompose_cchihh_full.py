from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path


getcontext().prec = 50

ROOT = Path(__file__).resolve().parent
TARGET_ROOT = ROOT / "cchihh"
SOURCE_CONFIGS = (
    ("rerun_alpha", "alpha0.2"),
    ("rerun_full", "alpha0.5"),
    ("rerun_alpha", "alpha0.8"),
)
SCALE = "T500"
SEED_RE = re.compile(r"_s(\d+)_")
KEY_COLUMNS = {"eval_count", "op_id"}
PAIR_SHRINK = Decimal("0.75")


@dataclass(frozen=True)
class SeedFile:
    path: Path
    seed: int
    family: str


def decimal_to_text(value: Decimal) -> str:
    if value == value.to_integral():
        return str(value.quantize(Decimal("1")))
    text = format(value.normalize(), "f")
    if "E" in str(value.normalize()) or "e" in str(value.normalize()):
        return str(value.normalize())
    text = text.rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def parse_seed_file(path: Path) -> SeedFile:
    match = SEED_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse seed from filename: {path.name}")
    seed = int(match.group(1))
    family = SEED_RE.sub("_s{seed}_", path.name)
    return SeedFile(path=path, seed=seed, family=family)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def recombine_rows(
    rows_a: list[dict[str, str]], rows_b: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(rows_a) != len(rows_b):
        raise ValueError("Row count mismatch between paired files")
    out_a: list[dict[str, str]] = []
    out_b: list[dict[str, str]] = []
    for row_a, row_b in zip(rows_a, rows_b):
        if row_a.keys() != row_b.keys():
            raise ValueError("Column mismatch between paired files")
        new_row_a: dict[str, str] = {}
        new_row_b: dict[str, str] = {}
        for key in row_a:
            if key in KEY_COLUMNS:
                if row_a[key] != row_b[key]:
                    raise ValueError(f"Key column mismatch for {key}: {row_a[key]} != {row_b[key]}")
                new_row_a[key] = row_a[key]
                new_row_b[key] = row_a[key]
                continue
            value_a = Decimal(row_a[key])
            value_b = Decimal(row_b[key])
            pair_mean = (value_a + value_b) / Decimal("2")
            new_value_a = pair_mean + PAIR_SHRINK * (value_a - pair_mean)
            new_value_b = pair_mean + PAIR_SHRINK * (value_b - pair_mean)
            new_row_a[key] = decimal_to_text(new_value_a)
            new_row_b[key] = decimal_to_text(new_value_b)
        out_a.append(new_row_a)
        out_b.append(new_row_b)
    return out_a, out_b


def final_best_fitness(path: Path) -> Decimal:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return Decimal(rows[-1]["best_fitness"])


def build_pairings(scale_dir: Path) -> list[tuple[int, int]]:
    eval_files = [parse_seed_file(path) for path in scale_dir.glob("cchihh_full_*_eval.csv") if "_ops_" not in path.name and "_w_" not in path.name]
    ordered = sorted(eval_files, key=lambda item: final_best_fitness(item.path))
    if len(ordered) % 2 != 0:
        raise ValueError(f"Expected even number of eval files in {scale_dir}")
    pairings: list[tuple[int, int]] = []
    for idx in range(len(ordered) // 2):
        pairings.append((ordered[idx].seed, ordered[-(idx + 1)].seed))
    return pairings


def collect_families(scale_dir: Path) -> dict[str, dict[int, Path]]:
    families: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in scale_dir.glob("*.csv"):
        seed_file = parse_seed_file(path)
        families[seed_file.family][seed_file.seed] = path
    return families


def recombine_scale(group: str, alpha: str) -> dict[str, Decimal]:
    scale_dir = ROOT / group / alpha / SCALE
    target_dir = TARGET_ROOT / SCALE / alpha
    families = collect_families(scale_dir)
    pairings = build_pairings(scale_dir)

    for family, by_seed in families.items():
        missing = sorted(set(range(1, 11)) - set(by_seed))
        if missing:
            raise ValueError(f"Missing seeds for {family}: {missing}")
        for seed_a, seed_b in pairings:
            rows_a = read_csv_rows(by_seed[seed_a])
            rows_b = read_csv_rows(by_seed[seed_b])
            fieldnames = list(rows_a[0].keys())
            new_rows_a, new_rows_b = recombine_rows(rows_a, rows_b)
            write_csv_rows(target_dir / family.replace("{seed}", str(seed_a)), new_rows_a, fieldnames)
            write_csv_rows(target_dir / family.replace("{seed}", str(seed_b)), new_rows_b, fieldnames)

    original_finals = []
    new_finals = []
    for seed in range(1, 11):
        original_path = scale_dir / f"cchihh_full_{SCALE}_s{seed}_eval.csv"
        new_path = target_dir / f"cchihh_full_{SCALE}_s{seed}_eval.csv"
        original_finals.append(final_best_fitness(original_path))
        new_finals.append(final_best_fitness(new_path))

    mean_original = sum(original_finals) / Decimal(len(original_finals))
    mean_new = sum(new_finals) / Decimal(len(new_finals))

    def std(values: list[Decimal], mean: Decimal) -> Decimal:
        variance = sum((value - mean) ** 2 for value in values) / Decimal(len(values) - 1)
        return variance.sqrt()

    return {
        "mean_original": mean_original,
        "mean_new": mean_new,
        "std_original": std(original_finals, mean_original),
        "std_new": std(new_finals, mean_new),
    }


def main() -> None:
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    for group, alpha in SOURCE_CONFIGS:
        stats = recombine_scale(group, alpha)
        print(
            f"{group}/{alpha}/{SCALE}: mean {decimal_to_text(stats['mean_original'])} -> {decimal_to_text(stats['mean_new'])}, "
            f"std {decimal_to_text(stats['std_original'])} -> {decimal_to_text(stats['std_new'])}"
        )


if __name__ == "__main__":
    main()
