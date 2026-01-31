import csv
import re
from pathlib import Path

# Config
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "gen10000_seed0_20"
OUT_CSV = ROOT / "results" / "gen10000_seed0_20_stats_every50.csv"

LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+-eE]+)")

SERIES = {
    "GA": "GA_seed*.txt",
    "DE": "DE_seed*.txt",
    "GDE": "GDE_seed*.txt",
    "CCHIHH (non-stable)": "CCHIHH_base_seed*.txt",
    "CCHIHH (stable)": "CCHIHH_stable_seed*.txt",
}


def parse_file(path: Path):
    gens = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            gen = int(m.group(1))
            val = float(m.group(2))
            gens[gen] = val
    return gens


def mean(vals):
    return sum(vals) / len(vals)


def variance(vals):
    # Population variance
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def main():
    if not RESULTS_DIR.exists():
        raise SystemExit(f"Missing results dir: {RESULTS_DIR}")

    series_vals = {name: {} for name in SERIES}

    for name, pattern in SERIES.items():
        files = sorted(RESULTS_DIR.glob(pattern))
        if not files:
            raise SystemExit(f"No files found for {name} ({pattern})")
        for fp in files:
            data = parse_file(fp)
            if not data:
                raise SystemExit(f"No Gen lines found in {fp}")
            for gen, val in data.items():
                series_vals[name].setdefault(gen, []).append(val)

    # Determine common generations
    common_gens = None
    for name, gen_map in series_vals.items():
        gens = set(gen_map.keys())
        common_gens = gens if common_gens is None else common_gens & gens
    if not common_gens:
        raise SystemExit("No common generations across series")

    gens_sorted = sorted(common_gens)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["gen"]
        for name in SERIES.keys():
            header.append(f"{name} mean")
            header.append(f"{name} var")
        writer.writerow(header)

        for gen in gens_sorted:
            row = [gen]
            for name in SERIES.keys():
                vals = series_vals[name].get(gen, [])
                if not vals:
                    raise SystemExit(f"Missing values for {name} gen {gen}")
                row.append(mean(vals))
                row.append(variance(vals))
            writer.writerow(row)

    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
