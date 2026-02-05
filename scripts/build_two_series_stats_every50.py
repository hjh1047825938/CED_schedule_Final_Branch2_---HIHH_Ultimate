import argparse
import csv
import re
from pathlib import Path

LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+-eE]+)")


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
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--pattern_a", required=True)
    parser.add_argument("--label_a", required=True)
    parser.add_argument("--pattern_b", required=True)
    parser.add_argument("--label_b", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Missing results dir: {results_dir}")

    series = {
        args.label_a: args.pattern_a,
        args.label_b: args.pattern_b,
    }

    series_vals = {name: {} for name in series}

    for name, pattern in series.items():
        files = sorted(results_dir.glob(pattern))
        if not files:
            raise SystemExit(f"No files found for {name} ({pattern})")
        for fp in files:
            data = parse_file(fp)
            if not data:
                raise SystemExit(f"No Gen lines found in {fp}")
            for gen, val in data.items():
                series_vals[name].setdefault(gen, []).append(val)

    common_gens = None
    for gen_map in series_vals.values():
        gens = set(gen_map.keys())
        common_gens = gens if common_gens is None else common_gens & gens
    if not common_gens:
        raise SystemExit("No common generations across series")

    gens_sorted = sorted(common_gens)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gen", f"{args.label_a} mean", f"{args.label_a} var", f"{args.label_b} mean", f"{args.label_b} var"])
        for gen in gens_sorted:
            row = [gen]
            for label in [args.label_a, args.label_b]:
                vals = series_vals[label].get(gen, [])
                if not vals:
                    raise SystemExit(f"Missing values for {label} gen {gen}")
                row.append(mean(vals))
                row.append(variance(vals))
            writer.writerow(row)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
