import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+-eE]+)")

SERIES = {
    "GA": "GA_seed*.txt",
    "DE": "DE_seed*.txt",
    "GDE": "GDE_seed*.txt",
    "CCHIHH_base": "CCHIHH_base_seed*.txt",
    "CCHIHH_stable": "CCHIHH_stable_seed*.txt",
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
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def build_stats_and_plot(results_dir: Path, out_prefix: Path, title: str, n: int):
    if not results_dir.exists():
        raise SystemExit(f"Missing results dir: {results_dir}")

    series_vals = {name: {} for name in SERIES}

    for name, pattern in SERIES.items():
        files = sorted(results_dir.glob(pattern))
        if not files:
            raise SystemExit(f"No files found for {name} ({pattern}) in {results_dir}")
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

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
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

    z = 1.96
    n = max(1, n)
    plt.figure(figsize=(9, 5.5))
    for name in SERIES.keys():
        means = [mean(series_vals[name][g]) for g in gens_sorted]
        vars_ = [variance(series_vals[name][g]) for g in gens_sorted]
        ci = [z * math.sqrt(max(v, 0.0) / n) for v in vars_]
        lower = [m - c for m, c in zip(means, ci)]
        upper = [m + c for m, c in zip(means, ci)]
        plt.plot(gens_sorted, means, label=name, linewidth=2)
        plt.fill_between(gens_sorted, lower, upper, alpha=0.18)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    print(f"Wrote: {csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--title", default="Mean with 95% CI")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    build_stats_and_plot(
        results_dir=Path(args.results_dir),
        out_prefix=Path(args.out_prefix),
        title=args.title,
        n=args.n,
    )


if __name__ == "__main__":
    main()
