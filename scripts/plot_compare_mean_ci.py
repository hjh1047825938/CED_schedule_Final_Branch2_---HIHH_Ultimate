import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No data in {path}")
    return rows, reader.fieldnames


def get_series(rows, fields, label):
    mean_col = f"{label} mean"
    var_col = f"{label} var"
    if mean_col not in fields or var_col not in fields:
        raise SystemExit(f"Missing columns for label: {label}")
    gens = [int(r["gen"]) for r in rows]
    means = [float(r[mean_col]) for r in rows]
    vars_ = [float(r[var_col]) for r in rows]
    return gens, means, vars_


def plot_two(csv_a, label_a, csv_b, label_b, out_path, title, ylabel, n):
    rows_a, fields_a = load_rows(Path(csv_a))
    rows_b, fields_b = load_rows(Path(csv_b))
    gens_a, means_a, vars_a = get_series(rows_a, fields_a, label_a)
    gens_b, means_b, vars_b = get_series(rows_b, fields_b, label_b)

    z = 1.96
    n = max(1, n)
    ci_a = [z * math.sqrt(max(v, 0.0) / n) for v in vars_a]
    ci_b = [z * math.sqrt(max(v, 0.0) / n) for v in vars_b]

    plt.figure(figsize=(9, 5.5))
    plt.plot(gens_a, means_a, label=label_a, linewidth=2)
    plt.fill_between(gens_a, [m - c for m, c in zip(means_a, ci_a)],
                     [m + c for m, c in zip(means_a, ci_a)], alpha=0.18)

    plt.plot(gens_b, means_b, label=label_b, linewidth=2)
    plt.fill_between(gens_b, [m - c for m, c in zip(means_b, ci_b)],
                     [m + c for m, c in zip(means_b, ci_b)], alpha=0.18)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=9)
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_a", required=True)
    parser.add_argument("--label_a", required=True)
    parser.add_argument("--csv_b", required=True)
    parser.add_argument("--label_b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Mean with 95% CI")
    parser.add_argument("--ylabel", default="Value")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    plot_two(
        args.csv_a,
        args.label_a,
        args.csv_b,
        args.label_b,
        args.out,
        args.title,
        args.ylabel,
        args.n,
    )


if __name__ == "__main__":
    main()
