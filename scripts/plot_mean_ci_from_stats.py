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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Mean with 95% CI")
    parser.add_argument("--ylabel", default="Value")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    rows, fields = load_rows(Path(args.csv))
    gens = [int(r["gen"]) for r in rows]

    labels = []
    for name in fields:
        if name.endswith(" mean") and name != "gen mean":
            labels.append(name[: -len(" mean")])

    if not labels:
        raise SystemExit("No '<label> mean' columns found")

    plt.figure(figsize=(9, 5.5))
    z = 1.96
    n = max(1, args.n)

    for label in labels:
        mean_col = f"{label} mean"
        var_col = f"{label} var"
        if mean_col not in fields or var_col not in fields:
            continue
        means = [float(r[mean_col]) for r in rows]
        vars_ = [float(r[var_col]) for r in rows]
        ci = [z * math.sqrt(max(v, 0.0) / n) for v in vars_]
        lower = [m - c for m, c in zip(means, ci)]
        upper = [m + c for m, c in zip(means, ci)]
        plt.plot(gens, means, label=label, linewidth=2)
        plt.fill_between(gens, lower, upper, alpha=0.18)

    plt.title(args.title)
    plt.xlabel("Generation")
    plt.ylabel(args.ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
