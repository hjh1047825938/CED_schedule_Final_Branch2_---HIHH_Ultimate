import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


SCALES = ["T100", "T200", "T500"]
Z95 = 1.96
GEN_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([0-9.+\-eE]+)")
FINAL_RE = re.compile(r"The\s+best\s+solution\s*=\s*([0-9.+\-eE]+)")


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    return float(x.strip())


def read_text_auto(path: Path) -> str:
    b = path.read_bytes()
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return b.decode("utf-16", errors="ignore")
    for enc in ("utf-8", "utf-16", "gbk", "latin1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin1", errors="ignore")


def parse_log(path: Path):
    txt = read_text_auto(path)
    series = {}
    final = None
    for ln in txt.splitlines():
        s = ln.strip()
        m = GEN_RE.match(s)
        if m:
            series[int(m.group(1))] = float(m.group(2))
            continue
        mf = FINAL_RE.search(s)
        if mf:
            final = float(mf.group(1))
    if final is None and series:
        final = series[max(series.keys())]
    if final is None:
        raise RuntimeError(f"No final value parsed: {path}")
    return series, final


def mean(xs):
    return sum(xs) / len(xs)


def std_sample(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def var_pop(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def load_group(scale_dir: Path, pattern: str, n: int):
    finals = []
    by_gen = {}
    for seed in range(1, n + 1):
        p = scale_dir / pattern.format(seed=seed)
        if not p.exists():
            raise RuntimeError(f"Missing file: {p}")
        series, final = parse_log(p)
        finals.append(final)
        for g, v in series.items():
            by_gen.setdefault(g, []).append(v)
    return finals, by_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandit_root", default="results/cchihh_ablation_suite/bandit")
    parser.add_argument("--out_dir", default="results/bandit_on_vs_fixed_ops")
    parser.add_argument("--n", type=int, default=10, help="Seed count for CI")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    bandit_root = root / args.bandit_root
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build summary table
    table_rows = []
    for scale in SCALES:
        scale_dir = bandit_root / scale
        finals_on, by_gen_on = load_group(scale_dir, "bandit_adaptive_seed{seed}.txt", args.n)
        finals_fx, by_gen_fx = load_group(scale_dir, "fixed_ops_seed{seed}.txt", args.n)

        mean_on = mean(finals_on)
        std_on = std_sample(finals_on)
        mean_fx = mean(finals_fx)
        std_fx = std_sample(finals_fx)

        # For minimization: positive means bandit_on is better than fixed_ops
        improvement = (mean_fx - mean_on) / mean_fx * 100.0

        # Mann-Whitney U p-value (same as existing suite convention)
        from scipy.stats import mannwhitneyu  # local import to avoid unnecessary import at module load

        p_value = mannwhitneyu(finals_on, finals_fx, alternative="two-sided", method="auto").pvalue

        table_rows.append(
            [
                scale,
                f"{mean_on:.6f} ± {std_on:.6f}",
                f"{mean_fx:.6f} ± {std_fx:.6f}",
                f"{improvement:.4f}",
                f"{p_value:.9g}",
            ]
        )

    table_path = out_dir / "table_bandit_on_vs_fixed_ops.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Problem",
                "use_bandit=on (Mean±Std)",
                "fixed_ops (Mean±Std)",
                "Improvement (%)",
                "p-value",
            ]
        )
        w.writerows(table_rows)

    # Plot
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "figure.figsize": (12, 4),
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    for i, scale in enumerate(SCALES):
        scale_dir = bandit_root / scale
        _, by_gen_on = load_group(scale_dir, "bandit_adaptive_seed{seed}.txt", args.n)
        _, by_gen_fx = load_group(scale_dir, "fixed_ops_seed{seed}.txt", args.n)

        gens = sorted(set(by_gen_on.keys()) & set(by_gen_fx.keys()))
        mean_on = [mean(by_gen_on[g]) for g in gens]
        var_on = [var_pop(by_gen_on[g]) for g in gens]
        mean_fx = [mean(by_gen_fx[g]) for g in gens]
        var_fx = [var_pop(by_gen_fx[g]) for g in gens]

        ci_on = [Z95 * math.sqrt(max(v, 0.0) / args.n) for v in var_on]
        ci_fx = [Z95 * math.sqrt(max(v, 0.0) / args.n) for v in var_fx]

        ax = axes[i]
        ax.plot(gens, mean_on, label="use_bandit=on", color="#1f77b4")
        ax.plot(gens, mean_fx, label="fixed_ops", color="#d62728")
        ax.fill_between(
            gens,
            [m - c for m, c in zip(mean_on, ci_on)],
            [m + c for m, c in zip(mean_on, ci_on)],
            color="#1f77b4",
            alpha=0.12,
        )
        ax.fill_between(
            gens,
            [m - c for m, c in zip(mean_fx, ci_fx)],
            [m + c for m, c in zip(mean_fx, ci_fx)],
            color="#d62728",
            alpha=0.15,
        )

        ax.set_title(scale, fontweight="bold")
        ax.set_xlabel("Generation")
        if i == 0:
            ax.set_ylabel("Best fitness")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    pdf_path = out_dir / "use_bandit_on_vs_fixed_ops_mean_ci.pdf"
    png_path = out_dir / "use_bandit_on_vs_fixed_ops_mean_ci.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)

    print(f"Saved: {table_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
