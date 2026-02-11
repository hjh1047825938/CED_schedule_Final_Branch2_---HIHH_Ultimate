import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, median, stdev

from scipy.stats import mannwhitneyu


GEN_LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([0-9.+\-eE]+)")
NSUB_SEED_RE = re.compile(r"nsubpop(\d+)_seed(\d+)", re.IGNORECASE)


def parse_txt_curve(path: Path):
    curve = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GEN_LINE_RE.match(line.strip())
            if m:
                curve[int(m.group(1))] = float(m.group(2))
    return curve


def parse_csv_curve(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        fields = [x.strip() for x in reader.fieldnames]
        lower = {k.lower(): k for k in fields}
        gen_col = None
        for c in ("gen", "generation"):
            if c in lower:
                gen_col = lower[c]
                break
        fit_col = None
        for k in fields:
            lk = k.lower()
            if "best" in lk and ("fit" in lk or "fitness" in lk):
                fit_col = k
                break
        if gen_col is None or fit_col is None:
            return {}
        curve = {}
        for row in reader:
            try:
                g = int(float(row[gen_col]))
                v = float(row[fit_col])
            except (ValueError, KeyError, TypeError):
                continue
            curve[g] = v
        return curve


def parse_curve_auto(path: Path):
    if path.suffix.lower() == ".txt":
        return parse_txt_curve(path)
    if path.suffix.lower() == ".csv":
        return parse_csv_curve(path)
    return {}


def std_or_zero(xs):
    return stdev(xs) if len(xs) > 1 else 0.0


def cliffs_delta(a, b):
    gt = 0
    lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    n = len(a) * len(b)
    if n == 0:
        return 0.0
    return (gt - lt) / n


def cliffs_label(delta):
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def p_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def table_to_latex(headers, rows, caption, label):
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{" + "l" * len(headers) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(str(x) for x in r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def print_table(title, headers, rows):
    print(f"\n{title}")
    print("-" * len(title))
    print(" | ".join(headers))
    for r in rows:
        print(" | ".join(str(x) for x in r))


def discover_runs(base_dir: Path):
    runs = {}
    for fp in sorted(base_dir.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in (".txt", ".csv"):
            continue
        m = NSUB_SEED_RE.search(fp.name)
        if not m:
            continue
        nsub = int(m.group(1))
        seed = int(m.group(2))
        if nsub not in (1, 8):
            continue
        scale = fp.parent.name
        curve = parse_curve_auto(fp)
        if not curve:
            continue
        runs.setdefault(scale, {}).setdefault(nsub, {})[seed] = {"path": fp, "curve": curve}
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="results/ablation_nsubpop_1_vs_8_multiscale",
        help="Directory containing nsubpop=1/8 logs.",
    )
    parser.add_argument("--output_dir", default="", help="Defaults to <input_dir>/stats")
    parser.add_argument("--max_gen", type=int, default=10000)
    args = parser.parse_args()

    base_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (base_dir / "stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(base_dir)
    if not runs:
        raise SystemExit(f"No valid runs found in: {base_dir}")

    scales = sorted(runs.keys())
    final_headers = ["scale", "config", "n", "Mean", "Std", "Best", "Median", "Worst"]
    final_rows = []

    wil_headers = [
        "scale",
        "group_a",
        "group_b",
        "n_a",
        "n_b",
        "U_stat",
        "p_value",
        "sig",
        "cliffs_delta",
        "effect_size",
    ]
    wil_rows = []

    conv_headers = ["scale", "config", "threshold", "MeanGen", "StdGen", "Mean+-Std"]
    conv_rows = []

    for scale in scales:
        if 1 not in runs[scale] or 8 not in runs[scale]:
            continue
        seeds1 = sorted(runs[scale][1].keys())
        seeds8 = sorted(runs[scale][8].keys())
        common_seeds = sorted(set(seeds1) & set(seeds8))
        if not common_seeds:
            continue

        finals = {1: [], 8: []}
        curves = {1: {}, 8: {}}
        for nsub in (1, 8):
            for s in common_seeds:
                c = runs[scale][nsub][s]["curve"]
                last_gen = max(c.keys())
                finals[nsub].append(c[last_gen])
                curves[nsub][s] = c

        for nsub in (1, 8):
            arr = finals[nsub]
            final_rows.append(
                [
                    scale,
                    f"nsubpop={nsub}",
                    len(arr),
                    fmt(mean(arr), 4),
                    fmt(std_or_zero(arr), 4),
                    fmt(min(arr), 4),
                    fmt(median(arr), 4),
                    fmt(max(arr), 4),
                ]
            )

        # Baseline threshold: median final fitness of nsubpop=8 (CCHIHH baseline)
        threshold = median(finals[8])

        conv = {1: [], 8: []}
        for nsub in (1, 8):
            for s in common_seeds:
                c = curves[nsub][s]
                reached = args.max_gen
                for g in sorted(c.keys()):
                    if c[g] <= threshold:
                        reached = g
                        break
                conv[nsub].append(reached)

            conv_mean = mean(conv[nsub])
            conv_std = std_or_zero(conv[nsub])
            conv_rows.append(
                [
                    scale,
                    f"nsubpop={nsub}",
                    fmt(threshold, 4),
                    fmt(conv_mean, 1),
                    fmt(conv_std, 1),
                    f"{fmt(conv_mean,1)} +- {fmt(conv_std,1)}",
                ]
            )

        u = mannwhitneyu(finals[1], finals[8], alternative="two-sided", method="auto")
        p = float(u.pvalue)
        delta = cliffs_delta(finals[1], finals[8])
        wil_rows.append(
            [
                scale,
                "nsubpop=1",
                "nsubpop=8",
                len(finals[1]),
                len(finals[8]),
                fmt(float(u.statistic), 3),
                f"{p:.6g}",
                p_stars(p),
                fmt(delta, 4),
                cliffs_label(delta),
            ]
        )

    # Write CSV
    final_csv = out_dir / "final_fitness_summary.csv"
    wil_csv = out_dir / "wilcoxon_ranksum_cliffs_delta.csv"
    conv_csv = out_dir / "convergence_speed.csv"

    for path, headers, rows in [
        (final_csv, final_headers, final_rows),
        (wil_csv, wil_headers, wil_rows),
        (conv_csv, conv_headers, conv_rows),
    ]:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)

    # Write LaTeX
    final_tex = out_dir / "final_fitness_summary.tex"
    wil_tex = out_dir / "wilcoxon_ranksum_cliffs_delta.tex"
    conv_tex = out_dir / "convergence_speed.tex"

    final_tex.write_text(
        table_to_latex(
            final_headers,
            final_rows,
            "Final fitness summary at generation 10000 (CCHIHH stable + migration + gate, nsubpop=1 vs 8)",
            "tab:nsubpop_final",
        ),
        encoding="utf-8",
    )
    wil_tex.write_text(
        table_to_latex(
            wil_headers,
            wil_rows,
            "Wilcoxon rank-sum (Mann-Whitney U) and Cliff's delta for final fitness",
            "tab:nsubpop_wilcoxon",
        ),
        encoding="utf-8",
    )
    conv_tex.write_text(
        table_to_latex(
            conv_headers,
            conv_rows,
            "Convergence speed (first generation reaching threshold; unreached set to 10000)",
            "tab:nsubpop_convergence",
        ),
        encoding="utf-8",
    )

    print(f"Input dir : {base_dir}")
    print(f"Output dir: {out_dir}")
    print("Threshold rule: per scale, threshold = median final fitness of nsubpop=8.")
    print("Unreached threshold is recorded as max_gen = 10000.")

    print_table("Final Fitness Summary", final_headers, final_rows)
    print_table("Wilcoxon Rank-Sum + Cliff's Delta", wil_headers, wil_rows)
    print_table("Convergence Speed", conv_headers, conv_rows)

    print("\nWrote files:")
    for p in [final_csv, final_tex, wil_csv, wil_tex, conv_csv, conv_tex]:
        print(f"- {p}")


if __name__ == "__main__":
    main()
