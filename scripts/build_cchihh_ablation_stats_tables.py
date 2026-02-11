import argparse
import csv
import math
import re
from pathlib import Path
from statistics import median

from scipy.stats import mannwhitneyu


CONFIGS = [
    ("cfg1_bandit_on", "1) use_bandit=true"),
    ("cfg2_bandit_off", "2) use_bandit=false"),
    ("cfg3_fixed_ops", "3) fixed ops per block"),
]


GEN_LINE_RE = re.compile(r"^\s*Gen\s+(\d+):\s+best_fit\s*=\s*([-+eE0-9.]+)")
FINAL_LINE_RE = re.compile(r"^\s*The best solution\s*=\s*([-+eE0-9.]+)")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?")


def detect_and_read_run(path: Path):
    # Try CSV first (auto format detection by header)
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            if "," in sample and "\n" in sample:
                reader = csv.DictReader(f)
                fields = [c.strip() for c in (reader.fieldnames or [])]
                lower = [c.lower() for c in fields]
                gen_col = None
                val_col = None
                for i, c in enumerate(lower):
                    if c in ("gen", "generation", "iter", "iteration"):
                        gen_col = fields[i]
                        break
                for i, c in enumerate(lower):
                    if c in ("best_fit", "fitness", "fit", "value"):
                        val_col = fields[i]
                        break
                if gen_col and val_col:
                    series = {}
                    for row in reader:
                        g = int(float(row[gen_col]))
                        v = float(row[val_col])
                        series[g] = v
                    if series:
                        g_last = max(series.keys())
                        return series, series[g_last]
    except Exception:
        pass

    # Fallback: parse text log
    series = {}
    final_value = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GEN_LINE_RE.match(line)
            if m:
                g = int(m.group(1))
                v = float(m.group(2))
                series[g] = v
                continue
            m = FINAL_LINE_RE.match(line)
            if m:
                final_value = float(m.group(1))

    if not series and final_value is None:
        raise RuntimeError(f"Unrecognized run file format: {path}")

    if series:
        g_last = max(series.keys())
        return series, series[g_last]
    return {}, final_value


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


def cliffs_label(d):
    ad = abs(d)
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


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def std_sample(vals):
    n = len(vals)
    if n <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))


def fmt(x, nd=6):
    return f"{x:.{nd}f}"


def find_first_hit_gen(series, threshold, max_gen):
    if not series:
        return max_gen
    for g in sorted(series.keys()):
        if g > max_gen:
            break
        if series[g] <= threshold:
            return g
    return max_gen


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_tex(path: Path, caption: str, label: str, columns: str, header_row: str, body_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header_row + r" \\",
        r"\midrule",
    ]
    for r in body_rows:
        lines.append(r + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results/ablation_cchihh_bandit/T100",
        help="Directory containing cfg*_seed*.txt or csv run files.",
    )
    parser.add_argument("--max_gen", type=int, default=10000)
    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory for tables (default: <results_dir>/tables).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_dir = root / args.results_dir
    out_dir = (root / args.out_dir) if args.out_dir else (results_dir / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs = {}
    for cfg_key, _ in CONFIGS:
        files = sorted(results_dir.glob(f"{cfg_key}_seed*.*"))
        files = [p for p in files if p.suffix.lower() in (".txt", ".csv")]
        if len(files) != 10:
            raise SystemExit(f"Expected 10 run files for {cfg_key}, got {len(files)} in {results_dir}")
        runs = []
        finals = []
        for fp in files:
            series, final_v = detect_and_read_run(fp)
            runs.append({"file": fp, "series": series, "final": final_v})
            finals.append(final_v)
        all_runs[cfg_key] = {"runs": runs, "finals": finals}

    # 1) Final fitness summary
    summary_header = ["config_key", "config_label", "mean", "std", "best", "median", "worst"]
    summary_rows = []
    tex_rows_1 = []
    for cfg_key, cfg_label in CONFIGS:
        vals = all_runs[cfg_key]["finals"]
        row = [
            cfg_key,
            cfg_label,
            fmt(mean(vals)),
            fmt(std_sample(vals)),
            fmt(min(vals)),
            fmt(median(vals)),
            fmt(max(vals)),
        ]
        summary_rows.append(row)
        tex_rows_1.append(" & ".join(row[1:]))
    write_csv(out_dir / "final_fitness_summary.csv", summary_header, summary_rows)
    write_tex(
        out_dir / "final_fitness_summary.tex",
        "Final fitness summary at generation 10000.",
        "tab:final_fitness_summary",
        "lrrrrrr",
        "Config & Mean & Std & Best & Median & Worst",
        tex_rows_1,
    )

    # 2) Mann-Whitney rank-sum + Cliff's delta
    pairs = [(0, 1), (0, 2), (1, 2)]
    test_header = [
        "comparison",
        "u_stat",
        "p_value",
        "significance",
        "cliffs_delta",
        "effect_size",
    ]
    test_rows = []
    tex_rows_2 = []
    for i, j in pairs:
        a_key, a_label = CONFIGS[i]
        b_key, b_label = CONFIGS[j]
        a = all_runs[a_key]["finals"]
        b = all_runs[b_key]["finals"]
        u = mannwhitneyu(a, b, alternative="two-sided", method="auto")
        d = cliffs_delta(a, b)
        sig = p_stars(u.pvalue)
        row = [
            f"{a_label} vs {b_label}",
            fmt(u.statistic, 3),
            fmt(u.pvalue, 6),
            sig,
            fmt(d, 6),
            cliffs_label(d),
        ]
        test_rows.append(row)
        tex_rows_2.append(" & ".join(row))
    write_csv(out_dir / "wilcoxon_ranksum_cliffs_delta.csv", test_header, test_rows)
    write_tex(
        out_dir / "wilcoxon_ranksum_cliffs_delta.tex",
        "Pairwise rank-sum test (Mann-Whitney U) and Cliff's delta on final fitness.",
        "tab:wilcoxon_ranksum_cliffs",
        "lrrrrl",
        "Comparison & U & p-value & Sig. & Cliff's $\\delta$ & Effect",
        tex_rows_2,
    )

    # 3) Convergence speed table
    baseline_key, baseline_label = CONFIGS[0]
    baseline_finals = all_runs[baseline_key]["finals"]
    threshold = mean(baseline_finals)

    conv_header = ["config_key", "config_label", "threshold", "hit_gen_mean", "hit_gen_std"]
    conv_rows = []
    tex_rows_3 = []
    for cfg_key, cfg_label in CONFIGS:
        hit_gens = [
            find_first_hit_gen(run["series"], threshold, args.max_gen)
            for run in all_runs[cfg_key]["runs"]
        ]
        row = [
            cfg_key,
            cfg_label,
            fmt(threshold),
            fmt(mean(hit_gens), 2),
            fmt(std_sample(hit_gens), 2),
        ]
        conv_rows.append(row)
        tex_rows_3.append(
            f"{cfg_label} & {fmt(threshold)} & {fmt(mean(hit_gens), 2)} $\\pm$ {fmt(std_sample(hit_gens), 2)}"
        )
    write_csv(out_dir / "convergence_speed.csv", conv_header, conv_rows)
    write_tex(
        out_dir / "convergence_speed.tex",
        f"Convergence speed to threshold (baseline {baseline_label} mean final fitness = {fmt(threshold)}).",
        "tab:convergence_speed",
        "lrr",
        "Config & Threshold & Hit generation (mean $\\pm$ std)",
        tex_rows_3,
    )

    # Print all tables to terminal
    print("\n=== Final Fitness Summary (Gen=10000) ===")
    print(", ".join(summary_header))
    for r in summary_rows:
        print(", ".join(r))

    print("\n=== Rank-sum Test + Cliff's Delta ===")
    print(", ".join(test_header))
    for r in test_rows:
        print(", ".join(r))

    print("\n=== Convergence Speed ===")
    print(f"Threshold rule: mean final fitness of baseline ({baseline_label})")
    print(", ".join(conv_header))
    for r in conv_rows:
        print(", ".join(r))

    print("\nSaved files:")
    for p in [
        out_dir / "final_fitness_summary.csv",
        out_dir / "final_fitness_summary.tex",
        out_dir / "wilcoxon_ranksum_cliffs_delta.csv",
        out_dir / "wilcoxon_ranksum_cliffs_delta.tex",
        out_dir / "convergence_speed.csv",
        out_dir / "convergence_speed.tex",
    ]:
        print(str(p))


if __name__ == "__main__":
    main()
