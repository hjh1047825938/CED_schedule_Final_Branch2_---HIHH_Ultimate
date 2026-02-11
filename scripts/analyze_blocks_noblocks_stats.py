import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu


GEN_LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+\-eE]+)")


def read_series_auto(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}

    # Format 1: text log lines like "Gen 50: best_fit = 54.9"
    gen_vals = {}
    for ln in lines:
        m = GEN_LINE_RE.match(ln)
        if m:
            gen_vals[int(m.group(1))] = float(m.group(2))
    if gen_vals:
        return gen_vals

    # Format 2: CSV (auto-detect columns)
    sample = lines[0].lower()
    if "," in sample:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return {}
            fields = [x.strip().lower() for x in reader.fieldnames]

            gen_col = None
            fit_col = None
            for c in reader.fieldnames:
                cl = c.strip().lower()
                if cl in ("gen", "generation"):
                    gen_col = c
                    break
            for c in reader.fieldnames:
                cl = c.strip().lower()
                if "best_fit" in cl or cl == "fitness" or cl.endswith(" mean"):
                    fit_col = c
                    break
            if gen_col is None or fit_col is None:
                return {}

            out = {}
            for row in reader:
                try:
                    g = int(float(row[gen_col]))
                    v = float(row[fit_col])
                except Exception:
                    continue
                out[g] = v
            return out

    return {}


def summarize(vals):
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "best": float(np.min(arr)),
        "median": float(np.median(arr)),
        "worst": float(np.max(arr)),
    }


def p_to_sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x = len(x)
    n_y = len(y)
    gt = 0
    lt = 0
    for xv in x:
        gt += np.sum(xv > y)
        lt += np.sum(xv < y)
    delta = (gt - lt) / (n_x * n_y)
    ad = abs(delta)
    if ad < 0.147:
        mag = "negligible"
    elif ad < 0.33:
        mag = "small"
    elif ad < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return float(delta), mag


def first_hit_gen(series, threshold, max_gen):
    for g in sorted(series.keys()):
        if g <= max_gen and series[g] <= threshold:
            return g
    return max_gen


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def save_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def latex_table(path: Path, caption: str, label: str, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(columns) + " \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(x) for x in r) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def print_table(title, header, rows):
    print(f"\n=== {title} ===")
    print(",".join(header))
    for r in rows:
        print(",".join(str(x) for x in r))


def collect_runs(scale_dir: Path, max_gen: int):
    runs = {"blocks": [], "no_blocks": []}
    series_map = {"blocks": [], "no_blocks": []}
    for seed in range(1, 11):
        b = scale_dir / f"CCHIHH_stable_gate_blocks_seed{seed}.txt"
        n = scale_dir / f"CCHIHH_stable_gate_noblocks_seed{seed}.txt"
        sb = read_series_auto(b)
        sn = read_series_auto(n)
        if not sb or not sn:
            raise RuntimeError(f"Failed to parse seed {seed} in {scale_dir}")
        if max_gen not in sb or max_gen not in sn:
            raise RuntimeError(f"Missing gen={max_gen} in seed {seed} at {scale_dir}")
        runs["blocks"].append(sb[max_gen])
        runs["no_blocks"].append(sn[max_gen])
        series_map["blocks"].append(sb)
        series_map["no_blocks"].append(sn)
    return runs, series_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default="results/ablation_blocks_vs_noblocks_stable_gate_gen10000_seed1_10",
    )
    parser.add_argument("--max_gen", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.results_root)
    scales = ["T100", "T200", "T500"]

    final_rows = []
    test_rows = []
    conv_rows = []

    for sc in scales:
        scale_dir = root / sc
        runs, series_map = collect_runs(scale_dir, args.max_gen)

        s_blocks = summarize(runs["blocks"])
        s_nob = summarize(runs["no_blocks"])

        final_rows.append(
            [sc, "blocks", fmt(s_blocks["mean"]), fmt(s_blocks["std"]), fmt(s_blocks["best"]), fmt(s_blocks["median"]), fmt(s_blocks["worst"])]
        )
        final_rows.append(
            [sc, "no_blocks", fmt(s_nob["mean"]), fmt(s_nob["std"]), fmt(s_nob["best"]), fmt(s_nob["median"]), fmt(s_nob["worst"])]
        )

        # Wilcoxon rank-sum (Mann-Whitney U, two-sided)
        u, p = mannwhitneyu(runs["blocks"], runs["no_blocks"], alternative="two-sided")
        d, mag = cliffs_delta(runs["blocks"], runs["no_blocks"])
        test_rows.append(
            [sc, f"{u:.3f}", f"{p:.6g}", p_to_sig(p), fmt(d), mag]
        )

        # Threshold based on CCHIHH blocks final median
        threshold = s_blocks["median"]
        b_hit = [first_hit_gen(s, threshold, args.max_gen) for s in series_map["blocks"]]
        n_hit = [first_hit_gen(s, threshold, args.max_gen) for s in series_map["no_blocks"]]
        b_stat = summarize(b_hit)
        n_stat = summarize(n_hit)
        conv_rows.append(
            [sc, "blocks", fmt(threshold), f"{b_stat['mean']:.1f} +/- {b_stat['std']:.1f}"]
        )
        conv_rows.append(
            [sc, "no_blocks", fmt(threshold), f"{n_stat['mean']:.1f} +/- {n_stat['std']:.1f}"]
        )

    out_dir = root / "stats_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_header = ["scale", "config", "mean", "std", "best", "median", "worst"]
    test_header = ["scale", "U_stat", "p_value", "sig", "cliffs_delta", "effect"]
    conv_header = ["scale", "config", "threshold", "first_hit_gen_mean_std"]

    save_csv(out_dir / "final_fitness_summary.csv", final_header, final_rows)
    save_csv(out_dir / "wilcoxon_cliffs_delta.csv", test_header, test_rows)
    save_csv(out_dir / "convergence_speed.csv", conv_header, conv_rows)

    latex_table(
        out_dir / "final_fitness_summary.tex",
        "Final fitness summary at generation 10000.",
        "tab:final_fitness_summary",
        final_header,
        final_rows,
    )
    latex_table(
        out_dir / "wilcoxon_cliffs_delta.tex",
        "Mann-Whitney U (Wilcoxon rank-sum) and Cliff's delta between blocks and no_blocks.",
        "tab:wilcoxon_cliffs",
        test_header,
        test_rows,
    )
    latex_table(
        out_dir / "convergence_speed.tex",
        "Convergence speed to threshold (threshold = blocks median at generation 10000).",
        "tab:convergence_speed",
        conv_header,
        conv_rows,
    )

    print_table("Final Fitness Summary", final_header, final_rows)
    print_table("Wilcoxon Rank-Sum + Cliff's Delta", test_header, test_rows)
    print_table("Convergence Speed", conv_header, conv_rows)

    print("\nSaved files:")
    print(out_dir / "final_fitness_summary.csv")
    print(out_dir / "final_fitness_summary.tex")
    print(out_dir / "wilcoxon_cliffs_delta.csv")
    print(out_dir / "wilcoxon_cliffs_delta.tex")
    print(out_dir / "convergence_speed.csv")
    print(out_dir / "convergence_speed.tex")


if __name__ == "__main__":
    main()
