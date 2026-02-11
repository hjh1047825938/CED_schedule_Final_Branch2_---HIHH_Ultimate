import argparse
import csv
import math
import re
from pathlib import Path
from itertools import combinations
from statistics import median

import numpy as np
from scipy.stats import mannwhitneyu

GEN_LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+-eE]+)")
SEED_FILE_RE = re.compile(r"^(?P<cfg>.+?)_seed(?P<seed>\d+)\.(txt|csv)$")


def read_text_series(path: Path):
    series = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            m = GEN_LINE_RE.match(raw.strip())
            if m:
                series[int(m.group(1))] = float(m.group(2))
    return series


def detect_csv_columns(fieldnames):
    if not fieldnames:
        return None, None
    lower = [x.lower() for x in fieldnames]
    gen_idx = None
    val_idx = None

    gen_keys = ("gen", "generation", "iter", "iteration", "step")
    val_keys = ("best_fit", "fitness", "best", "value", "score")

    for i, name in enumerate(lower):
        if gen_idx is None and any(k in name for k in gen_keys):
            gen_idx = i
        if val_idx is None and any(k in name for k in val_keys):
            val_idx = i
    if gen_idx is None or val_idx is None:
        return None, None
    return fieldnames[gen_idx], fieldnames[val_idx]


def read_csv_series(path: Path):
    series = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        gen_col, val_col = detect_csv_columns(reader.fieldnames)
        if not gen_col or not val_col:
            return {}
        for row in reader:
            try:
                g = int(float(row[gen_col]))
                v = float(row[val_col])
            except (ValueError, TypeError, KeyError):
                continue
            series[g] = v
    return series


def read_whitespace_series(path: Path):
    series = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                g = int(float(parts[0]))
                v = float(parts[1])
            except ValueError:
                continue
            series[g] = v
    return series


def load_series_auto(path: Path):
    # 1) Native solver txt logs: "Gen k: best_fit = x"
    series = read_text_series(path)
    if series:
        return series

    # 2) CSV with generation/value columns
    if path.suffix.lower() == ".csv":
        series = read_csv_series(path)
        if series:
            return series

    # 3) Generic whitespace two-column numeric format
    series = read_whitespace_series(path)
    return series


def format_sig(p):
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def cliffs_delta(x, y):
    gt = 0
    lt = 0
    for a in x:
        for b in y:
            if a > b:
                gt += 1
            elif a < b:
                lt += 1
    n = len(x) * len(y)
    if n == 0:
        return 0.0
    return (gt - lt) / n


def cliffs_magnitude(delta):
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def cfg_sort_key(name):
    m = re.match(r"cfg(\d+)_", name)
    if m:
        return (int(m.group(1)), name)
    return (999, name)


def pretty_cfg(name):
    mapping = {
        "cfg1_stable_on_gate_on": "1: stable on, gate on",
        "cfg2_stable_off_gate_on": "2: stable off, gate on",
        "cfg3_stable_on_gate_off": "3: stable on, gate off",
        "cfg4_stable_off_gate_off": "4: stable off, gate off",
    }
    return mapping.get(name, name)


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def latex_escape(s: str):
    return s.replace("_", r"\_")


def write_latex_table(path: Path, caption: str, label: str, columns: str, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{columns}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(str(x) for x in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def print_section(title, header, rows):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(" | ".join(header))
    for r in rows:
        print(" | ".join(str(x) for x in r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results/ablation_cchihh_stable_gate_gen10000_seed1_10",
        help="Directory containing cfg*_seed*.txt/csv logs.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/ablation_cchihh_stable_gate_gen10000_seed1_10/stats",
        help="Output folder for CSV/TEX tables.",
    )
    parser.add_argument("--max_gen", type=int, default=10000)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    if not results_dir.exists():
        raise SystemExit(f"Missing results dir: {results_dir}")

    cfg_runs = {}
    for fp in sorted(results_dir.glob("*_seed*.*")):
        m = SEED_FILE_RE.match(fp.name)
        if not m:
            continue
        cfg = m.group("cfg")
        seed = int(m.group("seed"))
        series = load_series_auto(fp)
        if not series:
            continue
        cfg_runs.setdefault(cfg, {})[seed] = series

    if not cfg_runs:
        raise SystemExit("No valid run files found after auto-detection.")

    cfg_names = sorted(cfg_runs.keys(), key=cfg_sort_key)
    baseline_cfg = cfg_names[0]
    for c in cfg_names:
        if c.startswith("cfg1_"):
            baseline_cfg = c
            break

    # Final fitness table
    final_header = ["Config", "N", "Mean", "Std", "Best", "Median", "Worst"]
    final_rows = []
    final_vals = {}
    for cfg in cfg_names:
        vals = []
        for seed in sorted(cfg_runs[cfg].keys()):
            s = cfg_runs[cfg][seed]
            v = s.get(args.max_gen, s[max(s.keys())])
            vals.append(v)
        arr = np.array(vals, dtype=float)
        final_vals[cfg] = arr
        final_rows.append(
            [
                pretty_cfg(cfg),
                len(arr),
                f"{arr.mean():.6f}",
                f"{arr.std(ddof=1) if len(arr) > 1 else 0.0:.6f}",
                f"{arr.min():.6f}",
                f"{median(arr.tolist()):.6f}",
                f"{arr.max():.6f}",
            ]
        )

    # Wilcoxon rank-sum (Mann-Whitney U) vs baseline
    wil_header = ["Baseline", "Compared", "U", "p-value", "Sig", "Cliff's delta", "Magnitude"]
    wil_rows = []
    base_vals = final_vals[baseline_cfg]
    for cfg in cfg_names:
        if cfg == baseline_cfg:
            continue
        cur = final_vals[cfg]
        u, p = mannwhitneyu(base_vals, cur, alternative="two-sided", method="auto")
        d = cliffs_delta(base_vals, cur)
        wil_rows.append(
            [
                pretty_cfg(baseline_cfg),
                pretty_cfg(cfg),
                f"{u:.4f}",
                f"{p:.6g}",
                format_sig(p),
                f"{d:.6f}",
                cliffs_magnitude(d),
            ]
        )

    # Full pairwise Mann-Whitney table
    pair_header = ["A", "B", "U", "p-value", "Sig", "Cliff's delta", "Magnitude"]
    pair_rows = []
    for a, b in combinations(cfg_names, 2):
        xa = final_vals[a]
        xb = final_vals[b]
        u, p = mannwhitneyu(xa, xb, alternative="two-sided", method="auto")
        d = cliffs_delta(xa, xb)
        pair_rows.append(
            [
                pretty_cfg(a),
                pretty_cfg(b),
                f"{u:.4f}",
                f"{p:.6g}",
                format_sig(p),
                f"{d:.6f}",
                cliffs_magnitude(d),
            ]
        )

    # Convergence speed table
    # Threshold chosen from baseline final-fitness median.
    threshold = float(np.median(base_vals))
    conv_header = ["Config", "Threshold", "Mean hit gen", "Std hit gen", "Details"]
    conv_rows = []
    for cfg in cfg_names:
        hit_gens = []
        for seed in sorted(cfg_runs[cfg].keys()):
            series = cfg_runs[cfg][seed]
            g_hit = args.max_gen
            for g in sorted(series.keys()):
                if series[g] <= threshold:
                    g_hit = g
                    break
            hit_gens.append(g_hit)
        hit_arr = np.array(hit_gens, dtype=float)
        conv_rows.append(
            [
                pretty_cfg(cfg),
                f"{threshold:.6f}",
                f"{hit_arr.mean():.2f}",
                f"{hit_arr.std(ddof=1) if len(hit_arr) > 1 else 0.0:.2f}",
                f"{hit_arr.mean():.2f} +/- {hit_arr.std(ddof=1) if len(hit_arr) > 1 else 0.0:.2f}",
            ]
        )

    # Write CSV
    write_csv(out_dir / "final_fitness_summary.csv", final_header, final_rows)
    write_csv(out_dir / "wilcoxon_mannwhitney_vs_baseline.csv", wil_header, wil_rows)
    write_csv(out_dir / "wilcoxon_mannwhitney_pairwise.csv", pair_header, pair_rows)
    write_csv(out_dir / "convergence_speed_threshold.csv", conv_header, conv_rows)

    # Write LaTeX (booktabs)
    write_latex_table(
        out_dir / "final_fitness_summary.tex",
        "Final fitness summary at generation 10000.",
        "tab:final_fitness_summary",
        "lrrrrrr",
        final_header,
        final_rows,
    )
    write_latex_table(
        out_dir / "wilcoxon_mannwhitney_vs_baseline.tex",
        "Mann-Whitney U (rank-sum) test versus baseline with significance and Cliff's delta.",
        "tab:wilcoxon_vs_baseline",
        "llrrrrl",
        wil_header,
        wil_rows,
    )
    write_latex_table(
        out_dir / "wilcoxon_mannwhitney_pairwise.tex",
        "Pairwise Mann-Whitney U (rank-sum) tests with significance and Cliff's delta.",
        "tab:wilcoxon_pairwise",
        "llrrrrl",
        pair_header,
        pair_rows,
    )
    write_latex_table(
        out_dir / "convergence_speed_threshold.tex",
        "Convergence speed to threshold (threshold is baseline median final fitness).",
        "tab:convergence_speed",
        "lrrrl",
        conv_header,
        conv_rows,
    )

    # Terminal output
    print(f"Baseline config: {pretty_cfg(baseline_cfg)}")
    print(f"Convergence threshold: {threshold:.6f} (baseline median final fitness)")
    print_section("Table 1: Final Fitness Summary", final_header, final_rows)
    print_section("Table 2: Wilcoxon Rank-Sum (Mann-Whitney U) vs Baseline", wil_header, wil_rows)
    print_section("Table 2b: Wilcoxon Rank-Sum (Mann-Whitney U) Pairwise", pair_header, pair_rows)
    print_section("Table 3: Convergence Speed", conv_header, conv_rows)
    print("\nOutputs:")
    print(f"- {out_dir / 'final_fitness_summary.csv'}")
    print(f"- {out_dir / 'final_fitness_summary.tex'}")
    print(f"- {out_dir / 'wilcoxon_mannwhitney_vs_baseline.csv'}")
    print(f"- {out_dir / 'wilcoxon_mannwhitney_vs_baseline.tex'}")
    print(f"- {out_dir / 'wilcoxon_mannwhitney_pairwise.csv'}")
    print(f"- {out_dir / 'wilcoxon_mannwhitney_pairwise.tex'}")
    print(f"- {out_dir / 'convergence_speed_threshold.csv'}")
    print(f"- {out_dir / 'convergence_speed_threshold.tex'}")


if __name__ == "__main__":
    main()
