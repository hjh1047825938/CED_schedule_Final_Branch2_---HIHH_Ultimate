import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, median, stdev

from scipy.stats import mannwhitneyu


GEN_LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def parse_text_log(path: Path):
    series = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GEN_LINE_RE.match(line.strip())
            if not m:
                continue
            g = int(m.group(1))
            v = float(m.group(2))
            series[g] = v
    return series


def parse_csv_generic(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        lower = {c.lower(): c for c in reader.fieldnames}
        if "gen" not in lower:
            return {}
        val_key = None
        for k in ("best_fit", "fitness", "value", "objective"):
            if k in lower:
                val_key = lower[k]
                break
        if val_key is None:
            return {}

        series = {}
        for row in reader:
            try:
                g = int(float(row[lower["gen"]]))
                v = float(row[val_key])
            except Exception:
                continue
            series[g] = v
        return series


def parse_series_auto(path: Path):
    # Format auto-detection by content
    txt = parse_text_log(path)
    if txt:
        return txt
    csv_series = parse_csv_generic(path)
    if csv_series:
        return csv_series
    return {}


def load_config_series(config_name: str, files):
    out = {}
    for fp in files:
        seed_m = re.search(r"seed(\d+)", fp.stem, re.IGNORECASE)
        if not seed_m:
            continue
        seed = int(seed_m.group(1))
        series = parse_series_auto(fp)
        if not series:
            raise SystemExit(f"[{config_name}] Cannot parse series from file: {fp}")
        out[seed] = series
    if not out:
        raise SystemExit(f"[{config_name}] No valid seed series.")
    return out


def get_final_values(seed_to_series, target_gen: int):
    finals = []
    for seed in sorted(seed_to_series.keys()):
        series = seed_to_series[seed]
        if target_gen in series:
            finals.append(series[target_gen])
        else:
            gmax = max(series.keys())
            finals.append(series[gmax])
    return finals


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


def delta_magnitude(delta):
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def sig_mark(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def to_latex_booktabs(headers, rows, caption, label):
    lines = []
    cols = "l" + "r" * (len(headers) - 1)
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        cells = [str(x).replace("_", "\\_") for x in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def fmt(x):
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def print_table(title, headers, rows):
    print(f"\n=== {title} ===")
    print(" | ".join(headers))
    print("-" * 100)
    for r in rows:
        print(" | ".join(fmt(v) for v in r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t100_dir",
        default="results/baseline_cchihh_stable_migration_gate_vs_ga_gde_ga_slhh_multiscale/T100",
    )
    parser.add_argument("--qphh_dir", default="results/qphh")
    parser.add_argument("--max_gen", type=int, default=10000)
    parser.add_argument(
        "--out_dir",
        default="results/baseline_cchihh_stable_migration_gate_vs_ga_gde_ga_slhh_multiscale/T100/stat_tables",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    t100_dir = root / args.t100_dir
    qphh_dir = root / args.qphh_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_files = {
        "CCHIHH_stable_migration_gate": sorted(t100_dir.glob("CCHIHH_stable_migration_gate_seed*.txt")),
        "GA": sorted(t100_dir.glob("GA_seed*.txt")),
        "GDE": sorted(t100_dir.glob("GDE_seed*.txt")),
        "GA-SLHH": sorted(t100_dir.glob("GA-SLHH_seed*.txt")),
        "QPHH": sorted(qphh_dir.glob("QHH_seed*.txt")),
    }

    data = {}
    for cfg, files in cfg_files.items():
        if not files:
            raise SystemExit(f"[{cfg}] No files found.")
        data[cfg] = load_config_series(cfg, files)

    # 1) Final fitness summary
    final_rows = []
    final_vals_by_cfg = {}
    for cfg in data:
        vals = get_final_values(data[cfg], args.max_gen)
        final_vals_by_cfg[cfg] = vals
        m = mean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        final_rows.append(
            [cfg, m, s, min(vals), median(vals), max(vals)]
        )
    final_rows.sort(key=lambda r: r[1])

    final_headers = ["Config", "Mean", "Std", "Best", "Median", "Worst"]
    write_csv(out_dir / "t100_final_fitness_summary.csv", final_headers, final_rows)
    (out_dir / "t100_final_fitness_summary.tex").write_text(
        to_latex_booktabs(
            final_headers,
            [[r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:.6f}"] for r in final_rows],
            "T100 final fitness summary at generation 10000.",
            "tab:t100_final_fitness_summary",
        ),
        encoding="utf-8",
    )

    # 2) Wilcoxon rank-sum (Mann-Whitney U) + Cliff's delta
    baseline = "CCHIHH_stable_migration_gate"
    bvals = final_vals_by_cfg[baseline]
    wilcox_rows = []
    for cfg in data:
        if cfg == baseline:
            continue
        x = bvals
        y = final_vals_by_cfg[cfg]
        u, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        d = cliffs_delta(x, y)  # baseline - cfg
        wilcox_rows.append(
            [
                f"{baseline} vs {cfg}",
                u,
                p,
                sig_mark(p),
                d,
                delta_magnitude(d),
            ]
        )
    wilcox_rows.sort(key=lambda r: r[2])

    wilcox_headers = ["Comparison", "U_stat", "p_value", "Sig", "Cliffs_delta", "Effect"]
    write_csv(out_dir / "t100_wilcoxon_rank_sum.csv", wilcox_headers, wilcox_rows)
    (out_dir / "t100_wilcoxon_rank_sum.tex").write_text(
        to_latex_booktabs(
            wilcox_headers,
            [
                [r[0], f"{r[1]:.3f}", f"{r[2]:.6g}", r[3], f"{r[4]:.6f}", r[5]]
                for r in wilcox_rows
            ],
            "T100 Wilcoxon rank-sum (Mann-Whitney U) tests vs CCHIHH baseline.",
            "tab:t100_wilcoxon_rank_sum",
        ),
        encoding="utf-8",
    )

    # 3) Convergence speed
    # Threshold chosen from baseline final mean.
    threshold = mean(bvals)
    conv_rows = []
    for cfg in data:
        reach = []
        for seed in sorted(data[cfg].keys()):
            series = data[cfg][seed]
            g_reach = args.max_gen
            for g in sorted(series.keys()):
                if g > args.max_gen:
                    continue
                if series[g] <= threshold:
                    g_reach = g
                    break
            reach.append(g_reach)
        m = mean(reach)
        s = stdev(reach) if len(reach) > 1 else 0.0
        conv_rows.append([cfg, threshold, m, s, min(reach), median(reach), max(reach)])
    conv_rows.sort(key=lambda r: r[2])

    conv_headers = ["Config", "Threshold", "ReachGen_Mean", "ReachGen_Std", "Best", "Median", "Worst"]
    write_csv(out_dir / "t100_convergence_speed.csv", conv_headers, conv_rows)
    (out_dir / "t100_convergence_speed.tex").write_text(
        to_latex_booktabs(
            conv_headers,
            [
                [r[0], f"{r[1]:.6f}", f"{r[2]:.2f}", f"{r[3]:.2f}", int(r[4]), f"{r[5]:.1f}", int(r[6])]
                for r in conv_rows
            ],
            "T100 convergence speed (first generation reaching threshold from CCHIHH final mean).",
            "tab:t100_convergence_speed",
        ),
        encoding="utf-8",
    )

    print(f"Output directory: {out_dir}")
    print(f"Convergence threshold (from {baseline} final mean): {threshold:.6f}")

    print_table("Final Fitness Summary", final_headers, final_rows)
    print_table("Wilcoxon Rank-Sum vs CCHIHH Baseline", wilcox_headers, wilcox_rows)
    print_table("Convergence Speed", conv_headers, conv_rows)


if __name__ == "__main__":
    main()
