import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, median, stdev

from scipy.stats import mannwhitneyu


GEN_LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+\-eE]+)")


def parse_txt_series(path: Path):
    series = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GEN_LINE_RE.match(line.strip())
            if m:
                series[int(m.group(1))] = float(m.group(2))
    return series


def parse_csv_series(path: Path):
    series = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return series

        low_fields = [x.strip().lower() for x in reader.fieldnames]
        gen_idx = None
        val_idx = None

        for i, name in enumerate(low_fields):
            if "gen" in name or "generation" in name:
                gen_idx = i
                break

        for i, name in enumerate(low_fields):
            if i == gen_idx:
                continue
            if "best_fit" in name or "fitness" in name or "best" in name:
                val_idx = i
                break

        if gen_idx is None:
            return series

        if val_idx is None:
            for i, name in enumerate(low_fields):
                if i != gen_idx:
                    val_idx = i
                    break
        if val_idx is None:
            return series

        gen_col = reader.fieldnames[gen_idx]
        val_col = reader.fieldnames[val_idx]

        for row in reader:
            try:
                g = int(float(row[gen_col]))
                v = float(row[val_col])
            except (ValueError, TypeError, KeyError):
                continue
            series[g] = v
    return series


def parse_series(path: Path):
    if path.suffix.lower() == ".csv":
        data = parse_csv_series(path)
        if data:
            return data
    data = parse_txt_series(path)
    if data:
        return data
    if path.suffix.lower() != ".csv":
        data = parse_csv_series(path)
    return data


def final_value(series, target_gen: int):
    if not series:
        return None
    if target_gen in series:
        return series[target_gen]
    valid = [g for g in series.keys() if g <= target_gen]
    if not valid:
        return series[max(series.keys())]
    return series[max(valid)]


def first_hit_generation(series, threshold: float, max_gen: int):
    if not series:
        return max_gen
    for g in sorted(series.keys()):
        if g > max_gen:
            break
        if series[g] <= threshold:
            return g
    return max_gen


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


def cliffs_label(delta: float):
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def signif_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt(x, digits=4):
    return f"{x:.{digits}f}"


def ensure_nonempty(arr, name):
    if not arr:
        raise SystemExit(f"No valid samples for {name}.")


def write_csv(path: Path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_booktabs_tex(path: Path, caption: str, label: str, columns: str, header, rows):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{columns}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(title: str, header, rows):
    print(f"\n=== {title} ===")
    print(",".join(header))
    for r in rows:
        print(",".join(str(x) for x in r))


def collect_group(scale_dir: Path, key: str, target_gen: int):
    files = sorted(scale_dir.glob(f"*{key}_seed*.*"))
    values = []
    series_map = {}
    for fp in files:
        s = parse_series(fp)
        if not s:
            continue
        v = final_value(s, target_gen)
        if v is None:
            continue
        values.append(v)
        series_map[fp.name] = s
    return values, series_map, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results/ablation_migration_on_off_multiscale")
    parser.add_argument("--scale", type=str, default="T100")
    parser.add_argument("--target_gen", type=int, default=10000)
    parser.add_argument("--max_gen", type=int, default=10000)
    args = parser.parse_args()

    scale_dir = Path(args.root) / args.scale
    if not scale_dir.exists():
        raise SystemExit(f"Scale directory not found: {scale_dir}")

    out_dir = scale_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    on_vals, on_series, on_files = collect_group(scale_dir, "migration_on", args.target_gen)
    off_vals, off_series, off_files = collect_group(scale_dir, "migration_off", args.target_gen)
    ensure_nonempty(on_vals, "migration_on")
    ensure_nonempty(off_vals, "migration_off")

    final_header = ["Scale", "Config", "N", "Mean", "Std", "Best", "Median", "Worst"]
    final_rows = []

    def row_for(cfg, arr):
        arr_sorted = sorted(arr)
        return [
            args.scale,
            cfg,
            len(arr),
            fmt(mean(arr)),
            fmt(stdev(arr) if len(arr) > 1 else 0.0),
            fmt(arr_sorted[0]),
            fmt(median(arr_sorted)),
            fmt(arr_sorted[-1]),
        ]

    final_rows.append(row_for("CCHIHH_stable_migration_on", on_vals))
    final_rows.append(row_for("CCHIHH_stable_migration_off", off_vals))

    final_csv = out_dir / f"{args.scale}_final_fitness_summary.csv"
    final_tex = out_dir / f"{args.scale}_final_fitness_summary.tex"
    write_csv(final_csv, final_header, final_rows)
    write_booktabs_tex(
        final_tex,
        f"{args.scale} final fitness summary at generation {args.target_gen}.",
        f"tab:{args.scale.lower()}_final_fitness_summary",
        "llrrrrrr",
        final_header,
        final_rows,
    )

    u_stat, p_val = mannwhitneyu(on_vals, off_vals, alternative="two-sided")
    delta = cliffs_delta(on_vals, off_vals)
    delta_lbl = cliffs_label(delta)
    stars = signif_stars(p_val)
    wil_header = [
        "Scale",
        "Comparison",
        "Test",
        "U_stat",
        "p_value",
        "Significance",
        "Cliffs_delta",
        "Effect_size",
    ]
    wil_rows = [[
        args.scale,
        "migration_on vs migration_off",
        "Mann-Whitney U",
        fmt(float(u_stat), 2),
        f"{p_val:.6g}",
        stars,
        fmt(delta, 4),
        delta_lbl,
    ]]

    wil_csv = out_dir / f"{args.scale}_wilcoxon_ranksum_cliffs_delta.csv"
    wil_tex = out_dir / f"{args.scale}_wilcoxon_ranksum_cliffs_delta.tex"
    write_csv(wil_csv, wil_header, wil_rows)
    write_booktabs_tex(
        wil_tex,
        f"{args.scale} rank-sum test and Cliff's delta.",
        f"tab:{args.scale.lower()}_wilcoxon_ranksum_cliffs_delta",
        "lllrrlll",
        wil_header,
        wil_rows,
    )

    threshold = mean(on_vals)
    conv_on = [first_hit_generation(s, threshold, args.max_gen) for s in on_series.values()]
    conv_off = [first_hit_generation(s, threshold, args.max_gen) for s in off_series.values()]
    ensure_nonempty(conv_on, "convergence migration_on")
    ensure_nonempty(conv_off, "convergence migration_off")

    conv_header = ["Scale", "Threshold", "Config", "N", "HitGen_Mean", "HitGen_Std", "Mean+-Std"]
    conv_rows = []

    def conv_row(cfg, arr):
        m = mean(arr)
        sd = stdev(arr) if len(arr) > 1 else 0.0
        return [args.scale, fmt(threshold, 4), cfg, len(arr), fmt(m, 2), fmt(sd, 2), f"{m:.2f}+-{sd:.2f}"]

    conv_rows.append(conv_row("CCHIHH_stable_migration_on", conv_on))
    conv_rows.append(conv_row("CCHIHH_stable_migration_off", conv_off))

    conv_csv = out_dir / f"{args.scale}_convergence_speed.csv"
    conv_tex = out_dir / f"{args.scale}_convergence_speed.tex"
    write_csv(conv_csv, conv_header, conv_rows)
    write_booktabs_tex(
        conv_tex,
        f"{args.scale} convergence speed to threshold based on CCHIHH migration-on final mean.",
        f"tab:{args.scale.lower()}_convergence_speed",
        "llrlrrl",
        conv_header,
        conv_rows,
    )

    print_table(f"{args.scale} Final Fitness Summary", final_header, final_rows)
    print_table(f"{args.scale} Rank-Sum and Cliff's Delta", wil_header, wil_rows)
    print_table(f"{args.scale} Convergence Speed", conv_header, conv_rows)

    print("\nGenerated files:")
    print(final_csv)
    print(final_tex)
    print(wil_csv)
    print(wil_tex)
    print(conv_csv)
    print(conv_tex)

    print("\nDetected input files:")
    print(f"migration_on: {len(on_files)} files")
    print(f"migration_off: {len(off_files)} files")


if __name__ == "__main__":
    main()
