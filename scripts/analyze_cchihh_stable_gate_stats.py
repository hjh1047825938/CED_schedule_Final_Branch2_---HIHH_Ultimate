import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, median, pstdev

from scipy.stats import mannwhitneyu

CFG_LABELS = {
    "cfg1": "1) stable on, gate on",
    "cfg2": "2) stable off, gate on",
    "cfg3": "3) stable on, gate off",
    "cfg4": "4) stable off, gate off",
}

PAIRS = [
    ("cfg1", "cfg2", "stable contribution (gate on)"),
    ("cfg1", "cfg3", "gate contribution (stable on)"),
    ("cfg1", "cfg4", "overall contribution"),
    ("cfg2", "cfg4", "gate-only effect (stable off)"),
    ("cfg3", "cfg4", "stable-only effect (gate off)"),
]

GEN_LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+\-eE]+)")


def stars_from_p(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


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
    delta = (gt - lt) / n if n else 0.0
    ad = abs(delta)
    if ad < 0.147:
        mag = "negligible"
    elif ad < 0.33:
        mag = "small"
    elif ad < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return delta, mag


def parse_txt_curve(path: Path):
    curve = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GEN_LINE_RE.match(line.strip())
            if m:
                g = int(m.group(1))
                v = float(m.group(2))
                curve[g] = v
    return curve


def parse_csv_curve(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            fields = [x.strip().lower() for x in reader.fieldnames]
            gen_keys = ["gen", "generation", "iter", "iteration", "step"]
            fit_keys = ["best_fit", "fitness", "best", "value", "obj", "objective"]
            gen_key = next((reader.fieldnames[i] for i, k in enumerate(fields) if k in gen_keys), None)
            fit_key = next((reader.fieldnames[i] for i, k in enumerate(fields) if k in fit_keys), None)
            if gen_key and fit_key:
                curve = {}
                for row in reader:
                    try:
                        g = int(float(row[gen_key]))
                        v = float(row[fit_key])
                    except Exception:
                        continue
                    curve[g] = v
                return curve

    # fallback: numeric 2+ columns, use first two as gen/value
    curve = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 2:
                continue
            try:
                g = int(float(parts[0]))
                v = float(parts[1])
            except Exception:
                continue
            curve[g] = v
    return curve


def parse_curve(path: Path):
    if path.suffix.lower() == ".txt" or path.suffix.lower() == ".log":
        curve = parse_txt_curve(path)
        if curve:
            return curve
    if path.suffix.lower() == ".csv":
        curve = parse_csv_curve(path)
        if curve:
            return curve
    # try both parsers as fallback
    curve = parse_txt_curve(path)
    if curve:
        return curve
    return parse_csv_curve(path)


def cfg_from_name(name: str):
    m = re.search(r"(cfg[1-4])", name.lower())
    return m.group(1) if m else None


def seed_from_name(name: str):
    m = re.search(r"seed(\d+)", name.lower())
    return int(m.group(1)) if m else None


def discover_data_dir(root: Path):
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".csv", ".log"}]
    score = {}
    for fp in files:
        cfg = cfg_from_name(fp.name)
        seed = seed_from_name(fp.name)
        if not cfg or seed is None:
            continue
        d = fp.parent
        score.setdefault(d, {"cfg": set(), "files": 0, "seeds": set()})
        score[d]["cfg"].add(cfg)
        score[d]["files"] += 1
        score[d]["seeds"].add(seed)

    best_dir = None
    best_key = (-1, -1, -1)
    for d, s in score.items():
        key = (len(s["cfg"]), len(s["seeds"]), s["files"])
        if key > best_key:
            best_key = key
            best_dir = d

    if best_dir is None or best_key[0] < 4:
        raise SystemExit("Could not auto-detect a directory containing cfg1..cfg4 data files.")
    return best_dir


def load_runs(data_dir: Path):
    runs = {k: {} for k in CFG_LABELS}
    for fp in sorted(data_dir.glob("*")):
        if not fp.is_file() or fp.suffix.lower() not in {".txt", ".csv", ".log"}:
            continue
        cfg = cfg_from_name(fp.name)
        seed = seed_from_name(fp.name)
        if cfg not in runs or seed is None:
            continue
        curve = parse_curve(fp)
        if curve:
            runs[cfg][seed] = {"path": fp, "curve": curve}
    return runs


def get_final_value(curve: dict, final_gen: int):
    if final_gen in curve:
        return curve[final_gen]
    return curve[max(curve.keys())]


def first_hit_gen(curve: dict, threshold: float, max_gen: int):
    for g in sorted(curve.keys()):
        if curve[g] <= threshold:
            return g
    return max_gen


def fmt(x, nd=6):
    return f"{x:.{nd}f}"


def write_csv(path: Path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def latex_escape(s: str):
    return s.replace("_", "\\_")


def write_latex(path: Path, caption: str, label: str, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    col_spec = "l" + "r" * (len(headers) - 1)
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append(f"\\begin{{tabular}}{{{col_spec}}}")
    out.append("\\toprule")
    out.append(" & ".join(latex_escape(h) for h in headers) + " \\")
    out.append("\\midrule")
    for r in rows:
        out.append(" & ".join(latex_escape(str(r[h])) for h in headers) + " \\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def print_table(title: str, headers, rows):
    print(f"\n=== {title} ===")
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    head = " | ".join(h.ljust(widths[h]) for h in headers)
    print(head)
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(widths[h]) for h in headers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".", help="Directory containing raw run files; auto-detect subdir if needed")
    parser.add_argument("--output_dir", default="results/cchihh_stable_gate_stats")
    parser.add_argument("--final_gen", type=int, default=10000)
    parser.add_argument("--max_gen", type=int, default=10000)
    parser.add_argument("--thresholds", default="54.9,54.8,54.7,54.65")
    args = parser.parse_args()

    base = Path(args.data_dir).resolve()
    data_dir = discover_data_dir(base)
    out_dir = Path(args.output_dir).resolve()
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    print(f"Detected data directory: {data_dir}")

    runs = load_runs(data_dir)
    for cfg in ["cfg1", "cfg2", "cfg3", "cfg4"]:
        print(f"{cfg}: {len(runs[cfg])} runs")
        if len(runs[cfg]) == 0:
            raise SystemExit(f"No runs loaded for {cfg}")

    # 1) Final fitness summary
    final_rows = []
    finals = {}
    for cfg in ["cfg1", "cfg2", "cfg3", "cfg4"]:
        vals = [get_final_value(item["curve"], args.final_gen) for _, item in sorted(runs[cfg].items())]
        finals[cfg] = vals
        final_rows.append(
            {
                "Config": CFG_LABELS[cfg],
                "N": len(vals),
                "Mean": fmt(mean(vals)),
                "Std": fmt(pstdev(vals)),
                "Best": fmt(min(vals)),
                "Median": fmt(median(vals)),
                "Worst": fmt(max(vals)),
            }
        )

    final_headers = ["Config", "N", "Mean", "Std", "Best", "Median", "Worst"]
    write_csv(out_dir / "final_fitness_summary.csv", final_rows, final_headers)
    write_latex(
        out_dir / "final_fitness_summary.tex",
        "Final-generation fitness summary (Gen=10000).",
        "tab:cchihh_final_summary",
        final_headers,
        final_rows,
    )

    # 2) rank-sum + Cliff's delta
    stat_rows = []
    for a, b, note in PAIRS:
        x = finals[a]
        y = finals[b]
        _, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        delta, mag = cliffs_delta(x, y)
        stat_rows.append(
            {
                "Comparison": f"{a} vs {b}",
                "Meaning": note,
                "N1": len(x),
                "N2": len(y),
                "p_value": f"{p:.6g}",
                "Sig": stars_from_p(p),
                "Cliffs_delta": fmt(delta),
                "Effect": mag,
            }
        )

    stat_headers = ["Comparison", "Meaning", "N1", "N2", "p_value", "Sig", "Cliffs_delta", "Effect"]
    write_csv(out_dir / "wilcoxon_ranksum_cliffs_delta.csv", stat_rows, stat_headers)
    write_latex(
        out_dir / "wilcoxon_ranksum_cliffs_delta.tex",
        "Mann-Whitney rank-sum tests on final fitness with Cliff's delta effect sizes.",
        "tab:cchihh_wilcoxon",
        stat_headers,
        stat_rows,
    )

    # 3) convergence speed
    conv_rows = []
    conv_latex_rows = []
    conv_headers = ["Config"] + [f"T={t}" for t in thresholds]
    conv_csv_headers = ["Config", "Threshold", "N", "MeanGen", "StdGen"]
    conv_csv_rows = []

    for cfg in ["cfg1", "cfg2", "cfg3", "cfg4"]:
        row = {"Config": CFG_LABELS[cfg]}
        for t in thresholds:
            hit = [first_hit_gen(item["curve"], t, args.max_gen) for _, item in sorted(runs[cfg].items())]
            m = mean(hit)
            s = pstdev(hit)
            row[f"T={t}"] = f"{m:.2f} +- {s:.2f}"
            conv_csv_rows.append(
                {
                    "Config": CFG_LABELS[cfg],
                    "Threshold": t,
                    "N": len(hit),
                    "MeanGen": f"{m:.6f}",
                    "StdGen": f"{s:.6f}",
                }
            )
        conv_rows.append(row)

    write_csv(out_dir / "convergence_speed.csv", conv_csv_rows, conv_csv_headers)
    write_latex(
        out_dir / "convergence_speed.tex",
        "Convergence speed: first generation reaching threshold (mean +- std). Unreached runs are set to 10000.",
        "tab:cchihh_convergence_speed",
        conv_headers,
        conv_rows,
    )

    # print to terminal
    print_table("Final Fitness Summary", final_headers, final_rows)
    print_table("Rank-Sum Tests and Cliff's Delta", stat_headers, stat_rows)
    print_table("Convergence Speed (mean +- std generations)", conv_headers, conv_rows)

    print("\nOutput files:")
    for p in [
        out_dir / "final_fitness_summary.csv",
        out_dir / "final_fitness_summary.tex",
        out_dir / "wilcoxon_ranksum_cliffs_delta.csv",
        out_dir / "wilcoxon_ranksum_cliffs_delta.tex",
        out_dir / "convergence_speed.csv",
        out_dir / "convergence_speed.tex",
    ]:
        print(p)


if __name__ == "__main__":
    main()
