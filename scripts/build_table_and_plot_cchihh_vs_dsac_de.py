import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


SCALES = ["T100", "T200", "T500"]
SOLVERS = ["CCHIHH_Full", "DSAC_DE"]
SOLVER_LABELS = {
    "CCHIHH_Full": "CCHIHH-full",
    "DSAC_DE": "DSAC-DE",
}
SOLVER_COLORS = {
    "CCHIHH_Full": "#1f77b4",
    "DSAC_DE": "#ff7f0e",
}
LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+-eE]+)")
Z95 = 1.96


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
    out = {}
    txt = read_text_auto(path)
    for line in txt.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        out[int(m.group(1))] = float(m.group(2))
    return out


def mean(vals):
    return sum(vals) / len(vals)


def variance(vals):
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def load_solver_runs(scale: str, root_dir: Path, solver: str):
    runs = []
    scale_dir = root_dir / scale
    if solver == "CCHIHH_Full":
        pattern = "CCHIHH_full_seed*.txt"
        valid = re.compile(r"^CCHIHH_full_seed([1-9]|10)\.txt$")
    else:
        pattern = "DSAC_DE_seed*.txt"
        valid = re.compile(r"^DSAC_DE_seed([1-9]|10)\.txt$")

    for fp in sorted(scale_dir.glob(pattern)):
        if not valid.match(fp.name):
            continue
        series = parse_log(fp)
        if series:
            runs.append((fp, series))
    return runs


def build_scale_stats(scale: str, cchihh_root: Path, dsac_root: Path):
    per_solver = {}
    root_map = {
        "CCHIHH_Full": cchihh_root,
        "DSAC_DE": dsac_root,
    }
    for solver in SOLVERS:
        runs = load_solver_runs(scale, root_map[solver], solver)
        if not runs:
            raise SystemExit(f"No valid logs for {solver} in scale={scale}")

        common_gens = set(runs[0][1].keys())
        for _, series in runs[1:]:
            common_gens &= set(series.keys())
        if not common_gens:
            raise SystemExit(f"No common generations for {solver} in scale={scale}")

        gens = sorted(common_gens)
        stats = {}
        finals = []
        for g in gens:
            vals = [series[g] for _, series in runs]
            stats[g] = {"mean": mean(vals), "var": variance(vals)}
        for _, series in runs:
            finals.append(series[gens[-1]])

        per_solver[solver] = {
            "n": len(runs),
            "gens": gens,
            "stats": stats,
            "finals": finals,
            "final_mean": mean(finals),
            "final_std": math.sqrt(variance(finals)),
        }

    common_all = set(per_solver[SOLVERS[0]]["gens"])
    for s in SOLVERS[1:]:
        common_all &= set(per_solver[s]["gens"])
    if not common_all:
        raise SystemExit(f"No common generations between solvers in scale={scale}")

    return per_solver, sorted(common_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cchihh_root", default="outputs/results/cchihh_full_canonical_multiscale")
    parser.add_argument("--dsac_root", default="outputs/results/dsac_de_multiscale")
    parser.add_argument("--out_dir", default="outputs/results/baseline_cchihh_vs_dsac_de")
    args = parser.parse_args()

    root = Path.cwd()
    cchihh_root = root / args.cchihh_root
    dsac_root = root / args.dsac_root
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scale_data = {}
    table_rows = []
    for scale in SCALES:
        per_solver, gens_all = build_scale_stats(scale, cchihh_root, dsac_root)
        scale_data[scale] = (per_solver, gens_all)

        cchihh_mean = per_solver["CCHIHH_Full"]["final_mean"]
        cchihh_std = per_solver["CCHIHH_Full"]["final_std"]
        dsac_mean = per_solver["DSAC_DE"]["final_mean"]
        dsac_std = per_solver["DSAC_DE"]["final_std"]
        improve_pct = (dsac_mean - cchihh_mean) / dsac_mean * 100.0 if dsac_mean != 0 else 0.0
        p_val = mannwhitneyu(
            per_solver["CCHIHH_Full"]["finals"],
            per_solver["DSAC_DE"]["finals"],
            alternative="two-sided",
            method="exact",
        ).pvalue

        table_rows.append(
            [
                scale,
                f"{cchihh_mean:.6f} +- {cchihh_std:.6f}",
                f"{dsac_mean:.6f} +- {dsac_std:.6f}",
                f"{improve_pct:.4f}",
                f"{p_val:.9f}",
            ]
        )

    table_path = out_dir / "table_baseline_cchihh_vs_dsac_de.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Problem", "CCHIHH-full (Mean+-Std)", "DSAC-DE (Mean+-Std)", "Improvement (%)", "p-value"])
        w.writerows(table_rows)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "figure.figsize": (12, 4),
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    for i, scale in enumerate(SCALES):
        per_solver, gens_all = scale_data[scale]
        ax = axes[i]
        for s in SOLVERS:
            means = [per_solver[s]["stats"][g]["mean"] for g in gens_all]
            vars_ = [per_solver[s]["stats"][g]["var"] for g in gens_all]
            n = max(1, per_solver[s]["n"])
            cis = [Z95 * math.sqrt(max(v, 0.0) / n) for v in vars_]
            c = SOLVER_COLORS[s]
            ax.plot(gens_all, means, label=SOLVER_LABELS[s], color=c)
            ax.fill_between(
                gens_all,
                [m - ci for m, ci in zip(means, cis)],
                [m + ci for m, ci in zip(means, cis)],
                color=c,
                alpha=0.14,
            )

        ax.set_title(scale, fontweight="bold")
        ax.set_xlabel("Generation")
        if i == 0:
            ax.set_ylabel("Best fitness")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    pdf_path = out_dir / "baseline_cchihh_vs_dsac_de_mean_ci.pdf"
    png_path = out_dir / "baseline_cchihh_vs_dsac_de_mean_ci.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)

    print(f"Saved: {table_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
