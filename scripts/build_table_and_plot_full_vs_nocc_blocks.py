import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


GEN_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([0-9.+\-eE]+)")
FINAL_RE = re.compile(r"The\s+best\s+solution\s*=\s*([0-9.+\-eE]+)")
SCALES = ["T100", "T200", "T500"]
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


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dir", default="results/cchihh_full_canonical_multiscale")
    parser.add_argument("--nocc_dir", default="results/cchihh_nocc_blocks_migration_multiscale")
    parser.add_argument("--out_dir", default="results/cchihh_full_vs_nocc_blocks_ci")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    full_base = root / args.full_dir
    nocc_base = root / args.nocc_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Table 1
    table_rows = []

    # Plot style (as requested)
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
        full_dir = full_base / scale
        nocc_dir = nocc_base / scale

        finals_full, by_gen_full = load_group(full_dir, "CCHIHH_full_seed{seed}.txt", args.n)
        finals_nocc, by_gen_nocc = load_group(nocc_dir, "CCHIHH_noCC_seed{seed}.txt", args.n)

        m_full = mean(finals_full)
        s_full = stdev(finals_full) if len(finals_full) > 1 else 0.0
        m_nocc = mean(finals_nocc)
        s_nocc = stdev(finals_nocc) if len(finals_nocc) > 1 else 0.0

        p_value = mannwhitneyu(finals_full, finals_nocc, alternative="two-sided", method="auto").pvalue
        improvement = (m_nocc - m_full) / m_nocc * 100.0

        table_rows.append(
            [
                scale,
                f"{m_full:.6f} ± {s_full:.6f}",
                f"{m_nocc:.6f} ± {s_nocc:.6f}",
                f"{improvement:.4f}",
                f"{p_value:.9g}",
            ]
        )

        gens = sorted(set(by_gen_full.keys()) & set(by_gen_nocc.keys()))
        rows = []
        mean_full = []
        mean_nocc = []
        ci_full = []
        ci_nocc = []

        for g in gens:
            fvals = by_gen_full[g]
            nvals = by_gen_nocc[g]
            mf = mean(fvals)
            mn = mean(nvals)
            vf = var_pop(fvals)
            vn = var_pop(nvals)
            cf = Z95 * math.sqrt(max(vf, 0.0) / args.n)
            cn = Z95 * math.sqrt(max(vn, 0.0) / args.n)
            mean_full.append(mf)
            mean_nocc.append(mn)
            ci_full.append(cf)
            ci_nocc.append(cn)
            rows.append([g, mf, vf, cf, mn, vn, cn])

        write_csv(
            out_dir / f"{scale}_full_vs_nocc_blocks_mean_var_ci.csv",
            ["gen", "full_mean", "full_var", "full_ci95", "nocc_mean", "nocc_var", "nocc_ci95"],
            rows,
        )

        ax = axes[i]
        ax.plot(gens, mean_full, label="CC-HIHH-full", color="#1f77b4")
        ax.plot(gens, mean_nocc, label="CC-HIHH-noCC", color="#d62728")
        ax.fill_between(gens, [m - c for m, c in zip(mean_full, ci_full)], [m + c for m, c in zip(mean_full, ci_full)], color="#1f77b4", alpha=0.12)
        ax.fill_between(gens, [m - c for m, c in zip(mean_nocc, ci_nocc)], [m + c for m, c in zip(mean_nocc, ci_nocc)], color="#d62728", alpha=0.15)
        ax.set_title(scale, fontweight="bold")
        ax.set_xlabel("Generation")
        if i == 0:
            ax.set_ylabel("Best fitness")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    write_csv(
        out_dir / "table1_full_vs_nocc_blocks.csv",
        ["Problem", "CC-HIHH-full (Mean±Std)", "CC-HIHH-noCC (Mean±Std)", "Improvement (%)", "p-value"],
        table_rows,
    )

    pdf_path = out_dir / "CC-HIHH_Comparison_HighRes_blocks_definition.pdf"
    png_path = out_dir / "CC-HIHH_Comparison_HighRes_blocks_definition.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)

    print(f"Saved: {out_dir / 'table1_full_vs_nocc_blocks.csv'}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    for s in SCALES:
        print(f"Saved: {out_dir / f'{s}_full_vs_nocc_blocks_mean_var_ci.csv'}")


if __name__ == "__main__":
    main()
