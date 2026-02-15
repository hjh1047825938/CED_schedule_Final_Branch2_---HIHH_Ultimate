import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


GEN_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([0-9.+\-eE]+)")

SCALES = ["T100", "T200", "T500"]
Z_95 = 1.96


def read_text_auto(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="ignore")
    for enc in ("utf-8", "utf-16", "gbk", "latin1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin1", errors="ignore")


def parse_curve(path: Path):
    text = read_text_auto(path)
    curve = {}
    for line in text.splitlines():
        m = GEN_RE.match(line.strip())
        if m:
            curve[int(m.group(1))] = float(m.group(2))
    return curve


def mean(xs):
    return sum(xs) / len(xs)


def var_pop(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def load_group(scale_dir: Path, pattern: str, n: int):
    by_gen = {}
    for seed in range(1, n + 1):
        p = scale_dir / pattern.format(seed=seed)
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")
        curve = parse_curve(p)
        if not curve:
            raise SystemExit(f"No generation curve parsed from: {p}")
        for g, v in curve.items():
            by_gen.setdefault(g, []).append(v)
    return by_gen


def build_stats(gens, group_vals, n):
    means = []
    vars_ = []
    cis = []
    for g in gens:
        arr = group_vals[g]
        m = mean(arr)
        v = var_pop(arr)
        ci = Z_95 * math.sqrt(max(v, 0.0) / max(1, n))
        means.append(m)
        vars_.append(v)
        cis.append(ci)
    return means, vars_, cis


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dir", default="results/cchihh_full_canonical_multiscale")
    parser.add_argument("--nocc_dir", default="results/cchihh_nocc_multiscale")
    parser.add_argument("--out_dir", default="results/cchihh_full_vs_nocc_ci")
    parser.add_argument("--n", type=int, default=10, help="Number of seeds per group.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    full_base = root / args.full_dir
    nocc_base = root / args.nocc_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=False, sharey=False)

    for i, scale in enumerate(SCALES):
        full_dir = full_base / scale
        nocc_dir = nocc_base / scale

        full_vals = load_group(full_dir, "CCHIHH_full_seed{seed}.txt", args.n)
        nocc_vals = load_group(nocc_dir, "CCHIHH_noCC_seed{seed}.txt", args.n)

        gens = sorted(set(full_vals.keys()) & set(nocc_vals.keys()))
        if not gens:
            raise SystemExit(f"No common generations found for {scale}.")

        mean_full, var_full, ci_full = build_stats(gens, full_vals, args.n)
        mean_nocc, var_nocc, ci_nocc = build_stats(gens, nocc_vals, args.n)

        rows = []
        for j, g in enumerate(gens):
            rows.append(
                [
                    g,
                    mean_full[j],
                    var_full[j],
                    ci_full[j],
                    mean_nocc[j],
                    var_nocc[j],
                    ci_nocc[j],
                ]
            )
        write_csv(
            out_dir / f"{scale}_full_vs_nocc_mean_var_ci.csv",
            [
                "gen",
                "full_mean",
                "full_var",
                "full_ci95",
                "nocc_mean",
                "nocc_var",
                "nocc_ci95",
            ],
            rows,
        )

        ax = axes[i]
        ax.plot(gens, mean_full, color="#1f77b4", linewidth=2, label="CC-HIHH-full")
        ax.fill_between(
            gens,
            [m - c for m, c in zip(mean_full, ci_full)],
            [m + c for m, c in zip(mean_full, ci_full)],
            color="#1f77b4",
            alpha=0.18,
        )
        ax.plot(gens, mean_nocc, color="#d62728", linewidth=2, label="CC-HIHH-noCC")
        ax.fill_between(
            gens,
            [m - c for m, c in zip(mean_nocc, ci_nocc)],
            [m + c for m, c in zip(mean_nocc, ci_nocc)],
            color="#d62728",
            alpha=0.18,
        )
        ax.set_title(scale)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best fitness")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("CC-HIHH-full vs CC-HIHH-noCC (mean ± 95% CI, seeds=10)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = out_dir / "cchihh_full_vs_nocc_mean_ci_multiscale.png"
    out_pdf = out_dir / "cchihh_full_vs_nocc_mean_ci_multiscale.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    for scale in SCALES:
        print(f"Saved: {out_dir / f'{scale}_full_vs_nocc_mean_var_ci.csv'}")


if __name__ == "__main__":
    main()
