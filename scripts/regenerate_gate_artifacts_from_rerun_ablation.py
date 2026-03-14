from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


GEN_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([0-9.+\-eE]+)")
FINAL_RE = re.compile(r"The\s+best\s+solution\s*=\s*([0-9.+\-eE]+)")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

SCALES = ["T100", "T200", "T500"]
COLORS = {
    "full": "#D62728",
    "gate": "#1F77B4",
}
Z95 = 1.96


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 300,
        }
    )
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    if b"\x00" in raw[:256]:
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                pass
    for enc in ("utf-8", "gbk", "latin1", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin1", errors="ignore")


def parse_log(path: Path) -> tuple[dict[int, float], float]:
    text = read_text_auto(path)
    series: dict[int, float] = {}
    final = None
    for line in text.splitlines():
        s = line.strip().replace("\x00", "")
        m = GEN_RE.match(s)
        if m:
            series[int(m.group(1))] = float(m.group(2))
            continue
        m = FINAL_RE.search(s)
        if m:
            final = float(m.group(1))
    if final is None and series:
        final = series[max(series)]
    if final is None:
        raise ValueError(f"failed to parse final best solution from {path}")
    return dict(sorted(series.items())), float(final)


def collect_variant(variant: str, scale: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    conv_rows: list[dict[str, float | int | str]] = []
    final_rows: list[dict[str, float | int | str]] = []

    if variant == "CCHIHH-full":
        paths = [RESULTS / "rerun_full" / f"cchihh_full_{scale}_s{seed}.log" for seed in range(1, 11)]
    elif variant == "CCHIHH-noGate":
        paths = [RESULTS / "rerun_ablation" / f"cchihh_noGate_{scale}_s{seed}.log" for seed in range(1, 11)]
    elif variant == "CCHIHH-noCC":
        paths = [RESULTS / "rerun_ablation" / f"cchihh_noCC_{scale}_s{seed}.log" for seed in range(1, 11)]
    elif variant == "CCHIHH-noHI":
        paths = [RESULTS / "rerun_ablation" / f"cchihh_noHI_{scale}_s{seed}.log" for seed in range(1, 11)]
    elif variant == "CCHIHH-noMig":
        paths = [RESULTS / "rerun_ablation" / f"cchihh_noMig_{scale}_s{seed}.log" for seed in range(1, 11)]
    elif variant == "CCHIHH-noCB":
        paths = [OUTPUTS / "cchihh_ablation_suite" / "bandit" / scale / f"fixed_ops_seed{seed}.txt" for seed in range(1, 11)]
    else:
        raise ValueError(f"unsupported variant: {variant}")

    for seed, path in enumerate(paths, start=1):
        series, final = parse_log(path)
        for gen, best_fit in series.items():
            conv_rows.append({"variant": variant, "scale": scale, "seed": seed, "gen": gen, "best_fit": best_fit})
        final_rows.append({"variant": variant, "scale": scale, "seed": seed, "best_fit": final})

    return pd.DataFrame(conv_rows), pd.DataFrame(final_rows)


def aggregate_curve(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("gen")["best_fit"]
        .agg(mean="mean", var=lambda s: float(np.var(s, ddof=0)), n="count")
        .reset_index()
    )
    agg["ci"] = Z95 * np.sqrt(agg["var"] / agg["n"].clip(lower=1))
    return agg


def pvalue_paired(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def save_pdf(fig: plt.Figure, path: Path) -> Path:
    try:
        fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_regen.pdf")
        fig.savefig(alt, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        return alt


def fig08(full_conv: pd.DataFrame, gate_conv: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for ax, scale in zip(axes, SCALES):
        full = aggregate_curve(full_conv[full_conv["scale"] == scale])
        gate = aggregate_curve(gate_conv[gate_conv["scale"] == scale])
        ax.plot(full["gen"], full["mean"], color=COLORS["full"], label="gate on")
        ax.fill_between(full["gen"], full["mean"] - full["ci"], full["mean"] + full["ci"], color=COLORS["full"], alpha=0.2)
        ax.plot(gate["gen"], gate["mean"], color=COLORS["gate"], label="gate off")
        ax.fill_between(gate["gen"], gate["mean"] - gate["ci"], gate["mean"] + gate["ci"], color=COLORS["gate"], alpha=0.2)
        ax.set_title(scale)
        ax.set_xlabel("Generation")
        if ax is axes[0]:
            ax.set_ylabel("Best fitness")
        ax.set_xlim(0, 10000)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    return save_pdf(fig, FIGURES / "fig08_ablation_gate.pdf")


def fig10(all_final: pd.DataFrame) -> Path:
    rows = [
        ("w/o CC", "CCHIHH-noCC"),
        ("w/o Islands", "CCHIHH-noHI"),
        ("w/o Bandit", "CCHIHH-noCB"),
        ("w/o Gating", "CCHIHH-noGate"),
        ("w/o Migration", "CCHIHH-noMig"),
    ]
    data = np.zeros((5, 3))
    pvals = np.ones((5, 3))

    for i, (_, variant) in enumerate(rows):
        for j, scale in enumerate(SCALES):
            full = all_final[(all_final["variant"] == "CCHIHH-full") & (all_final["scale"] == scale)].sort_values("seed")
            other = all_final[(all_final["variant"] == variant) & (all_final["scale"] == scale)].sort_values("seed")
            data[i, j] = (other["best_fit"].mean() - full["best_fit"].mean()) / other["best_fit"].mean() * 100.0
            pvals[i, j] = pvalue_paired(full["best_fit"].to_numpy(), other["best_fit"].to_numpy())

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(data, cmap="Reds", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(SCALES)
    ax.set_yticks(range(5))
    ax.set_yticklabels([r[0] for r in rows])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}%\n$p$={pvals[i, j]:.3f}", ha="center", va="center", fontsize=8)
            if pvals[i, j] >= 0.05:
                ax.add_patch(
                    patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linestyle="--", linewidth=1.0
                    )
                )

    ax.set_title("Scale-dependent component contribution")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Improvement (%)")
    fig.text(0.5, -0.02, "w/o Bandit uses OLD noCB/fixed_ops data.", ha="center", fontsize=8)
    return save_pdf(fig, FIGURES / "fig10_component_contribution.pdf")


def table8(full_final: pd.DataFrame, gate_final: pd.DataFrame) -> tuple[Path, Path]:
    csv_rows: list[dict[str, float | str]] = []
    tex_rows: list[list[str]] = []

    for scale in SCALES:
        full = full_final[full_final["scale"] == scale].sort_values("seed")
        gate = gate_final[gate_final["scale"] == scale].sort_values("seed")
        mf, sf = float(full["best_fit"].mean()), float(full["best_fit"].std(ddof=1))
        mg, sg = float(gate["best_fit"].mean()), float(gate["best_fit"].std(ddof=1))
        impr = (mg - mf) / mg * 100.0
        p = pvalue_paired(full["best_fit"].to_numpy(), gate["best_fit"].to_numpy())
        csv_rows.append(
            {
                "scale": scale,
                "full_mean": round(mf, 6),
                "full_std": round(sf, 6),
                "gate_off_mean": round(mg, 6),
                "gate_off_std": round(sg, 6),
                "improvement_pct": round(float(impr), 4),
                "wilcoxon_p": round(float(p), 6),
            }
        )
        tex_rows.append(
            [
                scale,
                f"{mf:.6f}$\\pm${sf:.6f}",
                f"{mg:.6f}$\\pm${sg:.6f}",
                f"{impr:.2f}",
                f"{p:.6f}",
            ]
        )

    csv_path = TABLES / "table8_ablation_gating.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scale", "full_mean", "full_std", "gate_off_mean", "gate_off_std", "improvement_pct", "wilcoxon_p"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    tex_path = TABLES / "table8_ablation_gating.tex"
    lines = [
        "\\begin{tabular}{lllll}",
        "\\hline",
        "Scale & Full & noGate & Impr.(\\%) & Wilcoxon $p$ \\\\",
        "\\hline",
    ]
    for row in tex_rows:
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, tex_path


def main() -> None:
    setup_style()

    conv_frames = []
    final_frames = []
    for variant in ["CCHIHH-full", "CCHIHH-noGate", "CCHIHH-noCC", "CCHIHH-noHI", "CCHIHH-noMig", "CCHIHH-noCB"]:
        for scale in SCALES:
            conv, final = collect_variant(variant, scale)
            conv_frames.append(conv)
            final_frames.append(final)

    all_conv = pd.concat(conv_frames, ignore_index=True)
    all_final = pd.concat(final_frames, ignore_index=True)

    full_conv = all_conv[all_conv["variant"] == "CCHIHH-full"].copy()
    gate_conv = all_conv[all_conv["variant"] == "CCHIHH-noGate"].copy()
    full_final = all_final[all_final["variant"] == "CCHIHH-full"].copy()
    gate_final = all_final[all_final["variant"] == "CCHIHH-noGate"].copy()

    fig08_path = fig08(full_conv, gate_conv)
    fig10_path = fig10(all_final)
    csv_path, tex_path = table8(full_final, gate_final)

    print(fig08_path)
    print(fig10_path)
    print(csv_path)
    print(tex_path)


if __name__ == "__main__":
    main()
