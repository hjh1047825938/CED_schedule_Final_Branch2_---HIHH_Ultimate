import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


LOG_BEST_RE = re.compile(r"best_fit\s*=\s*([-+0-9.eE]+)")
FINAL_BEST_RE = re.compile(r"The best (?:scalar )?solution\s*=\s*([-+0-9.eE]+)")
TIME_RE = re.compile(r"Time\s*=\s*([-+0-9.eE]+)\s*s")
SCALE_RE = re.compile(r"_T(\d+)_seed(\d+)\.log$")
WEIGHT_RE = re.compile(r"weights_(full|shared)_(off|seq|dev)_T(\d+)_seed(\d+)\.csv$")
LEGACY_WEIGHT_RE = re.compile(r"op_weights_(offload|seq|dev)_T(\d+)_run(\d+)\.csv$")
REWARD_VAR_RE = re.compile(r"reward_var_(full|shared)_T(\d+)_seed(\d+)\.csv$")
LEGACY_FULL_LOG_RE = re.compile(r"CCHIHH_full_seed(\d+)\.txt$")
LEGACY_DSAC_RE = re.compile(r"DSAC_DE_seed(\d+)\.txt$")
LEGACY_REWARD_RE = re.compile(r"op_rewards_T(\d+)_run(\d+)\.csv$")


def parse_log_metrics(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\x00", "")
    bests = [float(m.group(1)) for m in LOG_BEST_RE.finditer(text)]
    finals = [float(m.group(1)) for m in FINAL_BEST_RE.finditer(text)]
    times = [float(m.group(1)) for m in TIME_RE.finditer(text)]
    if not bests and not finals:
        return None
    best = finals[-1] if finals else bests[-1]
    runtime = times[-1] if times else math.nan
    return {"BestFit": best, "Runtime_s": runtime}


def safe_wilcoxon(x, y):
    try:
        stat, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox", correction=False)
        return float(stat), float(p)
    except Exception:
        return math.nan, math.nan


def fmt_mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    std = arr.std(ddof=1) if arr.size > 1 else 0.0
    return f"{arr.mean():.6f}±{std:.6f}"


def improvement_pct(full_vals, other_vals):
    full_mean = float(np.mean(full_vals))
    other_mean = float(np.mean(other_vals))
    if other_mean == 0.0:
        return math.nan
    return (other_mean - full_mean) / other_mean * 100.0


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def collect_solver_results(results_dir: Path, full_log_dir: Path | None = None):
    dsac_rows = []
    shared_rows = []
    for path in sorted(results_dir.rglob("*.log")):
        m = SCALE_RE.search(path.name)
        if not m:
            continue
        scale = f"T{m.group(1)}"
        seed = int(m.group(2))
        metrics = parse_log_metrics(path)
        if metrics is None:
            continue
        if path.name.startswith("dsac_de_"):
            dsac_rows.append({"Scale": scale, "Seed": seed, **metrics})
        elif path.name.startswith("cchihh_full_"):
            shared_rows.append({"Scale": scale, "Variant": "full", "Seed": seed, "BestFit": metrics["BestFit"]})
        elif path.name.startswith("cchihh_shared_"):
            shared_rows.append({"Scale": scale, "Variant": "sharedBandit", "Seed": seed, "BestFit": metrics["BestFit"]})

    if full_log_dir and full_log_dir.exists():
        for path in sorted(full_log_dir.rglob("CCHIHH_full_seed*.txt")):
            m = LEGACY_FULL_LOG_RE.match(path.name)
            if not m:
                continue
            scale = path.parent.name
            if not re.fullmatch(r"T\d+", scale):
                continue
            metrics = parse_log_metrics(path)
            if metrics is None:
                continue
            shared_rows.append(
                {
                    "Scale": scale,
                    "Variant": "full",
                    "Seed": int(m.group(1)),
                    "BestFit": metrics["BestFit"],
                }
            )
    dsac_df = pd.DataFrame(dsac_rows, columns=["Scale", "Seed", "BestFit", "Runtime_s"])
    shared_df = pd.DataFrame(shared_rows, columns=["Scale", "Variant", "Seed", "BestFit"])
    return dsac_df, shared_df


def build_pairwise_tables(dsac_df: pd.DataFrame, shared_df: pd.DataFrame):
    shared_summary = []
    dsac_summary = []
    if shared_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    for scale in sorted(shared_df["Scale"].dropna().unique(), key=lambda x: int(x[1:])):
        full = shared_df[(shared_df["Scale"] == scale) & (shared_df["Variant"] == "full")].sort_values("Seed")
        shared = shared_df[(shared_df["Scale"] == scale) & (shared_df["Variant"] == "sharedBandit")].sort_values("Seed")
        if not full.empty and not shared.empty:
            _, p = safe_wilcoxon(full["BestFit"].to_list(), shared["BestFit"].to_list())
            shared_summary.append(
                {
                    "Problem": scale,
                    "CCHIHH-full (Mean±Std)": fmt_mean_std(full["BestFit"]),
                    "CCHIHH-sharedBandit (Mean±Std)": fmt_mean_std(shared["BestFit"]),
                    "Improvement(%)": improvement_pct(full["BestFit"], shared["BestFit"]),
                    "p-value": p,
                }
            )

        dsac = dsac_df[dsac_df["Scale"] == scale].sort_values("Seed")
        if not full.empty and not dsac.empty:
            common = sorted(set(full["Seed"]).intersection(set(dsac["Seed"])))
            full_common = full[full["Seed"].isin(common)].sort_values("Seed")
            dsac_common = dsac[dsac["Seed"].isin(common)].sort_values("Seed")
            _, p = safe_wilcoxon(full_common["BestFit"].to_list(), dsac_common["BestFit"].to_list())
            dsac_summary.append(
                {
                    "Problem": scale,
                    "CCHIHH-full (Mean±Std)": fmt_mean_std(full_common["BestFit"]),
                    "DSAC-DE (Mean±Std)": fmt_mean_std(dsac_common["BestFit"]),
                    "Improvement(%)": improvement_pct(full_common["BestFit"], dsac_common["BestFit"]),
                    "p-value": p,
                }
            )
    return pd.DataFrame(shared_summary), pd.DataFrame(dsac_summary)


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        " | ".join(df.columns).replace("|", "&") + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.6g}" if math.isfinite(v) else "--")
            else:
                vals.append(str(v).replace("+/-", "$\\pm$"))
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_boxplot(shared_df: pd.DataFrame, out_path: Path):
    scales = sorted(shared_df["Scale"].dropna().unique(), key=lambda x: int(x[1:]))
    fig, axes = plt.subplots(1, len(scales), figsize=(5 * max(len(scales), 1), 4), squeeze=False)
    for idx, scale in enumerate(scales):
        ax = axes[0][idx]
        sub = shared_df[shared_df["Scale"] == scale]
        data = [
            sub[sub["Variant"] == "full"]["BestFit"].to_list(),
            sub[sub["Variant"] == "sharedBandit"]["BestFit"].to_list(),
        ]
        ax.boxplot(data, tick_labels=["full", "sharedBandit"])
        ax.set_title(scale)
        ax.set_ylabel("BestFit")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_reward_variance_curves(results_dir: Path, full_reward_dir: Path | None = None):
    curves = {}
    for path in sorted(results_dir.rglob("reward_var_*.csv")):
        m = REWARD_VAR_RE.match(path.name)
        if not m:
            continue
        variant = m.group(1)
        scale = f"T{m.group(2)}"
        seed = int(m.group(3))
        df = pd.read_csv(path)
        df["Variant"] = variant
        df["Scale"] = scale
        df["Seed"] = seed
        curves.setdefault((variant, scale), []).append(df)

    if full_reward_dir and full_reward_dir.exists():
        for path in sorted(full_reward_dir.rglob("op_rewards_T*_run*.csv")):
            m = LEGACY_REWARD_RE.match(path.name)
            if not m:
                continue
            scale = f"T{m.group(1)}"
            seed = int(m.group(2)) + 1
            df = pd.read_csv(path)
            if df.empty or "gen" not in df.columns or "reward" not in df.columns:
                continue
            agg = (
                df.groupby(["gen", "block_id"])["reward"]
                .agg(["mean", "var", "min", "max"])
                .reset_index()
                .rename(
                    columns={
                        "gen": "generation",
                        "mean": "reward_mean",
                        "var": "reward_variance",
                        "min": "reward_min",
                        "max": "reward_max",
                    }
                )
            )
            agg["reward_variance"] = agg["reward_variance"].fillna(0.0)
            agg["Variant"] = "full"
            agg["Scale"] = scale
            agg["Seed"] = seed
            curves.setdefault(("full", scale), []).append(agg)
    return curves


def plot_reward_variance(results_dir: Path, out_dir: Path, full_reward_dir: Path | None = None):
    curves = load_reward_variance_curves(results_dir, full_reward_dir)
    scales = sorted({scale for (_, scale) in curves.keys()}, key=lambda x: int(x[1:]))
    for scale in scales:
        fig, ax = plt.subplots(figsize=(6, 4))
        full_parts = curves.get(("full", scale), [])
        if full_parts:
            full_df = pd.concat(full_parts, ignore_index=True)
            grouped = full_df.groupby(["generation", "block_id"], as_index=False)["reward_variance"].mean()
            label_map = {0: "full-offload", 1: "full-seq", 2: "full-dev", -1: "full"}
            for block_id, sub in grouped.groupby("block_id"):
                ax.plot(sub["generation"], sub["reward_variance"], label=label_map.get(int(block_id), f"full-{block_id}"))
        shared_parts = curves.get(("shared", scale), [])
        if shared_parts:
            shared_df = pd.concat(shared_parts, ignore_index=True)
            shared_grouped = shared_df.groupby("generation", as_index=False)["reward_variance"].mean()
            ax.plot(shared_grouped["generation"], shared_grouped["reward_variance"], label="sharedBandit", linewidth=2.2, linestyle="--")
        ax.set_title(f"Reward Variance - {scale}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Reward variance")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"reward_variance_{scale}.png", dpi=200)
        plt.close(fig)


def collect_weight_logs(results_dir: Path, full_weight_dir: Path | None = None):
    items = []
    for path in sorted(results_dir.rglob("weights_*.csv")):
        m = WEIGHT_RE.match(path.name)
        if not m:
            continue
        items.append(
            {
                "path": path,
                "variant": m.group(1),
                "block": m.group(2),
                "scale": f"T{m.group(3)}",
                "seed": int(m.group(4)),
            }
        )
    if full_weight_dir and full_weight_dir.exists():
        for path in sorted(full_weight_dir.rglob("op_weights_*_T*_run*.csv")):
            m = LEGACY_WEIGHT_RE.match(path.name)
            if not m:
                continue
            block = m.group(1)
            if block == "offload":
                block = "off"
            items.append(
                {
                    "path": path,
                    "variant": "full",
                    "block": block,
                    "scale": f"T{m.group(2)}",
                    "seed": int(m.group(3)) + 1,
                }
            )
    return items


def plot_weight_heatmaps(results_dir: Path, out_dir: Path, full_weight_dir: Path | None = None):
    for item in collect_weight_logs(results_dir, full_weight_dir):
        df = pd.read_csv(item["path"])
        if df.empty:
            continue
        last_gen = df["gen"].max()
        sub = df[df["gen"] == last_gen].sort_values("op_id")
        weight_cols = [c for c in sub.columns if re.fullmatch(r"w\d+", c)]
        if not weight_cols:
            continue
        matrix = sub[weight_cols].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        ax.set_title(f"{item['variant']} {item['block']} {item['scale']} seed{item['seed']}")
        ax.set_xlabel("State feature")
        ax.set_ylabel("Operator")
        ax.set_xticks(range(len(weight_cols)), weight_cols)
        ax.set_yticks(range(len(sub)), [f"op{int(v)}" for v in sub["op_id"]])
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{item['variant']}_{item['block']}_{item['scale']}_seed{item['seed']}.png", dpi=200)
        plt.close(fig)


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8")


def sanitize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    rename_map = {}
    for col in cleaned.columns:
        new_col = str(col)
        if "Mean" in new_col and "Std" in new_col:
            new_col = re.sub(r"Mean.*Std", "Mean+/-Std", new_col)
        rename_map[col] = new_col
    cleaned = cleaned.rename(columns=rename_map)
    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].astype(str).str.replace(
                r"(?<=\d)[^\dA-Za-z.+-]+(?=\d)", " +/- ", regex=True
            )
    return cleaned


def main():
    ap = argparse.ArgumentParser(description="Analyze DSAC-DE and shared-bandit experiments.")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="results/analysis")
    ap.add_argument("--full_log_dir", default="")
    ap.add_argument("--full_reward_dir", default="")
    ap.add_argument("--full_weight_dir", default="")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    full_log_dir = Path(args.full_log_dir) if args.full_log_dir else None
    full_reward_dir = Path(args.full_reward_dir) if args.full_reward_dir else None
    full_weight_dir = Path(args.full_weight_dir) if args.full_weight_dir else None
    ensure_dir(out_dir)

    dsac_df, shared_df = collect_solver_results(results_dir, full_log_dir)
    shared_summary_df, dsac_summary_df = build_pairwise_tables(dsac_df, shared_df)
    shared_summary_df = sanitize_summary_df(shared_summary_df)
    dsac_summary_df = sanitize_summary_df(dsac_summary_df)

    if not dsac_df.empty:
        write_csv(dsac_df.sort_values(["Scale", "Seed"]), out_dir / "dsac_de_results.csv")
    if not shared_df.empty:
        write_csv(shared_df.sort_values(["Scale", "Variant", "Seed"]), out_dir / "shared_bandit_comparison.csv")
    if not shared_summary_df.empty:
        write_csv(shared_summary_df, out_dir / "shared_bandit_summary.csv")
        write_latex_table(
            shared_summary_df,
            out_dir / "table_shared_bandit.tex",
            "Shared-bandit ablation results.",
            "tab:shared_bandit_ablation",
        )
    if not dsac_summary_df.empty:
        write_csv(dsac_summary_df, out_dir / "dsac_de_vs_full_summary.csv")
        write_latex_table(
            dsac_summary_df,
            out_dir / "table_dsac_de.tex",
            "CCHIHH-full versus DSAC-DE configured with paper parameters.",
            "tab:dsac_de_vs_full",
        )

    if not shared_df.empty:
        plot_boxplot(shared_df, out_dir / "shared_vs_per_block_boxplot.png")
    plot_reward_variance(results_dir, out_dir, full_reward_dir)
    plot_weight_heatmaps(results_dir, out_dir, full_weight_dir)

    print(out_dir / "dsac_de_results.csv")
    print(out_dir / "shared_bandit_comparison.csv")
    print(out_dir / "table_shared_bandit.tex")
    print(out_dir / "table_dsac_de.tex")


if __name__ == "__main__":
    main()
