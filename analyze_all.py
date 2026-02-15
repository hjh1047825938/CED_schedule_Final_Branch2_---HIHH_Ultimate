#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATA_DIR = "results"
OUTPUT_DIR = "figures"
SCALES = [100, 200, 500]
BLOCKS = ["offload", "seq", "dev"]
N_RUNS = 10

BLOCK_OPS = {
    "offload": ["GA", "DE", "BITFLIP", "RESAMPLE"],
    "seq": ["GA", "SWAP", "VNS", "RESAMPLE"],
    "dev": ["DE", "GDE", "LEVY", "RESAMPLE"],
}

STRATEGY_VARIANTS = {
    "CB": "full",
    "Random": "random",
    "RoundRobin": "roundrobin",
    "FixedBest": "fixedbest",
}


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def ci95(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.ndim == 1:
        a = a[:, None]
    n = a.shape[0]
    m = np.nanmean(a, axis=0)
    if n <= 1:
        return m, np.zeros_like(m)
    s = np.nanstd(a, axis=0, ddof=1)
    hw = 1.96 * s / np.sqrt(n)
    return m, hw


def parse_best_from_log(path: Path) -> Optional[float]:
    final = None
    p1 = re.compile(r"The best solution\s*=\s*([0-9eE+\-.]+)")
    p2 = re.compile(r"best_fit\s*=\s*([0-9eE+\-.]+)")
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for line in text.splitlines():
        m1 = p1.search(line)
        if m1:
            final = float(m1.group(1))
        m2 = p2.search(line)
        if m2:
            final = float(m2.group(1))
    return final


def parse_curve_from_log(path: Path) -> Optional[pd.DataFrame]:
    p = re.compile(r"Gen\s+(\d+):\s+best_fit\s*=\s*([0-9eE+\-.]+)")
    rows = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for line in text.splitlines():
        m = p.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["gen", "best_fit"])


class OperatorAnalyzer:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.style.use("seaborn-v0_8-paper")
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 12

    def _find_files(self, patterns: List[str]) -> List[Path]:
        out: List[Path] = []
        for pat in patterns:
            out.extend(self.data_dir.glob(pat))
            out.extend(self.data_dir.glob(f"**/{pat}"))
        uniq = sorted(set(out))
        return [p for p in uniq if p.exists()]

    def _variant_aliases(self, variant: str) -> List[str]:
        v = variant.lower()
        aliases = [v]
        if v == "full": aliases += ["bandit_adaptive", "cchihh_full"]
        if v == "random": aliases += ["rand", "random_ops"]
        if v == "roundrobin": aliases += ["rr", "round_robin"]
        if v == "fixedbest": aliases += ["fixed", "fixed_ops"]
        if v == "reduced": aliases += ["reducedops"]
        if v == "top": aliases += ["topops"]
        return list(dict.fromkeys(aliases))

    def load_frequency_data(self, variant: str, scale: int) -> List[pd.DataFrame]:
        files: List[Path] = []
        for a in self._variant_aliases(variant):
            files += self._find_files([
                f"op_freq_{a}_T{scale}_run*.csv",
                f"*{a}*opstats*T{scale}*seed*.csv",
                f"*{a}*opstats*seed*.csv",
            ])
        files = sorted(set([f for f in files if f.suffix.lower() == ".csv"]))
        out = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if "gen" in df.columns:
                    out.append(df)
            except Exception:
                pass
        return out

    def load_weight_data(self, block: str, scale: int) -> List[pd.DataFrame]:
        files = self._find_files([
            f"op_weights_{block}_T{scale}_run*.csv",
            f"weights_{block}_T{scale}_run*.csv",
            f"*weights*{block}*T{scale}*run*.csv",
        ])
        out = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if {"gen", "op_id", "norm"}.issubset(df.columns):
                    out.append(df)
            except Exception:
                pass
        return out

    def load_reward_data(self, scale: int) -> List[pd.DataFrame]:
        files = self._find_files([
            f"op_rewards_T{scale}_run*.csv",
            f"*rewards*T{scale}*run*.csv",
        ])
        out = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if {"gen", "block_id", "op_id", "reward"}.issubset(df.columns):
                    out.append(df)
            except Exception:
                pass
        return out

    def _frequency_columns(self, block: str) -> List[str]:
        return [f"{block}_{op}" for op in BLOCK_OPS[block]]

    def _stack_runs(self, runs: List[pd.DataFrame], cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not runs:
            return np.array([]), np.empty((0, 0))
        gens = sorted(set.intersection(*[set(df["gen"].tolist()) for df in runs]))
        if not gens:
            gens = sorted(set.union(*[set(df["gen"].tolist()) for df in runs]))
        mats = []
        for df in runs:
            d = df.set_index("gen").reindex(gens)
            mats.append(d[cols].to_numpy(dtype=float))
        arr = np.stack(mats, axis=0)
        return np.array(gens), arr

    def plot_frequency_evolution(self):
        fig, axes = plt.subplots(3, 3, figsize=(12, 8), dpi=300, sharex=False, sharey=False)
        cmap = plt.get_cmap("tab10")
        for i, block in enumerate(BLOCKS):
            cols = self._frequency_columns(block)
            for j, scale in enumerate(SCALES):
                ax = axes[i, j]
                runs = self.load_frequency_data("full", scale)
                if not runs:
                    ax.set_title(f"{block}-T{scale} (no data)")
                    continue
                gens, arr = self._stack_runs(runs, cols)
                if arr.size == 0:
                    ax.set_title(f"{block}-T{scale} (no aligned data)")
                    continue
                for k, op in enumerate(BLOCK_OPS[block]):
                    y = arr[:, :, k]
                    m, hw = ci95(y)
                    ax.plot(gens, m, color=cmap(k), linewidth=1.5, label=op)
                    ax.fill_between(gens, m - hw, m + hw, color=cmap(k), alpha=0.15)
                ax.set_title(f"{block.capitalize()} T{scale}")
                ax.grid(True, color="lightgray", alpha=0.3)
                if i == 2:
                    ax.set_xlabel("Generation")
                if j == 0:
                    ax.set_ylabel("Selection Frequency")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        out = self.output_dir / "fig_operator_frequency_evolution.pdf"
        fig.savefig(out, dpi=300)
        plt.close(fig)

    def plot_frequency_heatmap(self):
        stages = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000)]
        for scale in SCALES:
            runs = self.load_frequency_data("full", scale)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
            for i, block in enumerate(BLOCKS):
                cols = self._frequency_columns(block)
                vals = np.zeros((len(cols), len(stages)))
                if runs:
                    all_df = []
                    for r in runs:
                        rr = r[["gen"] + cols].copy()
                        rr["run"] = len(all_df)
                        all_df.append(rr)
                    cat = pd.concat(all_df, ignore_index=True)
                    for c, (g0, g1) in enumerate(stages):
                        sel = cat[(cat["gen"] >= g0) & (cat["gen"] <= g1)]
                        if len(sel) > 0:
                            vals[:, c] = sel[cols].mean().to_numpy()
                hm = pd.DataFrame(vals, index=BLOCK_OPS[block], columns=[f"{a//1000}-{b//1000}k" for a, b in stages])
                sns.heatmap(hm, cmap="YlOrRd", ax=axes[i], cbar=(i == 2))
                axes[i].set_title(block.capitalize())
                axes[i].set_xlabel("Stage")
                axes[i].set_ylabel("Operator")
            fig.tight_layout()
            fig.savefig(self.output_dir / f"fig_operator_heatmap_T{scale}.pdf", dpi=300)
            plt.close(fig)

    def generate_early_late_table(self):
        stage_map = {
            "early": (0, 2000),
            "late": (8000, 10000),
        }
        for block in BLOCKS:
            rows = []
            cols = self._frequency_columns(block)
            for op, col in zip(BLOCK_OPS[block], cols):
                early_vals = []
                late_vals = []
                for scale in SCALES:
                    runs = self.load_frequency_data("full", scale)
                    for r in runs:
                        early_vals.extend(r[(r["gen"] >= stage_map["early"][0]) & (r["gen"] <= stage_map["early"][1])][col].tolist())
                        late_vals.extend(r[(r["gen"] >= stage_map["late"][0]) & (r["gen"] <= stage_map["late"][1])][col].tolist())
                e = float(np.mean(early_vals)) if early_vals else np.nan
                l = float(np.mean(late_vals)) if late_vals else np.nan
                rows.append([op, e, l, l - e])
            df = pd.DataFrame(rows, columns=["Operator", "Early (0-2k)", "Late (8k-10k)", "Change"])
            out = self.output_dir / f"table_early_late_{block}.tex"
            ensure_parent(out)
            df.to_latex(out, index=False, float_format=lambda x: f"{x:.4f}")

    def plot_weight_evolution(self):
        scale = 500
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        cmap = plt.get_cmap("Set2")
        for i, block in enumerate(BLOCKS):
            runs = self.load_weight_data(block, scale)
            if not runs:
                axes[i].set_title(f"{block} (no data)")
                continue
            ops = sorted(set(int(x) for df in runs for x in df["op_id"].unique().tolist()))
            for oi, op in enumerate(ops):
                curves = []
                gens_ref = None
                for df in runs:
                    d = df[df["op_id"] == op][["gen", "norm"]].dropna()
                    if d.empty:
                        continue
                    if gens_ref is None:
                        gens_ref = d["gen"].to_numpy()
                    curves.append(d["norm"].to_numpy())
                if not curves:
                    continue
                min_len = min(len(c) for c in curves)
                arr = np.stack([c[:min_len] for c in curves], axis=0)
                gens = gens_ref[:min_len]
                m, hw = ci95(arr)
                axes[i].plot(gens, m, color=cmap(oi % 8), label=f"op{op}")
                axes[i].fill_between(gens, m - hw, m + hw, color=cmap(oi % 8), alpha=0.15)
            axes[i].axvline(2000, color="gray", linestyle="--", linewidth=1)
            axes[i].set_title(block.capitalize())
            axes[i].set_xlabel("Generation")
            axes[i].set_ylabel("Weight Norm")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        fig.savefig(self.output_dir / "fig_weight_evolution.pdf", dpi=300)
        plt.close(fig)

    def plot_weight_dimensions(self):
        target = {"offload": 1, "seq": 1, "dev": 1}
        fig, axes = plt.subplots(3, 7, figsize=(14, 8), dpi=300, sharex=False)
        for r, block in enumerate(BLOCKS):
            runs = self.load_weight_data(block, 500)
            if not runs:
                continue
            opid = target[block]
            for c in range(7):
                curves = []
                gens_ref = None
                col = f"w{c}"
                for df in runs:
                    d = df[df["op_id"] == opid][["gen", col]].dropna()
                    if d.empty:
                        continue
                    if gens_ref is None:
                        gens_ref = d["gen"].to_numpy()
                    curves.append(d[col].to_numpy())
                if not curves:
                    continue
                min_len = min(len(x) for x in curves)
                arr = np.stack([x[:min_len] for x in curves], axis=0)
                gens = gens_ref[:min_len]
                m, _ = ci95(arr)
                axes[r, c].plot(gens, m, linewidth=1.2)
                axes[r, c].set_title(f"{block}-w{c}")
                axes[r, c].grid(True, color="lightgray", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "fig_weight_dimensions.pdf", dpi=300)
        plt.close(fig)

    def calculate_contribution(self) -> pd.DataFrame:
        rows = []
        for scale in SCALES:
            freq_runs = self.load_frequency_data("full", scale)
            rew_runs = self.load_reward_data(scale)
            if not freq_runs or not rew_runs:
                continue
            freq_mean = pd.concat(freq_runs, ignore_index=True).mean(numeric_only=True)
            rew = pd.concat(rew_runs, ignore_index=True)
            for block_id, block in enumerate(BLOCKS):
                ops = BLOCK_OPS[block]
                for op_id, op_name in enumerate(ops):
                    col = f"{block}_{op_name}"
                    freq = float(freq_mean.get(col, np.nan))
                    rr = rew[(rew["block_id"] == block_id) & (rew["op_id"] == op_id)]
                    avg_reward = float(rr["reward"].mean()) if not rr.empty else np.nan
                    contribution = freq * avg_reward if not (np.isnan(freq) or np.isnan(avg_reward)) else np.nan
                    rows.append([scale, block, op_name, freq, avg_reward, contribution])
        out = pd.DataFrame(rows, columns=["scale", "block", "operator", "freq", "avg_reward", "contribution"])
        if out.empty:
            return out
        out["rank"] = out.groupby(["scale", "block"])["contribution"].rank(ascending=False, method="dense")
        out.to_csv(self.output_dir / "operator_contribution.csv", index=False)
        return out

    def generate_contribution_table(self):
        df = self.calculate_contribution()
        if df.empty:
            return
        table = df.copy()
        table["freq"] = table["freq"] * 100.0
        out = self.output_dir / "table_operator_contribution.tex"
        table.to_latex(out, index=False, float_format=lambda x: f"{x:.4f}")

    def plot_contribution_bars(self):
        df = self.calculate_contribution()
        if df.empty:
            return
        fig, axes = plt.subplots(3, 3, figsize=(12, 8), dpi=300)
        for i, block in enumerate(BLOCKS):
            for j, scale in enumerate(SCALES):
                ax = axes[i, j]
                d = df[(df["block"] == block) & (df["scale"] == scale)]
                if d.empty:
                    ax.set_title(f"{block}-T{scale} (no data)")
                    continue
                ax.bar(d["operator"], d["contribution"], color=sns.color_palette("tab10", len(d)))
                ax.set_title(f"{block.capitalize()} T{scale}")
                ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(self.output_dir / "fig_operator_contribution.pdf", dpi=300)
        plt.close(fig)

    def plot_cross_scale_comparison(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        for i, block in enumerate(BLOCKS):
            cols = self._frequency_columns(block)
            freq_by_scale = []
            for scale in SCALES:
                runs = self.load_frequency_data("full", scale)
                if not runs:
                    freq_by_scale.append(np.zeros(len(cols)))
                    continue
                d = pd.concat(runs, ignore_index=True)
                freq_by_scale.append(d[cols].mean().to_numpy())
            x = np.arange(len(cols))
            w = 0.25
            for j, scale in enumerate(SCALES):
                axes[i].bar(x + (j - 1) * w, freq_by_scale[j], width=w, label=f"T{scale}")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(BLOCK_OPS[block], rotation=30)
            axes[i].set_title(block.capitalize())
            axes[i].set_ylabel("Mean Frequency")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        fig.savefig(self.output_dir / "fig_cross_scale_comparison.pdf", dpi=300)
        plt.close(fig)

    def calculate_entropy(self, frequencies: np.ndarray) -> float:
        p = np.array(frequencies, dtype=float)
        s = p.sum()
        if s <= 0:
            return 0.0
        p = p / s
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def plot_diversity_vs_scale(self):
        rows = []
        for block in BLOCKS:
            cols = self._frequency_columns(block)
            for scale in SCALES:
                runs = self.load_frequency_data("full", scale)
                ents = []
                for r in runs:
                    ents.append(self.calculate_entropy(r[cols].mean().to_numpy()))
                rows.append([scale, block, float(np.mean(ents)) if ents else np.nan, float(np.std(ents, ddof=1)) if len(ents) > 1 else 0.0])
        df = pd.DataFrame(rows, columns=["scale", "block", "entropy", "std"])
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        for block in BLOCKS:
            d = df[df["block"] == block]
            ax.errorbar(d["scale"], d["entropy"], yerr=d["std"], marker="o", label=block)
        ax.set_xlabel("Problem Scale")
        ax.set_ylabel("Shannon Entropy")
        ax.grid(True, color="lightgray", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(self.output_dir / "fig_diversity_vs_scale.pdf", dpi=300)
        plt.close(fig)

        out_rows = []
        for _, r in df.iterrows():
            runs = self.load_frequency_data("full", int(r["scale"]))
            cols = self._frequency_columns(r["block"])
            dom = ""
            if runs:
                m = pd.concat(runs, ignore_index=True)[cols].mean()
                tops = []
                for col, v in m.items():
                    if v > 0.3:
                        tops.append(f"{col.split('_', 1)[1]}({v*100:.1f}%)")
                dom = ", ".join(tops) if tops else "Balanced"
            out_rows.append([int(r["scale"]), r["block"], dom, r["entropy"], ""])
        pd.DataFrame(out_rows, columns=["Scale", "Block", "Dominant Ops (>30%)", "Entropy", "Interpretation"]).to_latex(
            self.output_dir / "table_scale_dependent_entropy.tex", index=False, float_format=lambda x: f"{x:.4f}"
        )

    def _load_strategy_finals(self, variant: str, scale: int) -> List[float]:
        vals = []
        aliases = self._variant_aliases(variant)
        files = []
        for a in aliases:
            files += self._find_files([
                f"final_results_{a}_T{scale}.txt",
                f"*{a}*T{scale}*run*.txt",
                f"*{a}*seed*.txt",
            ])
        for f in sorted(set(files)):
            v = parse_best_from_log(f)
            if v is not None:
                vals.append(v)
        return vals

    def compare_selection_strategies(self):
        rows = []
        for scale in SCALES:
            base = self._load_strategy_finals("full", scale)
            for label, variant in STRATEGY_VARIANTS.items():
                vals = self._load_strategy_finals(variant, scale)
                if vals:
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                else:
                    mean, std = np.nan, np.nan
                pval = np.nan
                if label != "CB" and base and vals and len(base) == len(vals):
                    try:
                        _, pval = stats.wilcoxon(base, vals)
                    except Exception:
                        pval = np.nan
                rows.append([scale, label, mean, std, pval])
        df = pd.DataFrame(rows, columns=["Scale", "Strategy", "Mean", "Std", "p_value_vs_CB"])
        df.to_csv(self.output_dir / "strategy_comparison.csv", index=False)
        df.to_latex(self.output_dir / "table_strategy_comparison.tex", index=False, float_format=lambda x: f"{x:.6f}")

    def plot_strategy_convergence(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        for i, scale in enumerate(SCALES):
            ax = axes[i]
            for label, variant in STRATEGY_VARIANTS.items():
                curves = []
                for a in self._variant_aliases(variant):
                    files = self._find_files([f"*{a}*T{scale}*run*.txt", f"*{a}*seed*.txt"])
                    for f in files:
                        d = parse_curve_from_log(f)
                        if d is not None:
                            curves.append(d)
                if not curves:
                    continue
                common = sorted(set.intersection(*[set(c["gen"].tolist()) for c in curves]))
                if not common:
                    continue
                arr = np.stack([c.set_index("gen").reindex(common)["best_fit"].to_numpy() for c in curves], axis=0)
                m, hw = ci95(arr)
                ax.plot(common, m, label=label)
                ax.fill_between(common, m - hw, m + hw, alpha=0.15)
            ax.set_title(f"T{scale}")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Best Fitness")
            ax.grid(True, color="lightgray", alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        fig.savefig(self.output_dir / "fig_selection_strategy_comparison.pdf", dpi=300)
        plt.close(fig)

    def analyze_gating_effect(self):
        rows = []
        for scale in SCALES:
            files = self._find_files([f"global_stats_full_T{scale}_run*.csv", f"*global_stats*full*T{scale}*.csv"])
            if not files:
                continue
            agg = []
            for f in files:
                try:
                    d = pd.read_csv(f)
                except Exception:
                    continue
                if d.empty:
                    continue
                blk = float(d["gate_blocked_count"].iloc[-1]) if "gate_blocked_count" in d.columns else np.nan
                fb = float(d["gate_fallback_count"].iloc[-1]) if "gate_fallback_count" in d.columns else np.nan
                agg.append((blk, fb))
            if not agg:
                continue
            b = np.nanmean([x[0] for x in agg])
            f = np.nanmean([x[1] for x in agg])
            total = b + f
            rows.append([scale, total, b, f, (b / total * 100.0) if total > 0 else np.nan])
        if rows:
            pd.DataFrame(rows, columns=["Scale", "Total RESAMPLE Attempts", "Blocked", "Fallback", "Block Rate(%)"]).to_latex(
                self.output_dir / "table_gating_statistics.tex", index=False, float_format=lambda x: f"{x:.4f}"
            )

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        plotted = False
        for scale in SCALES:
            files = self._find_files([f"op_freq_full_T{scale}_run*.csv"])
            if not files:
                continue
            runs = [pd.read_csv(f) for f in files if f.exists()]
            if not runs:
                continue
            col_candidates = ["offload_RESAMPLE", "seq_RESAMPLE", "dev_RESAMPLE"]
            for c in col_candidates:
                if c not in runs[0].columns:
                    break
            else:
                common = sorted(set.intersection(*[set(r["gen"].tolist()) for r in runs]))
                arr = np.stack([r.set_index("gen").reindex(common)[col_candidates].mean(axis=1).to_numpy() for r in runs], axis=0)
                m, _ = ci95(arr)
                ax.plot(common, m, label=f"T{scale}")
                plotted = True
        if plotted:
            ax.set_xlabel("Generation")
            ax.set_ylabel("RESAMPLE frequency")
            ax.legend(frameon=False)
            ax.grid(True, color="lightgray", alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.output_dir / "fig_gating_effect.pdf", dpi=300)
        plt.close(fig)

    def generate_all_figures(self):
        self.plot_frequency_evolution()
        self.plot_frequency_heatmap()
        self.plot_weight_evolution()
        self.plot_weight_dimensions()
        self.plot_contribution_bars()
        self.plot_cross_scale_comparison()
        self.plot_diversity_vs_scale()
        self.plot_strategy_convergence()
        self.analyze_gating_effect()

    def generate_all_tables(self):
        self.generate_early_late_table()
        self.generate_contribution_table()
        self.compare_selection_strategies()
        # Placeholder table file for reduced vs top performance.
        p = self.output_dir / "table_reduced_vs_top_performance.tex"
        if not p.exists():
            pd.DataFrame(columns=["Problem", "Full", "ReducedOps", "TopOps", "p-value"]).to_latex(p, index=False)


def generate_demo_data(base: Path):
    rng = np.random.default_rng(42)
    base.mkdir(parents=True, exist_ok=True)
    for scale in SCALES:
        for run in range(N_RUNS):
            gens = np.arange(50, 10001, 50)
            df = pd.DataFrame({"gen": gens})
            for block in BLOCKS:
                ops = BLOCK_OPS[block]
                raw = []
                for gi, g in enumerate(gens):
                    phase = gi / max(1, len(gens) - 1)
                    pref = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
                    pref[1] += 0.3 * phase
                    pref[3] -= 0.2 * (1 - phase)
                    pref = np.clip(pref + rng.normal(0, 0.02, 4), 0.01, None)
                    pref = pref / pref.sum()
                    raw.append(pref)
                arr = np.array(raw)
                for j, op in enumerate(ops):
                    df[f"{block}_{op}"] = arr[:, j]
            # overall columns
            df["overall_GA"] = (df["offload_GA"] + df["seq_GA"]) / 3.0
            df["overall_DE"] = (df["offload_DE"] + df["dev_DE"]) / 3.0
            df["overall_GDE"] = df["dev_GDE"] / 3.0
            df["overall_BITFLIP"] = df["offload_BITFLIP"] / 3.0
            df["overall_SWAP"] = df["seq_SWAP"] / 3.0
            df["overall_VNS"] = df["seq_VNS"] / 3.0
            df["overall_LEVY"] = df["dev_LEVY"] / 3.0
            df["overall_RESAMPLE"] = (df["offload_RESAMPLE"] + df["seq_RESAMPLE"] + df["dev_RESAMPLE"]) / 3.0
            df.to_csv(base / f"op_freq_full_T{scale}_run{run}.csv", index=False)

            # rewards
            rr = []
            for g in gens:
                for b in range(3):
                    for isl in range(8):
                        op = int(rng.integers(0, 4))
                        imp = float(max(0.0, rng.normal(0.015, 0.01)))
                        dv = float(rng.normal(0.0, 0.02))
                        rew = float(np.clip(imp + 0.1 * dv, -0.2, 0.2))
                        rr.append([g, b, isl, op, rew, imp, dv])
            pd.DataFrame(rr, columns=["gen", "block_id", "island_id", "op_id", "reward", "improvement", "diversity_change"]).to_csv(
                base / f"op_rewards_T{scale}_run{run}.csv", index=False
            )

            # global stats
            gs = pd.DataFrame({
                "gen": gens,
                "best_fitness": np.linspace(2.0, 1.0, len(gens)) + rng.normal(0, 0.02, len(gens)),
                "avg_fitness": np.linspace(3.0, 1.4, len(gens)) + rng.normal(0, 0.03, len(gens)),
                "diversity": np.linspace(0.2, 0.05, len(gens)) + rng.normal(0, 0.01, len(gens)),
                "epsilon": np.maximum(0.02, 0.2 * np.exp(-0.01 * gens / 10000.0)),
                "stagnation": np.clip(np.linspace(0, 20, len(gens)) + rng.normal(0, 2, len(gens)), 0, None),
                "gate_blocked_count": np.cumsum(rng.integers(0, 3, len(gens))),
                "gate_fallback_count": np.cumsum(rng.integers(0, 3, len(gens))),
            })
            gs.to_csv(base / f"global_stats_full_T{scale}_run{run}.csv", index=False)

            for block in BLOCKS:
                wr = []
                for g in gens:
                    for op_id in range(4):
                        ws = rng.normal(0.0 + 0.0002 * g, 0.2, 7)
                        norm = float(np.linalg.norm(ws))
                        wr.append([g, op_id] + ws.tolist() + [norm])
                pd.DataFrame(wr, columns=["gen", "op_id", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "norm"]).to_csv(
                    base / f"op_weights_{block}_T{scale}_run{run}.csv", index=False
                )

            # strategy logs
            for label, v in STRATEGY_VARIANTS.items():
                lines = []
                best = 2.5
                for g in gens:
                    decay = {"full": 0.012, "random": 0.009, "roundrobin": 0.008, "fixedbest": 0.010}[v]
                    best = max(0.9, best * (1 - decay) + rng.normal(0, 0.002))
                    lines.append(f"Gen {g}: best_fit = {best:.6f}")
                lines.append(f"The best solution = {best:.6f}")
                (base / f"final_{v}_T{scale}_run{run}.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CC-HIHH operator analysis")
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--demo", action="store_true", help="Generate synthetic data then run analysis")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.demo:
        demo_data = output_dir / "demo_data"
        generate_demo_data(demo_data)
        data_dir = demo_data

    analyzer = OperatorAnalyzer(str(data_dir), str(output_dir))

    print("=== 生成频率分析图表 ===")
    analyzer.plot_frequency_evolution()
    analyzer.plot_frequency_heatmap()
    analyzer.generate_early_late_table()

    print("=== 生成权重演化图表 ===")
    analyzer.plot_weight_evolution()
    analyzer.plot_weight_dimensions()

    print("=== 生成贡献度分析 ===")
    analyzer.calculate_contribution()
    analyzer.generate_contribution_table()
    analyzer.plot_contribution_bars()

    print("=== 生成规模依赖分析 ===")
    analyzer.plot_cross_scale_comparison()
    analyzer.plot_diversity_vs_scale()

    print("=== 生成策略对比 ===")
    analyzer.compare_selection_strategies()
    analyzer.plot_strategy_convergence()

    print("=== 生成Gating分析 ===")
    analyzer.analyze_gating_effect()

    print(f"完成！所有图表已保存至 {output_dir}")


if __name__ == "__main__":
    main()
