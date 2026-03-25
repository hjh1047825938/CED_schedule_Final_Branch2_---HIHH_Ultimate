"""Analyze per-block bandit vs shared/global bandit comparison logs."""
import re
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results" / "shared_global_vs_per_block"
SCALES = ["T100", "T200", "T500"]
SEEDS = range(1, 11)


def extract_final_fitness(log_path: Path):
    patterns = [
        re.compile(r"The best solution\s*=\s*([0-9.eE+\-]+)"),
        re.compile(r"best_fit\s*=\s*([0-9.eE+\-]+)"),
    ]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    best = None
    for line in text.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if not match:
                continue
            value = float(match.group(1))
            if best is None or value < best:
                best = value
    return best


def collect(scale: str, variant: str):
    fits = {}
    for seed in SEEDS:
        log_path = RESULT_DIR / f"{variant}_{scale}_seed{seed}.log"
        if not log_path.exists():
            continue
        fit = extract_final_fitness(log_path)
        if fit is not None:
            fits[seed] = fit
    return fits


def main():
    if not RESULT_DIR.exists():
        raise FileNotFoundError(f"Missing result directory: {RESULT_DIR}")

    for scale in SCALES:
        per_block = collect(scale, "per_block_bandit")
        shared = collect(scale, "shared_global_bandit")
        common = sorted(set(per_block) & set(shared))

        print("=" * 72)
        print(scale)
        print("=" * 72)
        print(f"per-block bandit seeds: {len(per_block)}")
        print(f"shared/global bandit seeds: {len(shared)}")

        if per_block:
            vals = np.array([per_block[s] for s in sorted(per_block)])
            print(f"per-block bandit:      mean={vals.mean():.6f}  sem={vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0:.6f}")
        if shared:
            vals = np.array([shared[s] for s in sorted(shared)])
            print(f"shared/global bandit:  mean={vals.mean():.6f}  sem={vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0:.6f}")

        if len(common) >= 5:
            a = np.array([per_block[s] for s in common])
            b = np.array([shared[s] for s in common])
            stat, pvalue = wilcoxon(a, b)
            diff = b.mean() - a.mean()
            better = "per-block bandit better" if diff > 0 else "shared/global bandit better"
            print(f"Wilcoxon p-value: {pvalue:.6f}")
            print(f"Mean(shared - per_block): {diff:.6f}")
            print(f"Conclusion: {better}")
        else:
            print("Not enough paired seeds for Wilcoxon test.")


if __name__ == "__main__":
    main()
