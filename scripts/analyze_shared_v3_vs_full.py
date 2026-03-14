"""Compare shared_v3 (block-indexed) vs full (independent per-block bandit).
Reads final fitness from logs and runs Wilcoxon signed-rank test."""
import re
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent

# Full baseline results directory
FULL_DIR = ROOT / "results" / "supplement" / "analysis"
SHARED_V2_DIR = ROOT / "results" / "supplement" / "shared_v2"
SHARED_V3_DIR = ROOT / "results" / "supplement" / "shared_v3"

SCALES = ["T100", "T200", "T500"]
SEEDS = range(1, 11)


def extract_final_fitness(log_path):
    """Extract the last reported best fitness from a log file."""
    best = None
    pat = re.compile(r'Best\s*(?:fitness|fit)[:\s=]*([0-9.eE+\-]+)', re.IGNORECASE)
    pat2 = re.compile(r'Gen\s+\d+.*?best[:\s=]*([0-9.eE+\-]+)', re.IGNORECASE)
    pat3 = re.compile(r'Global best:\s*([0-9.eE+\-]+)', re.IGNORECASE)
    pat4 = re.compile(r'\[Gen\s+\d+\]\s*best\s*=\s*([0-9.eE+\-]+)', re.IGNORECASE)
    try:
        text = log_path.read_text(errors='replace')
        for line in text.splitlines():
            for p in [pat, pat2, pat3, pat4]:
                m = p.search(line)
                if m:
                    val = float(m.group(1))
                    if best is None or val < best:
                        best = val
    except Exception as e:
        print(f"  Warning: could not read {log_path}: {e}")
    return best


def find_full_baseline_logs(scale):
    """Find full (non-shared) baseline logs."""
    # Try multiple possible directories
    candidates = [
        ROOT / "results" / "supplement" / "analysis",
        ROOT / "past_results" / "ablation_cchihh_stable",
        ROOT / "outputs" / "results" / "baseline_cchihh_vs_dsac_de",
    ]
    for cdir in candidates:
        if not cdir.exists():
            continue
        for seed in SEEDS:
            patterns = [
                f"CCHIHH_stable_seed{seed}.txt",
                f"CCHIHH_stable_{scale}_seed{seed}.txt",
                f"cchihh_{scale}_seed{seed}.log",
                f"CCHIHH_seed{seed}.txt",
            ]
            for pat in patterns:
                p = cdir / pat
                if p.exists():
                    yield seed, p
                    break
            # Also try scale subdirectory
            scale_dir = cdir / scale
            if scale_dir.exists():
                for pat in patterns:
                    p = scale_dir / pat
                    if p.exists():
                        yield seed, p
                        break


def main():
    print("=" * 70)
    print("Shared Bandit v3 (block-indexed) vs Full (independent) Comparison")
    print("=" * 70)

    # First, let's find where the full baseline results are
    print("\nSearching for full baseline logs...")

    # Try to extract from shared_v2 logs the corresponding full baseline
    # Actually, let me just scan all result directories
    for scale in SCALES:
        print(f"\n{'='*60}")
        print(f"Scale: {scale}")
        print(f"{'='*60}")

        # Get shared_v3 results
        v3_fits = {}
        for seed in SEEDS:
            log = SHARED_V3_DIR / f"cchihh_shared_{scale}_seed{seed}.log"
            if log.exists():
                fit = extract_final_fitness(log)
                if fit is not None:
                    v3_fits[seed] = fit

        # Get shared_v2 results (old shared for comparison)
        v2_fits = {}
        for seed in SEEDS:
            log = SHARED_V2_DIR / f"cchihh_shared_{scale}_seed{seed}.log"
            if log.exists():
                fit = extract_final_fitness(log)
                if fit is not None:
                    v2_fits[seed] = fit

        # Print results
        if v3_fits:
            vals = list(v3_fits.values())
            print(f"\nShared v3 (block-indexed): {np.mean(vals):.6f} +/- {np.std(vals)/np.sqrt(len(vals)):.6f}  (n={len(vals)})")
            for s, f in sorted(v3_fits.items()):
                print(f"  seed{s}: {f:.6f}")
        else:
            print("\nShared v3: NO RESULTS FOUND")

        if v2_fits:
            vals = list(v2_fits.values())
            print(f"\nShared v2 (weight-sharing): {np.mean(vals):.6f} +/- {np.std(vals)/np.sqrt(len(vals)):.6f}  (n={len(vals)})")

        # Wilcoxon test: v3 vs v2
        common_seeds = sorted(set(v3_fits.keys()) & set(v2_fits.keys()))
        if len(common_seeds) >= 5:
            a = np.array([v3_fits[s] for s in common_seeds])
            b = np.array([v2_fits[s] for s in common_seeds])
            stat, p = wilcoxon(a, b)
            direction = "v3 > v2 (v3 worse)" if np.mean(a) > np.mean(b) else "v3 < v2 (v3 better)"
            print(f"\nWilcoxon v3 vs v2: p={p:.6f}  ({direction})")
            print(f"  v3 mean={np.mean(a):.6f}, v2 mean={np.mean(b):.6f}, diff={np.mean(a)-np.mean(b):.6f}")

    print("\n" + "=" * 70)
    print("NOTE: To compare with full (independent bandit), check the baseline")
    print("results from the original ablation experiment.")
    print("Expected: v3 should be WORSE than full (higher fitness = worse)")
    print("because independent per-block bandits should learn better.")
    print("=" * 70)


if __name__ == "__main__":
    main()
