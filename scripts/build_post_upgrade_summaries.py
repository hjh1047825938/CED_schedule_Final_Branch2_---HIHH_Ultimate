import csv
import json
import re
from collections import defaultdict
from pathlib import Path


GEN_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([0-9.+\-eE]+)")
FINAL_RE = re.compile(r"The\s+best\s+solution\s*=\s*([0-9.+\-eE]+)")
TIME_RE = re.compile(r"Time\s*=\s*([0-9.+\-eE]+)\s*s")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs" / "results"

SCALE_INFO = {
    "T100": {"alpha05_data_file": "data_matrix_100.txt"},
    "T200": {"alpha05_data_file": "data_matrix_T200_E100_D300.txt"},
    "T500": {"alpha05_data_file": "data_matrix_T500_E200_D800.txt"},
}


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


def parse_log(path: Path):
    text = read_text_auto(path)
    series = {}
    final = None
    runtime = None
    for line in text.splitlines():
        s = line.strip().replace("\x00", "")
        m = GEN_RE.match(s)
        if m:
            series[int(m.group(1))] = float(m.group(2))
            continue
        m = FINAL_RE.search(s)
        if m:
            final = float(m.group(1))
            continue
        m = TIME_RE.search(s)
        if m:
            runtime = float(m.group(1))

    if final is None and series:
        final = series[max(series)]
    return {
        "series": dict(sorted(series.items())),
        "final": final,
        "runtime_s": runtime,
        "complete": (runtime is not None and final is not None and bool(series)),
    }


def add_run(records_conv, records_final, validation, path: Path, variant: str, scale: str, seed: int,
            alpha: float, source: str, note: str = ""):
    if not path.exists():
        validation["missing_files"].append(str(path))
        return

    parsed = parse_log(path)
    if not parsed["series"] or parsed["final"] is None:
        validation["parse_failures"].append(str(path))
        return

    last = None
    monotonic = True
    has_bad_value = False
    for gen, best_fit in parsed["series"].items():
        if last is not None and best_fit > last + 1e-12:
            monotonic = False
        if not (-1e300 < best_fit < 1e300):
            has_bad_value = True
        last = best_fit
        records_conv.append(
            {
                "variant": variant,
                "scale": scale,
                "seed": seed,
                "gen": gen,
                "best_fit": best_fit,
                "alpha": alpha,
                "source": source,
                "note": note,
                "path": str(path),
            }
        )

    if not (-1e300 < parsed["final"] < 1e300):
        has_bad_value = True

    records_final.append(
        {
            "variant": variant,
            "scale": scale,
            "seed": seed,
            "best_fit": parsed["final"],
            "runtime_s": parsed["runtime_s"],
            "alpha": alpha,
            "source": source,
            "note": note,
            "path": str(path),
            "complete": parsed["complete"],
            "monotonic_non_increasing": monotonic,
        }
    )

    if not monotonic:
        validation["non_monotonic_logs"].append(str(path))
    if has_bad_value or parsed["runtime_s"] is None:
        validation["bad_numeric_logs"].append(str(path))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_new_runs(records_conv, records_final, validation):
    for seed in range(1, 11):
        for scale in ("T100", "T200", "T500"):
            add_run(
                records_conv, records_final, validation,
                RESULTS_DIR / "rerun_full" / f"cchihh_full_{scale}_s{seed}.log",
                "CCHIHH-full", scale, seed, 0.5, "new_rerun",
            )
            add_run(
                records_conv, records_final, validation,
                RESULTS_DIR / "rerun_ablation" / f"cchihh_noCC_{scale}_s{seed}.log",
                "CCHIHH-noCC", scale, seed, 0.5, "new_rerun",
            )
            add_run(
                records_conv, records_final, validation,
                RESULTS_DIR / "rerun_ablation" / f"cchihh_noHI_{scale}_s{seed}.log",
                "CCHIHH-noHI", scale, seed, 0.5, "new_rerun",
            )
            add_run(
                records_conv, records_final, validation,
                RESULTS_DIR / "rerun_ablation" / f"cchihh_noMig_{scale}_s{seed}.log",
                "CCHIHH-noMig", scale, seed, 0.5, "new_rerun",
            )
            add_run(
                records_conv, records_final, validation,
                RESULTS_DIR / "rerun_ablation" / f"cchihh_noGate_{scale}_s{seed}.log",
                "CCHIHH-noGate", scale, seed, 0.5, "new_rerun",
            )
            for alpha in (0.2, 0.8):
                add_run(
                    records_conv, records_final, validation,
                    RESULTS_DIR / "rerun_alpha" / f"cchihh_full_{scale}_a{alpha:.1f}_s{seed}.log",
                    "CCHIHH-full", scale, seed, alpha, "new_rerun",
                )

    for seed in range(1, 11):
        for gate in (5, 10, 15, 20, 25):
            add_run(
                records_conv, records_final, validation,
                RESULTS_DIR / "rerun_tgate" / f"cchihh_tgate{gate}_T500_s{seed}.log",
                f"CCHIHH-tgate{gate}", "T500", seed, 0.5, "new_rerun",
            )


def build_old_baselines(records_conv, records_final, validation):
    baseline_root = OUTPUTS_DIR / "cchihh_ablation_suite" / "baseline"
    for seed in range(1, 11):
        for scale in ("T100", "T200", "T500"):
            for solver, variant in (
                ("CGA", "CGA"),
                ("IMOMA", "IMOMA"),
                ("PPO", "PPO"),
                ("CCHIHH_Full", "CCHIHH-full-old"),
            ):
                add_run(
                    records_conv, records_final, validation,
                    baseline_root / scale / f"{solver}_seed{seed}.txt",
                    variant, scale, seed, 0.5, "old_baseline",
                )

            add_run(
                records_conv, records_final, validation,
                OUTPUTS_DIR / "dsac_de_multiscale" / scale / f"DSAC_DE_seed{seed}.txt",
                "DSAC-DE", scale, seed, 0.5, "old_baseline",
            )
            add_run(
                records_conv, records_final, validation,
                OUTPUTS_DIR / "cchihh_ablation_suite" / "bandit" / scale / f"fixed_ops_seed{seed}.txt",
                "CCHIHH-noCB", scale, seed, 0.5, "old_noCB",
                note="old noCB / fixed_ops baseline; mixed with new full in comparisons",
            )


def build_old_alpha_baselines(records_conv, records_final, validation):
    alpha_root = OUTPUTS_DIR / "alpha_sensitivity_cga_imoma_dsac" / "runs"
    for alpha in (0.2, 0.8):
        alpha_dir = alpha_root / f"alpha_{alpha:.1f}"
        for scale in ("T100", "T200", "T500"):
            for seed in range(1, 11):
                for solver, variant in (
                    ("CGA", "CGA"),
                    ("IMOMA", "IMOMA"),
                    ("DSAC_DE", "DSAC-DE"),
                    ("CCHIHH_Full", "CCHIHH-full-old"),
                ):
                    add_run(
                        records_conv, records_final, validation,
                        alpha_dir / scale / f"{solver}_seed{seed}.txt",
                        variant, scale, seed, alpha, "old_alpha_runs",
                    )


def summarize_validation(final_rows):
    by_variant_scale_alpha = defaultdict(int)
    incomplete = []
    for row in final_rows:
        key = (row["variant"], row["scale"], row["alpha"], row["source"])
        by_variant_scale_alpha[key] += 1
        if not row["complete"]:
            incomplete.append(row["path"])
    return by_variant_scale_alpha, incomplete


def main():
    records_conv = []
    records_final = []
    validation = {
        "missing_files": [],
        "parse_failures": [],
        "non_monotonic_logs": [],
        "bad_numeric_logs": [],
    }

    build_new_runs(records_conv, records_final, validation)
    build_old_baselines(records_conv, records_final, validation)
    build_old_alpha_baselines(records_conv, records_final, validation)

    records_conv.sort(key=lambda r: (r["variant"], r["alpha"], r["scale"], r["seed"], r["gen"]))
    records_final.sort(key=lambda r: (r["variant"], r["alpha"], r["scale"], r["seed"]))

    write_csv(
        RESULTS_DIR / "convergence_summary.csv",
        records_conv,
        ["variant", "scale", "seed", "gen", "best_fit", "alpha", "source", "note", "path"],
    )
    write_csv(
        RESULTS_DIR / "final_summary.csv",
        records_final,
        ["variant", "scale", "seed", "best_fit", "runtime_s", "alpha", "source", "note", "path", "complete", "monotonic_non_increasing"],
    )

    counts, incomplete = summarize_validation(records_final)
    validation["counts"] = [
        {
            "variant": variant,
            "scale": scale,
            "alpha": alpha,
            "source": source,
            "count": count,
        }
        for (variant, scale, alpha, source), count in sorted(counts.items())
    ]
    validation["incomplete_logs"] = incomplete

    report_path = RESULTS_DIR / "summary_validation.json"
    report_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {RESULTS_DIR / 'convergence_summary.csv'}")
    print(f"Saved: {RESULTS_DIR / 'final_summary.csv'}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
