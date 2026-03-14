"""Run shared_bandit v3 (block-indexed operators) ablation: T100/T200/T500 × 10 seeds."""
import subprocess
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / "Release" / "CED_Schedule.exe"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "supplement" / "shared_v7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALES = {
    "T100": {"data_file": "data_matrix_100.txt",            "tnum": 100, "cnum": 100, "enum": 100, "dnum": 300},
    "T200": {"data_file": "data_matrix_T200_E100_D300.txt",  "tnum": 200, "cnum": 100, "enum": 100, "dnum": 300},
    "T500": {"data_file": "data_matrix_T500_E200_D800.txt",  "tnum": 500, "cnum": 200, "enum": 200, "dnum": 800},
}

SEEDS = range(1, 11)
GENERATIONS = 10000
POPSIZE = 40
NSUBPOP = 8
LOG_EVERY = 50
MAX_WORKERS = 15  # parallel jobs


def build_cmd(scale_name, cfg, seed):
    tag = f"shared_{scale_name}_seed{seed}"
    return [
        str(EXE),
        "--solver", "CCHIHH",
        "--data_dir", str(DATA_DIR),
        "--data_file", cfg["data_file"],
        "--tnum", str(cfg["tnum"]),
        "--mopt", "5",
        "--cnum", str(cfg["cnum"]),
        "--enum", str(cfg["enum"]),
        "--dnum", str(cfg["dnum"]),
        "--generations", str(GENERATIONS),
        "--popsize", str(POPSIZE),
        "--nsubpop", str(NSUBPOP),
        "--seed", str(seed),
        "--stable",
        "--shared_bandit",
        "--log_every", str(LOG_EVERY),
        "--reward_variance_log",
        str(OUT_DIR / f"reward_var_{tag}.csv"),
        "--cchihh_weight_log_offload",
        str(OUT_DIR / f"weights_shared_off_{tag}.csv"),
        "--cchihh_weight_log_seq",
        str(OUT_DIR / f"weights_shared_seq_{tag}.csv"),
        "--cchihh_weight_log_dev",
        str(OUT_DIR / f"weights_shared_dev_{tag}.csv"),
    ], tag


def run_one(args):
    cmd, tag = args
    log_path = OUT_DIR / f"cchihh_{tag}.log"
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    elapsed = time.time() - t0
    return tag, proc.returncode, elapsed


def main():
    tasks = []
    for scale_name, cfg in SCALES.items():
        for seed in SEEDS:
            cmd, tag = build_cmd(scale_name, cfg, seed)
            tasks.append((cmd, tag))

    print(f"Total tasks: {len(tasks)}, max parallel: {MAX_WORKERS}")
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(run_one, t): t[1] for t in tasks}
        for fut in as_completed(futs):
            tag = futs[fut]
            try:
                tag, rc, elapsed = fut.result()
                done += 1
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[{done}/{len(tasks)}] {tag}: {status}  ({elapsed:.1f}s)")
            except Exception as e:
                done += 1
                print(f"[{done}/{len(tasks)}] {tag}: ERROR {e}")

    print("All done.")


if __name__ == "__main__":
    main()
