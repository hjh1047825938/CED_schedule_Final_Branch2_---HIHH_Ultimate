"""Run per-block bandit vs shared/global bandit comparison for CCHIHH."""
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "shared_global_vs_per_block"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALES = {
    "T100": {"data_file": "data_matrix_100.txt", "tnum": 100, "cnum": 100, "enum": 100, "dnum": 300},
    "T200": {"data_file": "data_matrix_T200_E100_D300.txt", "tnum": 200, "cnum": 100, "enum": 100, "dnum": 300},
    "T500": {"data_file": "data_matrix_T500_E200_D800.txt", "tnum": 500, "cnum": 200, "enum": 200, "dnum": 800},
}

SOLVERS = {
    "per_block_bandit": "CCHIHH",
    "shared_global_bandit": "CCHIHH_shared_bandit",
}

SEEDS = range(1, 11)
GENERATIONS = 10000
POPSIZE = 40
NSUBPOP = 8
LOG_EVERY = 50
MAX_WORKERS = 12


def find_exe():
    candidates = [
        ROOT / "build_codex" / "Release" / "CED_Schedule.exe",
        ROOT / "build" / "Release" / "CED_Schedule.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def build_cmd(variant_name, solver_name, scale_name, cfg, seed):
    exe = find_exe()
    tag = f"{variant_name}_{scale_name}_seed{seed}"
    return [
        str(exe),
        "--solver",
        solver_name,
        "--data_dir",
        str(DATA_DIR),
        "--data_file",
        cfg["data_file"],
        "--tnum",
        str(cfg["tnum"]),
        "--mopt",
        "5",
        "--cnum",
        str(cfg["cnum"]),
        "--enum",
        str(cfg["enum"]),
        "--dnum",
        str(cfg["dnum"]),
        "--generations",
        str(GENERATIONS),
        "--popsize",
        str(POPSIZE),
        "--nsubpop",
        str(NSUBPOP),
        "--seed",
        str(seed),
        "--stable",
        "--log_every",
        str(LOG_EVERY),
    ], tag


def run_one(task):
    cmd, tag = task
    log_path = OUT_DIR / f"{tag}.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    return tag, proc.returncode, time.time() - t0


def main():
    exe = find_exe()
    if not exe.exists():
        raise FileNotFoundError(f"Missing executable: {exe}")

    tasks = []
    for scale_name, cfg in SCALES.items():
        for variant_name, solver_name in SOLVERS.items():
            for seed in SEEDS:
                tasks.append(build_cmd(variant_name, solver_name, scale_name, cfg, seed))

    print(f"Total tasks: {len(tasks)}, max parallel: {MAX_WORKERS}")
    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, task): task[1] for task in tasks}
        for fut in as_completed(futures):
            tag = futures[fut]
            completed += 1
            try:
                _, rc, elapsed = fut.result()
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[{completed}/{len(tasks)}] {tag}: {status} ({elapsed:.1f}s)")
            except Exception as exc:
                print(f"[{completed}/{len(tasks)}] {tag}: ERROR {exc}")


if __name__ == "__main__":
    main()
