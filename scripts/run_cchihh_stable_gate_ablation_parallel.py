import argparse
import shlex
import subprocess
import time
from pathlib import Path


CONFIGS = [
    ("cfg1_stable_on_gate_on", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15"]),
    ("cfg2_stable_off_gate_on", ["--solver", "CCHIHH", "--resample_gate", "15"]),
    ("cfg3_stable_on_gate_off", ["--solver", "CCHIHH", "--stable", "--resample_gate", "0"]),
    ("cfg4_stable_off_gate_off", ["--solver", "CCHIHH", "--resample_gate", "0"]),
]


def build_tasks(
    root: Path,
    generations: int,
    log_every: int,
    data_file: str,
    cnum: int,
    enum: int,
    dnum: int,
    tnum: int,
    mopt: int,
    results_dir: Path,
):
    exe = root / "build_ablation" / "Release" / "CED_Schedule.exe"
    if not exe.exists():
        exe = root / "build" / "Release" / "CED_Schedule.exe"
    data_dir = root / "data"
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for seed in range(1, 11):
        for cfg_name, cfg_args in CONFIGS:
            cmd = [
                str(exe),
                *cfg_args,
                "--seed",
                str(seed),
                "--generations",
                str(generations),
                "--log_every",
                str(log_every),
                "--data_dir",
                str(data_dir),
                "--data_file",
                data_file,
                "--cnum",
                str(cnum),
                "--enum",
                str(enum),
                "--dnum",
                str(dnum),
                "--tnum",
                str(tnum),
                "--mopt",
                str(mopt),
            ]
            tasks.append(
                {
                    "name": cfg_name,
                    "seed": seed,
                    "cmd": cmd,
                    "log_path": results_dir / f"{cfg_name}_seed{seed}.txt",
                }
            )
    return tasks


def run_parallel(tasks, max_parallel: int, force: bool):
    running = []
    completed = 0
    total = len(tasks)

    def start_task(task):
        log_path = task["log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if log_path.exists() and not force:
            return None

        f = log_path.open("w", encoding="utf-8")
        cmd_str = shlex.join(task["cmd"])
        f.write(f"Command: {cmd_str}\n")
        f.flush()
        proc = subprocess.Popen(task["cmd"], stdout=f, stderr=subprocess.STDOUT, shell=False)
        return {"proc": proc, "log": f, "task": task}

    queue = tasks[:]
    while queue or running:
        while queue and len(running) < max_parallel:
            task = queue.pop(0)
            launched = start_task(task)
            if launched is None:
                completed += 1
                print(f"[skip] {task['name']} seed {task['seed']} (log exists)")
            else:
                running.append(launched)
                print(f"[start] {task['name']} seed {task['seed']}")

        time.sleep(1)
        still_running = []
        for item in running:
            proc = item["proc"]
            if proc.poll() is None:
                still_running.append(item)
                continue

            exit_code = proc.returncode
            item["log"].write(f"\nexit code {exit_code}\n")
            item["log"].close()
            completed += 1
            task = item["task"]
            print(f"[done] {task['name']} seed {task['seed']} (exit {exit_code})")

        running = still_running

    print(f"All done. Completed {completed}/{total}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--data_file", default="data_matrix_100.txt")
    parser.add_argument("--cnum", type=int, default=100)
    parser.add_argument("--enum", type=int, default=100)
    parser.add_argument("--dnum", type=int, default=300)
    parser.add_argument("--tnum", type=int, default=100)
    parser.add_argument("--mopt", type=int, default=5)
    parser.add_argument(
        "--results_dir",
        default="results/ablation_cchihh_stable_gate_gen10000_seed1_10",
        help="Directory to store logs.",
    )
    parser.add_argument("--max_parallel", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Overwrite existing logs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    tasks = build_tasks(
        root=root,
        generations=args.generations,
        log_every=args.log_every,
        data_file=args.data_file,
        cnum=args.cnum,
        enum=args.enum,
        dnum=args.dnum,
        tnum=args.tnum,
        mopt=args.mopt,
        results_dir=(root / args.results_dir),
    )
    run_parallel(tasks, args.max_parallel, args.force)


if __name__ == "__main__":
    main()
