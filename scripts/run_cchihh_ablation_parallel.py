import argparse
import shlex
import subprocess
import time
from pathlib import Path


def build_tasks(root: Path, generations: int, log_every: int, data_file: str, nsubpop: int):
    exe = root / "build_local" / "Release" / "CED_Schedule.exe"
    if not exe.exists():
        exe = root / "build" / "Release" / "CED_Schedule.exe"
    data_dir = root / "data"
    base_dir = root / "past_results"

    dirs = {
        "baseline": base_dir / "ablation_cchihh_stable",
        "nomig": base_dir / "ablation_cchihh_nomig",
        "randop": base_dir / "ablation_cchihh_random_ops",
        "noblock": base_dir / "ablation_cchihh_no_blocks",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    tasks = []
    for seed in range(1, 11):
        # baseline (stable) + op stats
        op_stats_path = dirs["baseline"] / f"CCHIHH_stable_opstats_seed{seed}.csv"
        tasks.append(
            {
                "name": "CCHIHH_stable",
                "seed": seed,
                "cmd": [
                    str(exe),
                    "--solver",
                    "CCHIHH",
                    "--stable",
                    "--seed",
                    str(seed),
                    "--generations",
                    str(generations),
                    "--log_every",
                    str(log_every),
                    "--nsubpop",
                    str(nsubpop),
                    "--cchihh_op_stats",
                    str(op_stats_path),
                    "--cchihh_op_stats_every",
                    str(log_every),
                    "--data_dir",
                    str(data_dir),
                    "--data_file",
                    data_file,
                ],
                "log_path": dirs["baseline"] / f"CCHIHH_stable_seed{seed}.txt",
            }
        )

        # migration off
        tasks.append(
            {
                "name": "CCHIHH_stable_nomig",
                "seed": seed,
                "cmd": [
                    str(exe),
                    "--solver",
                    "CCHIHH",
                    "--stable",
                    "--cchihh_no_migration",
                    "--seed",
                    str(seed),
                    "--generations",
                    str(generations),
                    "--log_every",
                    str(log_every),
                    "--nsubpop",
                    str(nsubpop),
                    "--data_dir",
                    str(data_dir),
                    "--data_file",
                    data_file,
                ],
                "log_path": dirs["nomig"] / f"CCHIHH_stable_nomig_seed{seed}.txt",
            }
        )

        # random operator selection (no bandit)
        tasks.append(
            {
                "name": "CCHIHH_stable_randop",
                "seed": seed,
                "cmd": [
                    str(exe),
                    "--solver",
                    "CCHIHH",
                    "--stable",
                    "--cchihh_random_ops",
                    "--seed",
                    str(seed),
                    "--generations",
                    str(generations),
                    "--log_every",
                    str(log_every),
                    "--nsubpop",
                    str(nsubpop),
                    "--data_dir",
                    str(data_dir),
                    "--data_file",
                    data_file,
                ],
                "log_path": dirs["randop"] / f"CCHIHH_stable_randop_seed{seed}.txt",
            }
        )

        # no-block ablation (single population)
        tasks.append(
            {
                "name": "CCHIHH_stable_noblock",
                "seed": seed,
                "cmd": [
                    str(exe),
                    "--solver",
                    "CCHIHH",
                    "--stable",
                    "--cchihh_no_blocks",
                    "--cchihh_no_migration",
                    "--nsubpop",
                    "1",
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
                ],
                "log_path": dirs["noblock"] / f"CCHIHH_stable_noblock_seed{seed}.txt",
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
    parser.add_argument("--nsubpop", type=int, default=8)
    parser.add_argument("--max_parallel", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Overwrite existing logs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    tasks = build_tasks(root, args.generations, args.log_every, args.data_file, args.nsubpop)
    run_parallel(tasks, args.max_parallel, args.force)


if __name__ == "__main__":
    main()
