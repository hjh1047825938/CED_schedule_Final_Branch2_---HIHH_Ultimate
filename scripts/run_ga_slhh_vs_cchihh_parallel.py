import argparse
import shlex
import subprocess
import time
from pathlib import Path


def build_tasks(root: Path, generations: int, log_every: int, data_file: str, results_dir: Path):
    exe = root / "build" / "Release" / "CED_Schedule.exe"
    data_dir = root / "data"
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for seed in range(1, 11):
        for name, solver_args in [
            ("GA-SLHH", ["--solver", "GA-SLHH"]),
            ("CCHIHH_stable", ["--solver", "CCHIHH", "--stable"]),
        ]:
            log_name = f"{name}_seed{seed}.txt"
            cmd = [
                str(exe),
                *solver_args,
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
            ]
            tasks.append(
                {
                    "name": name,
                    "seed": seed,
                    "cmd": cmd,
                    "log_path": results_dir / log_name,
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
    parser.add_argument("--max_parallel", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Overwrite existing logs")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir

    tasks = build_tasks(root, args.generations, args.log_every, args.data_file, results_dir)
    run_parallel(tasks, args.max_parallel, args.force)


if __name__ == "__main__":
    main()
