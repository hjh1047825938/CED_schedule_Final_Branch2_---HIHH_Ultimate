#!/usr/bin/env python3
import argparse
import csv
import re
import shlex
import subprocess
import time
from pathlib import Path


EVAL_RE = re.compile(r"^Eval\s+(\d+):\s+best_fit\s*=\s*([0-9.+\-eE]+)")

SCALE_CONFIGS = [
    {
        "name": "T100",
        "data_file": "data_matrix_100.txt",
        "cnum": 100,
        "enum": 100,
        "dnum": 300,
        "tnum": 100,
        "mopt": 5,
    },
    {
        "name": "T200",
        "data_file": "data_matrix_T200_E100_D300.txt",
        "cnum": 100,
        "enum": 100,
        "dnum": 300,
        "tnum": 200,
        "mopt": 5,
    },
    {
        "name": "T500",
        "data_file": "data_matrix_T500_E200_D800.txt",
        "cnum": 200,
        "enum": 200,
        "dnum": 800,
        "tnum": 500,
        "mopt": 5,
    },
]

SOLVERS = [
    {"group": "lpsr", "label": "LPSR", "solver": "L-SRTDE"},
    {"group": "nlpsr", "label": "NLPSR", "solver": "NL-SHADE-LBC"},
]

ALPHAS = [0.2, 0.5, 0.8]
SEEDS = list(range(1, 11))


def build_tasks(root: Path, exe: Path, max_evals: int, log_every: int) -> list[dict]:
    data_dir = root / "data"
    tasks: list[dict] = []
    for solver in SOLVERS:
        for scale in SCALE_CONFIGS:
            for alpha in ALPHAS:
                for seed in SEEDS:
                    label = solver["label"]
                    group = solver["group"]
                    log_path = (
                        root
                        / "results"
                        / "eval"
                        / "raw"
                        / group
                        / f"alpha{alpha:.1f}"
                        / scale["name"]
                        / f"{label}_{scale['name']}_s{seed}.log"
                    )
                    csv_path = (
                        root
                        / "results"
                        / "eval"
                        / group
                        / f"alpha{alpha:.1f}"
                        / scale["name"]
                        / f"{label}_{scale['name']}_s{seed}_eval.csv"
                    )
                    cmd = [
                        str(exe),
                        "--solver",
                        solver["solver"],
                        "--seed",
                        str(seed),
                        "--alpha",
                        f"{alpha:.1f}",
                        "--generations",
                        "999999",
                        "--max_evals",
                        str(max_evals),
                        "--log_every",
                        str(log_every),
                        "--data_dir",
                        str(data_dir),
                        "--data_file",
                        scale["data_file"],
                        "--cnum",
                        str(scale["cnum"]),
                        "--enum",
                        str(scale["enum"]),
                        "--dnum",
                        str(scale["dnum"]),
                        "--tnum",
                        str(scale["tnum"]),
                        "--mopt",
                        str(scale["mopt"]),
                    ]
                    tasks.append(
                        {
                            "group": group,
                            "label": label,
                            "solver": solver["solver"],
                            "scale": scale["name"],
                            "alpha": alpha,
                            "seed": seed,
                            "cmd": cmd,
                            "log_path": log_path,
                            "csv_path": csv_path,
                        }
                    )
    return tasks


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_eval_curve_from_log(log_path: Path, csv_path: Path) -> None:
    rows: list[tuple[int, float]] = []
    best_so_far = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EVAL_RE.match(line.strip())
            if not match:
                continue
            eval_count = int(match.group(1))
            best_fit = float(match.group(2))
            best_so_far = best_fit if best_so_far is None else min(best_so_far, best_fit)
            rows.append((eval_count, best_so_far))
    if not rows:
        raise ValueError(f"No Eval lines found in {log_path}")

    ensure_parent(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_count", "best_fitness"])
        for eval_count, best_fit in rows:
            writer.writerow([eval_count, f"{best_fit:.15g}"])


def write_index(root: Path, tasks: list[dict]) -> None:
    index_path = root / "results" / "eval" / "replacement_index.csv"
    ensure_parent(index_path)
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "label", "solver", "scale", "alpha", "seed", "log_path", "csv_path"],
        )
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "group": task["group"],
                    "label": task["label"],
                    "solver": task["solver"],
                    "scale": task["scale"],
                    "alpha": f"{task['alpha']:.1f}",
                    "seed": task["seed"],
                    "log_path": str(task["log_path"]),
                    "csv_path": str(task["csv_path"]),
                }
            )


def launch_task(task: dict, force: bool) -> dict | None:
    if task["csv_path"].exists() and not force:
        return None
    ensure_parent(task["log_path"])
    f = task["log_path"].open("w", encoding="utf-8", newline="")
    f.write(f"Command: {shlex.join(task['cmd'])}\n")
    f.flush()
    proc = subprocess.Popen(task["cmd"], stdout=f, stderr=subprocess.STDOUT, shell=False)
    return {"task": task, "proc": proc, "file": f}


def run_parallel(tasks: list[dict], max_parallel: int, force: bool) -> None:
    queue = tasks[:]
    running: list[dict] = []
    done = 0
    total = len(tasks)

    while queue or running:
        while queue and len(running) < max_parallel:
            task = queue.pop(0)
            launched = launch_task(task, force=force)
            if launched is None:
                done += 1
                print(f"[skip] {task['label']} {task['scale']} alpha={task['alpha']:.1f} seed={task['seed']}")
                continue
            running.append(launched)
            print(f"[start] {task['label']} {task['scale']} alpha={task['alpha']:.1f} seed={task['seed']}")

        time.sleep(1)
        still_running: list[dict] = []
        for item in running:
            if item["proc"].poll() is None:
                still_running.append(item)
                continue
            task = item["task"]
            code = item["proc"].returncode
            item["file"].write(f"\nexit code {code}\n")
            item["file"].close()
            if code != 0:
                raise RuntimeError(
                    f"Task failed: {task['label']} {task['scale']} alpha={task['alpha']:.1f} seed={task['seed']} rc={code}"
                )
            export_eval_curve_from_log(task["log_path"], task["csv_path"])
            done += 1
            print(f"[done] {task['label']} {task['scale']} alpha={task['alpha']:.1f} seed={task['seed']} ({done}/{total})")
        running = still_running


def find_exe(root: Path) -> Path:
    candidates = [
        root / "build" / "Release" / "CED_Schedule.exe",
        root / "build_copy" / "Release" / "CED_Schedule.exe",
        root / "build_local" / "Release" / "CED_Schedule.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit("Cannot find CED_Schedule.exe in known build folders.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LPSR/NLPSR max-eval experiments with 2-way parallelism.")
    parser.add_argument("--max_evals", type=int, default=400000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = find_exe(root)
    tasks = build_tasks(root=root, exe=exe, max_evals=args.max_evals, log_every=args.log_every)
    write_index(root, tasks)
    run_parallel(tasks, max_parallel=args.max_parallel, force=args.force)


if __name__ == "__main__":
    main()
