import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from generate_ced_data import generate_data
from build_stats_and_plot_multi import build_stats_and_plot


INSTANCES = [
    ("inst1_small", 50, 50, 150),
    ("inst2_medium", 200, 100, 300),
    ("inst3_large", 300, 150, 500),
    ("inst4_xlarge", 500, 200, 800),
]

SOLVERS = [
    ("GA", ["--solver", "GA"]),
    ("DE", ["--solver", "DE"]),
    ("GDE", ["--solver", "GDE"]),
    ("CCHIHH_base", ["--solver", "CCHIHH"]),
    ("CCHIHH_stable", ["--solver", "CCHIHH", "--stable"]),
]


def build_tasks(root: Path, generations: int, log_every: int, data_file: str, cnum: int, enum: int, dnum: int, tnum: int, mopt: int):
    exe = root / "build" / "Release" / "CED_Schedule.exe"
    data_dir = root / "data"
    results_dir = root / "results" / f"gen10000_seed1_10_T{tnum}_E{enum}_D{dnum}"
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for seed in range(1, 11):
        for name, solver_args in SOLVERS:
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
                    "name": name,
                    "seed": seed,
                    "cmd": cmd,
                    "log_path": results_dir / log_name,
                }
            )

    return tasks, results_dir


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
    parser.add_argument("--max_parallel", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Overwrite existing logs")
    parser.add_argument("--data_seed", type=int, default=1)
    parser.add_argument("--mopt", type=int, default=5)
    parser.add_argument("--dep_prob", type=float, default=0.0)
    parser.add_argument("--max_dep", type=int, default=3)
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--no_postprocess", action="store_true")
    parser.add_argument("--only", default="", help="Comma-separated instance names")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    if args.only:
        selected = set(x.strip() for x in args.only.split(",") if x.strip())
    else:
        selected = None

    for idx, (name, tnum, enum, dnum) in enumerate(INSTANCES):
        if selected is not None and name not in selected:
            continue

        cnum = enum
        data_file = f"data_matrix_T{tnum}_E{enum}_D{dnum}.txt"
        data_path = data_dir / data_file

        if not args.skip_generate and not data_path.exists():
            seed = args.data_seed + idx
            generate_data(
                out_path=data_path,
                cnum=cnum,
                enum=enum,
                dnum=dnum,
                tnum=tnum,
                mopt=args.mopt,
                seed=seed,
                dep_prob=args.dep_prob,
                max_dep=args.max_dep,
            )
            print(f"[data] generated {data_path}")
        elif not args.skip_generate and args.force:
            seed = args.data_seed + idx
            generate_data(
                out_path=data_path,
                cnum=cnum,
                enum=enum,
                dnum=dnum,
                tnum=tnum,
                mopt=args.mopt,
                seed=seed,
                dep_prob=args.dep_prob,
                max_dep=args.max_dep,
            )
            print(f"[data] regenerated {data_path}")
        else:
            print(f"[data] exists {data_path}")

        tasks, results_dir = build_tasks(
            root=root,
            generations=args.generations,
            log_every=args.log_every,
            data_file=data_file,
            cnum=cnum,
            enum=enum,
            dnum=dnum,
            tnum=tnum,
            mopt=args.mopt,
        )
        run_parallel(tasks, args.max_parallel, args.force)

        if not args.no_postprocess:
            out_prefix = results_dir / "mean_ci"
            title = f"Mean with 95% CI (T={tnum}, E={enum}, D={dnum})"
            build_stats_and_plot(
                results_dir=results_dir,
                out_prefix=out_prefix,
                title=title,
                n=10,
            )


if __name__ == "__main__":
    sys.exit(main())
