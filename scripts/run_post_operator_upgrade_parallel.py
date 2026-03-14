import argparse
import shlex
import subprocess
import time
from pathlib import Path


SCALES = [
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


def find_exe(root: Path, exe_arg: str) -> Path:
    if exe_arg:
        exe = Path(exe_arg)
        if not exe.is_absolute():
            exe = (root / exe).resolve()
        if exe.exists():
            return exe
        raise SystemExit(f"Executable not found: {exe}")

    candidates = [
        root / "build_codex" / "Release" / "CED_Schedule.exe",
        root / "build" / "Release" / "CED_Schedule.exe",
        root / "build_ablation" / "Release" / "CED_Schedule.exe",
        root / "build_local" / "Release" / "CED_Schedule.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit("Cannot find CED_Schedule.exe in known build folders.")


def common_args(exe: Path, data_dir: Path, scale: dict, seed: int, generations: int, log_every: int):
    return [
        str(exe),
        "--solver",
        "CCHIHH",
        "--stable",
        "--seed",
        str(seed),
        "--generations",
        str(generations),
        "--popsize",
        "40",
        "--nsubpop",
        "8",
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


def add_task(tasks: list, label: str, cmd: list[str], log_path: Path):
    tasks.append({"label": label, "cmd": cmd, "log_path": log_path})


def with_nsubpop(cmd: list[str], value: int) -> list[str]:
    updated = cmd[:]
    idx = updated.index("--nsubpop")
    updated[idx + 1] = str(value)
    return updated


def build_tasks(root: Path, exe: Path, phases: set[str], generations: int, log_every: int):
    data_dir = root / "data"
    out_full = root / "results" / "rerun_full"
    out_ablation = root / "results" / "rerun_ablation"
    out_alpha = root / "results" / "rerun_alpha"
    out_tgate = root / "results" / "rerun_tgate"
    out_schedule = root / "results" / "rerun_schedule"

    for path in (out_full, out_ablation, out_alpha, out_tgate, out_schedule):
        path.mkdir(parents=True, exist_ok=True)

    tasks = []
    for scale in SCALES:
        for seed in range(1, 11):
            base = common_args(exe, data_dir, scale, seed, generations, log_every)
            scale_name = scale["name"]

            if "full" in phases:
                add_task(
                    tasks,
                    f"full_{scale_name}_s{seed}",
                    base
                    + [
                        "--cchihh_op_stats",
                        str(out_full / f"cchihh_full_{scale_name}_s{seed}_ops.csv"),
                        "--cchihh_weight_log_offload",
                        str(out_full / f"cchihh_full_{scale_name}_s{seed}_w_off.csv"),
                        "--cchihh_weight_log_seq",
                        str(out_full / f"cchihh_full_{scale_name}_s{seed}_w_seq.csv"),
                        "--cchihh_weight_log_dev",
                        str(out_full / f"cchihh_full_{scale_name}_s{seed}_w_dev.csv"),
                    ],
                    out_full / f"cchihh_full_{scale_name}_s{seed}.log",
                )

            if "ablation" in phases:
                add_task(
                    tasks,
                    f"noCC_{scale_name}_s{seed}",
                    base + ["--no_blocks"],
                    out_ablation / f"cchihh_noCC_{scale_name}_s{seed}.log",
                )
                add_task(
                    tasks,
                    f"noHI_{scale_name}_s{seed}",
                    with_nsubpop(base, 1) + ["--cchihh_no_migration"],
                    out_ablation / f"cchihh_noHI_{scale_name}_s{seed}.log",
                )
                add_task(
                    tasks,
                    f"noMig_{scale_name}_s{seed}",
                    base + ["--cchihh_no_migration"],
                    out_ablation / f"cchihh_noMig_{scale_name}_s{seed}.log",
                )
                add_task(
                    tasks,
                    f"noGate_{scale_name}_s{seed}",
                    base + ["--resample_gate", "0"],
                    out_ablation / f"cchihh_noGate_{scale_name}_s{seed}.log",
                )

            if "alpha" in phases:
                for alpha in (0.2, 0.8):
                    add_task(
                        tasks,
                        f"alpha{alpha:.1f}_{scale_name}_s{seed}",
                        base + ["--alpha", f"{alpha:.1f}"],
                        out_alpha / f"cchihh_full_{scale_name}_a{alpha:.1f}_s{seed}.log",
                    )

        if "tgate" in phases and scale["name"] == "T500":
            for gate in (5, 10, 15, 20, 25):
                for seed in range(1, 11):
                    base = common_args(exe, data_dir, scale, seed, generations, log_every)
                    add_task(
                        tasks,
                        f"tgate{gate}_{scale['name']}_s{seed}",
                        base + ["--resample_gate", str(gate)],
                        out_tgate / f"cchihh_tgate{gate}_{scale['name']}_s{seed}.log",
                    )

        if "gantt" in phases and scale["name"] == "T200":
            add_task(
                tasks,
                "gantt_probe_T200_seed1",
                common_args(exe, data_dir, scale, 1, generations, log_every)
                + ["--schedule_export", str(out_schedule / "cchihh_full_T200_s1_schedule.csv")],
                out_schedule / "cchihh_full_T200_s1_schedule.log",
            )

    return tasks


def launch_task(task: dict, force: bool):
    log_path = task["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and not force and log_is_complete(log_path):
        return None

    log_file = log_path.open("w", encoding="utf-8")
    log_file.write(f"Command: {shlex.join(task['cmd'])}\n")
    log_file.flush()
    proc = subprocess.Popen(task["cmd"], stdout=log_file, stderr=subprocess.STDOUT, shell=False)
    return {"proc": proc, "log_file": log_file, "task": task}


def run_parallel(tasks: list[dict], max_parallel: int, force: bool):
    queue = tasks[:]
    running = []
    completed = 0
    total = len(tasks)

    while queue or running:
        while queue and len(running) < max_parallel:
            task = queue.pop(0)
            launched = launch_task(task, force)
            if launched is None:
                completed += 1
                print(f"[skip] {task['label']}")
            else:
                running.append(launched)
                print(f"[start] {task['label']}")

        time.sleep(1)
        still_running = []
        for item in running:
            if item["proc"].poll() is None:
                still_running.append(item)
                continue

            code = item["proc"].returncode
            item["log_file"].write(f"\nexit code {code}\n")
            item["log_file"].close()
            completed += 1
            print(f"[done] {item['task']['label']} (exit {code})")
        running = still_running

    print(f"All done. Completed {completed}/{total}.")


def log_is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "Time =" in text and "exit code 0" in text


def run_one(tasks: list[dict], force: bool):
    for task in tasks:
        if not force and log_is_complete(task["log_path"]):
            print(f"[skip-complete] {task['label']}")
            continue

        print(f"[run-one] {task['label']}")
        launched = launch_task(task, force=True if force else False)
        if launched is None:
            continue
        code = launched["proc"].wait()
        launched["log_file"].write(f"\nexit code {code}\n")
        launched["log_file"].close()
        print(f"[done] {task['label']} (exit {code})")
        return

    print("No pending tasks found.")


def parse_phases(raw: str) -> set[str]:
    parts = {x.strip().lower() for x in raw.split(",") if x.strip()}
    valid = {"full", "ablation", "alpha", "tgate", "gantt", "all"}
    invalid = sorted(parts - valid)
    if invalid:
        raise SystemExit(f"Unknown phases: {', '.join(invalid)}")
    if "all" in parts:
        return {"full", "ablation", "alpha", "tgate", "gantt"}
    return parts


def main():
    parser = argparse.ArgumentParser(description="Run post-operator-upgrade CCHIHH experiment batches.")
    parser.add_argument("--exe", default="", help="Path to CED_Schedule.exe")
    parser.add_argument("--phases", default="full", help="Comma-separated: full,ablation,alpha,tgate,gantt,all")
    parser.add_argument("--generations", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_parallel", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Overwrite existing logs")
    parser.add_argument("--run_one", action="store_true", help="Run only the next pending task and exit")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = find_exe(root, args.exe)
    phases = parse_phases(args.phases)
    tasks = build_tasks(root, exe, phases, args.generations, args.log_every)
    print(f"Executable: {exe}")
    print(f"Phases: {', '.join(sorted(phases))}")
    print(f"Tasks: {len(tasks)}")
    if args.run_one:
        run_one(tasks, force=args.force)
    else:
        run_parallel(tasks, max_parallel=args.max_parallel, force=args.force)


if __name__ == "__main__":
    main()
