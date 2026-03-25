from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path


SCALES = [
    {
        "name": "T100",
        "budget_seconds": 30.0,
        "data_file": "data_matrix_100.txt",
        "cnum": 100,
        "enum": 100,
        "dnum": 300,
        "tnum": 100,
        "mopt": 5,
    },
    {
        "name": "T200",
        "budget_seconds": 60.0,
        "data_file": "data_matrix_T200_E100_D300.txt",
        "cnum": 100,
        "enum": 100,
        "dnum": 300,
        "tnum": 200,
        "mopt": 5,
    },
    {
        "name": "T500",
        "budget_seconds": 300.0,
        "data_file": "data_matrix_T500_E200_D800.txt",
        "cnum": 200,
        "enum": 200,
        "dnum": 800,
        "tnum": 500,
        "mopt": 5,
    },
]

BASELINE_ALGOS = [
    ("CCHIHH", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15"]),
    ("DSAC_DE", ["--solver", "DSAC-DE"]),
    ("CGA", ["--solver", "CGA"]),
    ("IMOMA", ["--solver", "IMOMA"]),
]


def find_executable(root: Path, rel_paths: list[str]) -> Path:
    for rel in rel_paths:
        path = root / rel
        if path.exists():
            return path
    raise SystemExit(f"Executable not found. Tried: {', '.join(rel_paths)}")


def parse_alpha_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("No alpha values provided.")
    return values


def parse_scale_list(raw: str) -> set[str]:
    names = {part.strip() for part in raw.split(",") if part.strip()}
    valid = {scale["name"] for scale in SCALES}
    if not names:
        return valid
    unknown = names - valid
    if unknown:
        raise SystemExit(f"Unknown scales: {', '.join(sorted(unknown))}")
    return names


def alpha_dir(alpha: float) -> str:
    return f"alpha{alpha:.1f}"


def run_parallel(tasks: list[dict], max_parallel_non_dsac: int, max_parallel_dsac: int, force: bool) -> None:
    pending = tasks[:]
    running: list[dict] = []
    finished = 0

    def launch(task: dict) -> dict | None:
        out_csv = task["csv_path"]
        if out_csv.exists() and not force:
            return None
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        task["log_path"].parent.mkdir(parents=True, exist_ok=True)
        f = task["log_path"].open("w", encoding="utf-8")
        f.write(f"Command: {shlex.join(task['cmd'])}\n")
        f.flush()
        proc = subprocess.Popen(task["cmd"], stdout=f, stderr=subprocess.STDOUT, shell=False)
        return {"proc": proc, "file": f, "task": task}

    def can_launch(task: dict) -> bool:
        dsac_running = sum(1 for item in running if item["task"]["group"] == "dsac")
        other_running = sum(1 for item in running if item["task"]["group"] == "other")
        if task["group"] == "dsac":
            return dsac_running < max_parallel_dsac
        return other_running < max_parallel_non_dsac

    while pending or running:
        launched = False
        for idx, task in enumerate(pending):
            if not can_launch(task):
                continue
            pending.pop(idx)
            handle = launch(task)
            if handle is None:
                finished += 1
                print(f"[skip] {task['label']}")
            else:
                running.append(handle)
                print(f"[start] {task['label']}")
            launched = True
            break
        if launched:
            continue

        time.sleep(1.0)
        next_running: list[dict] = []
        for handle in running:
            if handle["proc"].poll() is None:
                next_running.append(handle)
                continue
            code = handle["proc"].returncode
            handle["file"].write(f"\nexit code {code}\n")
            handle["file"].close()
            finished += 1
            print(f"[done] {handle['task']['label']} code={code}")
            if code != 0:
                raise SystemExit(f"Task failed: {handle['task']['label']}")
        running = next_running

    print(f"Completed {finished} tasks.")


def build_tasks(
    root: Path,
    exe: Path,
    ppo_exe: Path | None,
    out_root: Path,
    alphas: list[float],
    scales: set[str],
    seeds: int,
    include_ppo: bool,
    skip_dsac: bool,
    only_dsac: bool,
) -> list[dict]:
    other_tasks: list[dict] = []
    dsac_tasks: list[dict] = []
    data_dir = root / "data"

    for alpha in alphas:
        for scale in SCALES:
            if scale["name"] not in scales:
                continue
            scale_dir = out_root / alpha_dir(alpha) / scale["name"]
            for seed in range(1, seeds + 1):
                for algo_name, solver_args in BASELINE_ALGOS:
                    if skip_dsac and algo_name == "DSAC_DE":
                        continue
                    if only_dsac and algo_name != "DSAC_DE":
                        continue
                    csv_name = f"{algo_name}_seed{seed}.csv"
                    csv_path = scale_dir / csv_name
                    log_path = scale_dir / f"{algo_name}_seed{seed}.log"
                    cmd = [
                        str(exe),
                        *solver_args,
                        "--seed",
                        str(seed),
                        "--alpha",
                        f"{alpha:.1f}",
                        "--time_budget_seconds",
                        str(scale["budget_seconds"]),
                        "--convergence_csv",
                        str(csv_path),
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
                    task = (
                        {
                            "label": f"alpha={alpha:.1f} {scale['name']} {algo_name} seed{seed}",
                            "cmd": cmd,
                            "csv_path": csv_path,
                            "log_path": log_path,
                            "group": "dsac" if algo_name == "DSAC_DE" else "other",
                        }
                    )
                    if algo_name == "DSAC_DE":
                        dsac_tasks.append(task)
                    else:
                        other_tasks.append(task)

                if include_ppo and not only_dsac:
                    if ppo_exe is None:
                        raise SystemExit("PPO requested but ppo_scheduler.exe was not found.")
                    csv_path = scale_dir / f"PPO_seed{seed}.csv"
                    log_path = scale_dir / f"PPO_seed{seed}.log"
                    cmd = [
                        str(ppo_exe),
                        "--seed",
                        str(seed),
                        "--alpha",
                        f"{alpha:.1f}",
                        "--time_budget_seconds",
                        str(scale["budget_seconds"]),
                        "--convergence_csv",
                        str(csv_path),
                        "--results_dir",
                        str(scale_dir),
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
                        "--scale_tag",
                        scale["name"],
                    ]
                    other_tasks.append(
                        {
                            "label": f"alpha={alpha:.1f} {scale['name']} PPO seed{seed}",
                            "cmd": cmd,
                            "csv_path": csv_path,
                            "log_path": log_path,
                            "group": "other",
                        }
                    )

    return other_tasks + dsac_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run wall-clock matched CED scheduling experiments.")
    parser.add_argument("--alphas", default="0.2,0.5,0.8")
    parser.add_argument("--scales", default="T100,T200,T500")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_parallel_non_dsac", type=int, default=2)
    parser.add_argument("--max_parallel_dsac", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include_ppo", action="store_true", help="Run PPO with wall-clock budget as well.")
    parser.add_argument("--skip_dsac", action="store_true", help="Skip DSAC-DE tasks in this batch.")
    parser.add_argument("--only_dsac", action="store_true", help="Run only DSAC-DE tasks in this batch.")
    parser.add_argument("--out_dir", default="results/wallclock")
    parser.add_argument("--exe", default="")
    parser.add_argument("--ppo_exe", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = Path(args.exe) if args.exe else find_executable(root, ["build_codex/Release/CED_Schedule.exe", "build/Release/CED_Schedule.exe"])
    if not exe.is_absolute():
        exe = root / exe
    ppo_exe = None
    if args.include_ppo:
        ppo_exe = Path(args.ppo_exe) if args.ppo_exe else find_executable(root, ["build_codex/Release/ppo_scheduler.exe", "build/Release/ppo_scheduler.exe"])
        if not ppo_exe.is_absolute():
            ppo_exe = root / ppo_exe

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = root / out_root

    if args.skip_dsac and args.only_dsac:
        raise SystemExit("--skip_dsac and --only_dsac cannot be used together.")

    tasks = build_tasks(
        root,
        exe,
        ppo_exe,
        out_root,
        parse_alpha_list(args.alphas),
        parse_scale_list(args.scales),
        args.seeds,
        args.include_ppo,
        args.skip_dsac,
        args.only_dsac,
    )
    run_parallel(
        tasks,
        max_parallel_non_dsac=args.max_parallel_non_dsac,
        max_parallel_dsac=args.max_parallel_dsac,
        force=args.force,
    )


if __name__ == "__main__":
    main()
