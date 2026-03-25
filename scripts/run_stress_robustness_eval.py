#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import time
from pathlib import Path

try:
    from scripts.stress_robustness_common import (
        DEFAULT_SCENARIO_LEVELS,
        ScenarioDefinition,
        make_scenario_definitions,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from stress_robustness_common import (  # type: ignore
        DEFAULT_SCENARIO_LEVELS,
        ScenarioDefinition,
        make_scenario_definitions,
    )


EVAL_BUDGET = 400000
ALPHA = 0.5
SEEDS_1_TO_10 = list(range(1, 11))
EVAL_RE = re.compile(r"^Eval\s+(\d+):\s+best_fit\s*=\s*([0-9.+\-eE]+)")
FINAL_RE = re.compile(r"^(?:The best(?: scalar)? solution|Best fitness)\s*=\s*([0-9.+\-eE]+)")
TIME_RE = re.compile(r"^Time\s*=\s*([0-9.+\-eE]+)\s*s")

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

ALGORITHMS = [
    {
        "name": "CCHIHH",
        "solver": "CCHIHH",
        "extra_args": ["--stable", "--migration", "--resample_gate", "15"],
        "output_group": "cchihh",
    },
    {
        "name": "CGA",
        "solver": "CGA",
        "extra_args": [],
        "output_group": "cga",
    },
    {
        "name": "IMOMA",
        "solver": "IMOMA",
        "extra_args": [],
        "output_group": "imoma",
    },
    {
        "name": "DSAC-DE",
        "solver": "DSAC-DE",
        "extra_args": [],
        "output_group": "dsac_de",
    },
]

SHARED_BANDIT_ALGORITHM = {
    "name": "CCHIHH_shared_bandit",
    "solver": "CCHIHH_shared_bandit",
    "extra_args": ["--stable", "--migration", "--resample_gate", "15"],
    "output_group": "sharedbandit",
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def find_exe(root: Path) -> Path:
    candidates = [
        root / "build_stress" / "Release" / "CED_Schedule.exe",
        root / "build_codex" / "Release" / "CED_Schedule.exe",
        root / "build" / "Release" / "CED_Schedule.exe",
        root / "build_copy" / "Release" / "CED_Schedule.exe",
        root / "build_local" / "Release" / "CED_Schedule.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit("Cannot find CED_Schedule.exe in known build folders.")


def export_eval_curve_from_log(log_path: Path, csv_path: Path) -> tuple[int, float]:
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
    return rows[-1]


def parse_final_metrics(log_path: Path) -> tuple[float | None, float | None]:
    final_best = None
    runtime_sec = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            match = FINAL_RE.match(line)
            if match:
                final_best = float(match.group(1))
                continue
            match = TIME_RE.match(line)
            if match:
                runtime_sec = float(match.group(1))
    return final_best, runtime_sec


def read_eval_csv_tail(csv_path: Path) -> tuple[int, float]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty eval csv: {csv_path}")
    last = rows[-1]
    return int(last["eval_count"]), float(last["best_fitness"])


def build_task(
    root: Path,
    exe: Path,
    scale: dict,
    algorithm: dict,
    scenario: ScenarioDefinition,
    seed: int,
) -> dict:
    stem = (
        f"{algorithm['name']}_{scale['name']}_{scenario.family}_{scenario.severity_label}_s{seed}"
        .replace("-", "_")
    )
    log_path = (
        root
        / "results"
        / "stress_robustness"
        / "logs"
        / scale["name"]
        / scenario.family
        / scenario.severity_label
        / f"{stem}.log"
    )
    eval_csv_path = (
        root
        / "results"
        / "eval"
        / "stress_robustness"
        / algorithm["output_group"]
        / f"alpha{ALPHA:.1f}"
        / scale["name"]
        / scenario.family
        / scenario.severity_label
        / f"{stem}_eval.csv"
    )
    cmd = [
        str(exe),
        "--solver",
        algorithm["solver"],
        *algorithm["extra_args"],
        "--seed",
        str(seed),
        "--alpha",
        f"{ALPHA:.1f}",
        "--generations",
        "999999",
        "--max_evals",
        str(EVAL_BUDGET),
        "--log_every",
        "2000",
        "--data_dir",
        str(root / "data"),
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
        "--stress_family",
        scenario.family,
        "--stress_severity",
        scenario.severity_label,
        "--stress_cloud_scale",
        f"{scenario.cloud_scale:.6f}",
        "--stress_edge_scale",
        f"{scenario.edge_scale:.6f}",
        "--stress_device_scale",
        f"{scenario.device_scale:.6f}",
        "--stress_comm_scale",
        f"{scenario.comm_scale:.6f}",
    ]
    return {
        "instance": scale["name"],
        "algorithm": algorithm["name"],
        "scenario_family": scenario.family,
        "severity": scenario.severity_label,
        "severity_value": scenario.severity_value,
        "seed": seed,
        "eval_budget": EVAL_BUDGET,
        "stress_cloud_scale": scenario.cloud_scale,
        "stress_edge_scale": scenario.edge_scale,
        "stress_device_scale": scenario.device_scale,
        "stress_comm_scale": scenario.comm_scale,
        "cmd": cmd,
        "log_path": log_path,
        "eval_csv_path": eval_csv_path,
    }


def build_tasks(
    root: Path,
    exe: Path,
    scales: list[dict] | None = None,
    algorithms: list[dict] | None = None,
    seeds: list[int] | None = None,
    scenario_levels: dict[str, list[float]] | None = None,
    include_shared_bandit: bool = False,
    reuse_nominal: bool = True,
) -> list[dict]:
    selected_scales = SCALE_CONFIGS if scales is None else scales
    selected_algorithms = list(ALGORITHMS if algorithms is None else algorithms)
    if include_shared_bandit:
        selected_algorithms.append(SHARED_BANDIT_ALGORITHM)
    selected_seeds = SEEDS_1_TO_10 if seeds is None else seeds
    scenarios = make_scenario_definitions(scenario_levels or DEFAULT_SCENARIO_LEVELS)

    tasks: list[dict] = []
    for scale in selected_scales:
        for algorithm in selected_algorithms:
            for scenario in scenarios:
                if reuse_nominal and scenario.is_nominal:
                    continue
                for seed in selected_seeds:
                    tasks.append(build_task(root, exe, scale, algorithm, scenario, seed))
    return tasks


def nominal_eval_csv_path(root: Path, algorithm_name: str, instance: str, seed: int) -> Path:
    if algorithm_name == "CCHIHH":
        return root / "results" / "eval" / "rerun_full" / "alpha0.5" / instance / f"cchihh_full_{instance}_s{seed}_eval.csv"
    if algorithm_name == "CGA":
        return root / "results" / "eval" / "cga_baseline" / "alpha0.5" / instance / f"CGA_{instance}_s{seed}_eval.csv"
    if algorithm_name == "IMOMA":
        return root / "results" / "eval" / "imoma_baseline" / "alpha0.5" / instance / f"IMOMA_{instance}_s{seed}_eval.csv"
    if algorithm_name == "DSAC-DE":
        return root / "results" / "eval" / "dsac_de_multiscale" / "alpha0.5" / instance / f"DSAC_DE_{instance}_s{seed}_eval.csv"
    raise ValueError(f"Unsupported nominal algorithm: {algorithm_name}")


def load_existing_nominal_rows(root: Path, scales: list[dict], algorithms: list[dict], seeds: list[int]) -> list[dict]:
    final_summary_path = root / "results" / "final_summary.csv"
    if not final_summary_path.exists():
        raise FileNotFoundError(f"Missing nominal summary: {final_summary_path}")
    with final_summary_path.open("r", encoding="utf-8", newline="") as f:
        summary_rows = list(csv.DictReader(f))

    variant_map = {
        "CCHIHH": ("CCHIHH-full", "new_rerun"),
        "CGA": ("CGA", "old_baseline"),
        "IMOMA": ("IMOMA", "old_baseline"),
        "DSAC-DE": ("DSAC-DE", "old_baseline"),
    }
    lookup = {}
    for row in summary_rows:
        key = (row["variant"], row["source"], row["scale"], row["seed"], row["alpha"])
        lookup[key] = row

    nominal_rows: list[dict] = []
    for scale in scales:
        for algorithm in algorithms:
            variant, source = variant_map[algorithm["name"]]
            for seed in seeds:
                row = lookup.get((variant, source, scale["name"], str(seed), "0.5"))
                if row is None:
                    raise KeyError(
                        f"Missing nominal row for {algorithm['name']} {scale['name']} seed {seed} in {final_summary_path}"
                    )
                nominal_rows.append(
                    {
                        "algorithm": algorithm["name"],
                        "instance": scale["name"],
                        "scenario_family": "nominal",
                        "severity": "nominal",
                        "severity_value": 0.0,
                        "seed": seed,
                        "best_fitness": row["best_fit"],
                        "runtime_sec": row["runtime_s"],
                        "eval_budget": EVAL_BUDGET,
                        "final_eval": EVAL_BUDGET,
                        "stress_cloud_scale": 1.0,
                        "stress_edge_scale": 1.0,
                        "stress_device_scale": 1.0,
                        "stress_comm_scale": 1.0,
                        "log_path": row["path"],
                        "eval_csv_path": str(nominal_eval_csv_path(root, algorithm["name"], scale["name"], seed)),
                    }
                )
    return nominal_rows


def write_index(index_path: Path, rows: list[dict]) -> None:
    ensure_parent(index_path)
    unique_rows: dict[tuple[str, str, str, str, int], dict] = {}
    for row in rows:
        key = (
            str(row["algorithm"]),
            str(row["instance"]),
            str(row["scenario_family"]),
            str(row["severity"]),
            int(row["seed"]),
        )
        unique_rows[key] = row
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "instance",
                "scenario_family",
                "severity",
                "severity_value",
                "seed",
                "best_fitness",
                "runtime_sec",
                "eval_budget",
                "final_eval",
                "stress_cloud_scale",
                "stress_edge_scale",
                "stress_device_scale",
                "stress_comm_scale",
                "log_path",
                "eval_csv_path",
            ],
        )
        writer.writeheader()
        for row in sorted(
            unique_rows.values(),
            key=lambda item: (
                item["instance"],
                item["scenario_family"],
                float(item["severity_value"]),
                item["algorithm"],
                int(item["seed"]),
            ),
        ):
            writer.writerow(row)


def materialize_existing_task(task: dict) -> dict:
    final_eval, final_curve_best = read_eval_csv_tail(task["eval_csv_path"])
    final_best, runtime_sec = parse_final_metrics(task["log_path"]) if task["log_path"].exists() else (None, None)
    return {
        "algorithm": task["algorithm"],
        "instance": task["instance"],
        "scenario_family": task["scenario_family"],
        "severity": task["severity"],
        "severity_value": task["severity_value"],
        "seed": task["seed"],
        "best_fitness": f"{(final_best if final_best is not None else final_curve_best):.15g}",
        "runtime_sec": "" if runtime_sec is None else f"{runtime_sec:.15g}",
        "eval_budget": task["eval_budget"],
        "final_eval": final_eval,
        "stress_cloud_scale": task["stress_cloud_scale"],
        "stress_edge_scale": task["stress_edge_scale"],
        "stress_device_scale": task["stress_device_scale"],
        "stress_comm_scale": task["stress_comm_scale"],
        "log_path": str(task["log_path"]),
        "eval_csv_path": str(task["eval_csv_path"]),
    }


def launch_task(task: dict, force: bool) -> dict | None:
    if task["eval_csv_path"].exists() and not force:
        return {"task": task, "existing": True}
    ensure_parent(task["log_path"])
    f = task["log_path"].open("w", encoding="utf-8", newline="")
    f.write(f"Command: {shlex.join(task['cmd'])}\n")
    f.flush()
    proc = subprocess.Popen(task["cmd"], stdout=f, stderr=subprocess.STDOUT, shell=False)
    return {"task": task, "proc": proc, "file": f}


def process_completed_task(task: dict) -> dict:
    final_eval, final_curve_best = export_eval_curve_from_log(task["log_path"], task["eval_csv_path"])
    final_best, runtime_sec = parse_final_metrics(task["log_path"])
    return {
        "algorithm": task["algorithm"],
        "instance": task["instance"],
        "scenario_family": task["scenario_family"],
        "severity": task["severity"],
        "severity_value": task["severity_value"],
        "seed": task["seed"],
        "best_fitness": f"{(final_best if final_best is not None else final_curve_best):.15g}",
        "runtime_sec": "" if runtime_sec is None else f"{runtime_sec:.15g}",
        "eval_budget": task["eval_budget"],
        "final_eval": final_eval,
        "stress_cloud_scale": task["stress_cloud_scale"],
        "stress_edge_scale": task["stress_edge_scale"],
        "stress_device_scale": task["stress_device_scale"],
        "stress_comm_scale": task["stress_comm_scale"],
        "log_path": str(task["log_path"]),
        "eval_csv_path": str(task["eval_csv_path"]),
    }


def run_parallel(tasks: list[dict], max_parallel: int, force: bool, index_path: Path, initial_rows: list[dict] | None = None) -> None:
    queue = tasks[:]
    running: list[dict] = []
    completed_rows: list[dict] = list(initial_rows or [])
    done = 0
    total = len(tasks)
    if completed_rows:
        write_index(index_path, completed_rows)

    while queue or running:
        while queue and len(running) < max_parallel:
            task = queue.pop(0)
            launched = launch_task(task, force=force)
            if launched is None:
                done += 1
                print(
                    f"[skip] {task['instance']} {task['scenario_family']} "
                    f"{task['severity']} {task['algorithm']} seed={task['seed']}"
                )
                continue
            if launched.get("existing"):
                completed_rows.append(materialize_existing_task(task))
                write_index(index_path, completed_rows)
                done += 1
                print(
                    f"[resume-skip] {task['instance']} {task['scenario_family']} "
                    f"{task['severity']} {task['algorithm']} seed={task['seed']} ({done}/{total})"
                )
                continue
            running.append(launched)
            print(
                f"[start] {task['instance']} {task['scenario_family']} "
                f"{task['severity']} {task['algorithm']} seed={task['seed']}"
            )

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
                    f"Task failed: {task['instance']} {task['scenario_family']} "
                    f"{task['severity']} {task['algorithm']} seed={task['seed']} rc={code}"
                )
            completed_rows.append(process_completed_task(task))
            write_index(index_path, completed_rows)
            done += 1
            print(
                f"[done] {task['instance']} {task['scenario_family']} "
                f"{task['severity']} {task['algorithm']} seed={task['seed']} ({done}/{total})"
            )
        running = still_running


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval-budgeted stress robustness experiments.")
    parser.add_argument("--instances", default="T200,T500", help="Comma-separated: T100,T200,T500")
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include_shared_bandit", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = find_exe(root)
    selected_names = {item.strip() for item in args.instances.split(",") if item.strip()}
    selected_scales = [cfg for cfg in SCALE_CONFIGS if cfg["name"] in selected_names]
    if not selected_scales:
        raise SystemExit("No valid instances selected.")

    tasks = build_tasks(
        root=root,
        exe=exe,
        scales=selected_scales,
        scenario_levels=DEFAULT_SCENARIO_LEVELS,
        include_shared_bandit=args.include_shared_bandit,
        reuse_nominal=True,
    )
    index_path = root / "results" / "eval" / "stress_robustness" / "index.csv"
    nominal_rows = load_existing_nominal_rows(root, selected_scales, ALGORITHMS, SEEDS_1_TO_10)
    run_parallel(tasks, max_parallel=args.max_parallel, force=args.force, index_path=index_path, initial_rows=nominal_rows)


if __name__ == "__main__":
    main()
