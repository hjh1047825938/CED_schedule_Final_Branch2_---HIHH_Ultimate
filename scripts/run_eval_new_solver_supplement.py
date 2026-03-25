#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

try:
    from scripts.stress_robustness_common import DEFAULT_SCENARIO_LEVELS, non_nominal_scenarios
except ModuleNotFoundError:  # pragma: no cover
    from stress_robustness_common import DEFAULT_SCENARIO_LEVELS, non_nominal_scenarios  # type: ignore


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

SOLVER_METADATA = {
    "rde": {
        "solver": "rde",
        "solver_key": "rde",
        "display_name": "RDE",
        "source_files": ["src/solver_rde.cpp", "include/solver_rde.h"],
    },
    "L-SRTDE": {
        "solver": "L-SRTDE",
        "solver_key": "l_srtde",
        "display_name": "L-SRTDE",
        "source_files": ["src/L_SRTDE.cpp", "include/L_SRTDE.h"],
    },
    "NL-SHADE-LBC": {
        "solver": "NL-SHADE-LBC",
        "solver_key": "nl_shade_lbc",
        "display_name": "NL-SHADE-LBC",
        "source_files": ["src/NL_SHADE_LBC.cpp", "include/NL_SHADE_LBC.h"],
    },
}

DEFAULT_NEW_SOLVER_ORDER = ["rde", "L-SRTDE", "NL-SHADE-LBC"]


def alpha_tag(alpha: float) -> str:
    return f"alpha{alpha:.1f}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_eval_dirs(root: Path) -> dict[str, Path]:
    base = root / "results" / "eval"
    dirs = {
        "base": base,
        "main": base / "main",
        "degradation": base / "degradation",
        "alpha": base / "alpha",
        "traces": base / "traces",
        "logs": base / "logs",
        "summaries": base / "summaries",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def find_exe(root: Path, exe_arg: str = "") -> Path:
    if exe_arg:
        candidate = Path(exe_arg)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate
        raise SystemExit(f"Executable not found: {candidate}")
    for candidate in (
        root / "build" / "Release" / "CED_Schedule.exe",
        root / "build_stress" / "Release" / "CED_Schedule.exe",
        root / "build_copy" / "Release" / "CED_Schedule.exe",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit("Cannot find CED_Schedule.exe in known build folders.")


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    if b"\x00" in raw[:256]:
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                pass
    for enc in ("utf-8", "gbk", "latin1", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin1", errors="ignore")


def auto_detect_new_solvers(root: Path) -> list[str]:
    main_cpp = (root / "src" / "main.cpp").read_text(encoding="utf-8", errors="ignore")
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8", errors="ignore")
    detected: list[str] = []
    for key in DEFAULT_NEW_SOLVER_ORDER:
        meta = SOLVER_METADATA[key]
        files_exist = all((root / rel).exists() for rel in meta["source_files"])
        in_registry = meta["solver"] in main_cpp and any(rel.replace("\\", "/") in cmake for rel in meta["source_files"])
        if files_exist and in_registry:
            detected.append(key)
    return detected if len(detected) == 3 else DEFAULT_NEW_SOLVER_ORDER


def parse_solver_list(raw: str | None, root: Path) -> list[str]:
    if raw:
        items = [item.strip() for item in raw.split(",") if item.strip()]
        unknown = [item for item in items if item not in SOLVER_METADATA]
        if unknown:
            raise SystemExit(f"Unknown solver(s): {', '.join(unknown)}")
        return items
    return auto_detect_new_solvers(root)


def parse_scale_list(raw: str) -> list[dict]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    lookup = {cfg["name"]: cfg for cfg in SCALE_CONFIGS}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise SystemExit(f"Unknown scale(s): {', '.join(missing)}")
    return [lookup[name] for name in names]


def parse_log_metrics(log_path: Path) -> tuple[float | None, float | None]:
    final_best = None
    runtime_sec = None
    for raw_line in read_text_auto(log_path).splitlines():
        line = raw_line.strip().replace("\x00", "")
        match = FINAL_RE.match(line)
        if match:
            final_best = float(match.group(1))
            continue
        match = TIME_RE.match(line)
        if match:
            runtime_sec = float(match.group(1))
    return final_best, runtime_sec


def build_trace_rows(trace_csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with trace_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "eval_count": int(float(row["eval_count"])),
                    "best_fitness": float(row["best_fitness"]),
                    "elapsed_seconds": float(row["time_seconds"]),
                }
            )
    return rows


def base_command(exe: Path, root: Path, solver_key: str, scale: dict, seed: int, alpha: float, eval_budget: int, log_every: int) -> list[str]:
    solver_name = SOLVER_METADATA[solver_key]["solver"]
    return [
        str(exe),
        "--solver",
        solver_name,
        "--seed",
        str(seed),
        "--alpha",
        f"{alpha:.1f}",
        "--generations",
        "999999",
        "--max_evals",
        str(eval_budget),
        "--log_every",
        str(log_every),
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
    ]


def result_file_name(solver_key: str, scale_name: str, alpha: float, seed: int, scenario_family: str = "nominal", severity: str = "nominal") -> str:
    if scenario_family == "nominal":
        return f"{solver_key}_{scale_name}_{alpha_tag(alpha)}_seed{seed}.json"
    return f"{solver_key}_{scale_name}_{scenario_family}_{severity}_seed{seed}.json"


def trace_file_name(solver_key: str, scale_name: str, seed: int, tag: str) -> str:
    return f"{solver_key}_{scale_name}_{tag}_seed{seed}.csv"


def log_file_name(solver_key: str, scale_name: str, seed: int, tag: str) -> str:
    return f"{solver_key}_{scale_name}_{tag}_seed{seed}.log"


def build_main_tasks(root: Path, exe: Path, solvers: list[str], scales: list[dict], seed: int, eval_budget: int, log_every: int) -> list[dict]:
    dirs = ensure_eval_dirs(root)
    tasks: list[dict] = []
    for solver_key in solvers:
        for scale in scales:
            result_path = dirs["main"] / result_file_name(SOLVER_METADATA[solver_key]["solver_key"], scale["name"], 0.5, seed)
            trace_path = dirs["traces"] / trace_file_name(SOLVER_METADATA[solver_key]["solver_key"], scale["name"], seed, "main_alpha0.5")
            log_path = dirs["logs"] / log_file_name(SOLVER_METADATA[solver_key]["solver_key"], scale["name"], seed, "main_alpha0.5")
            cmd = base_command(exe, root, solver_key, scale, seed, 0.5, eval_budget, log_every) + ["--convergence_csv", str(trace_path)]
            tasks.append(
                {
                    "kind": "main",
                    "solver": SOLVER_METADATA[solver_key]["display_name"],
                    "solver_key": SOLVER_METADATA[solver_key]["solver_key"],
                    "scale": scale["name"],
                    "alpha": 0.5,
                    "seed": seed,
                    "scenario_family": "nominal",
                    "severity": "nominal",
                    "scenario_label": "main",
                    "cmd": cmd,
                    "result_path": result_path,
                    "trace_path": trace_path,
                    "log_path": log_path,
                }
            )
    return tasks


def build_degradation_tasks(root: Path, exe: Path, solvers: list[str], scales: list[dict], seed: int, eval_budget: int, log_every: int) -> list[dict]:
    dirs = ensure_eval_dirs(root)
    tasks: list[dict] = []
    for solver_key in solvers:
        for scale in scales:
            for scenario in non_nominal_scenarios(DEFAULT_SCENARIO_LEVELS):
                tag = f"degradation_{scenario.family}_{scenario.severity_label}_{alpha_tag(0.5)}"
                solver_tag = SOLVER_METADATA[solver_key]["solver_key"]
                result_path = dirs["degradation"] / result_file_name(solver_tag, scale["name"], 0.5, seed, scenario.family, scenario.severity_label)
                trace_path = dirs["traces"] / trace_file_name(solver_tag, scale["name"], seed, tag)
                log_path = dirs["logs"] / log_file_name(solver_tag, scale["name"], seed, tag)
                cmd = base_command(exe, root, solver_key, scale, seed, 0.5, eval_budget, log_every) + [
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
                    "--convergence_csv",
                    str(trace_path),
                ]
                tasks.append(
                    {
                        "kind": "degradation",
                        "solver": SOLVER_METADATA[solver_key]["display_name"],
                        "solver_key": solver_tag,
                        "scale": scale["name"],
                        "alpha": 0.5,
                        "seed": seed,
                        "scenario_family": scenario.family,
                        "severity": scenario.severity_label,
                        "scenario_label": f"{scenario.family}_{scenario.severity_label}",
                        "cmd": cmd,
                        "result_path": result_path,
                        "trace_path": trace_path,
                        "log_path": log_path,
                        "stress_cloud_scale": scenario.cloud_scale,
                        "stress_edge_scale": scenario.edge_scale,
                        "stress_device_scale": scenario.device_scale,
                        "stress_comm_scale": scenario.comm_scale,
                    }
                )
    return tasks


def build_alpha_tasks(root: Path, exe: Path, solvers: list[str], scales: list[dict], seed: int, eval_budget: int, log_every: int, alphas: list[float]) -> list[dict]:
    dirs = ensure_eval_dirs(root)
    tasks: list[dict] = []
    for solver_key in solvers:
        for scale in scales:
            for alpha in alphas:
                solver_tag = SOLVER_METADATA[solver_key]["solver_key"]
                result_path = dirs["alpha"] / result_file_name(solver_tag, scale["name"], alpha, seed)
                trace_path = dirs["traces"] / trace_file_name(solver_tag, scale["name"], seed, alpha_tag(alpha))
                log_path = dirs["logs"] / log_file_name(solver_tag, scale["name"], seed, alpha_tag(alpha))
                cmd = base_command(exe, root, solver_key, scale, seed, alpha, eval_budget, log_every) + ["--convergence_csv", str(trace_path)]
                tasks.append(
                    {
                        "kind": "alpha",
                        "solver": SOLVER_METADATA[solver_key]["display_name"],
                        "solver_key": solver_tag,
                        "scale": scale["name"],
                        "alpha": alpha,
                        "seed": seed,
                        "scenario_family": "nominal",
                        "severity": "nominal",
                        "scenario_label": f"alpha_{alpha:.1f}",
                        "cmd": cmd,
                        "result_path": result_path,
                        "trace_path": trace_path,
                        "log_path": log_path,
                    }
                )
    return tasks


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_task(task: dict, force: bool) -> dict:
    if task["result_path"].exists() and not force:
        return load_json(task["result_path"])
    ensure_parent(task["log_path"])
    with task["log_path"].open("w", encoding="utf-8", newline="") as f:
        f.write(f"Command: {shlex.join(task['cmd'])}\n")
        f.flush()
        proc = subprocess.run(task["cmd"], stdout=f, stderr=subprocess.STDOUT, shell=False)
        f.write(f"\nexit code {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Task failed: {task['log_path']}")

    final_fitness, runtime_sec = parse_log_metrics(task["log_path"])
    if final_fitness is None:
        raise RuntimeError(f"Final fitness not found in {task['log_path']}")
    payload = {
        "kind": task["kind"],
        "solver": task["solver"],
        "solver_key": task["solver_key"],
        "scale": task["scale"],
        "alpha": task["alpha"],
        "seed": task["seed"],
        "scenario_family": task["scenario_family"],
        "severity": task["severity"],
        "scenario_label": task["scenario_label"],
        "eval_budget": 400000,
        "final_fitness": final_fitness,
        "wall_clock_seconds": runtime_sec,
        "trace_path": str(task["trace_path"]),
        "log_path": str(task["log_path"]),
        "result_path": str(task["result_path"]),
    }
    if "stress_cloud_scale" in task:
        payload["stress_cloud_scale"] = task["stress_cloud_scale"]
        payload["stress_edge_scale"] = task["stress_edge_scale"]
        payload["stress_device_scale"] = task["stress_device_scale"]
        payload["stress_comm_scale"] = task["stress_comm_scale"]
    dump_json(task["result_path"], payload)
    return payload


def wait_for_slot(running: list[dict], completed: list[dict]) -> list[dict]:
    still_running: list[dict] = []
    for item in running:
        if item["proc"].poll() is None:
            still_running.append(item)
            continue
        item["file"].write(f"\nexit code {item['proc'].returncode}\n")
        item["file"].close()
        if item["proc"].returncode != 0:
            raise RuntimeError(f"Task failed: {item['task']['log_path']}")
        task = item["task"]
        final_fitness, runtime_sec = parse_log_metrics(task["log_path"])
        if final_fitness is None:
            raise RuntimeError(f"Final fitness not found in {task['log_path']}")
        payload = {
            "kind": task["kind"],
            "solver": task["solver"],
            "solver_key": task["solver_key"],
            "scale": task["scale"],
            "alpha": task["alpha"],
            "seed": task["seed"],
            "scenario_family": task["scenario_family"],
            "severity": task["severity"],
            "scenario_label": task["scenario_label"],
            "eval_budget": 400000,
            "final_fitness": final_fitness,
            "wall_clock_seconds": runtime_sec,
            "trace_path": str(task["trace_path"]),
            "log_path": str(task["log_path"]),
            "result_path": str(task["result_path"]),
        }
        if "stress_cloud_scale" in task:
            payload["stress_cloud_scale"] = task["stress_cloud_scale"]
            payload["stress_edge_scale"] = task["stress_edge_scale"]
            payload["stress_device_scale"] = task["stress_device_scale"]
            payload["stress_comm_scale"] = task["stress_comm_scale"]
        dump_json(task["result_path"], payload)
        completed.append(payload)
        print(
            f"[done] {task['kind']} {task['solver_key']} {task['scale']} "
            f"{task['scenario_label']} seed={task['seed']}"
        )
    return still_running


def run_tasks(tasks: list[dict], max_parallel: int, force: bool) -> list[dict]:
    queue = tasks[:]
    running: list[dict] = []
    completed: list[dict] = []

    while queue or running:
        while queue and len(running) < max_parallel:
            task = queue.pop(0)
            if task["result_path"].exists() and not force:
                completed.append(load_json(task["result_path"]))
                print(
                    f"[skip] {task['kind']} {task['solver_key']} {task['scale']} "
                    f"{task['scenario_label']} seed={task['seed']}"
                )
                continue
            ensure_parent(task["log_path"])
            handle = task["log_path"].open("w", encoding="utf-8", newline="")
            handle.write(f"Command: {shlex.join(task['cmd'])}\n")
            handle.flush()
            proc = subprocess.Popen(task["cmd"], stdout=handle, stderr=subprocess.STDOUT, shell=False)
            running.append({"task": task, "proc": proc, "file": handle})
            print(
                f"[start] {task['kind']} {task['solver_key']} {task['scale']} "
                f"{task['scenario_label']} seed={task['seed']}"
            )
        if running:
            time.sleep(1)
            running = wait_for_slot(running, completed)
    return completed


def make_alpha_result_from_main(main_result_path: Path, target_path: Path) -> dict:
    payload = load_json(main_result_path)
    reused = dict(payload)
    reused["kind"] = "alpha"
    reused["scenario_label"] = "alpha_0.5"
    reused["reused_from"] = str(main_result_path)
    reused["result_path"] = str(target_path)
    dump_json(target_path, reused)
    return reused


def compute_prr(nominal_fitness: float, degraded_fitness: float) -> float | None:
    if abs(nominal_fitness) <= 1e-12:
        return None
    return degraded_fitness / nominal_fitness


def attach_nominal_fields(degradation_payloads: list[dict], main_lookup: dict[tuple[str, str], dict]) -> list[dict]:
    enriched: list[dict] = []
    for payload in degradation_payloads:
        nominal = main_lookup[(payload["solver_key"], payload["scale"])]
        enriched_payload = dict(payload)
        enriched_payload["nominal_fitness"] = nominal["final_fitness"]
        enriched_payload["nominal_wall_clock_seconds"] = nominal.get("wall_clock_seconds")
        enriched_payload["nominal_result_path"] = nominal["result_path"]
        enriched_payload["prr"] = compute_prr(nominal["final_fitness"], payload["final_fitness"])
        dump_json(Path(payload["result_path"]), enriched_payload)
        enriched.append(enriched_payload)
    return enriched


def build_summary_record(payload: dict) -> dict:
    record = {
        "solver": payload["solver"],
        "solver_key": payload["solver_key"],
        "scale": payload["scale"],
        "alpha": payload["alpha"],
        "seed": payload["seed"],
        "scenario_family": payload["scenario_family"],
        "severity": payload["severity"],
        "final_fitness": payload["final_fitness"],
        "wall_clock_seconds": payload.get("wall_clock_seconds"),
        "trace_path": payload["trace_path"],
        "result_path": payload["result_path"],
    }
    if payload.get("kind") == "degradation" or "nominal_fitness" in payload:
        record["nominal_fitness"] = payload.get("nominal_fitness")
        record["degraded_fitness"] = payload["final_fitness"]
        record["prr"] = payload.get("prr")
    return record


def write_summary_files(csv_path: Path, json_path: Path, payloads: list[dict]) -> None:
    rows = [build_summary_record(payload) for payload in payloads]
    fieldnames = list(rows[0].keys()) if rows else [
        "solver", "solver_key", "scale", "alpha", "seed", "scenario_family", "severity",
        "final_fitness", "wall_clock_seconds", "trace_path", "result_path",
    ]
    ensure_parent(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    dump_json(json_path, rows)


def write_readme(root: Path, solvers: list[str], main_payloads: list[dict], degradation_payloads: list[dict], alpha_payloads: list[dict]) -> None:
    readme_path = root / "results" / "eval" / "summaries" / "README.md"
    lines = [
        "# Eval Supplement Runs",
        "",
        "This supplement only fills missing results for the three new solvers.",
        "",
        f"- New solvers: {', '.join(SOLVER_METADATA[key]['display_name'] for key in solvers)}",
        "- Seed: 1",
        "- Eval budget: 400000",
        "- Max parallel jobs: 2",
        "- Main alpha=0.5 results are reused for alpha sensitivity at alpha=0.5.",
        "- Main nominal results are reused as degradation baselines.",
        "",
        "## Generated Files",
        "",
        "- `results/eval/main/`: per-run main comparison JSON files",
        "- `results/eval/degradation/`: per-run degradation JSON files",
        "- `results/eval/alpha/`: per-run alpha sensitivity JSON files",
        "- `results/eval/traces/`: convergence traces with eval count and elapsed seconds",
        "- `results/eval/logs/`: raw run logs",
        "- `results/eval/summaries/`: merged CSV/JSON summaries",
        "",
        "## Counts",
        "",
        f"- Main runs: {len(main_payloads)}",
        f"- Degradation runs: {len(degradation_payloads)}",
        f"- Alpha records: {len(alpha_payloads)}",
        "",
        "## Re-run",
        "",
        "```powershell",
        "python scripts/run_eval_new_solver_supplement.py --max_parallel 2",
        "```",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run missing-only supplement experiments for the three new solvers.")
    parser.add_argument("--exe", default="")
    parser.add_argument("--new_solvers", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_budget", type=int, default=400000)
    parser.add_argument("--log_every", type=int, default=2000)
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--main_scales", default="T100,T200,T500")
    parser.add_argument("--degradation_scales", default="T200,T500")
    parser.add_argument("--alpha_scales", default="T100,T200,T500")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ensure_eval_dirs(root)
    exe = find_exe(root, args.exe)
    solvers = parse_solver_list(args.new_solvers or None, root)
    main_scales = parse_scale_list(args.main_scales)
    degradation_scales = parse_scale_list(args.degradation_scales)
    alpha_scales = parse_scale_list(args.alpha_scales)

    main_tasks = build_main_tasks(root, exe, solvers, main_scales, args.seed, args.eval_budget, args.log_every)
    main_payloads = run_tasks(main_tasks, args.max_parallel, args.force)

    main_lookup = {(payload["solver_key"], payload["scale"]): payload for payload in main_payloads}
    degradation_tasks = build_degradation_tasks(root, exe, solvers, degradation_scales, args.seed, args.eval_budget, args.log_every)
    degradation_payloads = attach_nominal_fields(
        run_tasks(degradation_tasks, args.max_parallel, args.force),
        main_lookup,
    )

    alpha_tasks = build_alpha_tasks(root, exe, solvers, alpha_scales, args.seed, args.eval_budget, args.log_every, [0.2, 0.8])
    alpha_payloads = run_tasks(alpha_tasks, args.max_parallel, args.force)
    for solver_key in solvers:
        for scale in alpha_scales:
            solver_tag = SOLVER_METADATA[solver_key]["solver_key"]
            main_result_path = root / "results" / "eval" / "main" / result_file_name(solver_tag, scale["name"], 0.5, args.seed)
            alpha_result_path = root / "results" / "eval" / "alpha" / result_file_name(solver_tag, scale["name"], 0.5, args.seed)
            alpha_payloads.append(make_alpha_result_from_main(main_result_path, alpha_result_path))

    summaries = root / "results" / "eval" / "summaries"
    write_summary_files(
        summaries / "main_new_solvers.csv",
        summaries / "main_new_solvers.json",
        main_payloads,
    )
    write_summary_files(
        summaries / "degradation_new_solvers.csv",
        summaries / "degradation_new_solvers.json",
        degradation_payloads,
    )
    write_summary_files(
        summaries / "alpha_new_solvers.csv",
        summaries / "alpha_new_solvers.json",
        alpha_payloads,
    )
    write_readme(root, solvers, main_payloads, degradation_payloads, alpha_payloads)


if __name__ == "__main__":
    main()
