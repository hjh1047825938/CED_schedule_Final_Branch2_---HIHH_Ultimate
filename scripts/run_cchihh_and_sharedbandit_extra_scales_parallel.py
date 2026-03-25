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
        "name": "T50",
        "data_file": "data_matrix_T50_E50_D150.txt",
        "cnum": 50,
        "enum": 50,
        "dnum": 150,
        "tnum": 50,
        "mopt": 5,
    },
    {
        "name": "T300",
        "data_file": "data_matrix_T300_E150_D500.txt",
        "cnum": 150,
        "enum": 150,
        "dnum": 500,
        "tnum": 300,
        "mopt": 5,
    },
    {
        "name": "T400",
        "data_file": "data_matrix_T400_E175_D650.txt",
        "cnum": 175,
        "enum": 175,
        "dnum": 650,
        "tnum": 400,
        "mopt": 5,
    },
]

SOLVER_CONFIGS = [
    {
        "solver_name": "CCHIHH",
        "group": "cchihh",
        "variant": "cchihh_full",
        "raw_dir": "cchihh",
    },
    {
        "solver_name": "CCHIHH_shared_bandit",
        "group": "sharedbandit",
        "variant": "cchihh_shared_bandit",
        "raw_dir": "sharedbandit",
    },
]

SEEDS = list(range(1, 11))
ALPHA = 0.5
POPSIZE = 40
NSUBPOP = 8
MAX_EVALS = 400000
LOG_EVERY_EVALS = 6000
LOG_EVERY_GENS = 50
INIT_EVALS = 40


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def find_exe(root: Path) -> Path:
    candidates = [
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
            eval_count = int(match.group(1)) + INIT_EVALS
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


def convert_gen_keyed_csv(input_path: Path, output_path: Path) -> bool:
    rows_to_write: list[list[str]] = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return False
        if header[0] != "gen":
            raise ValueError(f"Unsupported gen-keyed CSV header in {input_path}")
        out_header = ["eval_count", *header[1:]]
        for row in reader:
            if not row:
                continue
            gen = int(float(row[0]))
            eval_count = INIT_EVALS + gen * (POPSIZE * 3)
            if eval_count > MAX_EVALS:
                continue
            rows_to_write.append([str(eval_count), *row[1:]])

    if not rows_to_write:
        return False

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        writer.writerows(rows_to_write)
    return True


def build_task(root: Path, exe: Path, solver_cfg: dict, scale: dict, seed: int) -> dict:
    results_root = root / "results" / solver_cfg["raw_dir"]
    eval_root = root / "results" / "eval" / solver_cfg["group"] / f"alpha{ALPHA:.1f}" / scale["name"]

    stem = f"{solver_cfg['variant']}_{scale['name']}_s{seed}"
    log_path = results_root / f"{stem}.log"
    op_stats_path = results_root / f"{stem}_ops.csv"
    w_off_path = results_root / f"{stem}_w_off.csv"
    w_seq_path = results_root / f"{stem}_w_seq.csv"
    w_dev_path = results_root / f"{stem}_w_dev.csv"

    eval_csv_path = eval_root / f"{stem}_eval.csv"
    ops_eval_path = eval_root / f"{stem}_ops_eval.csv"
    w_off_eval_path = eval_root / f"{stem}_w_off_eval.csv"
    w_seq_eval_path = eval_root / f"{stem}_w_seq_eval.csv"
    w_dev_eval_path = eval_root / f"{stem}_w_dev_eval.csv"

    cmd = [
        str(exe),
        "--solver",
        solver_cfg["solver_name"],
        "--stable",
        "--seed",
        str(seed),
        "--alpha",
        f"{ALPHA:.1f}",
        "--generations",
        "999999",
        "--max_evals",
        str(MAX_EVALS),
        "--log_every",
        str(LOG_EVERY_EVALS),
        "--popsize",
        str(POPSIZE),
        "--nsubpop",
        str(NSUBPOP),
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
        "--cchihh_op_stats",
        str(op_stats_path),
        "--cchihh_op_stats_every",
        str(LOG_EVERY_GENS),
        "--cchihh_weight_log_offload",
        str(w_off_path),
        "--cchihh_weight_log_seq",
        str(w_seq_path),
        "--cchihh_weight_log_dev",
        str(w_dev_path),
        "--cchihh_weight_log_every",
        str(LOG_EVERY_GENS),
    ]

    return {
        "group": solver_cfg["group"],
        "variant": solver_cfg["variant"],
        "scale": scale["name"],
        "alpha": ALPHA,
        "seed": seed,
        "cmd": cmd,
        "log_path": log_path,
        "op_stats_path": op_stats_path,
        "w_off_path": w_off_path,
        "w_seq_path": w_seq_path,
        "w_dev_path": w_dev_path,
        "eval_csv_path": eval_csv_path,
        "ops_eval_path": ops_eval_path,
        "w_off_eval_path": w_off_eval_path,
        "w_seq_eval_path": w_seq_eval_path,
        "w_dev_eval_path": w_dev_eval_path,
    }


def launch_task(task: dict, force: bool) -> dict | None:
    if task["eval_csv_path"].exists() and not force:
        return None
    ensure_parent(task["log_path"])
    f = task["log_path"].open("w", encoding="utf-8", newline="")
    f.write(f"Command: {shlex.join(task['cmd'])}\n")
    f.flush()
    proc = subprocess.Popen(task["cmd"], stdout=f, stderr=subprocess.STDOUT, shell=False)
    return {"task": task, "proc": proc, "file": f}


def process_completed_task(task: dict) -> dict[str, object]:
    final_eval, final_best = export_eval_curve_from_log(task["log_path"], task["eval_csv_path"])
    convert_gen_keyed_csv(task["op_stats_path"], task["ops_eval_path"])
    convert_gen_keyed_csv(task["w_off_path"], task["w_off_eval_path"])
    convert_gen_keyed_csv(task["w_seq_path"], task["w_seq_eval_path"])
    convert_gen_keyed_csv(task["w_dev_path"], task["w_dev_eval_path"])
    with task["eval_csv_path"].open("r", encoding="utf-8") as f:
        points = sum(1 for _ in f) - 1
    return {
        "group": task["group"],
        "variant": task["variant"],
        "scale": task["scale"],
        "alpha": f"{task['alpha']:.1f}",
        "seed": task["seed"],
        "points": points,
        "final_eval": final_eval,
        "final_best_fitness": f"{final_best:.15g}",
        "input_path": str(task["log_path"]),
        "output_path": str(task["eval_csv_path"]),
    }


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge_rows(existing_rows: list[dict[str, str]], new_rows: list[dict[str, object]], target_scales: set[str]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for row in existing_rows:
        if row["scale"] in target_scales:
            continue
        merged.append(row)
    merged.extend(new_rows)
    merged.sort(key=lambda row: (row["scale"], int(row["seed"]), row["variant"]))
    return merged


def write_index(group_root: Path, rows: list[dict[str, object]]) -> None:
    path = group_root / "index.csv"
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "variant",
                "scale",
                "alpha",
                "seed",
                "points",
                "final_eval",
                "final_best_fitness",
                "input_path",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(group_root: Path, rows: list[dict[str, object]]) -> None:
    path = group_root / "summary.csv"
    ensure_parent(path)
    buckets: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        key = (row["group"], row["variant"], row["scale"], row["alpha"])
        buckets.setdefault(key, []).append(float(row["final_best_fitness"]))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "variant",
                "scale",
                "alpha",
                "num_seeds",
                "mean_final_best_fitness",
                "std_final_best_fitness",
                "min_final_best_fitness",
                "max_final_best_fitness",
            ],
        )
        writer.writeheader()
        for (group, variant, scale, alpha), values in sorted(buckets.items()):
            mean = sum(values) / len(values)
            if len(values) > 1:
                var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                std = var ** 0.5
            else:
                std = 0.0
            writer.writerow(
                {
                    "group": group,
                    "variant": variant,
                    "scale": scale,
                    "alpha": alpha,
                    "num_seeds": len(values),
                    "mean_final_best_fitness": f"{mean:.15g}",
                    "std_final_best_fitness": f"{std:.15g}",
                    "min_final_best_fitness": f"{min(values):.15g}",
                    "max_final_best_fitness": f"{max(values):.15g}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extra-scale CCHIHH/shared-bandit eval=400k experiments.")
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = find_exe(root)
    target_scales = {cfg["name"] for cfg in SCALE_CONFIGS}

    tasks = [
        build_task(root, exe, solver_cfg, scale, seed)
        for solver_cfg in SOLVER_CONFIGS
        for scale in SCALE_CONFIGS
        for seed in SEEDS
    ]

    queue = tasks[:]
    running: list[dict] = []
    completed_by_group: dict[str, list[dict[str, object]]] = {cfg["group"]: [] for cfg in SOLVER_CONFIGS}
    done = 0
    total = len(tasks)

    while queue or running:
        while queue and len(running) < args.max_parallel:
            task = queue.pop(0)
            launched = launch_task(task, force=args.force)
            if launched is None:
                done += 1
                print(f"[skip] {task['group']} {task['scale']} seed={task['seed']}")
                continue
            running.append(launched)
            print(f"[start] {task['group']} {task['scale']} seed={task['seed']}")

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
                raise RuntimeError(f"Task failed: {task['group']} {task['scale']} seed={task['seed']} rc={code}")

            row = process_completed_task(task)
            completed_by_group[task["group"]].append(row)

            group_root = root / "results" / "eval" / task["group"]
            existing_rows = load_existing_rows(group_root / "index.csv")
            merged_rows = merge_rows(existing_rows, completed_by_group[task["group"]], target_scales)
            write_index(group_root, merged_rows)
            write_summary(group_root, merged_rows)

            done += 1
            print(
                f"[done] {task['group']} {task['scale']} seed={task['seed']} "
                f"({done}/{total}) best={row['final_best_fitness']}"
            )

        running = still_running


if __name__ == "__main__":
    main()
