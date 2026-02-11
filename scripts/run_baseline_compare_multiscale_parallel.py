import argparse
import csv
import math
import re
import shlex
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt


LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+-eE]+)")

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

SOLVERS = [
    (
        "CCHIHH_stable_migration_gate",
        ["--solver", "CCHIHH", "--stable", "--migration", "--resample_gate", "15"],
    ),
    ("GA", ["--solver", "GA"]),
    ("GDE", ["--solver", "GDE"]),
    ("QHH", ["--solver", "QHH"]),
    ("GA-SLHH", ["--solver", "GA-SLHH"]),
]

COLORS = {
    "CCHIHH_stable_migration_gate": "#d62728",
    "GA": "#1f77b4",
    "GDE": "#2ca02c",
    "QHH": "#ff7f0e",
    "GA-SLHH": "#9467bd",
}


def find_exe(root: Path) -> Path:
    candidates = [
        root / "build_qhh" / "Release" / "CED_Schedule.exe",
        root / "build" / "Release" / "CED_Schedule.exe",
        root / "build2" / "Release" / "CED_Schedule.exe",
        root / "build_local" / "Release" / "CED_Schedule.exe",
        root / "build_ablation" / "Release" / "CED_Schedule.exe",
        root / "build_ablation_tmp_20260206" / "Release" / "CED_Schedule.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit("Cannot find CED_Schedule.exe in known build folders.")


def parse_log(path: Path):
    gens = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                gens[int(m.group(1))] = float(m.group(2))
    return gens


def mean(xs):
    return sum(xs) / len(xs)


def var_pop(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def build_tasks(root: Path, exe: Path, generations: int, log_every: int, n: int, out_base: Path):
    data_dir = root / "data"
    out_base.mkdir(parents=True, exist_ok=True)
    tasks = []

    for scale in SCALES:
        scale_dir = out_base / scale["name"]
        scale_dir.mkdir(parents=True, exist_ok=True)

        for seed in range(1, n + 1):
            for solver_name, solver_args in SOLVERS:
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
                        "scale": scale["name"],
                        "name": solver_name,
                        "seed": seed,
                        "cmd": cmd,
                        "log_path": scale_dir / f"{solver_name}_seed{seed}.txt",
                    }
                )

    return tasks


def run_parallel(tasks, max_parallel: int, force: bool):
    queue = tasks[:]
    running = []
    done = 0
    total = len(tasks)

    def launch(task):
        lp = task["log_path"]
        lp.parent.mkdir(parents=True, exist_ok=True)
        if lp.exists() and not force:
            return None

        f = lp.open("w", encoding="utf-8")
        f.write(f"Command: {shlex.join(task['cmd'])}\n")
        f.flush()
        p = subprocess.Popen(task["cmd"], stdout=f, stderr=subprocess.STDOUT, shell=False)
        return {"proc": p, "file": f, "task": task}

    while queue or running:
        while queue and len(running) < max_parallel:
            task = queue.pop(0)
            item = launch(task)
            if item is None:
                done += 1
                print(f"[skip] {task['scale']} {task['name']} seed {task['seed']}")
            else:
                running.append(item)
                print(f"[start] {task['scale']} {task['name']} seed {task['seed']}")

        time.sleep(1)
        nxt = []
        for it in running:
            if it["proc"].poll() is None:
                nxt.append(it)
                continue

            code = it["proc"].returncode
            it["file"].write(f"\nexit code {code}\n")
            it["file"].close()
            done += 1
            t = it["task"]
            print(f"[done] {t['scale']} {t['name']} seed {t['seed']} (exit {code})")
        running = nxt

    print(f"All done: {done}/{total}")


def aggregate_and_plot(out_base: Path, n: int):
    z = 1.96

    for scale in SCALES:
        scale_dir = out_base / scale["name"]
        values = {name: {} for name, _ in SOLVERS}

        for solver_name, _ in SOLVERS:
            for seed in range(1, n + 1):
                fp = scale_dir / f"{solver_name}_seed{seed}.txt"
                data = parse_log(fp)
                if not data:
                    raise SystemExit(f"Missing Gen logs: {fp}")
                for g, v in data.items():
                    values[solver_name].setdefault(g, []).append(v)

        common_gens = None
        for solver_name, _ in SOLVERS:
            gs = set(values[solver_name].keys())
            common_gens = gs if common_gens is None else (common_gens & gs)
        if not common_gens:
            raise SystemExit(f"No common generations in {scale['name']}")

        gens = sorted(common_gens)

        stats = {}
        for solver_name, _ in SOLVERS:
            means = []
            vars_ = []
            for g in gens:
                arr = values[solver_name][g]
                means.append(mean(arr))
                vars_.append(var_pop(arr))
            stats[solver_name] = {"mean": means, "var": vars_}

        csv_path = scale_dir / f"{scale['name']}_baseline_compare_mean_var.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["gen"]
            for solver_name, _ in SOLVERS:
                header.append(f"{solver_name} mean")
                header.append(f"{solver_name} var")
            w.writerow(header)
            for i, g in enumerate(gens):
                row = [g]
                for solver_name, _ in SOLVERS:
                    row.append(stats[solver_name]["mean"][i])
                    row.append(stats[solver_name]["var"][i])
                w.writerow(row)
        print(f"Wrote: {csv_path}")

        plt.figure(figsize=(10, 6))
        for solver_name, _ in SOLVERS:
            means = stats[solver_name]["mean"]
            vars_ = stats[solver_name]["var"]
            ci = [z * math.sqrt(max(v, 0.0) / max(1, n)) for v in vars_]
            lower = [m - c for m, c in zip(means, ci)]
            upper = [m + c for m, c in zip(means, ci)]
            color = COLORS.get(solver_name)

            plt.plot(gens, means, label=solver_name, linewidth=2, color=color)
            plt.fill_between(gens, lower, upper, alpha=0.18, color=color)

        plt.title(f"{scale['name']}: Mean +/- 95% CI (gen={gens[-1]}, seed=1-{n})")
        plt.xlabel("Generation")
        plt.ylabel("Best fitness")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()

        out_prefix = scale_dir / f"{scale['name']}_baseline_compare_mean_ci"
        plt.savefig(out_prefix.with_suffix(".png"), dpi=220)
        plt.savefig(out_prefix.with_suffix(".pdf"))
        print(f"Saved: {out_prefix.with_suffix('.png')}")
        print(f"Saved: {out_prefix.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_parallel", type=int, default=15)
    parser.add_argument("--n", type=int, default=10, help="seed count for CI")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--out_dir",
        default="results/baseline_cchihh_stable_migration_gate_vs_ga_gde_qhh_ga_slhh_multiscale",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = find_exe(root)
    out_base = Path(args.out_dir)
    if not out_base.is_absolute():
        out_base = root / out_base

    print(f"Using exe: {exe}")
    print(f"Output dir: {out_base}")

    tasks = build_tasks(
        root=root,
        exe=exe,
        generations=args.generations,
        log_every=args.log_every,
        n=args.n,
        out_base=out_base,
    )
    run_parallel(tasks, max_parallel=args.max_parallel, force=args.force)
    aggregate_and_plot(out_base=out_base, n=args.n)


if __name__ == "__main__":
    main()
