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


def find_exe(root: Path) -> Path:
    candidates = [
        root / "build_ablation_tmp_20260206" / "Release" / "CED_Schedule.exe",
        root / "build_ablation" / "Release" / "CED_Schedule.exe",
        root / "build_local" / "Release" / "CED_Schedule.exe",
        root / "build" / "Release" / "CED_Schedule.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit("Cannot find CED_Schedule.exe in known build folders.")


def parse_log(path: Path):
    out = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def mean(xs):
    return sum(xs) / len(xs)


def var_pop(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def build_tasks(root: Path, exe: Path, generations: int, log_every: int, resample_gate: int, n: int):
    data_dir = root / "data"
    out_base = root / "results" / "ablation_migration_on_off_multiscale"
    out_base.mkdir(parents=True, exist_ok=True)
    tasks = []

    for scale in SCALES:
        scale_dir = out_base / scale["name"]
        scale_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(1, n + 1):
            common = [
                str(exe),
                "--solver",
                "CCHIHH",
                "--stable",
                "--resample_gate",
                str(resample_gate),
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
                    "name": f"{scale['name']}_migration_on",
                    "seed": seed,
                    "cmd": common[:],
                    "log_path": scale_dir / f"CCHIHH_stable_migration_on_seed{seed}.txt",
                }
            )
            tasks.append(
                {
                    "name": f"{scale['name']}_migration_off",
                    "seed": seed,
                    "cmd": common[:] + ["--cchihh_no_migration"],
                    "log_path": scale_dir / f"CCHIHH_stable_migration_off_seed{seed}.txt",
                }
            )
    return tasks, out_base


def run_parallel(tasks, max_parallel: int, force: bool):
    running = []
    queue = tasks[:]
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
            t = queue.pop(0)
            item = launch(t)
            if item is None:
                done += 1
                print(f"[skip] {t['name']} seed {t['seed']}")
            else:
                running.append(item)
                print(f"[start] {t['name']} seed {t['seed']}")

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
            print(f"[done] {t['name']} seed {t['seed']} (exit {code})")
        running = nxt

    print(f"All done: {done}/{total}")


def aggregate_and_plot(out_base: Path, n: int):
    z = 1.96
    for scale in SCALES:
        scale_dir = out_base / scale["name"]
        on_vals = {}
        off_vals = {}

        for seed in range(1, n + 1):
            on = parse_log(scale_dir / f"CCHIHH_stable_migration_on_seed{seed}.txt")
            off = parse_log(scale_dir / f"CCHIHH_stable_migration_off_seed{seed}.txt")
            if not on or not off:
                raise SystemExit(f"Missing Gen logs in {scale['name']} seed {seed}")
            for g, v in on.items():
                on_vals.setdefault(g, []).append(v)
            for g, v in off.items():
                off_vals.setdefault(g, []).append(v)

        gens = sorted(set(on_vals.keys()) & set(off_vals.keys()))
        if not gens:
            raise SystemExit(f"No common generations for {scale['name']}")

        mean_on, var_on, mean_off, var_off = [], [], [], []
        for g in gens:
            on_arr = on_vals[g]
            off_arr = off_vals[g]
            mean_on.append(mean(on_arr))
            var_on.append(var_pop(on_arr))
            mean_off.append(mean(off_arr))
            var_off.append(var_pop(off_arr))

        stats_csv = scale_dir / f"{scale['name']}_migration_on_off_mean_var.csv"
        with stats_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gen", "migration_on mean", "migration_on var", "migration_off mean", "migration_off var"])
            for i, g in enumerate(gens):
                w.writerow([g, mean_on[i], var_on[i], mean_off[i], var_off[i]])
        print(f"Wrote: {stats_csv}")

        ci_on = [z * math.sqrt(max(v, 0.0) / max(1, n)) for v in var_on]
        ci_off = [z * math.sqrt(max(v, 0.0) / max(1, n)) for v in var_off]

        plt.figure(figsize=(9, 5.5))
        plt.plot(gens, mean_on, label="CCHIHH stable (migration on)", linewidth=2, color="#1f77b4")
        plt.fill_between(
            gens,
            [m - c for m, c in zip(mean_on, ci_on)],
            [m + c for m, c in zip(mean_on, ci_on)],
            alpha=0.2,
            color="#1f77b4",
        )
        plt.plot(gens, mean_off, label="CCHIHH stable (migration off)", linewidth=2, color="#d62728")
        plt.fill_between(
            gens,
            [m - c for m, c in zip(mean_off, ci_off)],
            [m + c for m, c in zip(mean_off, ci_off)],
            alpha=0.2,
            color="#d62728",
        )
        plt.title(f"{scale['name']}: migration on vs off (mean +/- 95% CI)")
        plt.xlabel("Generation")
        plt.ylabel("Best fitness")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()

        out_prefix = scale_dir / f"{scale['name']}_migration_on_off_mean_ci"
        plt.savefig(out_prefix.with_suffix(".png"), dpi=200)
        plt.savefig(out_prefix.with_suffix(".pdf"))
        print(f"Saved: {out_prefix.with_suffix('.png')}")
        print(f"Saved: {out_prefix.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--resample_gate", type=int, default=15)
    parser.add_argument("--max_parallel", type=int, default=15)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n", type=int, default=15, help="seed count for CI")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = find_exe(root)
    print(f"Using exe: {exe}")
    tasks, out_base = build_tasks(root, exe, args.generations, args.log_every, args.resample_gate, args.n)
    run_parallel(tasks, args.max_parallel, args.force)
    aggregate_and_plot(out_base, args.n)


if __name__ == "__main__":
    main()
