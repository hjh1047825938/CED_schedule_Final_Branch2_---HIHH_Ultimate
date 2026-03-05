import argparse
import csv
import math
import re
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median


LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([0-9.+\\-eE]+)")
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
ALGORITHMS = [
    {"name": "CCHIHH_Full", "solver_args": ["CCHIHH", "--stable", "--resample_gate", "15"], "file": "CCHIHH_Full_seed{seed}.txt"},
    {"name": "CGA", "solver_args": ["CGA"], "file": "CGA_seed{seed}.txt"},
    {"name": "IMOMA", "solver_args": ["IMOMA"], "file": "IMOMA_seed{seed}.txt"},
    {"name": "DSAC_DE", "solver_args": ["DSAC-DE"], "file": "DSAC_DE_seed{seed}.txt"},
]


def mean(xs):
    return sum(xs) / len(xs)


def var_pop(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def std_sample(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def parse_alpha_list(raw: str):
    vals = []
    for x in raw.split(","):
        t = x.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise SystemExit("No alpha values provided")
    return vals


def parse_scale_list(raw: str):
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names:
        raise SystemExit("No scales provided")
    valid = {s["name"] for s in SCALES}
    for n in names:
        if n not in valid:
            raise SystemExit(f"Unknown scale: {n}. Valid: {', '.join(sorted(valid))}")
    return names


def alpha_dir(alpha: float):
    return f"alpha_{alpha:.1f}"


def find_exe(root: Path, exe_arg: str):
    if exe_arg:
        p = Path(exe_arg)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return p
        raise SystemExit(f"Executable not found: {p}")

    candidates = [
        root / "build" / "Release" / "CED_Schedule.exe",
        root / "build_ablation" / "Release" / "CED_Schedule.exe",
        root / "build2" / "Release" / "CED_Schedule.exe",
        root / "build_local" / "Release" / "CED_Schedule.exe",
        root / "build_qhh" / "Release" / "CED_Schedule.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit("Cannot find CED_Schedule.exe in known build folders.")


def source_alpha05_path(root: Path, base_cga_imoma: Path, base_dsac: Path, scale: str, algo: str, seed: int):
    if algo in ("CCHIHH_Full", "CGA", "IMOMA"):
        return root / base_cga_imoma / scale / f"{algo}_seed{seed}.txt"
    if algo == "DSAC_DE":
        return root / base_dsac / scale / f"DSAC_DE_seed{seed}.txt"
    raise RuntimeError(f"Unknown algo: {algo}")


def target_path(out_root: Path, alpha: float, scale: str, algo: str, seed: int):
    fn = next(a["file"] for a in ALGORITHMS if a["name"] == algo).format(seed=seed)
    return out_root / "runs" / alpha_dir(alpha) / scale / fn


def resolve_result_path(root: Path, out_root: Path, base_cga_imoma: Path, base_dsac: Path, alpha: float, scale: str, algo: str, seed: int):
    if abs(alpha - 0.5) < 1e-9:
        p05 = source_alpha05_path(root, base_cga_imoma, base_dsac, scale, algo, seed)
        if p05.exists():
            return p05
    return target_path(out_root, alpha, scale, algo, seed)


def parse_series(path: Path):
    b = path.read_bytes()
    text = None
    # UTF-16 logs are common in existing baseline files; detect NUL-heavy payload first.
    if b"\x00" in b[:200]:
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                text = b.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    if text is None:
        for enc in ("utf-8", "gbk", "latin1", "utf-16"):
            try:
                text = b.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    if text is None:
        text = b.decode("latin1", errors="ignore")
    if "\x00" in text:
        text = text.replace("\x00", "")

    out = {}
    for line in text.splitlines():
        s = line.strip()
        m = LINE_RE.match(s)
        if not m:
            continue
        g = int(m.group(1))
        v = float(m.group(2))
        out[g] = v
    return out


def build_all_jobs(root: Path, exe: Path, out_root: Path, base_cga_imoma: Path, base_dsac: Path,
                   scales, alphas, seeds: int, generations: int, log_every: int):
    data_dir = root / "data"
    jobs = []
    for scale in [s for s in SCALES if s["name"] in set(scales)]:
        for alpha in alphas:
            for algo in ALGORITHMS:
                for seed in range(1, seeds + 1):
                    current_path = resolve_result_path(
                        root=root,
                        out_root=out_root,
                        base_cga_imoma=base_cga_imoma,
                        base_dsac=base_dsac,
                        alpha=alpha,
                        scale=scale["name"],
                        algo=algo["name"],
                        seed=seed,
                    )
                    if current_path.exists():
                        continue

                    out_log = target_path(out_root, alpha, scale["name"], algo["name"], seed)
                    cmd = [
                        str(exe),
                        "--solver",
                        algo["solver_args"][0],
                        *algo["solver_args"][1:],
                        "--seed",
                        str(seed),
                        "--alpha",
                        f"{alpha:.1f}",
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
                    jobs.append(
                        {
                            "alpha": alpha,
                            "scale": scale["name"],
                            "algo": algo["name"],
                            "seed": seed,
                            "cmd": cmd,
                            "log_path": str(out_log),
                        }
                    )
    return jobs


def run_one_job(job, retries: int):
    log_path = Path(job["log_path"])
    if log_path.exists():
        return {"status": "skip", "job": job, "code": 0}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_code = 1
    for attempt in range(retries + 1):
        with log_path.open("w", encoding="utf-8", newline="") as f:
            f.write(f"Command: {shlex.join(job['cmd'])}\n")
            p = subprocess.run(job["cmd"], stdout=f, stderr=subprocess.STDOUT, shell=False)
            last_code = int(p.returncode)
            f.write(f"\nexit code {last_code}\n")
        if last_code == 0:
            return {"status": "ok", "job": job, "code": 0}
    return {"status": "fail", "job": job, "code": last_code}


def write_missing_csv(path: Path, jobs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "scale", "algo", "seed", "log_path", "cmd"])
        for j in jobs:
            w.writerow([j["alpha"], j["scale"], j["algo"], j["seed"], j["log_path"], shlex.join(j["cmd"])])


def build_stats_inputs(root: Path, out_root: Path, base_cga_imoma: Path, base_dsac: Path, scales, alphas, seeds: int):
    agg_root = out_root / "aggregated"
    for alpha in alphas:
        alpha_root = agg_root / alpha_dir(alpha)
        for scale in [s for s in SCALES if s["name"] in set(scales)]:
            scale_name = scale["name"]
            scale_dir = alpha_root / scale_name
            scale_dir.mkdir(parents=True, exist_ok=True)

            values_by_algo = {a["name"]: {} for a in ALGORITHMS}
            final_by_algo = {a["name"]: [] for a in ALGORITHMS}

            for algo in ALGORITHMS:
                algo_name = algo["name"]
                runs = []
                for seed in range(1, seeds + 1):
                    fp = resolve_result_path(
                        root=root,
                        out_root=out_root,
                        base_cga_imoma=base_cga_imoma,
                        base_dsac=base_dsac,
                        alpha=alpha,
                        scale=scale_name,
                        algo=algo_name,
                        seed=seed,
                    )
                    if not fp.exists():
                        raise SystemExit(f"Missing result file for stats: {fp}")
                    series = parse_series(fp)
                    if not series:
                        raise SystemExit(f"No Gen series found: {fp}")
                    runs.append(series)

                common = set(runs[0].keys())
                for s in runs[1:]:
                    common &= set(s.keys())
                if not common:
                    raise SystemExit(f"No common generations for {algo_name} at {scale_name}, alpha={alpha:.1f}")

                max_common = max(common)
                for g in sorted(common):
                    vals = [r[g] for r in runs]
                    values_by_algo[algo_name][g] = vals
                final_by_algo[algo_name] = [r[max_common] for r in runs]

            common_all = None
            for algo_name in values_by_algo:
                gs = set(values_by_algo[algo_name].keys())
                common_all = gs if common_all is None else (common_all & gs)
            if not common_all:
                raise SystemExit(f"No cross-algorithm common generations at {scale_name}, alpha={alpha:.1f}")
            gens = sorted(common_all)

            with (scale_dir / f"{scale_name}_baseline_mean_var.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = ["gen"]
                for algo in ALGORITHMS:
                    header += [f"{algo['name']} mean", f"{algo['name']} var"]
                w.writerow(header)
                for g in gens:
                    row = [g]
                    for algo in ALGORITHMS:
                        vals = values_by_algo[algo["name"]][g]
                        row += [mean(vals), var_pop(vals)]
                    w.writerow(row)

            with (scale_dir / "final_performance_summary.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["config", "mean", "std", "best", "median", "worst", "mean±std"])
                for algo in ALGORITHMS:
                    vals = final_by_algo[algo["name"]]
                    w.writerow(
                        [
                            algo["name"],
                            mean(vals),
                            std_sample(vals),
                            min(vals),
                            median(vals),
                            max(vals),
                            f"{mean(vals):.6f} +- {std_sample(vals):.6f}",
                        ]
                    )


def render_group_tables(root: Path, out_root: Path, scales, alphas):
    render_script = root / "scripts" / "build_table_and_plot_baseline_four.py"
    if not render_script.exists():
        raise SystemExit(f"Missing render script: {render_script}")

    for alpha in alphas:
        baseline_root = out_root / "aggregated" / alpha_dir(alpha)
        group_out = out_root / "tables" / alpha_dir(alpha)
        group_out.mkdir(parents=True, exist_ok=True)
        tag = f"alpha_{alpha:.1f}_baseline_cchihh_cga_imoma_dsac".replace(".", "p")
        cmd = [
            sys.executable,
            str(render_script),
            "--baseline_root",
            str(baseline_root.relative_to(root)),
            "--out_dir",
            str(group_out.relative_to(root)),
            "--n",
            "10",
            "--scales",
            ",".join(scales),
            "--solvers",
            "CCHIHH_Full,CGA,IMOMA,DSAC_DE",
            "--tag",
            tag,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, shell=False)
        if p.returncode != 0:
            raise SystemExit(f"Table render failed for alpha={alpha:.1f}:\n{p.stdout}\n{p.stderr}")


def print_scan_summary(jobs, alphas):
    print("=== Missing Scan Summary ===")
    print(f"Total missing jobs: {len(jobs)}")
    for alpha in alphas:
        group = [j for j in jobs if abs(j["alpha"] - alpha) < 1e-9]
        print(f"  alpha={alpha:.1f}: {len(group)}")


def main():
    ap = argparse.ArgumentParser(description="Alpha sensitivity runner for CGA/IMOMA/DSAC-DE with missing-only parallel execution.")
    ap.add_argument("--stage", choices=["all", "scan", "run", "stats", "tables"], default="all")
    ap.add_argument("--exe", default="")
    ap.add_argument("--out_dir", default="outputs/results/alpha_sensitivity_cga_imoma_dsac")
    ap.add_argument("--base05_cga_imoma_root", default="outputs/results/cchihh_ablation_suite/baseline")
    ap.add_argument("--base05_dsac_root", default="outputs/results/dsac_de_multiscale")
    ap.add_argument("--alphas", default="0.2,0.5,0.8")
    ap.add_argument("--scales", default="T100,T200,T500")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--generations", type=int, default=10000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=15)
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_root = root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    base_cga_imoma = Path(args.base05_cga_imoma_root)
    base_dsac = Path(args.base05_dsac_root)
    alphas = parse_alpha_list(args.alphas)
    scales = parse_scale_list(args.scales)

    exe = find_exe(root, args.exe)
    jobs = build_all_jobs(
        root=root,
        exe=exe,
        out_root=out_root,
        base_cga_imoma=base_cga_imoma,
        base_dsac=base_dsac,
        scales=scales,
        alphas=alphas,
        seeds=args.seeds,
        generations=args.generations,
        log_every=args.log_every,
    )

    write_missing_csv(out_root / "missing_tasks.csv", jobs)
    print_scan_summary(jobs, alphas)
    print(f"Missing list written: {out_root / 'missing_tasks.csv'}")

    if args.stage in ("scan",):
        return

    if args.stage in ("all", "run"):
        if not jobs:
            print("No missing jobs to run.")
        else:
            print(f"Running {len(jobs)} missing jobs with jobs={args.jobs}, retries={args.retries}")
            ok = 0
            fail = 0
            executor_cls = ProcessPoolExecutor
            try:
                _probe = ProcessPoolExecutor(max_workers=1)
                _probe.shutdown(wait=True)
            except Exception:
                executor_cls = ThreadPoolExecutor
                print("[warn] ProcessPool unavailable; fallback to ThreadPool for subprocess fan-out.")

            with executor_cls(max_workers=args.jobs) as ex:
                futs = [ex.submit(run_one_job, j, args.retries) for j in jobs]
                for fut in as_completed(futs):
                    r = fut.result()
                    j = r["job"]
                    if r["status"] == "ok":
                        ok += 1
                        print(f"[done] alpha={j['alpha']:.1f} {j['scale']} {j['algo']} seed{j['seed']}")
                    elif r["status"] == "skip":
                        print(f"[skip] alpha={j['alpha']:.1f} {j['scale']} {j['algo']} seed{j['seed']}")
                    else:
                        fail += 1
                        print(f"[fail] alpha={j['alpha']:.1f} {j['scale']} {j['algo']} seed{j['seed']} code={r['code']}")
            print(f"Run summary: ok={ok}, fail={fail}, total={len(jobs)}")
            if fail > 0:
                raise SystemExit("Some jobs failed. Re-run with --stage run to resume missing tasks.")

    if args.stage in ("all", "stats"):
        build_stats_inputs(
            root=root,
            out_root=out_root,
            base_cga_imoma=base_cga_imoma,
            base_dsac=base_dsac,
            scales=scales,
            alphas=alphas,
            seeds=args.seeds,
        )
        print(f"Stats prepared under: {out_root / 'aggregated'}")

    if args.stage in ("all", "tables"):
        render_group_tables(root=root, out_root=out_root, scales=scales, alphas=alphas)
        print(f"Tables written under: {out_root / 'tables'}")


if __name__ == "__main__":
    main()
