import argparse
import csv
import math
import re
import shlex
import subprocess
import time
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


LINE_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+\-eE]+)")
FINAL_RE = re.compile(r"^The best solution\s*=\s*([0-9.+\-eE]+)")
GATE_BLOCK_RE = re.compile(r"^CCHIHH gate_blocked_total\s*=\s*(\d+)")
GATE_FALLBACK_RE = re.compile(r"^CCHIHH gate_fallback_total\s*=\s*(\d+)")


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


EXPERIMENTS = {
    "blocks": {
        "title": "Cooperative Coevolution Ablation",
        "configs": [
            ("use_blocks_true", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15"]),
            ("use_blocks_false", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15", "--cchihh_no_blocks"]),
        ],
    },
    "bandit": {
        "title": "Contextual Bandit Ablation",
        "configs": [
            ("bandit_adaptive", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15"]),
            ("random_ops", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15", "--cchihh_random_ops"]),
            ("fixed_ops", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15", "--cchihh_fixed_ops"]),
        ],
    },
    "migration": {
        "title": "Migration Ablation",
        "configs": [
            ("migration_true", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15"]),
            ("migration_false", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15", "--cchihh_no_migration"]),
        ],
    },
    "stability": {
        "title": "Stability Gating Ablation",
        "configs": [
            ("stable_true_gate15", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15"]),
            ("stable_false", ["--solver", "CCHIHH", "--resample_gate", "0"]),
        ],
    },
    "baseline": {
        "title": "Baseline Comparison",
        "configs": [
            ("CCHIHH_Full", ["--solver", "CCHIHH", "--stable", "--resample_gate", "15"]),
            ("GDE", ["--solver", "GDE"]),
            ("CGA", ["--solver", "CGA"]),
            ("IMOMA", ["--solver", "IMOMA"]),
        ],
    },
}


COLORS = {
    "use_blocks_true": "#1f77b4",
    "use_blocks_false": "#d62728",
    "bandit_adaptive": "#1f77b4",
    "random_ops": "#ff7f0e",
    "fixed_ops": "#2ca02c",
    "migration_true": "#1f77b4",
    "migration_false": "#d62728",
    "stable_true_gate15": "#1f77b4",
    "stable_false": "#d62728",
    "CCHIHH_Full": "#d62728",
    "GDE": "#2ca02c",
    "CGA": "#1f77b4",
    "IMOMA": "#ff7f0e",
}


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


def p_to_sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def find_exe(root: Path, exe_arg: str):
    if exe_arg:
        p = Path(exe_arg)
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise SystemExit(f"Executable not found: {p}")
        return p

    candidates = [
        root / "build_ablation" / "Release" / "CED_Schedule.exe",
        root / "build" / "Release" / "CED_Schedule.exe",
        root / "build2" / "Release" / "CED_Schedule.exe",
        root / "build_local" / "Release" / "CED_Schedule.exe",
        root / "build_qhh" / "Release" / "CED_Schedule.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit("Cannot find CED_Schedule.exe in known build folders.")


def parse_log(path: Path):
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="ignore")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("gbk")
            except UnicodeDecodeError:
                text = raw.decode("utf-16", errors="ignore")
    if "\x00" in text:
        text = text.replace("\x00", "")

    series = {}
    final = None
    gate_blocked = None
    gate_fallback = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        m = LINE_RE.match(line)
        if m:
            series[int(m.group(1))] = float(m.group(2))
            continue
        m = FINAL_RE.match(line)
        if m:
            final = float(m.group(1))
            continue
        m = GATE_BLOCK_RE.match(line)
        if m:
            gate_blocked = int(m.group(1))
            continue
        m = GATE_FALLBACK_RE.match(line)
        if m:
            gate_fallback = int(m.group(1))
            continue
    if final is None and series:
        final = series[max(series.keys())]
    return {
        "series": series,
        "final": final,
        "gate_blocked": gate_blocked,
        "gate_fallback": gate_fallback,
    }


def load_opstats(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No op-stats rows in {path}")
    return rows


def ensure_fixed_ops_support(root: Path):
    main_text = (root / "src" / "main.cpp").read_text(encoding="utf-8", errors="ignore")
    cc_text = (root / "src" / "CC_HIHH.cpp").read_text(encoding="utf-8", errors="ignore")

    if "--cchihh_fixed_ops" not in main_text:
        raise RuntimeError("Missing CLI flag --cchihh_fixed_ops in src/main.cpp")
    if "SetFixedOpsPerBlock" not in main_text:
        raise RuntimeError("Missing SetFixedOpsPerBlock wiring in src/main.cpp")
    if "if (fixed_ops_per_block)" not in cc_text:
        raise RuntimeError("Missing fixed_ops_per_block logic in src/CC_HIHH.cpp")
    if "op_sel = OFF_OP_GA" not in cc_text or "op_sel = SEQ_OP_GA" not in cc_text or "op_sel = DEV_OP_DE" not in cc_text:
        raise RuntimeError("Fixed-ops mapping (Offload=GA, Seq=GA, Dev=DE) not found in src/CC_HIHH.cpp")


def build_tasks(root: Path, exe: Path, out_root: Path, experiments, generations: int, log_every: int, seeds: int):
    data_dir = root / "data"
    tasks = []
    for exp_name in experiments:
        exp = EXPERIMENTS[exp_name]
        for scale in SCALES:
            scale_dir = out_root / exp_name / scale["name"]
            scale_dir.mkdir(parents=True, exist_ok=True)
            for seed in range(1, seeds + 1):
                for cfg_name, cfg_args in exp["configs"]:
                    cmd = [
                        str(exe),
                        *cfg_args,
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
                    op_stats_path = None
                    if exp_name == "bandit" and cfg_name == "bandit_adaptive":
                        op_stats_path = scale_dir / f"{cfg_name}_opstats_seed{seed}.csv"
                        cmd.extend([
                            "--cchihh_op_stats",
                            str(op_stats_path),
                            "--cchihh_op_stats_every",
                            str(log_every),
                        ])

                    tasks.append(
                        {
                            "exp": exp_name,
                            "scale": scale["name"],
                            "cfg": cfg_name,
                            "seed": seed,
                            "cmd": cmd,
                            "log_path": scale_dir / f"{cfg_name}_seed{seed}.txt",
                            "op_stats_path": op_stats_path,
                        }
                    )
    return tasks


def run_parallel(tasks, max_parallel: int, force: bool):
    queue = tasks[:]
    running = []
    done = 0
    total = len(tasks)

    def launch(task):
        log_path = task["log_path"]
        if log_path.exists() and not force:
            return None

        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = log_path.open("w", encoding="utf-8")
        f.write(f"Command: {shlex.join(task['cmd'])}\n")
        f.flush()
        p = subprocess.Popen(task["cmd"], stdout=f, stderr=subprocess.STDOUT, shell=False)
        return {"proc": p, "f": f, "task": task}

    while queue or running:
        while queue and len(running) < max_parallel:
            t = queue.pop(0)
            item = launch(t)
            if item is None:
                done += 1
                print(f"[skip] {t['exp']} {t['scale']} {t['cfg']} seed{t['seed']}")
            else:
                running.append(item)
                print(f"[start] {t['exp']} {t['scale']} {t['cfg']} seed{t['seed']}")

        time.sleep(1)

        nxt = []
        for it in running:
            if it["proc"].poll() is None:
                nxt.append(it)
                continue

            code = it["proc"].returncode
            it["f"].write(f"\nexit code {code}\n")
            it["f"].close()
            done += 1
            t = it["task"]
            print(f"[done] {t['exp']} {t['scale']} {t['cfg']} seed{t['seed']} (exit {code})")

        running = nxt

    print(f"All done: {done}/{total}")


def compute_series_stats(scale_dir: Path, configs, seeds: int):
    values_by_cfg = {cfg: {} for cfg, _ in configs}
    finals_by_cfg = {cfg: {} for cfg, _ in configs}
    gate_by_cfg = {cfg: {} for cfg, _ in configs}

    for cfg, _ in configs:
        for seed in range(1, seeds + 1):
            fp = scale_dir / f"{cfg}_seed{seed}.txt"
            if not fp.exists():
                raise RuntimeError(f"Missing log: {fp}")
            rec = parse_log(fp)
            if not rec["series"]:
                raise RuntimeError(f"No generation series parsed: {fp}")
            if rec["final"] is None:
                raise RuntimeError(f"No final value parsed: {fp}")
            finals_by_cfg[cfg][seed] = rec["final"]
            gate_by_cfg[cfg][seed] = (rec["gate_blocked"], rec["gate_fallback"])
            for g, v in rec["series"].items():
                values_by_cfg[cfg].setdefault(g, []).append(v)

    common_gens = None
    for cfg, _ in configs:
        gs = set(values_by_cfg[cfg].keys())
        common_gens = gs if common_gens is None else (common_gens & gs)
    if not common_gens:
        raise RuntimeError(f"No common generations in {scale_dir}")

    gens = sorted(common_gens)
    stats = {}
    for cfg, _ in configs:
        means = []
        vars_ = []
        for g in gens:
            arr = values_by_cfg[cfg][g]
            means.append(mean(arr))
            vars_.append(var_pop(arr))
        stats[cfg] = {"mean": means, "var": vars_}

    return gens, stats, finals_by_cfg, gate_by_cfg


def write_mean_var_csv(path: Path, gens, stats, configs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["gen"]
        for cfg, _ in configs:
            header += [f"{cfg} mean", f"{cfg} var"]
        w.writerow(header)
        for i, g in enumerate(gens):
            row = [g]
            for cfg, _ in configs:
                row.append(stats[cfg]["mean"][i])
                row.append(stats[cfg]["var"][i])
            w.writerow(row)


def plot_mean_ci(path_prefix: Path, title: str, gens, stats, configs, n: int):
    z = 1.96
    plt.figure(figsize=(10, 6))
    for cfg, _ in configs:
        means = stats[cfg]["mean"]
        vars_ = stats[cfg]["var"]
        ci = [z * math.sqrt(max(v, 0.0) / max(1, n)) for v in vars_]
        lo = [m - c for m, c in zip(means, ci)]
        hi = [m + c for m, c in zip(means, ci)]
        color = COLORS.get(cfg)
        plt.plot(gens, means, linewidth=2, label=cfg, color=color)
        plt.fill_between(gens, lo, hi, alpha=0.18, color=color)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path_prefix.with_suffix(".png"), dpi=220)
    plt.savefig(path_prefix.with_suffix(".pdf"))
    plt.close()


def write_final_summary(path: Path, finals_by_cfg, configs):
    rows = []
    for cfg, _ in configs:
        vals = [finals_by_cfg[cfg][s] for s in sorted(finals_by_cfg[cfg].keys())]
        rows.append(
            {
                "config": cfg,
                "mean": mean(vals),
                "std": std_sample(vals),
                "best": min(vals),
                "median": median(vals),
                "worst": max(vals),
                "text": f"{mean(vals):.6f} ± {std_sample(vals):.6f}",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "mean", "std", "best", "median", "worst", "mean±std"])
        for r in rows:
            w.writerow([r["config"], r["mean"], r["std"], r["best"], r["median"], r["worst"], r["text"]])


def safe_wilcoxon(a, b):
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", correction=False)
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def write_wilcoxon(path: Path, finals_by_cfg, configs):
    cfgs = [c for c, _ in configs]
    rows = []
    for i in range(len(cfgs)):
        for j in range(i + 1, len(cfgs)):
            a = cfgs[i]
            b = cfgs[j]
            seeds = sorted(set(finals_by_cfg[a].keys()) & set(finals_by_cfg[b].keys()))
            xa = [finals_by_cfg[a][s] for s in seeds]
            xb = [finals_by_cfg[b][s] for s in seeds]
            stat, p = safe_wilcoxon(xa, xb)
            rows.append([a, b, len(seeds), stat, p, p_to_sig(p)])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["A", "B", "n", "wilcoxon_stat", "p_value", "sig"])
        w.writerows(rows)


def write_baseline_pair_tables(scale_dir: Path, finals_by_cfg):
    baseline = "CCHIHH_Full"
    targets = ["GDE", "CGA", "IMOMA"]
    for t in targets:
        seeds = sorted(set(finals_by_cfg[baseline].keys()) & set(finals_by_cfg[t].keys()))
        xa = [finals_by_cfg[baseline][s] for s in seeds]
        xb = [finals_by_cfg[t][s] for s in seeds]
        stat, p = safe_wilcoxon(xa, xb)

        out = scale_dir / f"wilcoxon_{baseline}_vs_{t}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["A", "B", "n", "A_mean", "B_mean", "wilcoxon_stat", "p_value", "sig"])
            w.writerow([baseline, t, len(seeds), mean(xa), mean(xb), stat, p, p_to_sig(p)])


def aggregate_bandit_opfreq(scale_dir: Path, seeds: int):
    files = [scale_dir / f"bandit_adaptive_opstats_seed{s}.csv" for s in range(1, seeds + 1)]
    files = [p for p in files if p.exists()]
    if not files:
        return

    groups = {
        "offload": ["offload_GA", "offload_DE", "offload_BITFLIP", "offload_RESAMPLE"],
        "seq": ["seq_GA", "seq_SWAP", "seq_VNS", "seq_RESAMPLE"],
        "dev": ["dev_DE", "dev_GDE", "dev_LEVY", "dev_RESAMPLE"],
        "overall": [
            "overall_GA",
            "overall_DE",
            "overall_GDE",
            "overall_BITFLIP",
            "overall_SWAP",
            "overall_VNS",
            "overall_LEVY",
            "overall_RESAMPLE",
        ],
    }

    series = {k: {} for k in groups}
    gens_ref = None
    for fp in files:
        rows = load_opstats(fp)
        gens = [int(r["gen"]) for r in rows]
        if gens_ref is None:
            gens_ref = gens
        elif gens != gens_ref:
            raise RuntimeError(f"Generation mismatch in op-stats file: {fp}")

        for gkey, cols in groups.items():
            for col in cols:
                vals = [float(r[col]) for r in rows]
                series[gkey].setdefault(col, []).append(vals)

    out_dir = scale_dir / "opfreq"
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = {
        "GA": "#1f77b4",
        "DE": "#ff7f0e",
        "GDE": "#2ca02c",
        "BITFLIP": "#d62728",
        "SWAP": "#9467bd",
        "VNS": "#8c564b",
        "LEVY": "#e377c2",
        "RESAMPLE": "#7f7f7f",
    }

    z = 1.96
    for gkey, cols in groups.items():
        plt.figure(figsize=(10, 6))
        for col in cols:
            arrs = series[gkey][col]
            means = []
            vars_ = []
            for idx in range(len(gens_ref)):
                vals = [a[idx] for a in arrs]
                means.append(mean(vals))
                vars_.append(var_pop(vals))
            ci = [z * math.sqrt(max(v, 0.0) / max(1, len(arrs))) for v in vars_]
            lo = [m - c for m, c in zip(means, ci)]
            hi = [m + c for m, c in zip(means, ci)]
            label = col.split("_", 1)[1]
            color = palette.get(label)
            plt.plot(gens_ref, means, linewidth=2, label=label, color=color)
            plt.fill_between(gens_ref, lo, hi, alpha=0.18, color=color)

        plt.title(f"{scale_dir.name} Bandit Operator Frequency ({gkey}, mean ± 95% CI)")
        plt.xlabel("Generation")
        plt.ylabel("Selection frequency")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9, ncol=2)
        plt.tight_layout()

        prefix = out_dir / f"bandit_opfreq_{gkey}_mean_ci"
        plt.savefig(prefix.with_suffix(".png"), dpi=220)
        plt.savefig(prefix.with_suffix(".pdf"))
        plt.close()


def write_gate_summary(scale_dir: Path, gate_by_cfg, cfg_name: str):
    rows = []
    for seed, pair in sorted(gate_by_cfg[cfg_name].items()):
        blocked, fallback = pair
        if blocked is None or fallback is None:
            continue
        rows.append([seed, blocked, fallback])

    if not rows:
        return

    blocked_vals = [r[1] for r in rows]
    fallback_vals = [r[2] for r in rows]

    out = scale_dir / "gate_trigger_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "gate_blocked_total", "gate_fallback_total"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["metric", "mean", "std", "min", "max"])
        w.writerow(["gate_blocked_total", mean(blocked_vals), std_sample(blocked_vals), min(blocked_vals), max(blocked_vals)])
        w.writerow(["gate_fallback_total", mean(fallback_vals), std_sample(fallback_vals), min(fallback_vals), max(fallback_vals)])


def analyze_experiment(out_root: Path, exp_name: str, seeds: int):
    exp = EXPERIMENTS[exp_name]
    configs = exp["configs"]

    for scale in SCALES:
        scale_dir = out_root / exp_name / scale["name"]
        gens, stats, finals_by_cfg, gate_by_cfg = compute_series_stats(scale_dir, configs, seeds)

        write_mean_var_csv(scale_dir / f"{scale['name']}_{exp_name}_mean_var.csv", gens, stats, configs)
        plot_mean_ci(
            scale_dir / f"{scale['name']}_{exp_name}_mean_ci",
            f"{scale['name']} {exp['title']} (mean ± 95% CI)",
            gens,
            stats,
            configs,
            seeds,
        )
        write_final_summary(scale_dir / "final_performance_summary.csv", finals_by_cfg, configs)
        write_wilcoxon(scale_dir / "wilcoxon_pairwise.csv", finals_by_cfg, configs)

        if exp_name == "baseline":
            write_baseline_pair_tables(scale_dir, finals_by_cfg)

        if exp_name == "bandit":
            aggregate_bandit_opfreq(scale_dir, seeds)

        if exp_name == "stability":
            write_gate_summary(scale_dir, gate_by_cfg, "stable_true_gate15")


def parse_experiment_list(raw: str):
    if raw.strip().lower() == "all":
        return list(EXPERIMENTS.keys())
    out = []
    for x in raw.split(","):
        k = x.strip()
        if not k:
            continue
        if k not in EXPERIMENTS:
            raise SystemExit(f"Unknown experiment: {k}. Choices: {', '.join(EXPERIMENTS.keys())}")
        out.append(k)
    if not out:
        raise SystemExit("No experiment selected")
    return out


def main():
    parser = argparse.ArgumentParser(description="Run and analyze CCHIHH ablation suite.")
    parser.add_argument("--exe", default="", help="Path to CED_Schedule.exe (optional)")
    parser.add_argument("--generations", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_parallel", type=int, default=15)
    parser.add_argument("--experiments", default="all", help="Comma-separated: blocks,bandit,migration,stability,baseline or all")
    parser.add_argument("--stage", default="all", choices=["all", "run", "analyze"])
    parser.add_argument("--force", action="store_true", help="Rerun tasks even if log files exist")
    parser.add_argument("--out_dir", default="results/cchihh_ablation_suite")
    parser.add_argument("--skip_preflight", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    selected = parse_experiment_list(args.experiments)

    if not args.skip_preflight:
        ensure_fixed_ops_support(root)
        print("[preflight] fixed_ops flag and mapping check passed")

    exe = find_exe(root, args.exe)
    print(f"Using executable: {exe}")
    print(f"Output root: {out_root}")
    print(f"Experiments: {', '.join(selected)}")

    if args.stage in ("all", "run"):
        tasks = build_tasks(root, exe, out_root, selected, args.generations, args.log_every, args.seeds)
        run_parallel(tasks, args.max_parallel, args.force)

    if args.stage in ("all", "analyze"):
        for exp_name in selected:
            print(f"[analyze] {exp_name}")
            analyze_experiment(out_root, exp_name, args.seeds)

    print("Done.")


if __name__ == "__main__":
    main()
