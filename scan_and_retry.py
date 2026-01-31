import argparse
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime

# =========================
# Configurable markers
# =========================
SUCCESS_MARKERS = [
    "Final Results",
    "Training complete",
    "Finished",
    "exit code 0",
    "The best solution",
    "Time =",
]

ERROR_MARKERS = [
    "killed",
    "oom",
    "out of memory",
    "segfault",
    "segmentation fault",
    "traceback",
    "nccl",
    "timeout",
    "exception",
    "error:",
]

BACKOFF_SECONDS = [10, 30, 60, 120, 300]

DEFAULT_GENERATIONS = 10000
DEFAULT_LOG_EVERY = 50


def print_markers():
    print("Success markers:", SUCCESS_MARKERS)
    print("Error markers:", ERROR_MARKERS)


def read_file_lines(path, max_bytes=2_000_000):
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        text = data.decode("utf-8", errors="ignore")
        return text.splitlines()
    except Exception:
        return []


def tail_lines(path, n=200):
    lines = read_file_lines(path)
    return lines[-n:]


def find_command(lines):
    cmd_patterns = [
        re.compile(r"^(?:Command|Running|Args?)\s*[:=]\s*(.+)$", re.I),
    ]
    for line in lines:
        for pat in cmd_patterns:
            m = pat.search(line.strip())
            if m:
                return m.group(1).strip()
        if "CED_Schedule.exe" in line:
            return line.strip()
    return None


def sanitize_command(cmd):
    if not cmd:
        return None
    # Strip powershell Out-File or redirection
    for token in ["| Out-File", "|Out-File", ">"]:
        if token in cmd:
            cmd = cmd.split(token)[0].strip()
    return cmd.strip()


def extract_arg_from_cmd(cmd, key):
    if not cmd:
        return None
    m = re.search(rf"--{re.escape(key)}\s+([^\s]+)", cmd)
    if m:
        return m.group(1)
    return None


def parse_solver_seed(lines, filename):
    text = "\n".join(lines)
    solver = None
    seed = None
    stable = None

    solver_patterns = [
        re.compile(r"--solver\s+([A-Za-z0-9_]+)", re.I),
        re.compile(r"\bsolver\s*[:=]\s*([A-Za-z0-9_]+)", re.I),
    ]
    seed_patterns = [
        re.compile(r"--seed\s+(\d+)", re.I),
        re.compile(r"\bseed\s*[:=]\s*(\d+)", re.I),
        re.compile(r"\bSeed\s*[:=]\s*(\d+)", re.I),
    ]

    for pat in solver_patterns:
        m = pat.search(text)
        if m:
            solver = m.group(1)
            break
    for pat in seed_patterns:
        m = pat.search(text)
        if m:
            seed = int(m.group(1))
            break

    fname = os.path.basename(filename)
    if solver is None:
        for s in ["CCHIHH", "GA", "DE", "GDE"]:
            if re.search(rf"\b{s}\b", fname, re.I) or fname.upper().startswith(s):
                solver = s
                break
    if seed is None:
        m = re.search(r"seed[-_]?(\d+)", fname, re.I)
        if m:
            seed = int(m.group(1))

    if solver and solver.upper() == "CCHIHH":
        if re.search(r"stable", fname, re.I) or re.search(r"--stable", text, re.I):
            stable = True
        elif re.search(r"base", fname, re.I):
            stable = False

    return solver, seed, stable


def parse_expected_gen(lines, cmd):
    if cmd:
        g = extract_arg_from_cmd(cmd, "generations")
        if g and g.isdigit():
            return int(g)
    for line in lines:
        m = re.search(r"Generations\s*[:=]\s*(\d+)", line, re.I)
        if m:
            return int(m.group(1))
        m = re.search(r"Generation\s*=\s*(\d+)", line, re.I)
        if m:
            return int(m.group(1))
    return DEFAULT_GENERATIONS


def has_error(lines):
    text = "\n".join(lines).lower()
    for kw in ERROR_MARKERS:
        if kw.lower() in text:
            return True, kw
    return False, None


def last_gen(lines):
    last = None
    for line in lines:
        m = re.search(r"\bGen\s+(\d+)\s*:\s*best_fit", line)
        if m:
            last = int(m.group(1))
    return last


def is_success(lines, expected_gen):
    text = "\n".join(lines)
    for kw in SUCCESS_MARKERS:
        if kw in text:
            return True, "success_marker"
    err, err_kw = has_error(lines)
    if err:
        return False, f"error:{err_kw}"
    lg = last_gen(lines)
    if expected_gen and lg is not None and lg >= expected_gen:
        return True, "last_gen_reached"
    return False, "no_success_marker"


def build_default_command(solver, seed, stable):
    cmd = [
        r".\build\Release\CED_Schedule.exe",
        "--solver", solver,
        "--seed", str(seed),
        "--generations", str(DEFAULT_GENERATIONS),
        "--log_every", str(DEFAULT_LOG_EVERY),
        "--data_dir", r".\data",
        "--data_file", "data_matrix_100.txt",
    ]
    if stable:
        cmd.insert(2, "--stable")
    return cmd


def run_command(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        if isinstance(cmd, str):
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)
        else:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, shell=False)
        return_code = proc.wait()
    return return_code


def scan_logs(log_dir):
    files = []
    for root, _, filenames in os.walk(log_dir):
        for name in filenames:
            files.append(os.path.join(root, name))
    return files


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--retry_dir", required=True)
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--overwrite_failed", action="store_true", help="Overwrite failed logs in place")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N failed runs this invocation (0 = no limit)")
    args = parser.parse_args()

    print_markers()

    log_files = scan_logs(args.log_dir)
    print(f"Total log files scanned: {len(log_files)}")

    unresolved = []
    retry_plan = []
    success_count = 0

    for path in log_files:
        lines = read_file_lines(path)
        cmd = sanitize_command(find_command(lines))
        solver, seed, stable = parse_solver_seed(lines, path)
        if solver is None or seed is None:
            unresolved.append({
                "log_path": path,
                "solver": solver,
                "seed": seed,
            })
            continue

        expected_gen = parse_expected_gen(lines, cmd)
        ok, reason = is_success(lines, expected_gen)
        if ok:
            success_count += 1
            continue

        retry_plan.append({
            "log_path": path,
            "solver": solver,
            "seed": seed,
            "stable": stable,
            "cmd": cmd,
            "expected_gen": expected_gen,
            "reason": reason,
        })

    initial_failure_count = len(retry_plan)

    with open("retry_plan.json", "w", encoding="utf-8") as f:
        json.dump(retry_plan, f, indent=2, ensure_ascii=False)
    with open("unresolved.json", "w", encoding="utf-8") as f:
        json.dump(unresolved, f, indent=2, ensure_ascii=False)

    print(f"Initial failures: {initial_failure_count}")
    print(f"Unresolved: {len(unresolved)}")

    final_failures = []
    total_retries = 0

    for round_idx in range(1, args.max_retry + 1):
        if not retry_plan:
            break
        print(f"Retry round {round_idx}, remaining: {len(retry_plan)}")
        next_retry_plan = []

        processed = 0
        for item in retry_plan:
            if args.limit and processed >= args.limit:
                next_retry_plan.append(item)
                continue
            solver = item["solver"]
            seed = item["seed"]
            stable = item["stable"]
            expected_gen = item["expected_gen"]
            cmd = item["cmd"]

            run_id = f"{solver}_seed{seed}"
            if solver.upper() == "CCHIHH" and stable:
                run_id = f"{solver}_stable_seed{seed}"

            # Check existing retries for success
            existing_success = False
            for i in range(1, round_idx + 1):
                retry_log = os.path.join(args.retry_dir, f"{run_id}_retry{i}.txt")
                if os.path.exists(retry_log):
                    rlines = read_file_lines(retry_log)
                    ok, _ = is_success(rlines, expected_gen)
                    if ok:
                        existing_success = True
                        break
            if existing_success:
                continue

            retry_log = os.path.join(args.retry_dir, f"{run_id}_retry{round_idx}.txt")
            if args.overwrite_failed:
                retry_log = item["log_path"]
            if cmd:
                cmd_to_run = cmd
            else:
                cmd_to_run = build_default_command(solver, seed, stable)

            print(f"  Retrying {run_id} -> {retry_log}")
            total_retries += 1
            run_command(cmd_to_run, retry_log)
            processed += 1

            rlines = read_file_lines(retry_log)
            ok, reason = is_success(rlines, expected_gen)
            if not ok:
                next_retry_plan.append({**item, "reason": reason, "last_retry_log": retry_log})

        retry_plan = next_retry_plan
        if retry_plan and round_idx <= len(BACKOFF_SECONDS):
            time.sleep(BACKOFF_SECONDS[round_idx - 1])

    for item in retry_plan:
        log_path = item.get("last_retry_log") or item["log_path"]
        err_tail = tail_lines(log_path, 20)
        final_failures.append({
            "log_path": item["log_path"],
            "solver": item["solver"],
            "seed": item["seed"],
            "stable": item["stable"],
            "last_retry_log": log_path,
            "error_tail": err_tail,
            "reason": item["reason"],
        })

    final_report = {
        "timestamp": datetime.now().isoformat(),
        "total_scanned": len(log_files),
        "initial_success": success_count,
        "initial_failures": initial_failure_count,
        "total_retries": total_retries,
        "final_success": len(log_files) - len(unresolved) - len(final_failures),
        "final_failures": len(final_failures),
        "unresolved": len(unresolved),
        "failures": final_failures,
    }

    with open("final_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("Final report:")
    print(f"  Final success: {final_report['final_success']}")
    print(f"  Final failures: {final_report['final_failures']}")
    print(f"  Unresolved: {final_report['unresolved']}")

    if final_failures:
        print("Still failed runs:")
        for f in final_failures:
            print(f"  solver={f['solver']} seed={f['seed']} log={f['last_retry_log']}")
            for line in f["error_tail"]:
                print("    " + line)


if __name__ == "__main__":
    run()
