#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List


BEST_FIT_RE = re.compile(r"(?:Gen\s+\d+:\s+best_fit\s*=\s*|Eval\s+\d+:\s+best_fit\s*=\s*)([-+0-9.eE]+)")
FINAL_RE = re.compile(r"The best solution\s*=\s*([-+0-9.eE]+)")


@dataclass
class Case:
    name: str
    args: List[str]


def default_cases() -> List[Case]:
    return [
        Case("GA", ["--solver", "GA", "--seed", "1", "--generations", "300", "--log_every", "50"]),
        Case("DE", ["--solver", "DE", "--seed", "1", "--generations", "300", "--log_every", "50"]),
        Case("GDE", ["--solver", "GDE", "--seed", "1", "--generations", "300", "--log_every", "50"]),
        Case(
            "CCHIHH",
            [
                "--solver",
                "CCHIHH",
                "--stable",
                "--resample_gate",
                "15",
                "--migration",
                "--seed",
                "1",
                "--generations",
                "1000",
                "--log_every",
                "50",
                "--cnum",
                "100",
                "--enum",
                "100",
                "--dnum",
                "300",
                "--tnum",
                "100",
                "--mopt",
                "5",
            ],
        ),
        Case("GA-SLHH", ["--solver", "GA-SLHH", "--seed", "1", "--generations", "300", "--log_every", "50"]),
        Case("QHH", ["--solver", "QHH", "--seed", "1", "--generations", "300", "--log_every", "50", "--qphh_threads", "1"]),
        Case("IMOMA", ["--solver", "IMOMA", "--seed", "1", "--generations", "300", "--log_every", "50"]),
        Case("CGA", ["--solver", "CGA", "--seed", "1", "--generations", "300", "--log_every", "50"]),
    ]


def run_case(exe: pathlib.Path, data_dir: pathlib.Path, data_file: str, case: Case, env: Dict[str, str]) -> Dict[str, object]:
    cmd = [
        str(exe),
        "--data_dir",
        str(data_dir),
        "--data_file",
        data_file,
    ] + case.args
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    t1 = time.perf_counter()
    out = proc.stdout + "\n" + proc.stderr
    series = [float(m.group(1)) for m in BEST_FIT_RE.finditer(out)]
    fmatch = FINAL_RE.search(out)
    final = float(fmatch.group(1)) if fmatch else (series[-1] if series else None)
    return {
        "name": case.name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "seconds": t1 - t0,
        "series": series,
        "final": final,
        "stdout_path": None,
        "raw": out,
    }


def write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_raw(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: pathlib.Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(args: argparse.Namespace) -> int:
    exe = pathlib.Path(args.exe).resolve()
    data_dir = pathlib.Path(args.data_dir).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp_threads)

    all_results = []
    for case in default_cases():
        one = run_case(exe, data_dir, args.data_file, case, env)
        raw_path = out_dir / "logs" / f"{case.name}.log"
        save_raw(raw_path, str(one["raw"]))
        one["stdout_path"] = str(raw_path)
        one.pop("raw", None)
        all_results.append(one)
        print(f"[collect] {case.name}: rc={one['returncode']} final={one['final']} sec={one['seconds']:.3f}")

    payload = {
        "exe": str(exe),
        "data_dir": str(data_dir),
        "data_file": args.data_file,
        "omp_threads": args.omp_threads,
        "cases": all_results,
    }
    write_json(out_dir / "baseline.json", payload)
    print(f"[collect] wrote {out_dir / 'baseline.json'}")
    return 0


def compare(args: argparse.Namespace) -> int:
    baseline = load_json(pathlib.Path(args.baseline))
    exe = pathlib.Path(args.exe).resolve()
    data_dir = pathlib.Path(args.data_dir).resolve()
    report_dir = pathlib.Path(args.out_dir).resolve()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp_threads)

    fail = False
    report = []
    for b in baseline["cases"]:
        case = Case(b["name"], b["cmd"][5:])  # skip exe, data_dir, data_file tokens
        cur = run_case(exe, data_dir, args.data_file, case, env)
        expected_series = b["series"]
        expected_final = b["final"]
        same_series = cur["series"] == expected_series
        same_final = cur["final"] == expected_final
        ok = (cur["returncode"] == 0) and same_series and same_final
        if not ok:
            fail = True
        report.append(
            {
                "name": case.name,
                "ok": ok,
                "rc": cur["returncode"],
                "expected_final": expected_final,
                "actual_final": cur["final"],
                "series_len_expected": len(expected_series),
                "series_len_actual": len(cur["series"]),
                "same_series": same_series,
                "same_final": same_final,
            }
        )
        print(f"[compare] {case.name}: {'OK' if ok else 'FAIL'}")

    write_json(report_dir / "compare.json", {"report": report})
    print(f"[compare] wrote {report_dir / 'compare.json'}")
    return 1 if fail else 0


def perf(args: argparse.Namespace) -> int:
    exe = pathlib.Path(args.exe).resolve()
    data_dir = pathlib.Path(args.data_dir).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    runs = args.runs
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp_threads)

    rows = []
    for case in default_cases():
        times = []
        finals = []
        for _ in range(runs):
            one = run_case(exe, data_dir, args.data_file, case, env)
            times.append(one["seconds"])
            finals.append(one["final"])
        med = statistics.median(times)
        rows.append(
            {
                "name": case.name,
                "times": times,
                "median_seconds": med,
                "finals": finals,
            }
        )
        print(f"[perf] {case.name}: median={med:.3f}s")

    write_json(out_dir / "perf.json", {"runs": runs, "rows": rows})
    print(f"[perf] wrote {out_dir / 'perf.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="CED_Schedule baseline/consistency/perf regression helper")
    p.add_argument("--exe", required=True, help="Path to CED_Schedule.exe")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--data_file", default="data_matrix_100.txt")
    p.add_argument("--out_dir", default="results/perf_baseline")
    p.add_argument("--omp_threads", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--baseline", default="results/perf_baseline/baseline.json")
    p.add_argument("mode", choices=["collect", "compare", "perf"])
    args = p.parse_args()

    if args.mode == "collect":
        return collect(args)
    if args.mode == "compare":
        return compare(args)
    return perf(args)


if __name__ == "__main__":
    sys.exit(main())
