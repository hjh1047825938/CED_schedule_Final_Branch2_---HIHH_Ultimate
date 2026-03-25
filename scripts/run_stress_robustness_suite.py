#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], root: Path) -> None:
    result = subprocess.run(cmd, cwd=root, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot stress robustness runner + analyzer.")
    parser.add_argument("--instances", default="T200,T500")
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include_shared_bandit", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_cmd = [
        sys.executable,
        "scripts/run_stress_robustness_eval.py",
        "--instances",
        args.instances,
        "--max_parallel",
        str(args.max_parallel),
    ]
    if args.force:
        run_cmd.append("--force")
    if args.include_shared_bandit:
        run_cmd.append("--include_shared_bandit")

    run(run_cmd, root)
    run([sys.executable, "scripts/analyze_stress_robustness.py"], root)


if __name__ == "__main__":
    main()
