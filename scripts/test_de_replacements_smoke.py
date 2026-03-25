#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


EXPECTED_SOLVERS = ["L-SRTDE", "NL-SHADE-LBC"]


def run(cmd, cwd):
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def require(cond, message):
    if not cond:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description="Smoke test for DE replacement solvers.")
    parser.add_argument("--exe", required=True, help="Path to CED_Schedule executable")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--data_file", default="data_matrix_100.txt")
    parser.add_argument("--max_evals", type=int, default=80)
    args = parser.parse_args()

    exe = Path(args.exe).resolve()
    repo = Path(__file__).resolve().parents[1]
    require(exe.is_file(), f"Executable not found: {exe}")

    help_run = run([str(exe), "--help"], repo)
    require(help_run.returncode == 0, f"--help failed:\n{help_run.stdout}")
    for solver in EXPECTED_SOLVERS:
        require(
            solver in help_run.stdout,
            f"solver '{solver}' missing from --help output",
        )

    for solver in EXPECTED_SOLVERS:
        smoke = run(
            [
                str(exe),
                "--solver",
                solver,
                "--data_dir",
                args.data_dir,
                "--data_file",
                args.data_file,
                "--seed",
                "1",
                "--popsize",
                "40",
                "--generations",
                "999999",
                "--max_evals",
                str(args.max_evals),
                "--alpha",
                "0.5",
                "--log_every",
                "20",
            ],
            repo,
        )
        require(smoke.returncode == 0, f"{solver} smoke run failed:\n{smoke.stdout}")
        require(
            f"Solver: {solver}" in smoke.stdout,
            f"{solver} final output missing solver tag:\n{smoke.stdout}",
        )

    print("Smoke checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"SMOKE TEST FAILED: {exc}")
        sys.exit(1)
