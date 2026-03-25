import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def find_exe():
    candidates = [
        ROOT / "build_codex" / "Release" / "CED_Schedule.exe",
        ROOT / "build" / "Release" / "CED_Schedule.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def test_cchihh_shared_bandit_solver_uses_cchihh_branch():
    exe = find_exe()
    assert exe.exists(), f"Missing executable: {exe}"

    cmd = [
        str(exe),
        "--solver",
        "CCHIHH_shared_bandit",
        "--data_dir",
        str(ROOT / "data"),
        "--data_file",
        "data_matrix_100.txt",
        "--generations",
        "1",
        "--popsize",
        "20",
        "--nsubpop",
        "4",
        "--seed",
        "1",
        "--stable",
        "--log_every",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = proc.stdout + proc.stderr

    assert proc.returncode == 0, output
    assert "CCHIHH shared bandit: enabled" in output, output
    assert "CCHIHH gate_blocked_total" in output, output
