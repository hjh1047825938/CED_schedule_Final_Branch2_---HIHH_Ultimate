from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Dict, Iterable, List, Optional, Tuple

from scipy.stats import mannwhitneyu, wilcoxon


ROOT = Path(__file__).resolve().parent
SCALES = {
    "T100": {"data_file": "data_matrix_100.txt", "cnum": 100, "enum": 100, "dnum": 300, "tnum": 100, "mopt": 5},
    "T200": {"data_file": "data_matrix_T200_E100_D300.txt", "cnum": 100, "enum": 100, "dnum": 300, "tnum": 200, "mopt": 5},
    "T500": {"data_file": "data_matrix_T500_E200_D800.txt", "cnum": 200, "enum": 200, "dnum": 800, "tnum": 500, "mopt": 5},
}

PAPER_TABLES = {
    "CCHIHH-noCC": "tab:ablation_noCC",
    "CCHIHH-noHI": "tab:ablation_noHI",
    "CCHIHH-noCB": "tab:performance_comparison",
    "CCHIHH-noGate": "tab:ablation_gate",
    "CCHIHH-noMig": "tab:ablation_migration",
    "CGA": "tab:performance_comparison_CGA",
    "IMOMA": "tab:performance_comparison_IMOMA",
    "PPO": "tab:baseline_ppo",
    "DSAC-DE": "tab:baseline_dsac_de",
}

REQUESTED_ALGORITHMS = [
    "CCHIHH-full", "CCHIHH-noCC", "CCHIHH-noHI", "CCHIHH-noCB", "CCHIHH-noMig", "CCHIHH-noGate",
    "GA", "DE", "Gbest-DE", "CGA", "IMOMA", "PPO", "DSAC-DE",
]


@dataclass
class LogRecord:
    path: Path
    series: Dict[int, float]
    final: Optional[float]
    runtime_sec: Optional[float]
    normalization_f1: Optional[float]
    normalization_f2: Optional[float]
    gate_blocked_total: Optional[int]
    gate_fallback_total: Optional[int]
    seed: Optional[int]


GEN_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([0-9.+\-eE]+)")
FINAL_RE = re.compile(r"^The best solution\s*=\s*([0-9.+\-eE]+)")
TIME_RE = re.compile(r"^(?:Time|Total time)\s*=\s*([0-9.+\-eE]+)\s*s")
NORM_F1_RE = re.compile(r"^\[Normalization\]\s*f1_ref.*?=\s*([0-9.+\-eE]+)")
NORM_F2_RE = re.compile(r"^\[Normalization\]\s*f2_ref.*?=\s*([0-9.+\-eE]+)")
GATE_BLOCK_RE = re.compile(r"^CCHIHH gate_blocked_total\s*=\s*(\d+)")
GATE_FALLBACK_RE = re.compile(r"^CCHIHH gate_fallback_total\s*=\s*(\d+)")
SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)
PAPER_ROW_RE = re.compile(r"^\s*(T100|T200|T500)\s*&\s*([0-9.]+)\s*\\pm\s*([0-9.]+)\s*&\s*([0-9.]+)\s*\\pm\s*([0-9.]+)\s*&\s*([\-0-9.]+)\s*&\s*([0-9.]+)")


def read_text_auto(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="ignore")
    for enc in ("utf-8", "utf-16", "gbk", "latin1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin1", errors="ignore")


def parse_log(path: Path) -> LogRecord:
    text = read_text_auto(path).replace("\x00", "")
    series: Dict[int, float] = {}
    final = runtime = f1 = f2 = None
    gate_b = gate_f = None
    numeric_tail: List[float] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if re.fullmatch(r"[0-9.+\-eE]+", line):
            try:
                numeric_tail.append(float(line))
            except ValueError:
                pass
        for regex, target in (
            (GEN_RE, "gen"),
            (FINAL_RE, "final"),
            (TIME_RE, "time"),
            (NORM_F1_RE, "f1"),
            (NORM_F2_RE, "f2"),
            (GATE_BLOCK_RE, "gb"),
            (GATE_FALLBACK_RE, "gf"),
        ):
            m = regex.match(line)
            if not m:
                continue
            if target == "gen":
                series[int(m.group(1))] = float(m.group(2))
            elif target == "final":
                final = float(m.group(1))
            elif target == "time":
                runtime = float(m.group(1))
            elif target == "f1":
                f1 = float(m.group(1))
            elif target == "f2":
                f2 = float(m.group(1))
            elif target == "gb":
                gate_b = int(m.group(1))
            elif target == "gf":
                gate_f = int(m.group(1))
            break
    if final is None and series:
        final = series[max(series)]
    if final is None and numeric_tail:
        final = numeric_tail[-1]
    seed = None
    m = SEED_RE.search(path.name)
    if m:
        seed = int(m.group(1))
    return LogRecord(path, series, final, runtime, f1, f2, gate_b, gate_f, seed)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header: List[str], rows: Iterable[Iterable[object]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_logs(glob_pattern: str) -> Dict[int, LogRecord]:
    out: Dict[int, LogRecord] = {}
    for path in sorted(ROOT.glob(glob_pattern)):
        rec = parse_log(path)
        if rec.seed is not None and rec.final is not None:
            out[rec.seed] = rec
    return out


def algorithm_sources() -> Dict[str, Dict[str, str]]:
    return {
        "CCHIHH-full": {s: f"outputs/results/cchihh_ablation_suite/baseline/{s}/CCHIHH_Full_seed*.txt" for s in SCALES},
        "CCHIHH-noCC": {s: f"outputs/results/cchihh_nocc_multiscale/{s}/CCHIHH_noCC_seed*.txt" for s in SCALES},
        "CCHIHH-noHI": {s: "" for s in SCALES},
        "CCHIHH-noCB": {s: f"outputs/results/cchihh_ablation_suite/bandit/{s}/fixed_ops_seed*.txt" for s in SCALES},
        "CCHIHH-noMig": {s: f"outputs/results/cchihh_ablation_suite/migration/{s}/migration_false_seed*.txt" for s in SCALES},
        "CCHIHH-noGate": {s: f"outputs/results/cchihh_ablation_suite/stability/{s}/stable_false_seed*.txt" for s in SCALES},
        "GA": {s: "" for s in SCALES},
        "DE": {s: "" for s in SCALES},
        "Gbest-DE": {s: f"outputs/results/cchihh_ablation_suite/baseline/{s}/GDE_seed*.txt" for s in SCALES},
        "CGA": {s: f"outputs/results/cchihh_ablation_suite/baseline/{s}/CGA_seed*.txt" for s in SCALES},
        "IMOMA": {s: f"outputs/results/cchihh_ablation_suite/baseline/{s}/IMOMA_seed*.txt" for s in SCALES},
        "PPO": {s: f"outputs/results/PPO/{s}_seed*.txt" for s in SCALES},
        "DSAC-DE": {s: f"outputs/results/dsac_de_multiscale/{s}/DSAC_DE_seed*.txt" for s in SCALES},
    }


def collect_algorithm_data() -> Tuple[Dict[str, Dict[str, Dict[int, LogRecord]]], List[Dict[str, str]]]:
    data: Dict[str, Dict[str, Dict[int, LogRecord]]] = {}
    missing: List[Dict[str, str]] = []
    for alg, scale_map in algorithm_sources().items():
        data[alg] = {}
        for scale, pattern in scale_map.items():
            if not pattern:
                data[alg][scale] = {}
                missing.append({
                    "algorithm": alg,
                    "problem": scale,
                    "status": "REQUIRES RERUN" if alg in {"GA", "DE", "CCHIHH-noHI"} else "NOT FOUND",
                    "reason": "仓库中未找到与当前论文量级一致的 10-run 原始日志",
                })
                continue
            recs = collect_logs(pattern)
            data[alg][scale] = recs
            if len(recs) < 10:
                missing.append({
                    "algorithm": alg,
                    "problem": scale,
                    "status": "REQUIRES RERUN",
                    "reason": f"仅找到 {len(recs)} 个有效 seed 日志",
                })
    return data, missing


def exact_wilcoxon(a: List[float], b: List[float]) -> Tuple[Optional[float], Optional[float], str]:
    if len(a) != len(b) or not a:
        return None, None, "NOT FOUND"
    try:
        r = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", correction=False, method="exact")
        return float(r.statistic), float(r.pvalue), "exact"
    except Exception:
        try:
            r = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", correction=False, method="asymptotic")
            return float(r.statistic), float(r.pvalue), "asymptotic"
        except Exception:
            return None, None, "NOT FOUND"


def exact_mannwhitney(a: List[float], b: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not a or not b:
        return None, None
    try:
        r = mannwhitneyu(a, b, alternative="two-sided", method="exact")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return None, None


def parse_paper_tables() -> Dict[str, Dict[str, Dict[str, float]]]:
    text = (ROOT / "CCHIHH.tex").read_text(encoding="utf-8", errors="ignore")
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    current_label = None
    for line in text.splitlines():
        m = re.search(r"\\label\{([^}]+)\}", line)
        if m:
            current_label = m.group(1)
            continue
        m = PAPER_ROW_RE.match(line)
        if not m or current_label is None:
            continue
        out.setdefault(current_label, {})[m.group(1)] = {
            "full_mean": float(m.group(2)),
            "full_std": float(m.group(3)),
            "other_mean": float(m.group(4)),
            "other_std": float(m.group(5)),
            "improvement_pct": float(m.group(6)),
            "p_value": float(m.group(7)),
        }
    return out


def enumerate_exact_wilcoxon_pvalues(n: int = 10) -> List[float]:
    ranks = list(range(1, n + 1))
    total = sum(ranks)
    stat_to_count: Dict[int, int] = {}
    for mask in range(1 << n):
        wpos = sum(r for i, r in enumerate(ranks) if mask & (1 << i))
        w = min(wpos, total - wpos)
        stat_to_count[w] = stat_to_count.get(w, 0) + 1
    out = set()
    for w in stat_to_count:
        prob = sum(cnt for stat, cnt in stat_to_count.items() if stat <= w) / (2 ** n)
        out.add(round(min(1.0, 2.0 * prob), 12))
    return sorted(out)


def closest_match(value: float, candidates: List[float], tol: float = 1e-12) -> bool:
    return any(abs(value - c) <= tol for c in candidates)


def run_command(command: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd or ROOT, text=True, capture_output=True, check=False)


def parse_data_matrix(path: Path, scale: str) -> dict:
    cfg = SCALES[scale]
    nums = [int(x) if x.isdigit() else float(x) for x in path.read_text(encoding="utf-8", errors="ignore").split()]
    idx = 0

    def take(n: int):
        nonlocal idx
        out = nums[idx: idx + n]
        idx += n
        return out

    enum_n = cfg["enum"]
    dnum = cfg["dnum"]
    tnum = cfg["tnum"]
    mopt = cfg["mopt"]
    take(enum_n * dnum)
    take(dnum * dnum)
    mtask_time = take(tnum * mopt)
    ce_tasks = []
    for _ in range(tnum):
        computation = take(1)[0]
        communication = take(1)[0]
        precedence = take(int(take(1)[0]))
        interact = take(int(take(1)[0]))
        start_pre = take(int(take(1)[0]))
        end_pre = take(int(take(1)[0]))
        job_constraints = int(take(1)[0])
        ce_tasks.append({
            "computation": computation,
            "communication": communication,
            "precedence": precedence,
            "interact": interact,
            "start_pre": start_pre,
            "end_pre": end_pre,
            "job_constraints": job_constraints,
        })
    avail_devices = []
    for _ in range(tnum):
        for _ in range(mopt):
            avail_devices.append([int(v) for v in take(int(take(1)[0]))])
    avail_edges = []
    for _ in range(tnum):
        avail_edges.append([int(v) for v in take(int(take(1)[0]))])
    energy_list = take(11)
    return {
        "mtask_time": mtask_time,
        "ce_tasks": ce_tasks,
        "avail_devices": avail_devices,
        "avail_edges": avail_edges,
        "energy_list": energy_list,
    }
