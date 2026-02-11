#!/usr/bin/env python3
"""
Generate a scaled CED data matrix from an existing reference instance.

The output format follows the exact token order required by src/Multimethod.cpp::Initial():
1) EtoD_Distance           [Enum x Dnum]
2) DtoD_Distance           [Dnum x Dnum]
3) MTask_Time              [M_Jnum * M_OPTnum]
4) CETask records          [CE_Tnum], variable-length per record
5) AvailDeviceList         [M_Jnum * M_OPTnum], variable-length per op
6) AvailEdgeServerList     [CE_Tnum], variable-length per task
7) EnergyList              [11]
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence


@dataclass
class ParsedData:
    enum: int
    dnum: int
    tnum: int
    mopt: int
    eto_d: List[List[int]]
    dto_d: List[List[int]]
    mtask_time: List[float]
    tasks: List[Dict[str, object]]
    avail_devices: List[List[int]]
    avail_edges: List[List[int]]
    energy: List[float]


def _read_tokens(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").split()


def parse_matrix(path: Path, enum: int, dnum: int, tnum: int, mopt: int) -> ParsedData:
    tokens = _read_tokens(path)
    idx = 0

    def take(n: int) -> List[str]:
        nonlocal idx
        if idx + n > len(tokens):
            raise ValueError(f"Unexpected EOF while parsing {path}, need {n} tokens at {idx}.")
        out = tokens[idx : idx + n]
        idx += n
        return out

    eto_d = [[int(x) for x in take(dnum)] for _ in range(enum)]
    dto_d = [[int(x) for x in take(dnum)] for _ in range(dnum)]
    mtask_time = [float(x) for x in take(tnum * mopt)]

    tasks: List[Dict[str, object]] = []
    for _ in range(tnum):
        comp = float(take(1)[0])
        comm = float(take(1)[0])
        dep_lists: Dict[str, List[int]] = {}
        for name in ("precedence", "interact", "start_pre", "end_pre"):
            k = int(take(1)[0])
            dep_lists[name] = [int(x) for x in take(k)]
        job_constraint = int(take(1)[0])
        tasks.append(
            {
                "computation": comp,
                "communication": comm,
                "precedence": dep_lists["precedence"],
                "interact": dep_lists["interact"],
                "start_pre": dep_lists["start_pre"],
                "end_pre": dep_lists["end_pre"],
                "job_constraint": job_constraint,
            }
        )

    avail_devices = []
    for _ in range(tnum * mopt):
        k = int(take(1)[0])
        avail_devices.append([int(x) for x in take(k)])

    avail_edges = []
    for _ in range(tnum):
        k = int(take(1)[0])
        avail_edges.append([int(x) for x in take(k)])

    energy = [float(x) for x in take(11)]

    if idx != len(tokens):
        raise ValueError(f"Unconsumed tokens in {path}: {len(tokens) - idx}. Check source dimensions.")

    return ParsedData(
        enum=enum,
        dnum=dnum,
        tnum=tnum,
        mopt=mopt,
        eto_d=eto_d,
        dto_d=dto_d,
        mtask_time=mtask_time,
        tasks=tasks,
        avail_devices=avail_devices,
        avail_edges=avail_edges,
        energy=energy,
    )


def _density(dep_lists: Sequence[Sequence[int]]) -> float:
    n = len(dep_lists)
    possible = n * (n - 1) / 2.0
    if possible <= 0:
        return 0.0
    return sum(len(x) for x in dep_lists) / possible


def _clip_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _gaussian_count(rng: random.Random, mean_k: float, std_k: float, lo: int, hi: int) -> int:
    if std_k <= 1e-9:
        return _clip_int(int(round(mean_k)), lo, hi)
    return _clip_int(int(round(rng.gauss(mean_k, std_k))), lo, hi)


def _sample_dep_list(
    rng: random.Random,
    current_task: int,
    density: float,
    max_k: int,
) -> List[int]:
    if current_task <= 0 or density <= 0.0:
        return []
    candidates = list(range(current_task))
    chosen = [v for v in candidates if rng.random() < density]
    if max_k > 0 and len(chosen) > max_k:
        chosen = rng.sample(chosen, max_k)
    chosen.sort()
    return chosen


def _pick_weighted(rng: random.Random, value_counts: Dict[int, int]) -> int:
    values = list(value_counts.keys())
    weights = [value_counts[v] for v in values]
    return rng.choices(values, weights=weights, k=1)[0]


def suggest_scaled_sizes(src_tnum: int, src_cnum: int, src_enum: int, src_dnum: int, tgt_tnum: int):
    scale = tgt_tnum / float(src_tnum)
    infra_scale = 1.0 + 0.5 * (scale - 1.0)
    return (
        max(1, int(round(src_cnum * infra_scale))),
        max(1, int(round(src_enum * infra_scale))),
        max(1, int(round(src_dnum * infra_scale))),
    )


def generate_scaled(
    src: ParsedData,
    rng: random.Random,
    tgt_enum: int,
    tgt_dnum: int,
    tgt_tnum: int,
    tgt_mopt: int,
) -> ParsedData:
    src_eto_flat = [v for row in src.eto_d for v in row]
    src_dto_off = [src.dto_d[i][j] for i in range(src.dnum) for j in range(src.dnum) if i != j]
    src_dto_diag = [src.dto_d[i][i] for i in range(src.dnum)]

    eto_d = [[rng.choice(src_eto_flat) for _ in range(tgt_dnum)] for _ in range(tgt_enum)]

    dto_d = [[0] * tgt_dnum for _ in range(tgt_dnum)]
    for i in range(tgt_dnum):
        dto_d[i][i] = rng.choice(src_dto_diag)
    for i in range(tgt_dnum):
        for j in range(i + 1, tgt_dnum):
            v = rng.choice(src_dto_off)
            dto_d[i][j] = v
            dto_d[j][i] = v

    mtask_time = [rng.choice(src.mtask_time) for _ in range(tgt_tnum * tgt_mopt)]

    src_comp = [float(t["computation"]) for t in src.tasks]
    src_comm = [float(t["communication"]) for t in src.tasks]

    src_pre = [list(t["precedence"]) for t in src.tasks]
    src_inter = [list(t["interact"]) for t in src.tasks]
    src_start_pre = [list(t["start_pre"]) for t in src.tasks]
    src_end_pre = [list(t["end_pre"]) for t in src.tasks]

    pre_density = _density(src_pre)
    inter_density = _density(src_inter)
    start_pre_density = _density(src_start_pre)
    end_pre_density = _density(src_end_pre)

    max_pre = max((len(x) for x in src_pre), default=0)
    max_inter = max((len(x) for x in src_inter), default=0)
    max_start_pre = max((len(x) for x in src_start_pre), default=0)
    max_end_pre = max((len(x) for x in src_end_pre), default=0)

    jc_counts: Dict[int, int] = {}
    for t in src.tasks:
        jc = int(t["job_constraint"])
        jc_counts[jc] = jc_counts.get(jc, 0) + 1

    tasks: List[Dict[str, object]] = []
    for i in range(tgt_tnum):
        tasks.append(
            {
                "computation": rng.choice(src_comp),
                "communication": rng.choice(src_comm),
                "precedence": _sample_dep_list(rng, i, pre_density, max_pre),
                "interact": _sample_dep_list(rng, i, inter_density, max_inter),
                "start_pre": _sample_dep_list(rng, i, start_pre_density, max_start_pre),
                "end_pre": _sample_dep_list(rng, i, end_pre_density, max_end_pre),
                "job_constraint": _pick_weighted(rng, jc_counts),
            }
        )

    src_dev_lens = [len(x) for x in src.avail_devices]
    src_edge_lens = [len(x) for x in src.avail_edges]
    dev_ratio_mean = mean(src_dev_lens) / float(src.dnum)
    edge_ratio_mean = mean(src_edge_lens) / float(src.enum)
    dev_ratio_std = pstdev(src_dev_lens) / float(src.dnum) if len(src_dev_lens) > 1 else 0.0
    edge_ratio_std = pstdev(src_edge_lens) / float(src.enum) if len(src_edge_lens) > 1 else 0.0

    tgt_dev_mean = max(1.0, dev_ratio_mean * tgt_dnum)
    tgt_edge_mean = max(1.0, edge_ratio_mean * tgt_enum)
    tgt_dev_std = max(1.0, dev_ratio_std * tgt_dnum)
    tgt_edge_std = max(1.0, edge_ratio_std * tgt_enum)

    avail_devices: List[List[int]] = []
    for _ in range(tgt_tnum * tgt_mopt):
        k = _gaussian_count(rng, tgt_dev_mean, tgt_dev_std, 1, tgt_dnum)
        vals = sorted(rng.sample(range(tgt_dnum), k))
        avail_devices.append(vals)

    avail_edges: List[List[int]] = []
    for _ in range(tgt_tnum):
        k = _gaussian_count(rng, tgt_edge_mean, tgt_edge_std, 1, tgt_enum)
        vals = sorted(rng.sample(range(tgt_enum), k))
        avail_edges.append(vals)

    return ParsedData(
        enum=tgt_enum,
        dnum=tgt_dnum,
        tnum=tgt_tnum,
        mopt=tgt_mopt,
        eto_d=eto_d,
        dto_d=dto_d,
        mtask_time=mtask_time,
        tasks=tasks,
        avail_devices=avail_devices,
        avail_edges=avail_edges,
        energy=list(src.energy),
    )


def write_matrix(path: Path, data: ParsedData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in data.eto_d:
            f.write(" ".join(str(v) for v in row) + "\n")
        for row in data.dto_d:
            f.write(" ".join(str(v) for v in row) + "\n")
        for v in data.mtask_time:
            if float(v).is_integer():
                f.write(f"{int(v)}\n")
            else:
                f.write(f"{v:.6f}\n")
        for t in data.tasks:
            toks: List[str] = []
            comp = float(t["computation"])
            comm = float(t["communication"])
            toks.append(str(int(comp)) if comp.is_integer() else f"{comp:.6f}")
            toks.append(str(int(comm)) if comm.is_integer() else f"{comm:.6f}")
            for name in ("precedence", "interact", "start_pre", "end_pre"):
                vals = list(t[name])  # type: ignore[index]
                toks.append(str(len(vals)))
                toks.extend(str(int(x)) for x in vals)
            toks.append(str(int(t["job_constraint"])))
            f.write(" ".join(toks) + "\n")
        for vals in data.avail_devices:
            f.write(f"{len(vals)} {' '.join(str(v) for v in vals)}\n")
        for vals in data.avail_edges:
            f.write(f"{len(vals)} {' '.join(str(v) for v in vals)}\n")
        f.write(" ".join(str(int(v)) if float(v).is_integer() else f"{v:.6f}" for v in data.energy) + "\n")


def _stat_line(name: str, values: Sequence[float]) -> str:
    return (
        f"{name}: n={len(values)}, min={min(values):.4f}, max={max(values):.4f}, "
        f"mean={mean(values):.4f}, std={pstdev(values) if len(values)>1 else 0.0:.4f}"
    )


def summarize(label: str, data: ParsedData) -> None:
    eto_flat = [float(v) for row in data.eto_d for v in row]
    dto_flat = [float(v) for row in data.dto_d for v in row]
    dto_off = [float(data.dto_d[i][j]) for i in range(data.dnum) for j in range(data.dnum) if i != j]
    comp = [float(t["computation"]) for t in data.tasks]
    comm = [float(t["communication"]) for t in data.tasks]
    pre = [len(t["precedence"]) for t in data.tasks]
    ad = [len(x) for x in data.avail_devices]
    ae = [len(x) for x in data.avail_edges]
    print(f"\n[{label}] enum={data.enum}, dnum={data.dnum}, tnum={data.tnum}, mopt={data.mopt}")
    print(_stat_line("EtoD", eto_flat))
    print(_stat_line("DtoD_all", dto_flat))
    print(_stat_line("DtoD_offdiag", dto_off))
    print(_stat_line("MTask_Time", [float(x) for x in data.mtask_time]))
    print(_stat_line("Computation", comp))
    print(_stat_line("Communication", comm))
    print(_stat_line("Precedence_len", [float(x) for x in pre]))
    print(_stat_line("AvailDevice_len", [float(x) for x in ad]))
    print(_stat_line("AvailEdge_len", [float(x) for x in ae]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scaled data_matrix for CED scheduler.")
    parser.add_argument("--input", default="data/data_matrix_100.txt")
    parser.add_argument("--output", default="data/data_matrix_200.txt")
    parser.add_argument("--src-cnum", type=int, default=100)
    parser.add_argument("--src-enum", type=int, default=100)
    parser.add_argument("--src-dnum", type=int, default=300)
    parser.add_argument("--src-tnum", type=int, default=100)
    parser.add_argument("--src-mopt", type=int, default=5)
    parser.add_argument("--target-tnum", type=int, default=200)
    parser.add_argument("--target-mopt", type=int, default=5)
    parser.add_argument("--target-cnum", type=int, default=0, help="0 means auto-scale.")
    parser.add_argument("--target-enum", type=int, default=0, help="0 means auto-scale.")
    parser.add_argument("--target-dnum", type=int, default=0, help="0 means auto-scale.")
    parser.add_argument("--seed", type=int, default=20260206)
    args = parser.parse_args()

    src = parse_matrix(
        path=Path(args.input),
        enum=args.src_enum,
        dnum=args.src_dnum,
        tnum=args.src_tnum,
        mopt=args.src_mopt,
    )
    summarize("SOURCE", src)

    auto_c, auto_e, auto_d = suggest_scaled_sizes(
        src_tnum=args.src_tnum,
        src_cnum=args.src_cnum,
        src_enum=args.src_enum,
        src_dnum=args.src_dnum,
        tgt_tnum=args.target_tnum,
    )
    tgt_cnum = args.target_cnum if args.target_cnum > 0 else auto_c
    tgt_enum = args.target_enum if args.target_enum > 0 else auto_e
    tgt_dnum = args.target_dnum if args.target_dnum > 0 else auto_d

    rng = random.Random(args.seed)
    generated = generate_scaled(
        src=src,
        rng=rng,
        tgt_enum=tgt_enum,
        tgt_dnum=tgt_dnum,
        tgt_tnum=args.target_tnum,
        tgt_mopt=args.target_mopt,
    )
    summarize("TARGET", generated)

    write_matrix(Path(args.output), generated)
    print(f"\nGenerated file: {args.output}")
    print(
        "Recommended run args: "
        f"--cnum {tgt_cnum} --enum {tgt_enum} --dnum {tgt_dnum} --tnum {args.target_tnum} --mopt {args.target_mopt}"
    )


if __name__ == "__main__":
    main()

