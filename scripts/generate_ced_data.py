import argparse
import math
import random
from pathlib import Path


def _pick_k(rng: random.Random, min_k: int, max_k: int, upper: int) -> int:
    min_k = max(1, min_k)
    max_k = max(min_k, max_k)
    max_k = min(max_k, upper)
    return rng.randint(min_k, max_k)


def _pick_prev_indices(rng: random.Random, i: int, prob: float, max_k: int):
    if i <= 0 or prob <= 0.0:
        return []
    if rng.random() > prob:
        return []
    k = _pick_k(rng, 1, max_k, i)
    return rng.sample(range(i), k)


def generate_data(
    out_path: Path,
    cnum: int,
    enum: int,
    dnum: int,
    tnum: int,
    mopt: int,
    seed: int,
    dep_prob: float = 0.0,
    max_dep: int = 3,
):
    rng = random.Random(seed)

    edges = [(rng.uniform(0.0, 1000.0), rng.uniform(0.0, 1000.0)) for _ in range(enum)]
    devices = [(rng.uniform(0.0, 1000.0), rng.uniform(0.0, 1000.0)) for _ in range(dnum)]

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    eto_d = [
        [int(1000 + dist(edges[e], devices[d])) for d in range(dnum)]
        for e in range(enum)
    ]
    dto_d = [
        [int(1000 + dist(devices[i], devices[j])) for j in range(dnum)]
        for i in range(dnum)
    ]

    mtask_time = [round(rng.uniform(10.0, 100.0), 2) for _ in range(tnum * mopt)]

    task_computation = [round(rng.uniform(500.0, 2000.0), 2) for _ in range(tnum)]
    task_communication = [round(rng.uniform(100.0, 1000.0), 2) for _ in range(tnum)]

    def job_constraint():
        r = rng.random()
        if r < 0.6:
            return 0
        if r < 0.75:
            return 1
        if r < 0.9:
            return 2
        return 3

    avail_devices = []
    min_dev = max(3, int(dnum * 0.05))
    max_dev = max(min_dev, int(dnum * 0.15))
    max_dev = min(max_dev, dnum)

    for _ in range(tnum * mopt):
        k = _pick_k(rng, min_dev, max_dev, dnum)
        avail_devices.append(rng.sample(range(dnum), k))

    avail_edges = []
    min_edge = max(3, int(enum * 0.05))
    max_edge = max(min_edge, int(enum * 0.15))
    max_edge = min(max_edge, enum)
    for _ in range(tnum):
        k = _pick_k(rng, min_edge, max_edge, enum)
        avail_edges.append(rng.sample(range(enum), k))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in eto_d:
            f.write(" ".join(str(v) for v in row) + "\n")
        for row in dto_d:
            f.write(" ".join(str(v) for v in row) + "\n")

        f.write(" ".join(str(v) for v in mtask_time) + "\n")

        for i in range(tnum):
            pre = _pick_prev_indices(rng, i, dep_prob, max_dep)
            inter = _pick_prev_indices(rng, i, dep_prob, max_dep)
            start_pre = _pick_prev_indices(rng, i, dep_prob, max_dep)
            end_pre = _pick_prev_indices(rng, i, dep_prob, max_dep)
            tokens = [
                task_computation[i],
                task_communication[i],
                len(pre),
                *pre,
                len(inter),
                *inter,
                len(start_pre),
                *start_pre,
                len(end_pre),
                *end_pre,
                job_constraint(),
            ]
            f.write(" ".join(str(v) for v in tokens) + "\n")

        for ops in avail_devices:
            f.write(str(len(ops)) + " " + " ".join(str(v) for v in ops) + "\n")

        for edges_list in avail_edges:
            f.write(str(len(edges_list)) + " " + " ".join(str(v) for v in edges_list) + "\n")

        f.write(" ".join(str(i) for i in range(11)) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cnum", type=int, required=True)
    parser.add_argument("--enum", type=int, required=True)
    parser.add_argument("--dnum", type=int, required=True)
    parser.add_argument("--tnum", type=int, required=True)
    parser.add_argument("--mopt", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dep_prob", type=float, default=0.0)
    parser.add_argument("--max_dep", type=int, default=3)
    args = parser.parse_args()

    generate_data(
        out_path=Path(args.out),
        cnum=args.cnum,
        enum=args.enum,
        dnum=args.dnum,
        tnum=args.tnum,
        mopt=args.mopt,
        seed=args.seed,
        dep_prob=args.dep_prob,
        max_dep=args.max_dep,
    )
    print(f"Generated: {args.out}")


if __name__ == "__main__":
    main()
