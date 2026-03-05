#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def tail_rows(path: Path, n: int):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s == "r,l,t":
                continue
            rows.append(s.split(","))
    return rows[-n:]


def latest_row(path: Path):
    rows = tail_rows(path, 1)
    if not rows:
        return None
    r, l, t = rows[0]
    return {"reward": float(r), "len": int(float(l)), "sec": float(t)}


def file_age_sec(path: Path) -> float:
    if not path.exists():
        return 1e18
    return time.time() - path.stat().st_mtime


def summarize_one(base: Path, scale: str):
    mon = base / scale / "monitor.csv"
    row = latest_row(mon)
    age = file_age_sec(mon)
    if row is None:
        return {
            "scale": scale,
            "status": "waiting",
            "latest_reward": "",
            "episodes_logged": 0,
            "monitor_age_sec": round(age, 1),
            "sim_time_sec": "",
            "warn": "no_episode_yet",
        }

    rows = tail_rows(mon, 50)
    rewards = [float(x[0]) for x in rows] if rows else [row["reward"]]
    mean50 = sum(rewards) / len(rewards)
    warn = ""
    if age > 300:
        warn = "stale_monitor_over_5min"
    elif row["reward"] < -5:
        warn = "large_negative_reward_seen"

    return {
        "scale": scale,
        "status": "running_or_done",
        "latest_reward": f"{row['reward']:.6f}",
        "episodes_logged": len(tail_rows(mon, 1000000)),
        "monitor_age_sec": round(age, 1),
        "sim_time_sec": f"{row['sec']:.1f}",
        "mean_reward_last50": f"{mean50:.6f}",
        "warn": warn,
    }


def write_status(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scale",
                "status",
                "latest_reward",
                "episodes_logged",
                "monitor_age_sec",
                "sim_time_sec",
                "mean_reward_last50",
                "warn",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="outputs/results/ppo_sb3_subprocess_budgeted")
    p.add_argument("--interval_sec", type=int, default=60)
    p.add_argument("--once", action="store_true")
    p.add_argument("--status_csv", default="outputs/results/ppo_sb3_subprocess_budgeted/monitor_status.csv")
    args = p.parse_args()

    base = Path(args.base_dir)
    status_csv = Path(args.status_csv)

    while True:
        rows = [summarize_one(base, s) for s in ("T100", "T200", "T500")]
        write_status(status_csv, rows)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] status updated: {status_csv}")
        for r in rows:
            print(
                f"  {r['scale']}: status={r['status']} ep={r['episodes_logged']} "
                f"latest={r['latest_reward']} age={r['monitor_age_sec']}s warn={r['warn']}"
            )
        if args.once:
            break
        time.sleep(max(5, args.interval_sec))


if __name__ == "__main__":
    main()

