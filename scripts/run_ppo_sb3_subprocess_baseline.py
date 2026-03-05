#!/usr/bin/env python3
"""
SB3 PPO baseline using subprocess calls to CED_Schedule.exe as black-box evaluator.

Design note:
- This script does not modify any C++ code.
- Each RL step triggers one subprocess invocation and parses fitness from stdout.
- Action is MultiDiscrete: [tier(cloud/edge), server, device].
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'gymnasium'. Install with: pip install gymnasium"
    ) from exc

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
    from stable_baselines3.common.monitor import Monitor
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'stable-baselines3'. Install with: pip install stable-baselines3"
    ) from exc


FITNESS_PATTERNS = [
    re.compile(r"The best solution\s*=\s*([0-9eE+\-.]+)"),
    re.compile(r"Best fitness\s*=\s*([0-9eE+\-.]+)"),
    re.compile(r"best_fit\s*=\s*([0-9eE+\-.]+)"),
]


@dataclass(frozen=True)
class ScaleConfig:
    name: str
    data_file: str
    cnum: int
    enum: int
    dnum: int
    tnum: int
    mopt: int
    budget_min: float


SCALES: List[ScaleConfig] = [
    ScaleConfig("T100", "data_matrix_100.txt", 100, 100, 300, 100, 5, 30.0),
    ScaleConfig("T200", "data_matrix_T200_E100_D300.txt", 100, 100, 300, 200, 5, 60.0),
    ScaleConfig("T500", "data_matrix_T500_E200_D800.txt", 200, 200, 800, 500, 5, 120.0),
]


def find_exe(root: Path, user_exe: str) -> Path:
    if user_exe:
        p = Path(user_exe).resolve()
        if not p.exists():
            raise SystemExit(f"--exe not found: {p}")
        return p
    candidates = [
        root / "build" / "Release" / "CED_Schedule.exe",
        root / "build_local" / "Release" / "CED_Schedule.exe",
        root / "build_ablation" / "Release" / "CED_Schedule.exe",
        root / "build2" / "Release" / "CED_Schedule.exe",
        root / "build_qhh" / "Release" / "CED_Schedule.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit("Cannot find CED_Schedule.exe. Pass --exe explicitly.")


def parse_data_for_state_features(data_path: Path, cfg: ScaleConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = data_path.read_text(encoding="utf-8", errors="ignore")
    toks = text.split()
    p = 0

    def read_float() -> float:
        nonlocal p
        v = float(toks[p])
        p += 1
        return v

    def read_int() -> int:
        return int(round(read_float()))

    try:
        p += cfg.enum * cfg.dnum
        p += cfg.dnum * cfg.dnum
        p += cfg.tnum * cfg.mopt

        comp = np.zeros(cfg.tnum, dtype=np.float32)
        comm = np.zeros(cfg.tnum, dtype=np.float32)
        pred = np.zeros(cfg.tnum, dtype=np.float32)
        max_pred = 1

        for i in range(cfg.tnum):
            comp[i] = read_float()
            comm[i] = read_float()
            n_pre = read_int()
            pred[i] = float(n_pre)
            max_pred = max(max_pred, n_pre)
            p += n_pre
            n_inter = read_int()
            p += n_inter
            n_start = read_int()
            p += n_start
            n_end = read_int()
            p += n_end
            _ = read_float()  # job constraints

        comp = comp / (np.max(comp) + 1e-9)
        comm = comm / (np.max(comm) + 1e-9)
        pred = pred / float(max_pred)
        return comp, comm, pred
    except Exception:
        rng = np.random.default_rng(abs(hash(str(data_path))) % (2**32))
        comp = rng.random(cfg.tnum, dtype=np.float32)
        comm = rng.random(cfg.tnum, dtype=np.float32)
        pred = np.linspace(0.0, 1.0, cfg.tnum, dtype=np.float32)
        return comp, comm, pred


class TimeBudgetCallback(BaseCallback):
    def __init__(self, budget_seconds: float):
        super().__init__()
        self.budget_seconds = budget_seconds
        self.start_time = 0.0

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()

    def _on_step(self) -> bool:
        return (time.perf_counter() - self.start_time) < self.budget_seconds


class RealFitnessEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int,
        csv_path: Path,
        verbose: int = 1,
    ):
        super().__init__(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=1,
            deterministic=True,
            verbose=verbose,
        )
        self.csv_path = csv_path
        self.real_fitness_points: List[Tuple[int, float]] = []
        self.best_real_fitness = math.inf

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "real_fitness"])

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            fit = run_deterministic_eval(self.model, self.eval_env)
            self.real_fitness_points.append((int(self.num_timesteps), float(fit)))
            self.best_real_fitness = min(self.best_real_fitness, float(fit))
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([int(self.num_timesteps), f"{float(fit):.10f}"])
            if self.verbose > 0:
                print(
                    f"[EvalCallback] timesteps={int(self.num_timesteps)} "
                    f"real_fitness={float(fit):.10f} best={self.best_real_fitness:.10f}"
                )
        return True


class CEDSubprocessEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        exe_path: Path,
        data_dir: Path,
        cfg: ScaleConfig,
        episode_steps: int,
        base_seed: int,
    ):
        super().__init__()
        self.exe_path = exe_path
        self.data_dir = data_dir
        self.cfg = cfg
        self.episode_steps = max(4, episode_steps)
        self.base_seed = int(base_seed)

        self.comp_feat, self.comm_feat, self.pred_feat = parse_data_for_state_features(
            data_dir / cfg.data_file, cfg
        )

        self.max_servers = max(cfg.cnum, cfg.enum)
        self.action_space = spaces.MultiDiscrete(
            np.array([2, self.max_servers, cfg.dnum], dtype=np.int64)
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.step_idx = 0
        self.best_fit = math.inf
        self.last_fit = math.inf
        self.cloud_count = 0
        self.edge_count = 0
        self.device_hist = np.zeros(cfg.dnum, dtype=np.int32)

    def _obs(self) -> np.ndarray:
        task_i = min(self.step_idx, self.cfg.tnum - 1)
        done_steps = max(1, self.step_idx)
        cloud_load = min(1.0, self.cloud_count / max(1.0, done_steps))
        edge_load = min(1.0, self.edge_count / max(1.0, done_steps))
        device_load = min(1.0, float(np.count_nonzero(self.device_hist)) / float(self.cfg.dnum))
        progress = min(1.0, self.step_idx / float(self.episode_steps))
        return np.array(
            [
                progress,
                cloud_load,
                edge_load,
                device_load,
                float(self.comp_feat[task_i]),
                float(self.comm_feat[task_i]),
                float(self.pred_feat[task_i]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _parse_fitness(stdout: str) -> float:
        found: List[float] = []
        for line in stdout.splitlines():
            s = line.strip()
            for pat in FITNESS_PATTERNS:
                m = pat.search(s)
                if m:
                    try:
                        found.append(float(m.group(1)))
                    except ValueError:
                        pass
        if not found:
            return math.inf
        return found[-1]

    def _evaluate_action(self, tier: int, server: int, device: int) -> float:
        if tier == 0:
            solver = "GA"
            srv = int(server % max(1, self.cfg.cnum))
        else:
            solver = "GDE"
            srv = int(server % max(1, self.cfg.enum))

        h = hashlib.sha256(
            f"{self.base_seed}|{self.cfg.name}|{self.step_idx}|{tier}|{srv}|{device}".encode("utf-8")
        ).hexdigest()
        seed = int(h[:8], 16) % (2**31 - 1)
        popsize = 40
        generations = 1
        pini = 0.2 + 0.6 * (float(device) / max(1.0, self.cfg.dnum - 1))

        cmd = [
            str(self.exe_path),
            "--solver",
            solver,
            "--seed",
            str(seed),
            "--popsize",
            str(popsize),
            "--generations",
            str(generations),
            "--log_every",
            str(generations),
            "--pini",
            f"{pini:.6f}",
            "--data_dir",
            str(self.data_dir),
            "--data_file",
            self.cfg.data_file,
            "--cnum",
            str(self.cfg.cnum),
            "--enum",
            str(self.cfg.enum),
            "--dnum",
            str(self.cfg.dnum),
            "--tnum",
            str(self.cfg.tnum),
            "--mopt",
            str(self.cfg.mopt),
            "--alpha",
            "0.5",
        ]
        # Timeout guard for rare hangs in external process calls.
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, shell=False, timeout=60)
        except subprocess.TimeoutExpired:
            return math.inf
        if proc.returncode != 0:
            return math.inf
        return self._parse_fitness(proc.stdout)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.best_fit = math.inf
        self.last_fit = math.inf
        self.cloud_count = 0
        self.edge_count = 0
        self.device_hist.fill(0)
        return self._obs(), {}

    def step(self, action):
        tier, server, device = [int(x) for x in action]
        device = int(device % self.cfg.dnum)
        if tier == 0:
            self.cloud_count += 1
        else:
            self.edge_count += 1
        self.device_hist[device] += 1

        fit = self._evaluate_action(tier, server, device)
        if not np.isfinite(fit):
            fit = 1e6
        prev_best = self.best_fit
        self.best_fit = min(self.best_fit, fit)

        if np.isfinite(prev_best):
            improve = prev_best - self.best_fit
        else:
            improve = 0.0
        reward = float(np.clip(0.2 * improve - 0.001 * fit, -10.0, 10.0))
        self.last_fit = fit
        self.step_idx += 1
        terminated = self.step_idx >= self.episode_steps
        truncated = False
        info = {}
        if terminated:
            info["episode_best_fitness"] = float(self.best_fit)
        return self._obs(), reward, terminated, truncated, info


def read_monitor_rewards(monitor_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    rewards: List[float] = []
    lengths: List[int] = []
    with monitor_csv.open("r", encoding="utf-8") as f:
        _ = f.readline()
        _ = f.readline()
        for line in f:
            r_s, l_s, _t_s = line.strip().split(",")
            rewards.append(float(r_s))
            lengths.append(int(l_s))
    if not rewards:
        return np.array([]), np.array([])
    x = np.cumsum(np.asarray(lengths, dtype=np.int64))
    y = np.asarray(rewards, dtype=np.float64)
    return x, y


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if x.size == 0:
        return x
    k = max(1, min(k, x.size))
    kernel = np.ones(k, dtype=np.float64) / float(k)
    return np.convolve(x, kernel, mode="valid")


def run_deterministic_eval(model: PPO, env) -> float:
    # Gymnasium env branch
    if hasattr(env, "observation_space") and hasattr(env, "action_space") and not hasattr(env, "num_envs"):
        obs, _ = env.reset()
        done = False
        best_fit = math.inf
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            if isinstance(info, dict) and "episode_best_fitness" in info:
                best_fit = float(info["episode_best_fitness"])
        return best_fit

    # VecEnv branch (EvalCallback may wrap eval_env)
    obs = env.reset()
    done = [False]
    best_fit = math.inf
    while not bool(done[0]):
        action, _ = model.predict(obs, deterministic=True)
        obs, _rewards, done, infos = env.step(action)
        if infos and isinstance(infos[0], dict) and "episode_best_fitness" in infos[0]:
            best_fit = float(infos[0]["episode_best_fitness"])
    return best_fit


def read_real_fitness_curve(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.array([]), np.array([])
    xs: List[int] = []
    ys: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                xs.append(int(float(row["timesteps"])))
                ys.append(float(row["real_fitness"]))
            except Exception:
                continue
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.float64)


def train_one_scale(
    root: Path,
    exe: Path,
    out_root: Path,
    cfg: ScaleConfig,
    budget_min_override: float,
    base_seed: int,
    episode_steps: int,
) -> Dict[str, float]:
    budget_min = budget_min_override if budget_min_override > 0 else cfg.budget_min
    scale_out = out_root / cfg.name
    scale_out.mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"
    env_core = CEDSubprocessEnv(exe, data_dir, cfg, episode_steps=episode_steps, base_seed=base_seed)
    env = Monitor(env_core, filename=str(scale_out / "monitor.csv"))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=base_seed,
    )

    real_eval_csv = scale_out / "real_fitness_eval.csv"
    eval_env = CEDSubprocessEnv(exe, data_dir, cfg, episode_steps=episode_steps, base_seed=base_seed + 12345)
    real_eval_cb = RealFitnessEvalCallback(
        eval_env=eval_env,
        eval_freq=5000,
        csv_path=real_eval_csv,
        verbose=1,
    )
    cb = CallbackList(
        [
            TimeBudgetCallback(budget_seconds=budget_min * 60.0),
            real_eval_cb,
        ]
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=10**12, callback=cb, progress_bar=False)
    t1 = time.perf_counter()
    wall_min = (t1 - t0) / 60.0
    steps = int(model.num_timesteps)

    model_path = scale_out / "ppo_model.zip"
    model.save(str(model_path))

    if steps > 0 and (len(real_eval_cb.real_fitness_points) == 0 or real_eval_cb.real_fitness_points[-1][0] != steps):
        final_probe_fit = run_deterministic_eval(model, eval_env)
        with real_eval_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([int(steps), f"{float(final_probe_fit):.10f}"])
        real_eval_cb.real_fitness_points.append((int(steps), float(final_probe_fit)))

    final_fit = run_deterministic_eval(model, eval_env)

    mon = scale_out / "monitor.csv"
    x, y = read_monitor_rewards(mon)
    k = max(1, min(20, int(y.size) if y.size > 0 else 1))
    y_ma = moving_average(y, k)
    x_ma = x[k - 1 :] if y_ma.size > 0 else x

    curve_csv = scale_out / "convergence_curve.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timesteps", "episode_reward", "reward_ma20"])
        for i in range(y.size):
            ma = y_ma[i - k + 1] if i >= k - 1 and y_ma.size > 0 else ""
            w.writerow([int(x[i]), float(y[i]), ma])

    plt.figure(figsize=(8, 4.8))
    if y.size > 0:
        plt.plot(x, y, alpha=0.25, label="Episode reward")
    if y_ma.size > 0 and x_ma.size == y_ma.size:
        plt.plot(x_ma, y_ma, linewidth=2.0, label="MA(20)")
    plt.title(f"PPO Convergence ({cfg.name})")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    curve_png = scale_out / "convergence_curve.png"
    plt.savefig(curve_png, dpi=220)
    plt.close()

    x_fit, y_fit = read_real_fitness_curve(real_eval_csv)
    fitness_curve_png = scale_out / "fitness_convergence_curve.png"
    if x_fit.size > 0:
        plt.figure(figsize=(8, 4.8))
        plt.plot(x_fit, y_fit, marker="o", linewidth=1.8)
        plt.title(f"PPO Real Fitness Convergence ({cfg.name})")
        plt.xlabel("Training timesteps")
        plt.ylabel("Real fitness")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fitness_curve_png, dpi=220)
        plt.close()

    summary = {
        "problem_scale": cfg.name,
        "actual_training_steps": steps,
        "training_wall_clock_min": wall_min,
        "final_policy_fitness": float(final_fit),
        "time_budget_min": budget_min,
        "model_path": str(model_path),
        "monitor_csv": str(mon),
        "convergence_curve_csv": str(curve_csv),
        "convergence_curve_png": str(curve_png),
        "real_fitness_eval_csv": str(real_eval_csv),
        "real_fitness_curve_png": str(fitness_curve_png),
        "real_fitness_eval_points": int(x_fit.size),
    }
    (scale_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default="", help="Path to CED_Schedule.exe")
    parser.add_argument(
        "--out_dir",
        default="outputs/results/ppo_sb3_subprocess_budgeted",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base random seed")
    parser.add_argument("--episode_steps", type=int, default=32, help="RL episode length")
    parser.add_argument(
        "--budget_t100_min", type=float, default=-1.0, help="Override T100 budget minutes"
    )
    parser.add_argument(
        "--budget_t200_min", type=float, default=-1.0, help="Override T200 budget minutes"
    )
    parser.add_argument(
        "--budget_t500_min", type=float, default=-1.0, help="Override T500 budget minutes"
    )
    parser.add_argument(
        "--only_scale",
        default="",
        choices=["", "T100", "T200", "T500"],
        help="Run only one scale for debugging",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = find_exe(root, args.exe)
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    budget_override = {
        "T100": float(args.budget_t100_min),
        "T200": float(args.budget_t200_min),
        "T500": float(args.budget_t500_min),
    }

    selected = [s for s in SCALES if not args.only_scale or s.name == args.only_scale]
    all_summary: List[Dict[str, float]] = []

    print(f"Using exe: {exe}")
    print(f"Output: {out_root}")
    os.environ["OMP_NUM_THREADS"] = "1"

    for idx, cfg in enumerate(selected):
        print(f"\n=== Training PPO (SB3) for {cfg.name} ===")
        one = train_one_scale(
            root=root,
            exe=exe,
            out_root=out_root,
            cfg=cfg,
            budget_min_override=budget_override[cfg.name],
            base_seed=args.seed + idx * 1000,
            episode_steps=args.episode_steps,
        )
        all_summary.append(one)
        print(
            f"{cfg.name}: steps={one['actual_training_steps']}, "
            f"time={one['training_wall_clock_min']:.3f} min, "
            f"final_fitness={one['final_policy_fitness']:.10f}"
        )

    summary_csv = out_root / "ppo_sb3_budgeted_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "problem_scale",
                "actual_training_steps",
                "training_wall_clock_min",
                "final_policy_fitness",
                "time_budget_min",
                "convergence_curve_png",
                "real_fitness_eval_csv",
                "real_fitness_curve_png",
                "real_fitness_eval_points",
            ]
        )
        for row in all_summary:
            w.writerow(
                [
                    row["problem_scale"],
                    row["actual_training_steps"],
                    f"{row['training_wall_clock_min']:.6f}",
                    f"{row['final_policy_fitness']:.10f}",
                    f"{row['time_budget_min']:.2f}",
                    row["convergence_curve_png"],
                    row["real_fitness_eval_csv"],
                    row["real_fitness_curve_png"],
                    row["real_fitness_eval_points"],
                ]
            )
    (out_root / "ppo_sb3_budgeted_summary.json").write_text(
        json.dumps(all_summary, indent=2), encoding="utf-8"
    )
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()
