"""
RL model zoo 학습 (stable-baselines3).

    python rl_zoo/train_zoo.py --env Reach-v0 --algo SAC                 # 한 개 (기본 timesteps는 zoo_config)
    python rl_zoo/train_zoo.py --all --parallel 6                        # zoo_config.RUNS 전체를 병렬로
    python rl_zoo/train_zoo.py --env Push-v0 --algo SAC --timesteps 300000 --seed 1

출력: models_zoo/<env>__<algo>/model.zip, config.json, monitor.csv(에피소드 리턴), eval.json(학습 직후 20 에피소드),
      learning_curve.png
"""
import argparse
import json
import os
import subprocess
import sys
import time

# one BLAS/OpenMP thread per process: several runs train in parallel and MLP policies are tiny
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from rl_zoo.zoo_config import ALGO_KWARGS, MODELS_DIR, RUNS, run_name


def get_algo(name):
    import stable_baselines3 as sb3
    return getattr(sb3, name)


def make_env(env_id, seed=0, monitor_path=None):
    from stable_baselines3.common.monitor import Monitor
    from envs import make_env as _make
    env = _make(env_id)
    env.reset(seed=seed)
    return Monitor(env, filename=str(monitor_path) if monitor_path else None, info_keywords=("success", "dist"))


def evaluate(model, env_id, episodes=20, seed=10_000):
    from envs import make_env as _make
    env = _make(env_id)
    rets, succ, dists, cols = [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done, ret = False, 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            ret += r
            done = term or trunc
        rets.append(ret)
        succ.append(info["success"])
        dists.append(info["dist"])
        cols.append(info["collision"])
    env.close()
    return dict(episodes=episodes, mean_return=float(np.mean(rets)), std_return=float(np.std(rets)),
                success_rate=float(np.mean(succ)), mean_final_dist=float(np.mean(dists)),
                collision_rate=float(np.mean(cols)))


def plot_curve(monitor_csv, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(monitor_csv, skiprows=1)
    if len(df) == 0:
        return
    steps = df["l"].cumsum()
    k = max(1, len(df) // 50)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(steps, df["r"], color="#b9c3d6", lw=0.8)
    axes[0].plot(steps, df["r"].rolling(k, min_periods=1).mean(), color="#2a78d6", lw=2)
    axes[0].set_title("episode return")
    axes[1].plot(steps, df["success"].astype(float).rolling(max(k, 10), min_periods=1).mean() * 100, color="#1baf7a", lw=2)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("success rate (moving avg, %)")
    for a in axes:
        a.set_xlabel("env steps")
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def train_one(env_id, algo, timesteps, seed=0, threads=1):
    import torch
    torch.set_num_threads(threads)
    out = MODELS_DIR / run_name(env_id, algo)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(env_id, seed=seed, monitor_path=out / "monitor.csv")
    kwargs = dict(ALGO_KWARGS[algo])
    if algo == "TD3":
        from stable_baselines3.common.noise import NormalActionNoise
        n = env.action_space.shape[0]
        kwargs["action_noise"] = NormalActionNoise(np.zeros(n), 0.1 * np.ones(n))
    model = get_algo(algo)("MlpPolicy", env, seed=seed, verbose=0, device="cpu", **kwargs)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    train_time = time.time() - t0
    model.save(out / "model.zip")
    ev = evaluate(model, env_id)
    cfg = dict(env_id=env_id, algo=algo, timesteps=timesteps, seed=seed, train_time_s=train_time,
               obs_dim=int(env.observation_space.shape[0]), act_dim=int(env.action_space.shape[0]),
               algo_kwargs={k: (v if not isinstance(v, dict) else v) for k, v in ALGO_KWARGS[algo].items()})
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
    (out / "eval.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
    try:
        plot_curve(out / "monitor.csv", out / "learning_curve.png", f"{env_id} / {algo}")
    except Exception as e:  # plotting must never fail a run
        print(f"[{env_id}/{algo}] curve plot failed: {e}")
    env.close()
    print(f"[{env_id}/{algo}] {timesteps} steps in {train_time/60:.1f} min | success {ev['success_rate']*100:.0f}% "
          f"return {ev['mean_return']:.1f}", flush=True)
    return ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str)
    p.add_argument("--algo", type=str, choices=list(ALGO_KWARGS))
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1, help="torch threads per run")
    p.add_argument("--all", action="store_true", help="train every run in zoo_config.RUNS")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    if args.all:
        runs = [(e, a, t) for e, a, t in RUNS
                if not (args.skip_existing and (MODELS_DIR / run_name(e, a) / "model.zip").exists())]
        print(f"training {len(runs)} runs, {args.parallel} in parallel", flush=True)
        t0 = time.time()

        def launch(run):
            e, a, t = run
            cmd = [sys.executable, __file__, "--env", e, "--algo", a, "--timesteps", str(t),
                   "--seed", str(args.seed), "--threads", str(args.threads)]
            log = MODELS_DIR / run_name(e, a)
            log.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            with open(log / "train_log.txt", "w", encoding="utf-8") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
            print(f"finished {e}/{a} (exit {r.returncode}) at {(time.time()-t0)/60:.1f} min", flush=True)
            return r.returncode

        with ThreadPoolExecutor(args.parallel) as ex:
            codes = list(ex.map(launch, runs))
        print(f"all done in {(time.time()-t0)/60:.1f} min, failures: {sum(c != 0 for c in codes)}")
        return

    timesteps = args.timesteps or dict(((e, a), t) for e, a, t in RUNS).get((args.env, args.algo), 100_000)
    train_one(args.env, args.algo, timesteps, seed=args.seed, threads=args.threads)


if __name__ == "__main__":
    main()
