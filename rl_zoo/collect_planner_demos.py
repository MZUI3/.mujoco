"""
학습 궤적 플래너(trajectory_planning/)로 ReachMultiObstacle-v0 데모를 수집한다.

문제 N개를 플래너의 ProblemSampler로 뽑고, 각 문제의 B-spline 관절 궤적을 `plan_learned`(목표 IK + 신경망 1회 forward)로
계산한 뒤, tp_sim.SimScene.rollout 대신 **환경의 step()**으로 추종한다. 매 제어 주기(10 ms)마다
SimScene._control과 같은 computed-torque 식(M(q)·(qdd_d + kp·e + kd·ė) + bias − passive)으로 토크를 계산하고,
모터 gear로 나눠 [-1, 1]로 정규화한 값을 action으로 기록한다. 그래서 (obs, action) 쌍은 환경의 관측·행동 공간과 같다.

    python rl_zoo/collect_planner_demos.py                      # 2000문제 -> data/planner_demos.npz
    python rl_zoo/collect_planner_demos.py --n 500 --keep-all   # 실패 에피소드도 저장

출력 npz: obs (T, 29), actions (T, 4), episode_starts (T,) bool, 그리고 에피소드별 success/collision/final_dist/return,
문제(q0, goal, obstacles)와 제어점(ctrl_pts). 기본은 성공한 에피소드(3 cm 이내 + 접촉 0회)만 저장한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trajectory_planning"))

import mujoco
import numpy as np
import torch

from envs import make_env

ENV_ID = "ReachMultiObstacle-v0"
PLANNER_PATH = ROOT / "models_tp" / "planner.pt"
DEMO_PATH = ROOT / "data" / "planner_demos.npz"


class ComputedTorqueTracker:
    """Computed-torque controller (same law as tp_sim.SimScene._control) that returns the *normalised* action
    (torque / motor gear, clipped to [-1, 1]) the environment expects.

    The reference is a B-spline (ctrl_pts, basis) time-scaled to `duration` seconds; after that the final
    configuration is held with zero velocity/acceleration.

    mode="time": the reference at step k is the plan at time k*dt (SimScene.rollout behaviour).
    mode="path": path following. The reference is the plan sample closest (in joint space) to the current
        q, searched forward from the previous one within `window` steps, plus `lookahead` steps. Without
        perturbations this equals "time"; under perturbations the expert tracks the path geometry instead of
        a clock, so its action is a function of the state (and the plan) alone, which a state-only policy can
        imitate (behaviour cloning of a clock-indexed controller is not identifiable from the observation)."""

    def __init__(self, env, kp=400.0, kd=40.0, mode="time", lookahead=1, window=25):
        assert mode in ("time", "path")
        self.env = env.unwrapped
        m = self.env.model
        self.kp, self.kd, self.mode = kp, kd, mode
        self.lookahead, self.window = int(lookahead), int(window)
        self.gear = m.actuator_gear[self.env.arm_act, 0].copy()
        self._M = np.zeros((m.nv, m.nv))
        self._k_ref = 0

    def set_reference(self, ctrl_pts, basis, duration=None):
        T = float(duration or basis.duration)
        n = self.env.max_steps
        ts = np.arange(n) * self.env.dt
        s = np.clip(ts / T, 0.0, 1.0)
        moving = (ts < T)[:, None]
        scale = basis.duration / T                      # basis derivatives are per second of the basis' own duration
        self.Q = basis.matrix(s, 0) @ ctrl_pts
        self.QD = basis.matrix(s, 1) @ ctrl_pts * scale * moving
        self.QDD = basis.matrix(s, 2) @ ctrl_pts * scale ** 2 * moving
        self._k_ref = 0

    def action(self, k):
        """Normalised torque action for control step k (uses the env's current state)."""
        env, m, d = self.env, self.env.model, self.env.data
        q, qd = d.qpos[env.arm_q], d.qvel[env.arm_v]
        if self.mode == "path":
            lo = self._k_ref
            hi = min(lo + self.window, len(self.Q) - 1)
            j = lo + int(np.argmin(np.linalg.norm(self.Q[lo:hi + 1] - q, axis=1)))
            self._k_ref = j
            k = j + self.lookahead
        k = min(k, len(self.Q) - 1)
        mujoco.mj_forward(m, d)                          # fresh bias/passive forces for the current state
        mujoco.mj_fullM(m, d, self._M)
        Ma = self._M[np.ix_(env.arm_v, env.arm_v)]
        acc = self.QDD[k] + self.kp * (self.Q[k] - q) + self.kd * (self.QD[k] - qd)
        tau = Ma @ acc + d.qfrc_bias[env.arm_v] - d.qfrc_passive[env.arm_v]
        return np.clip(tau / self.gear, -1.0, 1.0)


def load_planner_stack(env):
    """(tp_core module, kin, cost, net, cfg) built on the environment's own MuJoCo model."""
    import tp_core
    kin = tp_core.ArmKinematics(env.unwrapped.model)
    net, cfg = tp_core.load_planner(PLANNER_PATH, kin)
    cost = tp_core.TrajectoryCost(kin, cfg)
    return tp_core, kin, cost, net, cfg


def plan_batched(tp_core, net, cost, q0, goal, obs, batch=256):
    out = []
    for i in range(0, len(q0), batch):
        ctrl, _ = tp_core.plan_learned(net, cost, q0[i:i + batch], goal[i:i + batch], obs[i:i + batch])
        out.append(ctrl)
    return torch.cat(out).numpy().astype(np.float64)


def track_episode(env, tracker, problem, ctrl_pts, basis, duration=None, exec_noise=0.0, rng=None, render_cb=None):
    """Replay `problem` in env and follow the plan with computed torque. Returns (obs list, action list, info, return)."""
    tracker.set_reference(ctrl_pts, basis, duration)
    obs, info = env.reset(options=dict(problem=problem))
    O, A, ret = [], [], 0.0
    for k in range(env.unwrapped.max_steps):
        a = tracker.action(k)
        O.append(np.asarray(obs, dtype=np.float32))
        A.append(a.astype(np.float32))
        a_exec = a if exec_noise <= 0 else np.clip(a + rng.normal(0, exec_noise, size=a.shape), -1, 1)
        if render_cb is not None:
            render_cb(k, info)
        obs, r, term, trunc, info = env.step(a_exec)
        ret += r
        if term or trunc:
            break
    return O, A, info, ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000, help="number of problems")
    p.add_argument("--seed", type=int, default=0, help="problem sampler seed")
    p.add_argument("--out", type=str, default=str(DEMO_PATH))
    p.add_argument("--duration", type=float, default=None,
                   help="seconds the 2 s plan is squeezed into (default: planner duration); the rest of the episode holds the goal")
    p.add_argument("--kp", type=float, default=400.0)
    p.add_argument("--kd", type=float, default=40.0)
    p.add_argument("--exec-noise", type=float, default=0.0,
                   help="Gaussian noise added to the executed action (DART-style); the clean expert action is recorded")
    p.add_argument("--track", type=str, default="time", choices=["time", "path"],
                   help="reference selection: plan at time k*dt, or closest plan sample (path following)")
    p.add_argument("--lookahead", type=int, default=1, help="path mode: reference = closest sample + this many steps")
    p.add_argument("--keep-all", action="store_true", help="also store failed episodes (default: successes only)")
    args = p.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    env = make_env(ENV_ID)
    tp_core, kin, cost, net, cfg = load_planner_stack(env)
    tracker = ComputedTorqueTracker(env, kp=args.kp, kd=args.kd, mode=args.track, lookahead=args.lookahead)
    rng = np.random.default_rng(args.seed + 1)

    t0 = time.perf_counter()
    q0, goal, obs = tp_core.ProblemSampler(kin, cost, seed=args.seed).sample(args.n)
    t_sample = time.perf_counter() - t0
    t0 = time.perf_counter()
    ctrl = plan_batched(tp_core, net, cost, q0, goal, obs)
    t_plan = time.perf_counter() - t0
    print(f"sampled {args.n} problems in {t_sample:.1f} s, planned in {t_plan:.1f} s "
          f"({1000 * t_plan / args.n:.1f} ms/problem batched)", flush=True)
    q0, goal, obs = q0.numpy().astype(np.float64), goal.numpy().astype(np.float64), obs.numpy().astype(np.float64)

    O, A, starts = [], [], []
    ep_success, ep_coll, ep_dist, ep_ret, kept = [], [], [], [], []
    t0 = time.perf_counter()
    for i in range(args.n):
        o, a, info, ret = track_episode(env, tracker, (q0[i], goal[i], obs[i]), ctrl[i], cost.basis,
                                        duration=args.duration, exec_noise=args.exec_noise, rng=rng)
        ep_success.append(bool(info["success"]))
        ep_coll.append(bool(info["collision"]))
        ep_dist.append(float(info["dist"]))
        ep_ret.append(float(ret))
        if info["success"] or args.keep_all:
            kept.append(i)
            starts += [True] + [False] * (len(o) - 1)
            O += o
            A += a
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{args.n}  success so far {100 * np.mean(ep_success):.1f}%  "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    env.close()

    O, A = np.asarray(O, dtype=np.float32), np.asarray(A, dtype=np.float32)
    kept = np.asarray(kept, dtype=np.int64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, obs=O, actions=A, episode_starts=np.asarray(starts, dtype=bool),
        episode_success=np.asarray(ep_success)[kept], episode_collision=np.asarray(ep_coll)[kept],
        episode_final_dist=np.asarray(ep_dist)[kept], episode_return=np.asarray(ep_ret)[kept],
        q0=q0[kept], goal=goal[kept], obstacles=obs[kept], ctrl_pts=ctrl[kept], kept_index=kept,
        meta=json.dumps(dict(env_id=ENV_ID, n_problems=args.n, seed=args.seed, planner=str(PLANNER_PATH.relative_to(ROOT)),
                             duration=args.duration or cfg.duration, kp=args.kp, kd=args.kd, exec_noise=args.exec_noise,
                             track=args.track, lookahead=args.lookahead,
                             keep_all=args.keep_all, all_success_rate=float(np.mean(ep_success)),
                             all_collision_rate=float(np.mean(ep_coll)), all_mean_final_dist=float(np.mean(ep_dist)),
                             all_mean_return=float(np.mean(ep_ret)))))
    sat = float(np.mean(np.abs(A) >= 0.999))
    print(f"planner tracked through env.step: success {100 * np.mean(ep_success):.1f}%  collision {100 * np.mean(ep_coll):.1f}%  "
          f"final dist {np.mean(ep_dist) * 100:.2f} cm  return {np.mean(ep_ret):.1f}")
    print(f"saved {out}: {len(kept)} episodes, {len(O)} transitions, obs {O.shape[1]}, act {A.shape[1]}, "
          f"|action| saturated {100 * sat:.1f}% of entries  ({(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
