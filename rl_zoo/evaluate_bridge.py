"""
플래너 ↔ RL 브리지 비교: ReachMultiObstacle-v0 의 같은 20문제(평가 seed 10000~)에서

    (a) 무작위 행동          (b) SAC 100k (models_zoo/ReachMultiObstacle-v0__SAC)
    (c) BC on planner demos  (models_zoo/ReachMultiObstacle-v0__BC)
    (d) 학습 플래너 자체     tp_sim.SimScene.rollout (computed-torque 2 ms 제어, 참고값)
    (e) 학습 플래너 + env    같은 B-spline 을 환경 step()으로 추종 (데모 수집과 같은 10 ms computed-torque)

을 성공률(3 cm 이내 + 접촉 0회)·충돌률·최종 거리·리턴으로 비교한다.

    python rl_zoo/evaluate_bridge.py --episodes 20
출력: results_zoo/planner_rl_bridge.md, planner_rl_bridge.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trajectory_planning"))

import numpy as np
import torch

from envs import make_env
from rl_zoo.collect_planner_demos import ComputedTorqueTracker, ENV_ID, load_planner_stack, plan_batched, track_episode
from rl_zoo.evaluate_zoo import load_model, model_file, rollouts
from rl_zoo.zoo_config import MODELS_DIR, RESULTS_DIR


def summarize(succ, coll, dist, ret=None):
    out = dict(success_rate=float(np.mean(succ)), collision_rate=float(np.mean(coll)), mean_final_dist=float(np.mean(dist)),
               episodes=len(succ))
    if ret is not None:
        out.update(mean_return=float(np.mean(ret)), std_return=float(np.std(ret)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--sac-run", type=str, default=f"{ENV_ID}__SAC")
    p.add_argument("--bc-run", type=str, default=f"{ENV_ID}__BC")
    p.add_argument("--naive-run", type=str, default=f"{ENV_ID}__BC_naive", help="pure BC ablation (obs only, no DAgger)")
    p.add_argument("--duration", type=float, default=None, help="plan time-scaling used for (e), as in collect_planner_demos")
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "planner_rl_bridge.md"))
    args = p.parse_args()

    env = make_env(ENV_ID)
    rows = []

    # the evaluation problems: what env.reset(seed=seed+ep) produces (same as evaluate_zoo.rollouts)
    problems = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        problems.append(env.problem)
    n_poles = [int((pr[2][:, 3] > 0).sum()) for pr in problems]

    print("(a) random", flush=True)
    r = rollouts(ENV_ID, lambda obs, e: e.action_space.sample(), args.episodes, args.seed)
    rows.append(dict(key="random", name="무작위 행동", data="-", note="기준선", **r))

    for key, run, label in (("SAC", args.sac_run, "SAC 100k"), ("BC_naive", args.naive_run, "BC (순수 행동 복제, obs만)"),
                            ("BC", args.bc_run, "BC + DAgger (플래너 데모 + 전문가 라벨링)")):
        d = MODELS_DIR / run
        if model_file(d) is None:
            print(f"({key}) {run} not found, skipped", flush=True)
            continue
        print(f"({key}) {run}", flush=True)
        model, cfg = load_model(d)
        r = rollouts(ENV_ID, lambda obs, e, m=model: m.predict(obs, deterministic=True)[0], args.episodes, args.seed,
                     context_wrapper=cfg.get("episode_context", False))
        if key == "SAC":
            data = f"{cfg['timesteps'] // 1000}k env steps, {cfg.get('train_time_s', 0) / 60:.0f} min"
            note = f"`{d.relative_to(ROOT).as_posix()}`"
        else:
            ak = cfg.get("algo_kwargs", {})
            data = (f"데모 {ak.get('n_demo_episodes', '?')} 에피소드 / {ak.get('n_demo_transitions', 0) // 1000}k 전이"
                    + (f" + DAgger {ak['dagger_iters']}×{ak['dagger_episodes']} 에피소드 라벨링" if ak.get("dagger_iters") else "")
                    + f", {cfg.get('train_time_s', 0) / 60:.0f} min")
            note = (f"`{d.relative_to(ROOT).as_posix()}`, MLP {ak.get('hidden')}, 입력 "
                    + ("obs + 시간 + q0" if cfg.get("episode_context") else "obs만"))
        rows.append(dict(key=key, name=label, data=data, note=note, **r))

    # planner on the same problems
    tp_core, kin, cost, net, cfg = load_planner_stack(env)
    q0 = torch.tensor(np.array([pr[0] for pr in problems]), dtype=torch.float32)
    goal = torch.tensor(np.array([pr[1] for pr in problems]), dtype=torch.float32)
    obs = torch.tensor(np.array([pr[2] for pr in problems]), dtype=torch.float32)
    ctrl = plan_batched(tp_core, net, cost, q0, goal, obs)
    lat = []
    for i in range(args.episodes):
        _, dt = tp_core.plan_learned(net, cost, q0[i:i + 1], goal[i:i + 1], obs[i:i + 1])
        lat.append(dt * 1000)

    print("(d) planner, SimScene.rollout", flush=True)
    import tp_sim
    sim = tp_sim.SimScene(env.unwrapped.model)
    s3, s2, c, dist = [], [], [], []
    for i, pr in enumerate(problems):
        m3, _ = sim.rollout(ctrl[i], cost.basis, *pr, success_tol=0.03)
        s3.append(m3["success"])
        s2.append(m3["final_err_mm"] < 20 and not m3["collision"])
        c.append(m3["collision"])
        dist.append(m3["final_err_mm"] / 1000)
    rows.append(dict(key="planner_sim", name="학습 플래너 (SimScene.rollout, 참고)", data="데모 없음 (self-supervised)",
                     note=f"2 ms computed-torque, 0.5 s settle; 플래너 기준(2 cm) 성공 {100 * np.mean(s2):.0f}%, "
                          f"계획 {np.median(lat):.0f} ms/문제",
                     **summarize(s3, c, dist)))

    print("(e) planner tracked through env.step", flush=True)
    tracker = ComputedTorqueTracker(env)
    s, c, dist, ret = [], [], [], []
    for i, pr in enumerate(problems):
        _, _, info, R = track_episode(env, tracker, pr, ctrl[i], cost.basis, duration=args.duration)
        s.append(info["success"])
        c.append(info["collision"])
        dist.append(info["dist"])
        ret.append(R)
    rows.append(dict(key="planner_env", name="학습 플래너 + computed torque (env.step, 데모 생성기)", data="데모 없음 (self-supervised)",
                     note="10 ms 제어, BC 데모와 같은 추종기 = BC의 상한", **summarize(s, c, dist, ret)))
    env.close()

    # context: the fixed-pole task result from the zoo table
    ctx = None
    zs = RESULTS_DIR / "zoo_summary.json"
    if zs.exists():
        z = json.loads(zs.read_text(encoding="utf-8"))
        for r in z.get("zoo", []):
            if r["env_id"] == "ReachObstacle-v0" and r["algo"] == "SAC":
                ctx = r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(dict(env_id=ENV_ID, episodes=args.episodes, seed=args.seed, n_poles=n_poles,
                                                        rows=rows, context_reach_obstacle_sac=ctx), indent=1, ensure_ascii=False),
                                        encoding="utf-8")

    def pct(x):
        return f"{100 * x:.0f}%"
    lines = [f"# 플래너 ↔ RL 브리지: `{ENV_ID}`", "",
             f"같은 {args.episodes}문제(평가 seed {args.seed}~, 기둥 수 1/2/3개 = "
             f"{n_poles.count(1)}/{n_poles.count(2)}/{n_poles.count(3)}), 결정적 정책, 200 step(2 s). "
             "성공 = 마지막 step에서 목표 3 cm 이내 **그리고** 에피소드 내 팔-기둥/바닥 접촉 0회.", "",
             "| 정책 | 학습 데이터 / 예산 | 성공률 | 충돌률 | 최종 거리 (cm) | 평균 리턴 | 비고 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        ret = f"{r['mean_return']:.1f} ± {r['std_return']:.1f}" if "mean_return" in r else "-"
        bold = r["key"] in ("BC", "BC_naive", "SAC")
        sr = f"**{pct(r['success_rate'])}**" if bold else pct(r["success_rate"])
        lines.append(f"| {r['name']} | {r['data']} | {sr} | {pct(r['collision_rate'])} | {100 * r['mean_final_dist']:.1f} | {ret} | {r['note']} |")
    lines += [""]
    if ctx is not None:
        lines += [f"참고: 기둥 1개 고정 크기(4 cm × 30 cm)인 `ReachObstacle-v0`에서 SAC {ctx['timesteps'] // 1000}k는 "
                  f"성공 {pct(ctx['success_rate'])}, 충돌 {pct(ctx['collision_rate'])} 였습니다 (`results_zoo/zoo_summary.md`).", ""]
    lines += ["재현:", "", "```bash",
              "python rl_zoo/collect_planner_demos.py                               # 플래너 데모 2000문제 -> data/planner_demos.npz",
              "python rl_zoo/train_bc.py                                            # BC + DAgger -> models_zoo/ReachMultiObstacle-v0__BC",
              "python rl_zoo/train_bc.py --dagger 0 --no-context --run ReachMultiObstacle-v0__BC_naive --algo-label BC_naive   # 순수 BC 대조군",
              "python rl_zoo/train_zoo.py --env ReachMultiObstacle-v0 --algo SAC    # SAC 100k",
              f"python rl_zoo/evaluate_bridge.py --episodes {args.episodes}                           # 이 표",
              "python rl_zoo/visualize_zoo.py --run ReachMultiObstacle-v0__BC       # GIF", "```", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
