"""
학습된 플래너를 사용자가 지정한 문제(시작 관절각, 목표 위치, 기둥 장애물)로 MuJoCo에서 실행.

    python trajectory_planning/simulate_planner.py --goal 0.35 -0.25 0.20 \
        --obstacle 0.30 0.05 0.04 0.30 --obstacle 0.15 -0.30 0.05 0.25 --viewer
    python trajectory_planning/simulate_planner.py --goal 0.35 -0.25 0.20 --obstacle 0.30 0.05 0.04 0.30 \
        --gif results_tp/videos/custom.gif --methods learned linear

옵션을 생략하면 시작 자세는 0, 장애물은 1개(기본 위치)이다. 결과(성공 여부, 최종 오차, 최소 거리)를 출력하고
계획한 관절 궤적을 --save-npz 로 저장할 수 있다(키: t, q, qd, qdd, ctrl_pts; 다른 제어기에서 추종할 때 사용).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from tp_core import MAX_OBS, ROOT, TrajectoryCost, load_planner, obstacle_mask
from tp_sim import SimScene
from evaluate_planner import METHODS, make_planners
from visualize_planner import record_gif, run_viewer


def build_problem(args, sim, cost):
    q0 = torch.tensor(np.deg2rad(args.q0), dtype=torch.float32)
    goal = torch.tensor(args.goal, dtype=torch.float32)
    obs = torch.zeros(MAX_OBS, 4)
    poles = args.obstacle or [[0.30, 0.05, 0.04, 0.30]]
    if len(poles) > MAX_OBS:
        raise SystemExit(f"at most {MAX_OBS} obstacles are supported")
    for j, pole in enumerate(poles):
        obs[j] = torch.tensor(pole, dtype=torch.float32)

    # sanity checks (the planner was trained on problems that satisfy these)
    with torch.no_grad():
        clr0 = float(cost.clearance(q0[None, None], obs[None]).min())
        ee0 = sim.kin.ee_pos(q0[None])[0]
    if clr0 < 0:
        print(f"warning: the start configuration is in collision (clearance {clr0*100:.1f} cm)")
    r_goal = float(torch.linalg.norm(goal[:2]))
    if not (0.22 < r_goal < 0.62 and 0.05 < goal[2] < 0.45):
        print("warning: goal outside the training workspace (radius 0.22-0.62 m, height 0.05-0.45 m)")
    for j in range(len(poles)):
        x, y, r, h = obs[j].tolist()
        if np.hypot(float(goal[0]) - x, float(goal[1]) - y) < r + 0.09 and float(goal[2]) < h + 0.09:
            print(f"warning: goal is very close to obstacle {j}; the gripper may not fit")
    print(f"start EE {ee0.numpy().round(3)} -> goal {goal.numpy().round(3)}, "
          f"{int(obstacle_mask(obs).sum())} obstacle(s)")
    return q0, goal, obs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default=str(ROOT / "models_tp" / "planner.pt"))
    p.add_argument("--q0", type=float, nargs=4, default=[0, 0, 0, 0], metavar="DEG",
                   help="start joint angles (shoulder_pan, shoulder_lift, elbow, wrist_roll) in degrees")
    p.add_argument("--goal", type=float, nargs=3, default=[0.35, -0.25, 0.20], metavar=("X", "Y", "Z"),
                   help="goal end-effector position in metres")
    p.add_argument("--obstacle", type=float, nargs=4, action="append", metavar=("X", "Y", "R", "H"),
                   help=f"pole obstacle: centre x y, radius, height (m); repeat up to {MAX_OBS} times")
    p.add_argument("--methods", nargs="+", default=["learned"], choices=METHODS)
    p.add_argument("--viewer", action="store_true", help="play back in the interactive MuJoCo viewer")
    p.add_argument("--gif", type=str, default=None, help="save a side-by-side GIF to this path")
    p.add_argument("--save-npz", type=str, default=None, help="save the planned joint trajectory (first method)")
    args = p.parse_args()

    torch.manual_seed(0)
    sim = SimScene()
    net, cfg = load_planner(args.model, sim.kin)
    cost = TrajectoryCost(sim.kin, cfg)
    planners = make_planners(net, cost)
    q0, goal, obs = build_problem(args, sim, cost)
    for m in args.methods:  # warm-up so the reported planning time excludes first-call overhead
        planners[m](q0[None], goal[None], obs[None])

    if args.viewer:
        run_viewer(sim, cost, planners, args.methods, [(q0, goal, obs)])
        return
    if args.gif:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        res = record_gif(sim, cost, planners, args.methods, q0, goal, obs, args.gif)
        print(f"saved {args.gif}")
    else:
        res = {}
        for m in args.methods:
            ctrl, dt = planners[m](q0[None], goal[None], obs[None])
            res[m], _ = sim.rollout(ctrl[0].numpy(), cost.basis, q0.numpy(), goal.numpy(), obs.numpy())
            res[m]["plan_ms"] = dt * 1000
    for m, r in res.items():
        print(f"[{m}] success={r['success']} final error={r['final_err_mm']:.1f} mm "
              f"min obstacle distance={r['min_obstacle_dist_cm']:.1f} cm collision={r['collision']}"
              + (f" plan={r['plan_ms']:.0f} ms" if "plan_ms" in r else ""))

    if args.save_npz:
        m = args.methods[0]
        ctrl, _ = planners[m](q0[None], goal[None], obs[None])
        ctrl = ctrl[0].numpy()
        t = np.arange(0, cfg.duration + 1e-9, sim.model.opt.timestep)
        s = np.clip(t / cfg.duration, 0, 1)
        np.savez(args.save_npz, t=t, q=cost.basis.matrix(s, 0) @ ctrl, qd=cost.basis.matrix(s, 1) @ ctrl,
                 qdd=cost.basis.matrix(s, 2) @ ctrl, ctrl_pts=ctrl, joint_names=np.array(
                     ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll"]))
        print(f"saved planned trajectory ({m}) to {args.save_npz}")


if __name__ == "__main__":
    main()
