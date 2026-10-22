"""
계획된 궤적을 MuJoCo로 실행하며 시각화.

    - 기본: 오프스크린 렌더링 -> 방법별 화면을 나란히 붙인 GIF 저장 (headless 서버에서도 동작)
      (얇은 선 = 계획된 EE 경로, 굵은 선 = 실제 추종된 EE 경로, 초록 구 = 목표, 빨간 기둥 = 장애물)
    - --viewer : MuJoCo 인터랙티브 뷰어에서 실시간 재생 (GUI 필요)

Usage:
    python trajectory_planning/visualize_planner.py --n 3 --methods learned linear
    python trajectory_planning/visualize_planner.py --viewer --methods learned
"""
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from tp_core import ROOT, ProblemSampler, TrajectoryCost, load_planner
from tp_sim import METHOD_COLORS, SimScene, add_polyline, default_camera, hex_rgba
from evaluate_planner import METHODS, make_planners


def planned_ee_path(sim, cost, ctrl):
    with torch.no_grad():
        q = cost.B0 @ ctrl
        return sim.kin.ee_pos(q).numpy()


def record_gif(sim, cost, planners, methods, q0, goal, obs, path, width=480, height=400, fps=30):
    renderer = mujoco.Renderer(sim.model, height, width)
    cam = default_camera()
    frame_every = max(1, int(round(1.0 / fps / sim.model.opt.timestep)))
    font = ImageFont.load_default(size=14)
    per_method = {}
    for m in methods:
        ctrl, dt = planners[m](q0[None], goal[None], obs[None])
        plan_pts = planned_ee_path(sim, cost, ctrl)[0]
        color = hex_rgba(METHOD_COLORS[m])
        frames = []

        def on_frame(t, ee_hist, m=m):
            renderer.update_scene(sim.data, cam)
            add_polyline(renderer.scene, plan_pts, color * [1, 1, 1, 0.5], width=0.003)
            add_polyline(renderer.scene, ee_hist, color, width=0.006, stride=10)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, 22], fill=(255, 255, 255))
            draw.text((8, 3), f"{m}   plan {dt*1000:.0f} ms   t={t:.2f}s", fill=(20, 20, 20), font=font)
            frames.append(img)

        metrics, _ = sim.rollout(ctrl[0].numpy(), cost.basis, q0.numpy(), goal.numpy(), obs.numpy(),
                                 frame_cb=on_frame, frame_every=frame_every)
        status = "SUCCESS" if metrics["success"] else ("COLLISION" if metrics["collision"] else "MISSED GOAL")
        for img in frames[-int(fps * 0.5):]:
            d = ImageDraw.Draw(img)
            d.rectangle([0, height - 24, width, height], fill=(255, 255, 255))
            d.text((8, height - 21), f"{status}  err={metrics['final_err_mm']:.1f}mm  "
                   f"clearance={metrics['min_obstacle_dist_cm']:.1f}cm",
                   fill=(0, 131, 0) if metrics["success"] else (227, 73, 72), font=font)
        per_method[m] = (frames, metrics)

    n = max(len(f) for f, _ in per_method.values())
    combined = []
    for k in range(n):
        canvas = Image.new("RGB", (width * len(methods), height), (255, 255, 255))
        for j, m in enumerate(methods):
            frames = per_method[m][0]
            canvas.paste(frames[min(k, len(frames) - 1)], (j * width, 0))
        combined.append(canvas)
    hold = [combined[-1]] * fps  # hold the last frame for 1 s
    combined[0].save(path, save_all=True, append_images=combined[1:] + hold, duration=int(1000 / fps), loop=0)
    renderer.close()
    return {m: v[1] for m, v in per_method.items()}


def run_viewer(sim, cost, planners, methods, problems):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        cam = default_camera()
        viewer.cam.lookat[:], viewer.cam.distance = cam.lookat, cam.distance
        viewer.cam.azimuth, viewer.cam.elevation = cam.azimuth, cam.elevation
        for i, (q0, goal, obs) in enumerate(problems):
            for m in methods:
                if not viewer.is_running():
                    return
                ctrl, _ = planners[m](q0[None], goal[None], obs[None])
                plan_pts = planned_ee_path(sim, cost, ctrl)[0]
                color = hex_rgba(METHOD_COLORS[m])
                print(f"problem {i} - {m}")
                wall0 = time.time()

                def on_frame(t, ee_hist):
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        add_polyline(viewer.user_scn, plan_pts, color * [1, 1, 1, 0.5], width=0.003)
                        add_polyline(viewer.user_scn, ee_hist, color, width=0.006, stride=10)
                    viewer.sync()
                    lag = t - (time.time() - wall0)   # real-time playback
                    if lag > 0:
                        time.sleep(lag)

                metrics, _ = sim.rollout(ctrl[0].numpy(), cost.basis, q0.numpy(), goal.numpy(), obs.numpy(),
                                         frame_cb=on_frame, frame_every=8)
                print(f"   success={metrics['success']} err={metrics['final_err_mm']:.1f}mm "
                      f"collision={metrics['collision']} clearance={metrics['min_obstacle_dist_cm']:.1f}cm")
                time.sleep(0.8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=str(ROOT / "models_tp" / "planner.pt"))
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026, help="same seed as evaluate_planner -> same problems")
    p.add_argument("--start", type=int, default=0, help="index of the first test problem to show")
    p.add_argument("--methods", nargs="+", default=["learned", "linear"], choices=METHODS)
    p.add_argument("--out", type=str, default=str(ROOT / "results_tp" / "videos"))
    p.add_argument("--viewer", action="store_true", help="interactive MuJoCo viewer instead of GIFs")
    args = p.parse_args()

    torch.manual_seed(0)
    sim = SimScene()
    net, cfg = load_planner(args.model, sim.kin)
    cost = TrajectoryCost(sim.kin, cfg)
    q0, goal, obs = ProblemSampler(sim.kin, cost, seed=args.seed).sample(args.start + args.n)
    planners = make_planners(net, cost)
    problems = [(q0[i], goal[i], obs[i]) for i in range(args.start, args.start + args.n)]
    for m in args.methods:  # warm-up so the displayed planning time excludes first-call overhead
        planners[m](q0[:1], goal[:1], obs[:1])

    if args.viewer:
        run_viewer(sim, cost, planners, args.methods, problems)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, (a, b, c) in enumerate(problems):
        idx = args.start + k
        path = out / f"problem_{idx:03d}_{'_vs_'.join(m.replace('+', '_') for m in args.methods)}.gif"
        res = record_gif(sim, cost, planners, args.methods, a, b, c, path)
        print(f"saved {path}  " + "  ".join(f"{m}: {'ok' if r['success'] else 'fail'}" for m, r in res.items()))


if __name__ == "__main__":
    main()
