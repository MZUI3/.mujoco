"""
zoo 모델을 MuJoCo에서 실행하며 시각화.

    python rl_zoo/visualize_zoo.py --run Reach-v0__SAC --episodes 2               # GIF (headless OK)
    python rl_zoo/visualize_zoo.py --run Push-v0__SAC --viewer                     # 인터랙티브 뷰어
    python rl_zoo/visualize_zoo.py --all                                           # 모든 모델 1 에피소드씩 GIF
    python rl_zoo/visualize_zoo.py --env Reach-v0 --random --episodes 1            # 무작위 정책 보기

GIF: results_zoo/videos/<run>.gif. 화면 위에 과제/알고리즘/step/거리, 마지막에 SUCCESS/FAIL 표시.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs import make_env
from rl_zoo.zoo_config import MODELS_DIR, RESULTS_DIR


def load_policy(run):
    import stable_baselines3 as sb3
    d = MODELS_DIR / run
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    model = getattr(sb3, cfg["algo"]).load(d / "model.zip", device="cpu")
    return cfg["env_id"], cfg["algo"], lambda obs: model.predict(obs, deterministic=True)[0]


def record(env_id, label, policy, episodes, path, seed=10_000, fps=25, every=4):
    env = make_env(env_id, render_mode="rgb_array")
    font = ImageFont.load_default(size=14)
    frames = []
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done, k, ret = False, 0, 0.0
        ep_frames = []
        while not done:
            if k % every == 0:
                img = Image.fromarray(env.render())
                dr = ImageDraw.Draw(img)
                dr.rectangle([0, 0, img.width, 22], fill=(255, 255, 255))
                dr.text((8, 3), f"{env_id}  {label}  ep {ep+1}  t={k*env.dt:.2f}s  dist={info['dist']:.3f}",
                        fill=(20, 20, 20), font=font)
                ep_frames.append(img)
            obs, r, term, trunc, info = env.step(policy(obs))
            ret += r
            k += 1
            done = term or trunc
        status = "SUCCESS" if info["success"] else "FAIL"
        for img in ep_frames[-fps // 2:]:
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, img.height - 24, img.width, img.height], fill=(255, 255, 255))
            dr.text((8, img.height - 21), f"{status}  return={ret:.1f}  final dist={info['dist']:.3f}"
                    + ("  collision" if info["collision"] else ""),
                    fill=(0, 131, 0) if info["success"] else (227, 73, 72), font=font)
        frames += ep_frames + [ep_frames[-1]] * (fps // 2)
        results.append(dict(success=info["success"], ret=ret, dist=info["dist"]))
    env.close()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return results


def view(env_id, policy, episodes, seed=10_000):
    env = make_env(env_id, render_mode="human")
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done, ret, t0 = False, 0.0, time.time()
        k = 0
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            ret += r
            k += 1
            env.render()
            lag = k * env.dt - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)
            done = term or trunc
        print(f"episode {ep+1}: success={info['success']} return={ret:.1f} dist={info['dist']:.3f}")
        time.sleep(0.5)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, help="models_zoo/<run> directory name, e.g. Reach-v0__SAC")
    p.add_argument("--env", type=str, help="with --random: environment id")
    p.add_argument("--random", action="store_true", help="random policy instead of a model")
    p.add_argument("--all", action="store_true", help="one GIF per trained model")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "videos"))
    args = p.parse_args()
    out = Path(args.out)

    if args.all:
        for d in sorted(MODELS_DIR.iterdir()):
            if not (d / "model.zip").exists():
                continue
            env_id, algo, pol = load_policy(d.name)
            res = record(env_id, algo, pol, args.episodes, out / f"{d.name}.gif", seed=args.seed)
            print(f"saved {out / (d.name + '.gif')}  " + " ".join("ok" if r["success"] else "fail" for r in res))
        return
    if args.random:
        env_id, label = args.env, "random"
        rng = np.random.default_rng(0)
        space = make_env(env_id).action_space
        pol = lambda obs: rng.uniform(-1, 1, size=space.shape).astype(np.float32)
        name = f"{env_id}__random"
    else:
        env_id, label, pol = load_policy(args.run)
        name = args.run
    if args.viewer:
        view(env_id, pol, args.episodes, seed=args.seed)
    else:
        res = record(env_id, label, pol, args.episodes, out / f"{name}.gif", seed=args.seed)
        print(f"saved {out / (name + '.gif')}  " + " ".join("ok" if r["success"] else "fail" for r in res))


if __name__ == "__main__":
    main()
