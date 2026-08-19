"""
Load a stable-baselines3 SAC model and run it in the manipulator environment, visualizing
via MuJoCo viewer if available, otherwise falling back to textual output.

Usage:
    python visualize_policy_sb3.py --model models/sac_manipulator.zip --episodes 3
"""
import argparse
import time
import numpy as np

from manipulator_env import DomainRandomizedEnv

try:
    from stable_baselines3 import SAC
except Exception:
    SAC = None

import mujoco


def run_episode(env, model=None, render=False):
    obs, _ = env.reset()
    total = 0.0
    for t in range(1000):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        total += float(reward)
        if render:
            # try to keep viewer in sync
            try:
                if hasattr(mujoco, 'viewer'):
                    mujoco.viewer.launch(env.model, env.data)
                    # on headless machines, viewer.launch likely fails and will be caught
                    time.sleep(0.001)
            except Exception:
                pass
        if term or trunc:
            break
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/sac_manipulator.zip')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env = DomainRandomizedEnv(randomize=False)

    model = None
    if SAC is not None and args.model and Path := None:
        try:
            model = SAC.load(args.model)
            print(f"Loaded model: {args.model}")
        except Exception as e:
            print(f"Could not load model {args.model}: {e}")
            model = None

    for ep in range(args.episodes):
        ret = run_episode(env, model=model, render=args.render)
        print(f"Episode {ep+1}: return={ret:.2f}")

    env.close()


if __name__ == '__main__':
    main()
