"""
Load a stable-baselines3 SAC model and run it in the manipulator environment, visualizing
via MuJoCo viewer if available, otherwise falling back to textual output.
Without stable-baselines3 or a model file it runs random actions so the scene can still be checked.

Usage:
    python visualize_policy_sb3.py --model models/sac_manipulator.zip --episodes 3
"""
import argparse
import time
from pathlib import Path

import numpy as np

from manipulator_env import DomainRandomizedEnv

try:
    from stable_baselines3 import SAC
except Exception:
    SAC = None

import mujoco
import mujoco.viewer


def run_episode(env, model=None, viewer=None, max_steps=1000):
    obs, _ = env.reset()
    total = 0.0
    for t in range(max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        total += float(reward)
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(env.model.opt.timestep)
        if term or trunc:
            break
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/sac_manipulator.zip')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env = DomainRandomizedEnv(randomize=False)

    model = None
    if SAC is None:
        print("stable-baselines3 is not installed (pip install stable-baselines3); using random actions")
    elif not Path(args.model).exists():
        print(f"Model not found: {args.model} (train with train_sb3_sac.py); using random actions")
    else:
        try:
            model = SAC.load(args.model)
            print(f"Loaded model: {args.model}")
        except Exception as e:
            print(f"Could not load model {args.model}: {e}")
            model = None

    viewer = None
    if args.render:
        try:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        except Exception as e:
            # on headless machines the viewer cannot open; fall back to text output
            print(f"Could not launch mujoco.viewer: {e}")

    try:
        for ep in range(args.episodes):
            ret = run_episode(env, model=model, viewer=viewer, max_steps=args.max_steps)
            print(f"Episode {ep+1}: return={ret:.2f}")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == '__main__':
    main()
