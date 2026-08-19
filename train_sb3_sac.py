"""
Train a SAC agent (stable-baselines3) on the DomainRandomizedEnv.
Usage (server):
    python train_sb3_sac.py --timesteps 200000 --save-dir models
"""
import argparse
from pathlib import Path

import numpy as np

from manipulator_env import DomainRandomizedEnv

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback
except Exception as e:
    raise RuntimeError("stable-baselines3 is required. Install with: pip install stable-baselines3")


def make_env(randomize=True):
    def _init():
        env = DomainRandomizedEnv(randomize=randomize)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=200000)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(randomize=True)])
    env = Monitor(env)

    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=str(save_dir / 'tb'))

    checkpoint_cb = CheckpointCallback(save_freq=50000, save_path=str(save_dir), name_prefix='sac_checkpoint')

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)

    model.save(str(save_dir / 'sac_manipulator'))
    print(f"Saved model to: {save_dir / 'sac_manipulator.zip'}")


if __name__ == '__main__':
    main()
