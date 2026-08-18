"""
Simple visualization for the manipulator using MuJoCo Python viewer.
Usage:
    python visualize_manipulator.py

This script creates the DomainRandomizedEnv and launches a MuJoCo viewer if available.
"""

import time
from pathlib import Path

try:
    import mujoco
    from manipulator_env import DomainRandomizedEnv
except Exception as e:
    print(f"Missing imports: {e}")
    raise


def main():
    # Create env with randomize=False for deterministic visualization
    env = DomainRandomizedEnv(randomize=False, render_mode=None)

    # If environment created, try to launch mujoco.viewer
    try:
        print("Launching MuJoCo viewer...")
        # viewer.launch accepts model and data in many mujoco versions
        mujoco.viewer.launch(env.model, env.data)
    except Exception as e:
        print(f"Could not launch mujoco.viewer: {e}")
        print("Fallback: step through a few steps and print states (no GUI)")
        obs, _ = env.reset()
        for i in range(100):
            a = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(a)
            print(f"Step {i+1}: reward={r:.3f}")
            time.sleep(0.01)
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == '__main__':
    main()
