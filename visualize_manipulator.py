"""Simple visualization for the manipulator using MuJoCo's Python viewer."""

import time

try:
    import mujoco
    from manipulator_env import DomainRandomizedEnv
except Exception as e:
    print(f"Missing imports: {e}")
    raise


def main():
    env = DomainRandomizedEnv(randomize=False, render_mode=None)
    obs, _ = env.reset()

    try:
        print("Launching MuJoCo viewer...")
        if hasattr(mujoco, "viewer") and hasattr(mujoco.viewer, "launch_passive"):
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            for _ in range(200):
                action = np.array([0.15, -0.4, 0.45, -0.2, 0.0, 0.0], dtype=np.float32)
                obs, reward, terminated, truncated, _ = env.step(action)
                viewer.sync()
                time.sleep(0.01)
                if terminated or truncated:
                    break
            viewer.close()
        else:
            raise RuntimeError("MuJoCo viewer is not available on this machine")
    except Exception as exc:
        print(f"Could not launch mujoco.viewer: {exc}")
        print("Fallback: step through a few deterministic states (no GUI)")
        for i in range(50):
            action = np.array([0.18 * np.sin(i / 8), -0.28, 0.35, -0.15, 0.0, 0.0], dtype=np.float32)
            obs, r, terminated, truncated, _ = env.step(action)
            print(f"Step {i + 1}: reward={r:.3f}, base={obs[:2]}")
            time.sleep(0.01)
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    import numpy as np
    main()
