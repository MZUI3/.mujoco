"""Visualize the best checkpoint in a selected MuJoCo condition."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

try:
    import mujoco
    import mujoco.viewer
    from manipulator_env import DomainRandomizedEnv
    from run_best_model_experiments import choose_best_model, load_ppo
except Exception as e:
    print(f"Missing imports: {e}")
    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=("nominal", "domain_randomized"),
        default="nominal",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    randomize = args.condition == "domain_randomized"
    env = DomainRandomizedEnv(randomize=randomize, render_mode=None)
    obs, _ = env.reset()
    device = torch.device(args.device)
    model_name, model_path, known_scores = choose_best_model(
        Path(__file__).parent / "results" / "compare_results.csv"
    )
    policy, policy_obs_dim = load_ppo(model_path, env, device)
    print(f"Loaded best model: {model_name} ({model_path.name}, {device})")
    if known_scores:
        print(f"Known comparison mean return: {np.mean(known_scores):.3f}")
    print(f"Visualization condition: {args.condition}")

    try:
        print("Launching MuJoCo viewer...")
        if hasattr(mujoco, "viewer") and hasattr(mujoco.viewer, "launch_passive"):
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            episode_return = 0.0
            for step in range(args.steps):
                policy_obs = obs[:policy_obs_dim]
                obs_tensor = torch.from_numpy(policy_obs).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    action = policy(obs_tensor)[0][0].cpu().numpy()
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                viewer.sync()
                time.sleep(0.01)
                if terminated or truncated:
                    break
            viewer.close()
            print(f"Visualization finished: steps={step + 1}, return={episode_return:.3f}")
        else:
            raise RuntimeError("MuJoCo viewer is not available on this machine")
    except Exception as exc:
        print(f"Could not launch mujoco.viewer: {exc}")
        print("Fallback: step through a few deterministic states (no GUI)")
        episode_return = 0.0
        for i in range(args.steps):
            policy_obs = obs[:policy_obs_dim]
            obs_tensor = torch.from_numpy(policy_obs).float().to(device).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_tensor)[0][0].cpu().numpy()
            obs, r, terminated, truncated, _ = env.step(action)
            episode_return += r
            print(f"Step {i + 1}: reward={r:.3f}, cumulative={episode_return:.3f}, base={obs[:2]}")
            time.sleep(0.01)
            if terminated or truncated:
                break
        print(f"Fallback finished: steps={i + 1}, return={episode_return:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
