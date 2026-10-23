"""Visualize the trained PPO policy in a selected MuJoCo condition."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

try:
    import mujoco
    import mujoco.viewer
    from manipulator_env import DomainRandomizedEnv
    from train_ppo import ActorCritic
except Exception as e:
    print(f"Missing imports: {e}")
    raise

ROOT = Path(__file__).parent


def choose_best_model(csv_path, model_path=None):
    """Return (name, checkpoint path, PPO returns recorded in the comparison CSV)."""
    model_path = Path(model_path) if model_path else ROOT / "models" / "ppo_policy.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_path} (python train_ppo.py 로 먼저 학습하세요)")
    known_scores = []
    if Path(csv_path).exists():
        with open(csv_path, newline="") as f:
            known_scores = [float(r["return"]) for r in csv.DictReader(f) if r["algorithm"] == "ppo"]
    return "ppo", model_path, known_scores


def load_ppo(model_path, env, device):
    """Build ActorCritic with the checkpoint's input size and load its weights."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    policy_obs_dim = checkpoint["net.0.weight"].shape[1]
    if policy_obs_dim > env.observation_space.shape[0]:
        raise ValueError(
            f"모델 입력 차원({policy_obs_dim})이 환경 관측 차원보다 큽니다 "
            f"({env.observation_space.shape[0]})"
        )
    policy = ActorCritic(policy_obs_dim, env.action_space.shape[0]).to(device)
    policy.load_state_dict(checkpoint)
    policy.eval()
    return policy, policy_obs_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=("nominal", "domain_randomized"),
        default="nominal",
    )
    parser.add_argument("--model", default=None, help="PPO checkpoint (default: models/ppo_policy.pt)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    randomize = args.condition == "domain_randomized"
    env = DomainRandomizedEnv(randomize=randomize, render_mode=None)
    obs, _ = env.reset()
    device = torch.device(args.device)
    model_name, model_path, known_scores = choose_best_model(
        ROOT / "results" / "compare_results.csv", args.model
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
