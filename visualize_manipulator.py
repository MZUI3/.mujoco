"""Simple visualization for the manipulator using MuJoCo's Python viewer."""

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


def main():
    env = DomainRandomizedEnv(randomize=False, render_mode=None)
    obs, _ = env.reset()
    model_path = Path(__file__).parent / "models" / "ppo_policy.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    policy_obs_dim = checkpoint["net.0.weight"].shape[1]
    if policy_obs_dim > env.observation_space.shape[0]:
        raise ValueError(
            f"모델 입력 차원({policy_obs_dim})이 환경 관측 차원보다 큽니다 "
            f"({env.observation_space.shape[0]})"
        )
    policy = ActorCritic(
        policy_obs_dim,
        env.action_space.shape[0],
    ).to(device)
    policy.load_state_dict(checkpoint)
    policy.eval()
    print(f"Loaded best model: {model_path.name} ({device})")

    try:
        print("Launching MuJoCo viewer...")
        if hasattr(mujoco, "viewer") and hasattr(mujoco.viewer, "launch_passive"):
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            for _ in range(200):
                policy_obs = obs[:policy_obs_dim]
                obs_tensor = torch.from_numpy(policy_obs).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    action = policy(obs_tensor)[0][0].cpu().numpy()
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
    main()
