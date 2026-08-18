"""
Record episodes for a given policy (diffusion or ppo) into MP4 files (offscreen rendering when available).
Requires imageio and imageio-ffmpeg for video writing. On headless servers, set MUJOCO_GL=egl or osmesa and ensure mujoco can render offscreen.

Usage examples:
  # Record 3 episodes for PPO
  python record_episodes.py --policy ppo --model models/ppo_policy.pt --episodes 3 --output_dir results/videos

  # Record 2 episodes for diffusion
  python record_episodes.py --policy diffusion --model models/diffusion_policy.pt --norm models/normalization.json --episodes 2
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch

try:
    import imageio
except Exception:
    imageio = None

try:
    import mujoco
except Exception:
    mujoco = None

from manipulator_env import DomainRandomizedEnv
from train_diffusion_policy import DiffusionPolicy
from train_ppo import ActorCritic


def try_render(env, width, height):
    """Try several mujoco render APIs to get an RGB image. Return None if none work."""
    if mujoco is None:
        return None
    try:
        # MuJoCo >=2.2/3.x python API
        if hasattr(mujoco, 'render'):
            img = mujoco.render(env.model, env.data, width=width, height=height)
            return img
    except Exception:
        pass
    try:
        # Older API: mujoco.viewer may provide render_frame
        if hasattr(mujoco, 'viewer') and hasattr(mujoco.viewer, 'render'):
            img = mujoco.viewer.render(env.model, env.data, width=width, height=height)
            return img
    except Exception:
        pass
    # If no render method works, return None
    return None


def record_policy(policy_type, model_path, norm_path=None, episodes=1, episode_length=200, device='cpu', output_dir='results/videos', fps=30, width=640, height=480):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = DomainRandomizedEnv(randomize=False, render_mode=None)

    if policy_type == 'ppo':
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        policy = ActorCritic(obs_dim, action_dim).to(device)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict):
            policy.load_state_dict(state)
        else:
            policy = state.to(device)
        policy.eval()
    elif policy_type == 'diffusion':
        if norm_path is None:
            raise ValueError('diffusion policy requires --norm normalization file')
        import json
        with open(norm_path, 'r') as f:
            norm = json.load(f)
        obs_mean = np.array(norm['obs_mean'], dtype=np.float32)
        obs_std = np.array(norm['obs_std'], dtype=np.float32)
        action_horizon = norm.get('action_horizon', 8)

        obs_dim = len(obs_mean)
        policy = DiffusionPolicy(obs_dim=obs_dim, action_dim=env.action_space.shape[0], action_horizon=action_horizon, time_steps=100).to(device)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict):
            policy.load_state_dict(state)
        else:
            policy = state.to(device)
        policy.eval()
    else:
        raise ValueError('Unknown policy type')

    # Determine rendering availability. If mujoco.render is not available, fall back to simple matplotlib-based visualization (joint positions) to still produce a video on headless servers.
    can_render = (imageio is not None) and (mujoco is not None) and (try_render(env, width, height) is not None)
    use_matplotlib_fallback = False
    if not can_render:
        try:
            import matplotlib
            # Use Agg backend for headless
            matplotlib.use('Agg')
            import matplotlib.pyplot as _plt
            use_matplotlib_fallback = True
            print('Offscreen rendering via mujoco not available. Will use matplotlib fallback visualization (joint positions) to create videos.')
        except Exception:
            print('Offscreen rendering not available and matplotlib fallback failed. Video recording will be skipped; saving only returns as text.')
            use_matplotlib_fallback = False

    for ep in range(episodes):
        obs, _ = env.reset()
        if policy_type == 'diffusion':
            # build obs history
            obs_history = np.tile(obs, (action_horizon, 1))

        frames = []
        returns = 0.0
        length = 0

        for t in range(episode_length):
            if policy_type == 'ppo':
                obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0)
                with torch.no_grad():
                    mu, std, val = policy(obs_tensor)
                    action = mu[0].cpu().numpy()
            else:
                # diffusion: use obs_history
                obs_seq = torch.from_numpy(((obs_history - obs_mean) / (obs_std + 1e-8)).astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    action_seq = policy.sample(obs_seq, num_steps=50)
                    action = action_seq[0, 0, :].cpu().numpy()

            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            if policy_type == 'diffusion':
                obs_history = np.vstack([obs_history[1:], obs.reshape(1, -1)])

            returns += float(reward)
            length += 1

            frame_added = False
            if can_render:
                img = try_render(env, width, height)
                if img is not None:
                    # Ensure uint8 image
                    img_u8 = (np.clip(img, 0, 255)).astype(np.uint8)
                    frames.append(img_u8)
                    frame_added = True

            # Matplotlib fallback: visualize joint positions as a bar plot
            if (not frame_added) and use_matplotlib_fallback:
                try:
                    import matplotlib.pyplot as plt
                    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

                    # obs layout: [qpos, qvel]
                    n_q = env.model.nq if hasattr(env.model, 'nq') else min(6, len(obs)//2)
                    qpos = obs[:n_q]

                    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
                    ax = fig.add_subplot(111)
                    ax.bar(range(len(qpos)), qpos, color='C0')
                    ax.set_ylim(-3.5, 3.5)
                    ax.set_title(f'Episode {ep+1} Step {t+1}')
                    ax.set_xlabel('Joint')
                    ax.set_ylabel('Position (rad)')
                    plt.tight_layout()

                    canvas = FigureCanvas(fig)
                    canvas.draw()
                    buf = canvas.buffer_rgba()
                    img_arr = np.asarray(buf)[:, :, :3]
                    frames.append(img_arr.astype(np.uint8))
                    plt.close(fig)
                except Exception as e:
                    # fallback failed; skip frame
                    pass

            if terminated or truncated:
                break

        print(f'Episode {ep+1}: return={returns:.3f}, length={length}')

        if can_render and frames:
            video_path = out_dir / f'{policy_type}_episode_{ep+1}.mp4'
            try:
                imageio.mimwrite(str(video_path), frames, fps=fps, codec='libx264')
                print(f'Saved video: {video_path}')
            except Exception as e:
                print(f'Could not write video (imageio error): {e}')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=['ppo', 'diffusion'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--norm', default=None, help='Normalization JSON (required for diffusion)')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=200)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='results/videos')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    record_policy(args.policy, args.model, norm_path=args.norm, episodes=args.episodes, episode_length=args.episode_length, device=args.device, output_dir=args.output_dir, fps=args.fps, width=args.width, height=args.height)
