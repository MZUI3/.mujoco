"""
Simple PPO trainer for DomainRandomizedEnv (MuJoCo)
- Lightweight PyTorch implementation
- Save trained model to models/ppo_policy.pt
- Save training returns plot to models/ppo_returns.png

Usage:
    python train_ppo.py --total_timesteps 20000
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from manipulator_env import DomainRandomizedEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size, action_dim)
        self.logstd = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logstd.exp(), self.value_head(h).squeeze(-1)


def rollout(env, policy, rollout_steps, device):
    obs_buf = []
    act_buf = []
    logp_buf = []
    val_buf = []
    rew_buf = []
    done_buf = []

    obs, _ = env.reset()
    obs = obs.astype(np.float32)
    steps = 0

    while steps < rollout_steps:
        obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        with torch.no_grad():
            mu, std, val = policy(obs_tensor)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)

        action_np = action.cpu().numpy().squeeze(0)
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = bool(terminated or truncated)

        obs_buf.append(obs.copy())
        act_buf.append(action_np)
        logp_buf.append(logp.cpu().item())
        val_buf.append(val.cpu().item())
        rew_buf.append(float(reward))
        done_buf.append(done)

        obs = next_obs.astype(np.float32)
        steps += 1

        if done:
            obs, _ = env.reset()
            obs = obs.astype(np.float32)

    # Convert to numpy
    return (
        np.array(obs_buf, dtype=np.float32),
        np.array(act_buf, dtype=np.float32),
        np.array(logp_buf, dtype=np.float32),
        np.array(val_buf, dtype=np.float32),
        np.array(rew_buf, dtype=np.float32),
        np.array(done_buf, dtype=bool),
        obs,  # last observation
    )


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(policy, optimizer, obs, acts, old_logps, returns, advantages, clip=0.2, epochs=10, batch_size=64, device="cpu"):
    obs_tensor = torch.from_numpy(obs).float().to(device)
    acts_tensor = torch.from_numpy(acts).float().to(device)
    old_logp_tensor = torch.from_numpy(old_logps).float().to(device)
    returns_tensor = torch.from_numpy(returns).float().to(device)
    adv_tensor = torch.from_numpy(advantages).float().to(device)

    dataset_size = obs_tensor.size(0)
    for _ in range(epochs):
        idxs = np.random.permutation(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mb_idx = idxs[start:end]

            mu, std, val = policy(obs_tensor[mb_idx])
            dist = torch.distributions.Normal(mu, std)
            logp = dist.log_prob(acts_tensor[mb_idx]).sum(-1)

            ratio = torch.exp(logp - old_logp_tensor[mb_idx])
            surr1 = ratio * adv_tensor[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_tensor[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((returns_tensor[mb_idx] - val) ** 2).mean()

            entropy = dist.entropy().sum(-1).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()


def train(
    total_timesteps=20000,
    rollout_steps=2048,
    epochs=10,
    clip=0.2,
    batch_size=64,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    device="cpu",
    output_dir="models",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    env = DomainRandomizedEnv(randomize=False, render_mode=None)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    timesteps_done = 0
    returns_history = []
    episode_rewards = []

    print(f"Training PPO on device={device}, obs_dim={obs_dim}, action_dim={action_dim}")

    start_time = time.time()
    while timesteps_done < total_timesteps:
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, last_obs = rollout(env, policy, rollout_steps, device)
        timesteps_done += len(rew_buf)

        # Estimate last value for bootstrap
        with torch.no_grad():
            last_obs_tensor = torch.from_numpy(last_obs).float().to(device).unsqueeze(0)
            _, _, last_val = policy(last_obs_tensor)
            last_val = last_val.cpu().item()

        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_val, gamma=gamma, lam=lam)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ppo_update(
            policy,
            optimizer,
            obs_buf,
            act_buf,
            logp_buf,
            returns,
            advantages,
            clip=clip,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )

        # Logging: compute episode returns from rewards (split by done)
        # For simplicity, approximate episodes by summing until done flags
        ep_rets = []
        cur = 0.0
        for r, d in zip(rew_buf, done_buf):
            cur += r
            if d:
                ep_rets.append(cur)
                cur = 0.0
        if cur != 0.0:
            ep_rets.append(cur)

        if ep_rets:
            returns_history.extend(ep_rets)
            mean_recent = float(np.mean(ep_rets))
        else:
            mean_recent = float(np.mean(rew_buf))

        print(f"Timesteps: {timesteps_done}/{total_timesteps}, Recent mean return: {mean_recent:.3f}")

    # Save model
    model_path = Path(output_dir) / "ppo_policy.pt"
    torch.save(policy.state_dict(), model_path)
    print(f"Saved PPO model: {model_path}")

    # Save returns plot
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(returns_history)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('PPO training returns')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'ppo_returns.png')
        plt.close()
        print(f"Saved returns plot: {Path(output_dir) / 'ppo_returns.png'}")
    except Exception as e:
        print(f"Could not save returns plot: {e}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=20000)
    parser.add_argument('--rollout_steps', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        output_dir='models',
    )
