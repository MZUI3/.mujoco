"""
Evaluate and compare Diffusion policy and PPO policy on the same environment.
Saves per-episode returns to CSV and a comparison plot (PNG).

Usage:
    python evaluate_compare.py --diffusion_model models/diffusion_policy.pt \
        --diffusion_norm models/normalization.json \
        --ppo_model models/ppo_policy.pt \
        --episodes 10
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from manipulator_env import DomainRandomizedEnv
from train_diffusion_policy import DiffusionPolicy
from train_ppo import ActorCritic


def evaluate_diffusion(model_path, norm_path, episodes=10, episode_length=200, device="cpu"):
    # Load normalization
    import json
    with open(norm_path, 'r') as f:
        norm = json.load(f)

    obs_mean = np.array(norm['obs_mean'], dtype=np.float32)
    obs_std = np.array(norm['obs_std'], dtype=np.float32)
    action_horizon = norm.get('action_horizon', 8)
    obs_dim = len(obs_mean)

    policy = DiffusionPolicy(obs_dim=obs_dim, action_dim=6, action_horizon=action_horizon, time_steps=100).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        policy.load_state_dict(state)
    else:
        # TorchScript
        policy = state.to(device)
    policy.eval()

    env = DomainRandomizedEnv(randomize=False, render_mode=None)

    returns = []
    lengths = []

    for ep in range(episodes):
        obs, _ = env.reset()
        obs_history = np.tile(obs, (action_horizon, 1))
        ep_return = 0.0
        ep_len = 0
        for t in range(episode_length):
            with torch.no_grad():
                obs_seq = torch.from_numpy((obs_history - obs_mean) / (obs_std + 1e-8)).float().unsqueeze(0).to(device)
                action_seq = policy.sample(obs_seq, num_steps=50)
                action = action_seq[0, 0, :].cpu().numpy()
                action = np.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, _ = env.step(action)
            obs_history = np.vstack([obs_history[1:], obs.reshape(1, -1)])
            ep_return += float(reward)
            ep_len += 1
            if terminated or truncated:
                break

        returns.append(ep_return)
        lengths.append(ep_len)

    env.close()
    return np.array(returns), np.array(lengths)


def evaluate_ppo(model_path, episodes=10, episode_length=200, device="cpu"):
    env = DomainRandomizedEnv(randomize=False, render_mode=None)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = ActorCritic(obs_dim, action_dim).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        policy.load_state_dict(state)
    else:
        policy = state.to(device)
    policy.eval()

    returns = []
    lengths = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_len = 0
        for t in range(episode_length):
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0)
            with torch.no_grad():
                mu, std, val = policy(obs_tensor)
                dist = torch.distributions.Normal(mu, std)
                action = dist.mean[0].cpu().numpy()  # deterministic evaluation
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            if terminated or truncated:
                break
        returns.append(ep_return)
        lengths.append(ep_len)

    env.close()
    return np.array(returns), np.array(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusion_model', type=str, default='models/diffusion_policy.pt')
    parser.add_argument('--diffusion_norm', type=str, default='models/normalization.json')
    parser.add_argument('--ppo_model', type=str, default='models/ppo_policy.pt')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--episode_length', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / 'compare_results.csv'

    rows = []

    if Path(args.diffusion_model).exists() and Path(args.diffusion_norm).exists():
        print('Evaluating diffusion policy...')
        d_returns, d_lengths = evaluate_diffusion(args.diffusion_model, args.diffusion_norm, episodes=args.episodes, episode_length=args.episode_length, device=args.device)
        for i, (r, l) in enumerate(zip(d_returns, d_lengths)):
            rows.append(['diffusion', i+1, float(r), int(l)])
    else:
        print('Warning: Diffusion model or normalization not found. Skipping diffusion evaluation.')

    if Path(args.ppo_model).exists():
        print('Evaluating PPO policy...')
        p_returns, p_lengths = evaluate_ppo(args.ppo_model, episodes=args.episodes, episode_length=args.episode_length, device=args.device)
        for i, (r, l) in enumerate(zip(p_returns, p_lengths)):
            rows.append(['ppo', i+1, float(r), int(l)])
    else:
        print('Warning: PPO model not found. Skipping PPO evaluation.')

    # Save CSV
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'episode', 'return', 'length'])
        writer.writerows(rows)

    print(f'Saved per-episode results to: {results_csv}')

    # Plot summary
    algs = sorted(list(set([r[0] for r in rows])))
    summary = {}
    for alg in algs:
        vals = [r[2] for r in rows if r[0] == alg]
        summary[alg] = np.array(vals)

    plt.figure(figsize=(6,4))
    for alg, vals in summary.items():
        plt.plot(vals, marker='o', label=f'{alg} (mean={vals.mean():.2f})')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Diffusion vs PPO: per-episode returns')
    plt.legend()
    plt.grid(True)
    plot_path = out_dir / 'compare_plot.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f'Saved comparison plot: {plot_path}')

    # Print summary stats + statistical tests
    print('\nSummary:')
    for alg, vals in summary.items():
        print(f"{alg}: mean={vals.mean():.3f}, std={vals.std():.3f}, max={vals.max():.3f}, min={vals.min():.3f}")

    # If both algorithms present, run statistical comparison
    if len(summary) >= 2:
        algs = list(summary.keys())
        a = summary[algs[0]]
        b = summary[algs[1]]

        # Compute 95% confidence intervals for the mean
        try:
            from scipy import stats
            use_scipy = True
        except Exception:
            stats = None
            use_scipy = False

        def ci_mean(x, alpha=0.05):
            n = len(x)
            mean = x.mean()
            se = x.std(ddof=1) / (n ** 0.5)
            if use_scipy:
                df = n - 1
                t_crit = stats.t.ppf(1 - alpha/2, df)
            else:
                # normal approximation
                t_crit = 1.96
            lo = mean - t_crit * se
            hi = mean + t_crit * se
            return lo, hi

        ci_a = ci_mean(a)
        ci_b = ci_mean(b)

        print(f"\n95% CI {algs[0]} mean: [{ci_a[0]:.3f}, {ci_a[1]:.3f}]")
        print(f"95% CI {algs[1]} mean: [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")

        # Welch's t-test
        if use_scipy:
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            print(f"\nWelch t-test: t={t_stat:.3f}, p={p_val:.5f}")
        else:
            # Compute Welch's t-statistic; p-value not available without scipy
            na = len(a); nb = len(b)
            ma = a.mean(); mb = b.mean()
            sa2 = a.var(ddof=1); sb2 = b.var(ddof=1)
            t_stat = (ma - mb) / ((sa2/na + sb2/nb) ** 0.5)
            # Welch-Satterthwaite df
            df = (sa2/na + sb2/nb) ** 2 / ((sa2**2)/((na**2)*(na-1)) + (sb2**2)/((nb**2)*(nb-1)))
            print(f"\nWelch t-statistic: t={t_stat:.3f}, df~={df:.1f} (p-value requires scipy)")


if __name__ == '__main__':
    main()
