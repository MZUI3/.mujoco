"""
Step 4: Simulation Policy Validation
=====================================
Validate the trained Diffusion Policy in MuJoCo environment
"""

import numpy as np
import torch
import json
from pathlib import Path
from manipulator_env import DomainRandomizedEnv
from train_diffusion_policy import DiffusionPolicy
from tqdm import tqdm


class PolicyValidator:
    """Validate trained policy"""
    
    def __init__(self, model_path, normalization_path, device="cpu"):
        self.device = device
        
        # Load normalization parameters
        with open(normalization_path, "r") as f:
            norm_data = json.load(f)
        
        self.obs_mean = np.array(norm_data["obs_mean"], dtype=np.float32)
        self.obs_std = np.array(norm_data["obs_std"], dtype=np.float32)
        self.action_horizon = norm_data["action_horizon"]
        self.obs_dim = len(self.obs_mean)
        self.action_dim = 6
        
        print("Load normalization parameters:")
        print(f"  obs_dim: {self.obs_dim}")
        print(f"  action_dim: {self.action_dim}")
        print(f"  action_horizon: {self.action_horizon}")
        
        # Create and load policy
        self.policy = DiffusionPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            time_steps=100,
        ).to(device)
        
        try:
            loaded = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(loaded, dict):
                self.policy.load_state_dict(loaded)
            else:
                print("TorchScript format loaded. Using model directly")
                self.policy = loaded.to(device)
        except Exception as e:
            print(f"Model load failed: {e}")
            epoch_checkpoint = Path("models/policy_epoch_150.pt")
            if epoch_checkpoint.exists():
                self.policy.load_state_dict(torch.load(epoch_checkpoint, map_location=device, weights_only=False))
            else:
                raise
        
        self.policy.eval()
        print(f"\nModel loaded: {model_path}")
    
    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
    
    @torch.no_grad()
    def get_action(self, obs_sequence):
        obs_seq = torch.from_numpy(self.normalize_obs(obs_sequence)).float().to(self.device)
        obs_seq = obs_seq.unsqueeze(0)
        action_seq = self.policy.sample(obs_seq, num_steps=50)
        action = action_seq[0, 0, :].cpu().numpy()
        action = np.clip(action, -1.0, 1.0)
        return action
    
    def validate(self, num_episodes=10, episode_length=200):
        env = DomainRandomizedEnv(randomize=False, render_mode=None)
        
        print(f"\n{'='*60}")
        print(f"Policy Validation (episodes: {num_episodes})")
        print(f"{'='*60}")
        
        episode_returns = []
        episode_lengths = []
        
        for ep in tqdm(range(num_episodes), desc="Validation"):
            obs, _ = env.reset()
            obs_history = np.tile(obs, (self.action_horizon, 1))
            ep_return = 0.0
            ep_length = 0
            
            for step in range(episode_length):
                try:
                    action = self.get_action(obs_history)
                    obs, reward, terminated, truncated, info = env.step(action)
                    obs_history = np.vstack([obs_history[1:], obs.reshape(1, -1)])
                    ep_return += reward
                    ep_length += 1
                    if terminated or truncated:
                        break
                except Exception as e:
                    print(f"\nEpisode {ep+1}, Step {step}: {e}")
                    break
            
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
        
        env.close()
        
        returns_array = np.array(episode_returns)
        lengths_array = np.array(episode_lengths)
        
        results = {
            "returns": returns_array,
            "lengths": lengths_array,
            "mean_return": float(returns_array.mean()),
            "std_return": float(returns_array.std()),
            "mean_length": float(lengths_array.mean()),
            "max_return": float(returns_array.max()),
            "min_return": float(returns_array.min()),
        }
        
        print(f"\n{'='*60}")
        print(f"Validation Results")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes}")
        print(f"\nReturn statistics:")
        print(f"  Mean: {results['mean_return']:.4f}")
        print(f"  Std: {results['std_return']:.4f}")
        print(f"  Min/Max: {results['min_return']:.4f} / {results['max_return']:.4f}")
        print(f"\nEpisode length:")
        print(f"  Mean: {results['mean_length']:.1f}")
        print(f"  Total steps: {int(lengths_array.sum())}")
        
        return results
    
    def validate_with_dr(self, num_episodes=5, episode_length=200):
        env = DomainRandomizedEnv(randomize=True, render_mode=None)
        
        print(f"\n{'='*60}")
        print(f"Domain Randomization Validation (episodes: {num_episodes})")
        print(f"{'='*60}")
        
        episode_returns = []
        episode_lengths = []
        
        for ep in tqdm(range(num_episodes), desc="DR Validation"):
            obs, _ = env.reset()
            obs_history = np.tile(obs, (self.action_horizon, 1))
            ep_return = 0.0
            ep_length = 0
            
            for step in range(episode_length):
                try:
                    action = self.get_action(obs_history)
                    obs, reward, terminated, truncated, info = env.step(action)
                    obs_history = np.vstack([obs_history[1:], obs.reshape(1, -1)])
                    ep_return += reward
                    ep_length += 1
                    if terminated or truncated:
                        break
                except Exception as e:
                    print(f"\nEpisode {ep+1}, Step {step}: {e}")
                    break
            
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
        
        env.close()
        
        returns_array = np.array(episode_returns)
        lengths_array = np.array(episode_lengths)
        
        results = {
            "returns": returns_array,
            "lengths": lengths_array,
            "mean_return": float(returns_array.mean()),
            "std_return": float(returns_array.std()),
            "mean_length": float(lengths_array.mean()),
        }
        
        print(f"\n{'='*60}")
        print(f"Domain Randomization Results")
        print(f"{'='*60}")
        print(f"Return: {results['mean_return']:.4f} +- {results['std_return']:.4f}")
        print(f"Episode length: {results['mean_length']:.1f}")
        
        return results


if __name__ == "__main__":
    model_dir = Path("models")
    model_path = model_dir / "diffusion_policy.pt"
    norm_path = model_dir / "normalization.json"
    
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        exit(1)
    
    if not norm_path.exists():
        print(f"ERROR: Normalization parameters not found: {norm_path}")
        exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validator = PolicyValidator(
        model_path=str(model_path),
        normalization_path=str(norm_path),
        device=device,
    )
    
    # Basic validation
    results_basic = validator.validate(num_episodes=10, episode_length=200)
    
    # DR validation
    results_dr = validator.validate_with_dr(num_episodes=5, episode_length=200)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Basic environment:")
    print(f"  Return: {results_basic['mean_return']:.4f} +- {results_basic['std_return']:.4f}")
    print(f"  Length: {results_basic['mean_length']:.1f}")
    
    print(f"\nDomain Randomization:")
    print(f"  Return: {results_dr['mean_return']:.4f} +- {results_dr['std_return']:.4f}")
    print(f"  Length: {results_dr['mean_length']:.1f}")
    
    drop = (results_basic['mean_return'] - results_dr['mean_return']) / (abs(results_basic['mean_return']) + 1e-8) * 100
    print(f"\nPerformance drop: {drop:.1f}%")
    
    if drop > 30:
        print(f"WARNING: Large performance drop. Step 6 retraining recommended")
    else:
        print(f"OK: Performance stable. Proceed to Step 5")
    
    print(f"\n{'='*60}")
    print(f"Step 4 completed: Policy validation done!")
    print(f"{'='*60}")
