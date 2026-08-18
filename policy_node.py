"""
Step 5: Policy Node - Policy Integration with Simulation
=========================================================
Integrate trained Diffusion Policy into simulation
Real-time policy execution with receding-horizon control
"""

import numpy as np
import torch
import json
from pathlib import Path
from manipulator_env import DomainRandomizedEnv
from train_diffusion_policy import DiffusionPolicy


class PolicyNode:
    """
    Real-time policy node for MuJoCo manipulation tasks
    Executes trained Diffusion Policy with receding-horizon control
    """
    
    def __init__(
        self,
        model_path="models/diffusion_policy.pt",
        normalization_path="models/normalization.json",
        device="cpu",
    ):
        self.device = device
        
        # Load normalization
        with open(normalization_path, "r") as f:
            norm_data = json.load(f)
        
        self.obs_mean = np.array(norm_data["obs_mean"], dtype=np.float32)
        self.obs_std = np.array(norm_data["obs_std"], dtype=np.float32)
        self.action_horizon = norm_data["action_horizon"]
        
        self.obs_dim = len(self.obs_mean)
        self.action_dim = 6
        
        print("Policy Node initialized")
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
        
        # Load model weights
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        
        print(f"Model loaded: {model_path}")
        print("="*60)
    
    def normalize_obs(self, obs):
        """Normalize observation using loaded statistics"""
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
    
    @torch.no_grad()
    def get_action(self, obs_sequence):
        """
        Get next action using receding-horizon control
        
        Args:
            obs_sequence: (action_horizon, obs_dim) observation history
        
        Returns:
            action: (action_dim,) next action
        """
        obs_norm = self.normalize_obs(obs_sequence)
        obs_tensor = torch.from_numpy(obs_norm).float().to(self.device).unsqueeze(0)
        
        # Sample action sequence from policy
        action_seq = self.policy.sample(obs_tensor, num_steps=50)
        
        # Use only first action (receding-horizon)
        action = action_seq[0, 0, :].cpu().numpy()
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def run_episode(self, env, episode_length=200, verbose=True):
        """
        Run single episode with policy
        
        Args:
            env: MuJoCo environment
            episode_length: Max steps
            verbose: Print statistics
        
        Returns:
            episode_return: Total return
            episode_length: Steps taken
            trajectory: Episode data
        """
        obs, _ = env.reset()
        obs_history = np.tile(obs, (self.action_horizon, 1))
        
        episode_return = 0.0
        step_count = 0
        trajectory = {
            "obs": [obs.copy()],
            "actions": [],
            "rewards": [],
            "obs_history": [obs_history.copy()],
        }
        
        for step in range(episode_length):
            # Get action from policy
            try:
                action = self.get_action(obs_history)
            except Exception as e:
                print(f"Error getting action at step {step}: {e}")
                break
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update observation history
            obs_history = np.vstack([obs_history[1:], obs.reshape(1, -1)])
            
            # Record trajectory
            trajectory["obs"].append(obs.copy())
            trajectory["actions"].append(action.copy())
            trajectory["rewards"].append(float(reward))
            trajectory["obs_history"].append(obs_history.copy())
            
            episode_return += reward
            step_count += 1
            
            if done:
                break
        
        if verbose:
            print(f"Episode finished: return={episode_return:.2f}, steps={step_count}")
        
        return episode_return, step_count, trajectory


def main():
    """Main execution: Run policy in simulation"""
    
    print("="*60)
    print("Step 5: Policy Node Integration")
    print("="*60)
    
    # Initialize policy node
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_node = PolicyNode(device=device)
    
    # Create environment
    print("\nInitializing environment...")
    env = DomainRandomizedEnv(randomize=False, render_mode=None)
    print("Environment ready")
    
    # Run episodes
    print("\n" + "="*60)
    print("Running policy episodes")
    print("="*60)
    
    num_episodes = 3
    returns = []
    lengths = []
    
    for ep in range(num_episodes):
        print(f"\nEpisode {ep+1}/{num_episodes}")
        ep_return, ep_length, trajectory = policy_node.run_episode(env, verbose=True)
        
        returns.append(ep_return)
        lengths.append(ep_length)
        
        # Save trajectory
        traj_path = Path("trajectories") / f"episode_{ep+1}.npz"
        traj_path.parent.mkdir(exist_ok=True)
        np.savez(
            traj_path,
            obs=np.array(trajectory["obs"]),
            actions=np.array(trajectory["actions"]),
            rewards=np.array(trajectory["rewards"]),
        )
        print(f"  Trajectory saved: {traj_path}")
    
    env.close()
    
    # Summary
    returns = np.array(returns)
    lengths = np.array(lengths)
    
    print("\n" + "="*60)
    print("Step 5 Summary")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Return: {returns.mean():.2f} +- {returns.std():.2f}")
    print(f"  Min/Max: {returns.min():.2f} / {returns.max():.2f}")
    print(f"Episode length: {lengths.mean():.1f}")
    print("="*60)
    print("Step 5 completed: Policy node integration done!")
    print("="*60)


if __name__ == "__main__":
    main()
