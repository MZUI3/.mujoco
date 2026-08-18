"""
Step 6: Domain Randomization Retraining
========================================
Retrain Diffusion Policy with domain randomization
for improved robustness to dynamics variations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from manipulator_env import DomainRandomizedEnv
from train_diffusion_policy import DiffusionPolicy, TrajectoryDataset


def collect_trajectories_with_dr(
    num_episodes=500,
    episode_length=200,
    device="cpu",
):
    """
    Collect trajectories using random actions with domain randomization
    
    Args:
        num_episodes: Number of episodes to collect
        episode_length: Max steps per episode
        device: Device (cpu or cuda)
    
    Returns:
        trajectories: List of trajectory dictionaries
    """
    env = DomainRandomizedEnv(randomize=True, render_mode=None)
    
    trajectories = []
    pbar = tqdm(range(num_episodes), desc="Collecting DR trajectories")
    
    for ep in pbar:
        obs, _ = env.reset()
        trajectory = {
            "obs": [obs],
            "actions": [],
            "next_obs": [],
            "rewards": [],
            "dones": [],
        }
        
        for step in range(episode_length):
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            trajectory["actions"].append(action)
            trajectory["next_obs"].append(obs)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)
            trajectory["obs"].append(obs)
            
            if done:
                break
        
        # Convert to numpy arrays
        trajectory["obs"] = np.array(trajectory["obs"][:-1])
        trajectory["actions"] = np.array(trajectory["actions"])
        trajectory["next_obs"] = np.array(trajectory["next_obs"])
        trajectory["rewards"] = np.array(trajectory["rewards"])
        trajectory["dones"] = np.array(trajectory["dones"])
        
        trajectories.append(trajectory)
    
    env.close()
    return trajectories


def train_with_dr(
    trajectories,
    obs_mean,
    obs_std,
    output_dir="models_dr",
    action_horizon=8,
    num_epochs=3000,
    batch_size=256,
    learning_rate=1e-4,
    device="cpu",
):
    """
    Train policy with domain randomized data
    
    Args:
        trajectories: List of trajectory dictionaries
        obs_mean: Observation normalization mean
        obs_std: Observation normalization std
        output_dir: Output directory
        action_horizon: Action sequence length
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device (cpu or cuda)
    
    Returns:
        policy: Trained policy model
        norm_params: Normalization parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine dimensions
    first_traj = trajectories[0]
    obs_dim = first_traj["obs"].shape[1]
    action_dim = first_traj["actions"].shape[1]
    
    print(f"Training with Domain Randomization")
    print(f"  obs_dim: {obs_dim}")
    print(f"  action_dim: {action_dim}")
    print(f"  action_horizon: {action_horizon}")
    print(f"  trajectories: {len(trajectories)}")
    
    # Create dataset
    dataset = TrajectoryDataset(trajectories, action_horizon, obs_mean, obs_std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"  dataset size: {len(dataset)} samples")
    print(f"  batches: {len(dataloader)}")
    
    # Create policy
    policy = DiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        time_steps=100,
    ).to(device)
    
    print(f"  parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    losses = []
    print(f"\nTraining for {num_epochs} epochs")
    print("="*60)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            obs = batch["obs"].to(device)
            actions = batch["actions"].to(device)
            
            # Random timesteps
            t = torch.randint(0, policy.time_steps, (obs.shape[0],), device=device)
            
            # Add noise
            noisy_actions, noise = policy.add_noise(actions, t)
            
            # Predict noise
            noise_pred = policy(obs, noisy_actions, t)
            
            # Loss
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
        # Checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"policy_epoch_{epoch+1}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = output_dir / "diffusion_policy_dr.pt"
    torch.save(policy.state_dict(), final_model_path)
    print(f"\nFinal DR model saved: {final_model_path}")
    
    # Save loss curve
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("DR Training Loss")
        plt.grid(True)
        plt.savefig(output_dir / "training_loss_dr.png")
        plt.close()
        print(f"Loss curve saved: {output_dir / 'training_loss_dr.png'}")
    except Exception as e:
        print(f"Could not save loss curve: {e}")
    
    return policy, {"obs_mean": obs_mean, "obs_std": obs_std, "action_horizon": action_horizon}


def main():
    """Main execution: Domain Randomization training"""
    
    print("="*60)
    print("Step 6: Domain Randomization Retraining")
    print("="*60)
    
    # Use existing normalization
    norm_path = Path("models/normalization.json")
    with open(norm_path, "r") as f:
        norm_data = json.load(f)
    
    obs_mean = np.array(norm_data["obs_mean"], dtype=np.float32)
    obs_std = np.array(norm_data["obs_std"], dtype=np.float32)
    
    print("\nUsing normalization from Step 3")
    print(f"  obs_mean: {obs_mean[:3]}...")
    print(f"  obs_std: {obs_std[:3]}...")
    
    # Collect DR trajectories
    print("\n" + "="*60)
    print("Collecting data with Domain Randomization")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trajectories = collect_trajectories_with_dr(
        num_episodes=200,
        episode_length=200,
        device=device,
    )
    
    print(f"\nCollected {len(trajectories)} episodes")
    print(f"Total transitions: {sum(len(t['actions']) for t in trajectories)}")
    
    # Train with DR
    print("\n" + "="*60)
    print("Training with Domain Randomized data")
    print("="*60)
    
    policy, norm_params = train_with_dr(
        trajectories=trajectories,
        obs_mean=obs_mean,
        obs_std=obs_std,
        output_dir="models_dr",
        action_horizon=8,
        num_epochs=30,
        batch_size=32,
        learning_rate=1e-4,
        device=device,
    )
    
    # Save normalization parameters
    norm_path_dr = Path("models_dr/normalization.json")
    norm_params["obs_mean"] = obs_mean.tolist()
    norm_params["obs_std"] = obs_std.tolist()
    with open(norm_path_dr, "w") as f:
        json.dump(norm_params, f)
    print(f"Normalization params saved: {norm_path_dr}")
    
    print("\n" + "="*60)
    print("Step 6 completed: Domain Randomization retraining done!")
    print("="*60)
    print("\nNext: Validate DR model with validate_policy_dr.py")


if __name__ == "__main__":
    main()
