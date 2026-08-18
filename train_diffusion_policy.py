"""
Step 3: Diffusion Policy Training
==================================
Train TemporalUNet + DDPM for action prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import argparse


# ========== Diffusion Model Components ==========

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.activation = nn.SiLU()
        
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x, time_emb):
        h = self.activation(self.norm1(self.conv1(x)))
        scale, shift = self.mlp(time_emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = self.activation(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class TemporalUNet(nn.Module):
    def __init__(self, obs_dim, action_dim, action_horizon, time_steps=100, channels=None):
        super().__init__()
        if channels is None:
            channels = [64, 128]
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        
        self.time_emb = SinusoidalPositionEmbedding(128)
        
        in_channels = obs_dim + action_dim
        self.input_conv = nn.Conv1d(in_channels, channels[0], 3, padding=1)
        
        self.down1 = ResidualBlock(channels[0], channels[1], 128)
        self.down_pool1 = nn.MaxPool1d(2)
        
        self.up_pool1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = ResidualBlock(channels[1], channels[0], 128)
        
        self.output_conv = nn.Conv1d(channels[0], action_dim, 3, padding=1)
    
    def forward(self, obs, actions, t):
        batch_size = obs.shape[0]
        
        time_emb = self.time_emb(t)
        
        x = torch.cat([obs, actions], dim=2)
        x = self.input_conv(x.transpose(1, 2))
        
        x = self.down1(x, time_emb)
        x = self.down_pool1(x)
        
        x = self.up_pool1(x)
        x = self.up1(x, time_emb)
        
        x = self.output_conv(x)
        x = x.transpose(1, 2)
        
        return x


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, action_horizon, time_steps=100):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.time_steps = time_steps
        
        self.model = TemporalUNet(obs_dim, action_dim, action_horizon, time_steps)
        
        # Noise schedule
        beta = torch.linspace(0.0001, 0.02, time_steps)
        alpha = 1 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1 - alpha_cumprod))
    
    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        x_noisy = (
            self.sqrt_alpha_cumprod[t, None, None] * x +
            self.sqrt_one_minus_alpha_cumprod[t, None, None] * noise
        )
        return x_noisy, noise
    
    def forward(self, obs, actions, t):
        return self.model(obs, actions, t)
    
    @torch.no_grad()
    def sample(self, obs, num_steps=50):
        batch_size, seq_len, _ = obs.shape
        
        x = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=obs.device
        )
        
        time_steps = torch.linspace(
            self.time_steps - 1,
            0,
            num_steps,
            device=obs.device,
            dtype=torch.long
        )
        
        for t in time_steps:
            t_batch = torch.full((batch_size,), t, device=obs.device, dtype=torch.long)
            
            noise_pred = self.model(obs, x, t_batch)
            
            alpha_t = self.alpha[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            alpha_cumprod_t_minus_1 = (
                self.alpha_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
            )
            
            beta_t = self.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(
                    (1 - alpha_cumprod_t_minus_1) /
                    (1 - alpha_cumprod_t) *
                    beta_t
                )
            else:
                noise = torch.zeros_like(x)
                sigma_t = 0
            
            x = (
                (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) /
                torch.sqrt(alpha_t) +
                sigma_t * noise
            )
        
        return x


# ========== Dataset ==========

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, action_horizon, obs_mean, obs_std):
        self.action_horizon = action_horizon
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        
        self.data = []
        
        for traj in trajectories:
            obs_seq = traj["obs"]
            actions_seq = traj["actions"]
            
            for i in range(len(obs_seq) - action_horizon):
                obs_window = obs_seq[i:i + action_horizon]
                action_window = actions_seq[i:i + action_horizon]
                
                obs_norm = (obs_window - obs_mean) / (obs_std + 1e-8)
                
                self.data.append({
                    "obs": torch.from_numpy(obs_norm).float(),
                    "actions": torch.from_numpy(action_window).float(),
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ========== Training ==========

def train_diffusion_policy(
    trajectories,
    obs_mean,
    obs_std,
    output_dir="models",
    action_horizon=8,
    num_epochs=5000,
    batch_size=64,
    learning_rate=1e-4,
    device="cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine dimensions
    first_traj = trajectories[0]
    obs_dim = first_traj["obs"].shape[1]
    action_dim = first_traj["actions"].shape[1]
    
    print(f"Model dimensions:")
    print(f"  obs_dim: {obs_dim}")
    print(f"  action_dim: {action_dim}")
    print(f"  action_horizon: {action_horizon}")
    
    # Create dataset and dataloader
    dataset = TrajectoryDataset(trajectories, action_horizon, obs_mean, obs_std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Dataloader: {len(dataloader)} batches of size {batch_size}")
    
    # Create policy
    policy = DiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        time_steps=100,
    ).to(device)
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
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
            t = torch.randint(
                0, policy.time_steps,
                (obs.shape[0],),
                device=device
            )
            
            # Add noise
            noisy_actions, noise = policy.add_noise(actions, t)
            
            # Predict noise
            noise_pred = policy(obs, noisy_actions, t)
            
            # MSE loss
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
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"policy_epoch_{epoch+1}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = output_dir / "diffusion_policy.pt"
    torch.save(policy.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    # Save loss curve
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.savefig(output_dir / "training_loss.png")
        plt.close()
        print(f"Loss curve saved: {output_dir / 'training_loss.png'}")
    except Exception as e:
        print(f"Could not save loss curve: {e}")
    
    return policy, {"obs_mean": obs_mean, "obs_std": obs_std, "action_horizon": action_horizon}


# ========== Main ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/trajectories.npz")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data: {args.data_path}")
    data = np.load(args.data_path, allow_pickle=True)
    trajectories = list(data["trajectories"])
    print(f"Loaded {len(trajectories)} episodes")
    
    # Compute normalization parameters
    print("Computing normalization parameters...")
    all_obs = np.concatenate([traj["obs"] for traj in trajectories], axis=0)
    obs_mean = all_obs.mean(axis=0).astype(np.float32)
    obs_std = all_obs.std(axis=0).astype(np.float32)
    print(f"  obs_mean: {obs_mean[:3]}...")
    print(f"  obs_std: {obs_std[:3]}...")
    
    # Train policy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    policy, norm_params = train_diffusion_policy(
        trajectories=trajectories,
        obs_mean=obs_mean,
        obs_std=obs_std,
        output_dir=args.output_dir,
        action_horizon=args.action_horizon,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )
    
    # Save normalization parameters
    norm_path = Path(args.output_dir) / "normalization.json"
    norm_params["obs_mean"] = obs_mean.tolist()
    norm_params["obs_std"] = obs_std.tolist()
    with open(norm_path, "w") as f:
        json.dump(norm_params, f)
    print(f"Normalization params saved: {norm_path}")
    
    print("="*60)
    print("Step 3 completed: Diffusion Policy training done!")
    print("="*60)
