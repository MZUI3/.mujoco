"""Training matrix for the RL model zoo: (env id, algorithm, timesteps).

Budgets are modest so the whole matrix trains on a laptop CPU in under an hour
(off-policy SAC/TD3 100k steps, on-policy PPO/A2C 300k steps; the hard tasks ReachSparse/ReachObstacle/Lift get 300k). Increase them for stronger models:
    python rl_zoo/train_zoo.py --env Reach-v0 --algo SAC --timesteps 300000
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models_zoo"
RESULTS_DIR = ROOT / "results_zoo"

OFF, ON = 100_000, 300_000
RUNS = [
    ("Reach-v0", "SAC", OFF),
    ("Reach-v0", "TD3", OFF),
    ("Reach-v0", "PPO", ON),
    ("Reach-v0", "A2C", ON),
    ("ReachSparse-v0", "SAC", 300_000),
    ("ReachSparse-v0", "SAC_HER", 300_000),   # HER: hindsight goal relabelling for the sparse reward
    ("ReachSparse-v0", "PPO", ON),
    ("ReachObstacle-v0", "SAC", 300_000),
    ("ReachObstacle-v0", "PPO", ON),
    ("ReachDR-v0", "SAC", OFF),
    ("ReachDR-v0", "PPO", ON),
    ("ReachMoving-v0", "SAC", OFF),
    ("Push-v0", "SAC", OFF),
    ("Lift-v0", "SAC", 300_000),
    ("TrackTrajectory-v0", "SAC", OFF),
    ("TrackTrajectory-v0", "PPO", ON),
    ("HoldPose-v0", "PPO", ON),
    ("ApproachObjects-v0", "SAC", OFF),
]

ALGO_KWARGS = {
    "SAC": dict(learning_rate=3e-4, buffer_size=200_000, learning_starts=2_000, batch_size=256, tau=0.005,
                gamma=0.98, train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256])),
    "TD3": dict(learning_rate=3e-4, buffer_size=200_000, learning_starts=2_000, batch_size=256, tau=0.005,
                gamma=0.98, train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256])),
    # SAC + HerReplayBuffer on the goal-conditioned (dict obs) wrapper; env is wrapped by train_zoo
    "SAC_HER": dict(learning_rate=3e-4, buffer_size=300_000, learning_starts=2_000, batch_size=256, tau=0.005,
                    gamma=0.98, train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256])),
    "PPO": dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.98, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[256, 256])),
    "A2C": dict(learning_rate=7e-4, n_steps=16, gamma=0.98, gae_lambda=0.95, ent_coef=0.0,
                policy_kwargs=dict(net_arch=[256, 256])),
}


def run_name(env_id: str, algo: str) -> str:
    return f"{env_id}__{algo}"


def needs_goal_wrapper(algo: str) -> bool:
    return algo.endswith("_HER")


def base_algo(algo: str) -> str:
    return algo.replace("_HER", "")
