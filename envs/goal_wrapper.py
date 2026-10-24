"""Goal-conditioned (dict observation) wrapper so SB3's HerReplayBuffer can relabel goals.

Works with the Reach-family tasks: achieved_goal = EE position, desired_goal = goal position, and
compute_reward reproduces the task reward from (achieved, desired) alone (sparse: 1 inside the tolerance).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GoalObsWrapper(gym.Wrapper):
    def __init__(self, env, sparse: bool = True):
        super().__init__(env)
        base = env.unwrapped
        self.tol = float(getattr(base, "success_tol", 0.03))
        self.sparse = sparse
        self.dt = getattr(base, "dt", 0.01)   # gymnasium wrappers do not forward attributes
        self.observation_space = spaces.Dict(dict(
            observation=env.observation_space,
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        ))

    def _wrap(self, obs):
        base = self.env.unwrapped
        return dict(observation=np.asarray(obs, dtype=np.float32),
                    achieved_goal=np.asarray(base.ee, dtype=np.float32),
                    desired_goal=np.asarray(base.goal, dtype=np.float32))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._wrap(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self._wrap(obs), r, term, trunc, info

    def compute_reward(self, achieved_goal, desired_goal, info, **kw):
        d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        if self.sparse:
            return (d < self.tol).astype(np.float32)
        return (-2.0 * d + np.exp(-10.0 * d) + 2.0 * (d < self.tol)).astype(np.float32)
