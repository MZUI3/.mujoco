"""Episode-context wrapper: appends normalised time k / max_steps and the start pose q0 to the observation.

A trajectory plan is a function of (q0, goal, obstacles) and is tracked against a clock, so a policy that
imitates "plan + tracking controller" needs both the clock and q0; neither is in the task observation.
Used by rl_zoo/train_bc.py (config "episode_context": true); evaluate_zoo / visualize_zoo wrap the env
accordingly. Both features are deterministic functions of the episode history, so the wrapped policy is a
reactive policy with a trivial memory (cf. SB3's TimeFeatureWrapper for fixed-horizon tasks)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

N_CONTEXT = 5   # time (1) + q0 (4)


class EpisodeContextWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped
        self.dt = getattr(base, "dt", 0.01)                 # gymnasium wrappers do not forward attributes
        self.max_steps = int(getattr(base, "max_steps", 200))
        n = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n + N_CONTEXT,), dtype=np.float32)
        self._k = 0
        self._q0 = np.zeros(4, dtype=np.float32)

    @property
    def problem(self):
        return self.env.unwrapped.problem

    def _wrap(self, obs):
        return np.concatenate([np.asarray(obs, dtype=np.float32), [self._k / self.max_steps], self._q0]).astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._k = 0
        self._q0 = np.asarray(self.env.unwrapped.q, dtype=np.float32).copy()
        return self._wrap(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._k += 1
        return self._wrap(obs), r, term, trunc, info


def context_features(episode_starts: np.ndarray, q0_per_episode: np.ndarray, max_steps: int = 200) -> np.ndarray:
    """Same features for an offline dataset: (T,) episode-start flags + (E, 4) start poses -> (T, N_CONTEXT)."""
    ep = np.cumsum(episode_starts) - 1
    k = np.arange(len(episode_starts)) - np.flatnonzero(episode_starts)[ep]
    return np.concatenate([(k / max_steps)[:, None], q0_per_episode[ep]], axis=1).astype(np.float32)
