"""Gymnasium registration for the manipulator task suite.

    import envs; env = gymnasium.make("Yang/Reach-v0")
or  from envs import make_env; env = make_env("Reach-v0")
"""
from gymnasium.envs.registration import register, registry

from .manipulator_tasks import TASKS, TASK_DESCRIPTIONS, make_env  # noqa: F401

for _tid, _cls in TASKS.items():
    _gym_id = f"Yang/{_tid}"
    if _gym_id not in registry:
        register(id=_gym_id, entry_point=f"envs.manipulator_tasks:{_cls.__name__}", max_episode_steps=200)
