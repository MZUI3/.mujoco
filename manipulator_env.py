import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


class DomainRandomizedEnv(gym.Env):
    """Stable MuJoCo manipulator environment with a fixed base and shaped rewards."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, xml_path=None, randomize=True, render_mode=None):
        super().__init__()

        self.randomize = randomize
        self.render_mode = render_mode
        self.viewer = None

        if xml_path is None:
            default_path = Path(__file__).parent / "mujoco200_win64" / "model" / "yang" / "yang_manipulator.xml"
            if not default_path.exists():
                raise FileNotFoundError(f"기본 모델을 찾을 수 없습니다: {default_path}")
            xml_path = str(default_path)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self._default_mass = self.model.body_mass.copy()
        self._default_damping = self.model.dof_damping.copy()
        self._default_friction = self.model.geom_friction.copy()

        self.n_dof = self.model.nv
        self.n_ctrl = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_ctrl,), dtype=np.float32)

        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Body/joint identifiers
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")

        # Two target objects (free joints named in XML)
        self.target_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free2"),
        ]
        # their qpos addresses (free joint qpos are 7 long: 3 pos + 4 quat)
        self.target_qpos_addrs = [int(self.model.jnt_qposadr[jid]) for jid in self.target_joint_ids]

        # obstacle body id
        self.obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")

        self.base_start_pos = self.data.xpos[self.base_body_id].copy()
        self.ctrl_scale = np.maximum(
            np.abs(self.model.actuator_ctrlrange[:, 0]),
            np.abs(self.model.actuator_ctrlrange[:, 1]),
        ).astype(np.float32)
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize target object positions within reachable workspace box
        if self.randomize:
            self._randomize_dynamics()

        # sample target positions in a semicircular workspace in front of the arm
        for adr in self.target_qpos_addrs:
            x = np.random.uniform(0.30, 0.55)
            y = np.random.uniform(-0.18, 0.18)
            z = np.random.uniform(0.06, 0.12)
            # set translation (3) and quaternion (4: identity)
            self.data.qpos[adr:adr+3] = np.array([x, y, z], dtype=np.float64)
            self.data.qpos[adr+3:adr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Small random perturbation to obstacle (static body pos change)
        obs_x = np.random.uniform(0.30, 0.40)
        obs_y = np.random.uniform(-0.05, 0.05)
        if self.obstacle_body_id >= 0:
            self.model.body_pos[self.obstacle_body_id, :2] = np.array([obs_x, obs_y])

        # forward to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action * self.ctrl_scale

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False
        truncated = self._step_count >= 1000
        return obs, reward, terminated, truncated, {}

    def _randomize_dynamics(self):
        mass_scale = np.random.uniform(0.85, 1.15, size=self._default_mass.shape)
        self.model.body_mass[:] = self._default_mass * mass_scale

        damping_scale = np.random.uniform(0.8, 1.2, size=self._default_damping.shape)
        self.model.dof_damping[:] = self._default_damping * damping_scale

        friction_scale = np.random.uniform(0.8, 1.15, size=self._default_friction.shape)
        self.model.geom_friction[:] = self._default_friction * friction_scale

    def _get_obs(self):
        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        if self.randomize:
            obs = obs + np.random.normal(0, 0.005, size=obs.shape).astype(np.float32)
        return obs

    def _compute_reward(self):
        base_xy = self.data.xpos[self.base_body_id][:2]
        gripper_xy = self.data.xpos[self.gripper_body_id][:2]
        target_xy = self.data.xpos[self.target_body_id][:2]

        distance_to_target = np.linalg.norm(gripper_xy - target_xy)
        approach_bonus = np.exp(-4.0 * distance_to_target)
        base_drift = np.linalg.norm(base_xy - self.base_start_pos[:2])
        action_penalty = 0.02 * np.sum(self.data.ctrl ** 2)
        velocity_penalty = 0.08 * np.sum(np.abs(self.data.qvel))

        reward = 2.5 * approach_bonus - 2.5 * base_drift - action_penalty - velocity_penalty
        return float(reward)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except AttributeError:
                    self.viewer = mujoco.viewer.launch(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


if __name__ == "__main__":
    print("=" * 60)
    print("MuJoCo 환경 단독 테스트 시작")
    print("=" * 60)

    env = DomainRandomizedEnv(randomize=False, render_mode=None)
    print(f"✓ 환경 생성 성공")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Control inputs (nu): {env.n_ctrl}")
    print(f"  - DoF/Velocities (nv): {env.n_dof}")

    obs, _ = env.reset()
    print(f"\n✓ reset() 성공")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Observation 첫 5개 값: {obs[:5]}")

    print(f"\n✓ step() 테스트 (10 스텝):")
    for step_idx in range(10):
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(random_action)
        base_xy = env.data.xpos[env.base_body_id][:2]
        print(f"  Step {step_idx+1}: reward={reward:.4f}, base_xy={np.round(base_xy, 4).tolist()}, truncated={truncated}")

    env.close()
    print(f"\n✓ 모든 테스트 완료!")
    print("=" * 60)