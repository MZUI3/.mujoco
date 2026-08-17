import numpy as np
import mujoco

class DomainRandomizedEnv(ManipulatorEnv):
    def __init__(self, xml_path="manipulator.xml", randomize=True):
        super().__init__(xml_path)
        self.randomize = randomize
        # 원본 파라미터 백업
        self._default_mass = self.model.body_mass.copy()
        self._default_damping = self.model.dof_damping.copy()
        self._default_friction = self.model.geom_friction.copy()

    def reset(self, seed=None, options=None):
        if self.randomize:
            self._randomize_dynamics()
        return super().reset(seed, options)

    def _randomize_dynamics(self):
        mass_scale = np.random.uniform(0.8, 1.2, size=self._default_mass.shape)
        self.model.body_mass[:] = self._default_mass * mass_scale

        damping_scale = np.random.uniform(0.7, 1.3, size=self._default_damping.shape)
        self.model.dof_damping[:] = self._default_damping * damping_scale

        friction_scale = np.random.uniform(0.8, 1.2, size=self._default_friction.shape)
        self.model.geom_friction[:] = self._default_friction * friction_scale

    def _get_obs(self):
        obs = super()._get_obs()
        if self.randomize:
            obs = obs + np.random.normal(0, 0.005, size=obs.shape)  # sensor noise
        return obs.astype(np.float32)