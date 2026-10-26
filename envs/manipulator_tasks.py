"""
yang_manipulator 로봇팔(4 DoF + 그리퍼)용 Gymnasium 과제 환경 모음.

모든 과제는 ManipulatorTaskEnv를 상속하고 같은 MuJoCo 모델을 쓴다. 과제마다 씬(물체/장애물/목표 마커),
관측, 보상, 성공 조건이 다르다. 행동은 팔 모터 4개(토크, -1~1)이고 Lift 과제만 그리퍼 1개가 추가된다.
제어 주기 10 ms(MuJoCo 2 ms × frame_skip 5), 에피소드 200 step(2초).

| id                    | 과제                                           | obs | act |
|-----------------------|------------------------------------------------|-----|-----|
| Reach-v0              | 무작위 목표 위치에 EE 도달 (dense)             | 17  | 4   |
| ReachSparse-v0        | 위와 같음, 희소 보상(3 cm 안에서만 +1)          | 17  | 4   |
| ReachObstacle-v0      | 기둥 장애물을 피해 도달, 접촉 패널티            | 21  | 4   |
| ReachMultiObstacle-v0 | 기둥 1~3개(크기 가변)를 피해 도달, 플래너 문제 분포 | 29  | 4   |
| ReachDR-v0            | Reach + 질량/댐핑/마찰/관측노이즈/행동스케일 랜덤 | 17  | 4   |
| ReachMoving-v0        | 원을 그리며 움직이는 목표 추적                  | 20  | 4   |
| Push-v0               | 바닥의 상자를 10~18 cm 떨어진 목표로 밀기       | 26  | 4   |
| Lift-v0               | 상자를 잡아 15 cm 이상 들기 (그리퍼 제어)       | 23  | 5   |
| TrackTrajectory-v0    | 무작위 B-spline 관절 궤적 추종                  | 21  | 4   |
| HoldPose-v0           | 외력 교란 속에서 목표 관절 자세 유지            | 19  | 4   |
| ApproachObjects-v0    | 기존 DomainRandomizedEnv (물체 2개 접근, 토크 6)| 38  | 6   |

info 에는 항상 success(bool), dist(float), collision(bool) 이 들어간다.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent.parent
ROBOT_XML = ROOT / "assets" / "yang_manipulator.xml"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll"]
FINGER_JOINTS = ["finger_left_joint", "finger_right_joint"]
MOVING_BODIES = ["link2", "link3", "wrist", "gripper_base", "finger_left", "finger_right"]


def _default_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.15, 0.0, 0.2]
    cam.distance = 1.6
    cam.azimuth = 150
    cam.elevation = -24
    return cam


class ManipulatorTaskEnv(gym.Env):
    """Base class: builds the scene, exposes helpers; subclasses define obs/reward/success."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 100}
    keep_box = False        # keep target_object1 (box) in the scene
    keep_obstacle = False   # keep the pole obstacle
    gripper_action = False  # 5th action controls the fingers
    max_steps = 200
    frame_skip = 5
    goal_marker_rgba = (0.2, 0.9, 0.3, 0.6)
    randomize_dynamics = False
    ground_is_hazard = True   # arm-ground contact counts as a collision (off for Push/Lift, where it is normal)

    def __init__(self, render_mode: str | None = None, seed: int | None = None, width=480, height=400):
        super().__init__()
        self.render_mode = render_mode
        self.width, self.height = width, height
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.dt = m.opt.timestep * self.frame_skip
        self.arm_jnt = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.arm_q = np.array([m.jnt_qposadr[j] for j in self.arm_jnt])
        self.arm_v = np.array([m.jnt_dofadr[j] for j in self.arm_jnt])
        self.q_lo, self.q_hi = m.jnt_range[self.arm_jnt, 0].copy(), m.jnt_range[self.arm_jnt, 1].copy()
        self.fing_jnt = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in FINGER_JOINTS]
        self.fing_q = np.array([m.jnt_qposadr[j] for j in self.fing_jnt])
        trn = m.actuator_trnid[:, 0]
        self.arm_act = np.array([int(np.where(trn == j)[0][0]) for j in self.arm_jnt])
        self.fing_act = np.array([int(np.where(trn == j)[0][0]) for j in self.fing_jnt])
        self.ee_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        self.goal_mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "goal")])
        self.ground_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        moving_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in MOVING_BODIES}
        self.moving_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in moving_ids}
        self.obstacle_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obstacle") if self.keep_obstacle else -1
        self.obstacle_geom = int(m.body_geomadr[self.obstacle_body]) if self.keep_obstacle else -1
        self.box_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_object1") if self.keep_box else -1
        if self.keep_box:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "target_free1")
            self.box_q, self.box_v = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
            self.box_geom = int(m.body_geomadr[self.box_body])
        self._default_mass = m.body_mass.copy()
        self._default_damping = m.dof_damping.copy()
        self._default_friction = m.geom_friction.copy()

        n_act = 5 if self.gripper_action else 4
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.goal = np.zeros(3)
        self._step = 0
        self._renderer = None
        self._viewer = None
        obs_dim = len(self._task_reset_and_obs(first=True))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ scene
    def _build_model(self):
        spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
        for name, keep in (("target_object1", self.keep_box), ("target_object2", False),
                           ("obstacle", self.keep_obstacle)):
            body = spec.body(name)
            if body is not None and not keep:
                spec.delete(body)
        if self.keep_obstacle:
            ob = spec.body("obstacle")
            ob.pos = [0.35, 0.0, 0.15]
            ob.geoms[0].size = [0.04, 0.15, 0.0]
            ob.geoms[0].rgba = [0.85, 0.35, 0.25, 1.0]
        goal = spec.worldbody.add_body(name="goal", mocap=True)
        goal.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], contype=0, conaffinity=0,
                      rgba=list(self.goal_marker_rgba))
        return spec.compile()

    # ------------------------------------------------------------------ helpers
    @property
    def q(self):
        return self.data.qpos[self.arm_q]

    @property
    def qd(self):
        return self.data.qvel[self.arm_v]

    @property
    def ee(self):
        return self.data.xpos[self.ee_body]

    def _fk(self, q):
        """EE position for arm angles q without disturbing the simulation state."""
        saved = self.data.qpos.copy()
        self.data.qpos[self.arm_q] = q
        mujoco.mj_kinematics(self.model, self.data)
        ee = self.data.xpos[self.ee_body].copy()
        self.data.qpos[:] = saved
        mujoco.mj_kinematics(self.model, self.data)
        return ee

    def _random_arm_q(self, lo_frac=0.85):
        lo = np.deg2rad([-170, -80, -140, -170])
        hi = np.deg2rad([170, 80, 140, 170])
        for _ in range(100):
            q = self.np_random.uniform(lo, hi)
            if self._fk(q)[2] > 0.10:
                return q
        return np.zeros(4)

    def _random_goal(self, z_lo=0.08, z_hi=0.40):
        """Reachable goal = EE of a random configuration inside the workspace box."""
        for _ in range(200):
            p = self._fk(self._random_arm_q())
            r = np.hypot(p[0], p[1])
            if z_lo < p[2] < z_hi and 0.25 < r < 0.55 and p[0] > -0.1:
                return p
        return np.array([0.4, 0.0, 0.2])

    def _set_goal(self, p):
        self.goal = np.asarray(p, dtype=np.float64)
        self.data.mocap_pos[self.goal_mocap] = self.goal

    def _randomize(self):
        m = self.model
        m.body_mass[:] = self._default_mass * self.np_random.uniform(0.8, 1.2, size=self._default_mass.shape)
        m.dof_damping[:] = self._default_damping * self.np_random.uniform(0.7, 1.3, size=self._default_damping.shape)
        m.geom_friction[:] = self._default_friction * self.np_random.uniform(0.8, 1.2, size=self._default_friction.shape)
        self._action_scale = self.np_random.uniform(0.85, 1.15, size=4)

    def _contacts(self):
        """(collision with ground/obstacle by moving arm links, contact between gripper and box)."""
        d = self.data
        hit, grasp = False, False
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            pair = {g1, g2}
            if pair & self.moving_geoms:
                if (self.ground_is_hazard and self.ground_geom in pair) or (self.keep_obstacle and self.obstacle_geom in pair):
                    hit = True
                if self.keep_box and self.box_geom in pair:
                    grasp = True
        return hit, grasp

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._action_scale = np.ones(4)
        if self.randomize_dynamics:
            self._randomize()
        self._step = 0
        self._collided = False
        obs = self._task_reset_and_obs()
        return obs, self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1, 1)
        self.data.ctrl[self.arm_act] = a[:4] * self._action_scale
        if self.gripper_action:
            self.data.ctrl[self.fing_act] = a[4]             # +1 open, -1 close
        else:
            self.data.ctrl[self.fing_act] = -0.3              # hold the fingers closed
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step += 1
        hit, grasp = self._contacts()
        self._hit, self._grasp = hit, grasp
        self._collided |= hit
        self._task_step()
        reward = self._reward(a)
        obs = self._obs()
        truncated = self._step >= self.max_steps
        return obs.astype(np.float32), float(reward), False, truncated, self._info()

    def _info(self):
        return dict(success=bool(self._success()), dist=float(self._dist()), collision=bool(self._collided))

    def _task_reset_and_obs(self, first=False):
        self._task_reset()
        self._hit = self._grasp = False
        self._collided = False
        return self._obs().astype(np.float32)

    def _task_step(self):
        pass

    def _base_obs(self):
        return [self.q, self.qd / 10.0, self.ee, self.goal, self.goal - self.ee]

    # ------------------------------------------------------------------ to override
    def _task_reset(self):
        raise NotImplementedError

    def _obs(self):
        raise NotImplementedError

    def _reward(self, a):
        raise NotImplementedError

    def _dist(self):
        return float(np.linalg.norm(self.ee - self.goal))

    def _success(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, self.height, self.width)
                self._cam = _default_camera()
            self._renderer.update_scene(self.data, self._cam)
            return self._renderer.render()
        if self.render_mode == "human":
            from mujoco import viewer as mj_viewer
            if self._viewer is None:
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
                cam = _default_camera()
                self._viewer.cam.lookat[:], self._viewer.cam.distance = cam.lookat, cam.distance
                self._viewer.cam.azimuth, self._viewer.cam.elevation = cam.azimuth, cam.elevation
            self._viewer.sync()
            return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# ====================================================================== tasks
class ReachEnv(ManipulatorTaskEnv):
    """Dense-reward reaching of a random goal."""
    success_tol = 0.03

    def _task_reset(self):
        self.data.qpos[self.arm_q] = self._random_arm_q()
        mujoco.mj_forward(self.model, self.data)
        self._set_goal(self._random_goal())

    def _obs(self):
        return np.concatenate(self._base_obs())

    def _reward(self, a):
        d = self._dist()
        return -2.0 * d + np.exp(-10.0 * d) + 2.0 * (d < self.success_tol) - 0.01 * float(a[:4] @ a[:4]) - 1.0 * self._hit

    def _success(self):
        return self._dist() < self.success_tol


class ReachSparseEnv(ReachEnv):
    """Same task, reward only inside the 3 cm goal ball."""

    def _reward(self, a):
        return 1.0 if self._dist() < self.success_tol else 0.0


class ReachObstacleEnv(ReachEnv):
    """Reach with a pole between start and goal; touching it costs reward and success."""
    keep_obstacle = True

    def _task_reset(self):
        super()._task_reset()
        s = self.np_random.uniform(0.35, 0.65)
        mid = s * self.ee[:2] + (1 - s) * self.goal[:2] + self.np_random.normal(0, 0.03, 2)
        r = np.linalg.norm(mid)
        if r < 0.22:
            mid = mid / max(r, 1e-6) * 0.22
        # keep the goal reachable: pole at least 9 cm from the goal in xy
        v = mid - self.goal[:2]
        if np.linalg.norm(v) < 0.13:
            mid = self.goal[:2] + v / max(np.linalg.norm(v), 1e-6) * 0.13
        self.model.body_pos[self.obstacle_body, :2] = mid
        mujoco.mj_forward(self.model, self.data)

    def _obs(self):
        ob = self.model.body_pos[self.obstacle_body]
        return np.concatenate(self._base_obs() + [ob[:2], [0.04, 0.30]])

    def _reward(self, a):
        return super()._reward(a) - 2.0 * self._hit

    def _success(self):
        return self._dist() < self.success_tol and not self._collided


class ReachMultiObstacleEnv(ReachObstacleEnv):
    """Reach around 1-3 poles of random size: the trajectory planner's problem distribution.

    Scene and problems come straight from trajectory_planning/tp_core.py (`build_model`, `ProblemSampler`):
    poles with radius 3-7 cm and height 15-40 cm, start/goal configurations collision-free. Observation =
    the ReachObstacle-v0 observation + 3 slots x (x, y, radius, height), empty slots all zero. Reward and
    success are those of ReachObstacle-v0 (contact penalty; success = within 3 cm with no contact).

    reset(options={"problem": (q0, goal, obstacles)}) replays a given problem (obstacles: (3, 4) rows
    (x, y, r, h), h = 0 for an empty slot); `env.problem` returns the current one."""
    keep_obstacle = False

    def _tp(self):
        import sys
        tp_dir = str(ROOT / "trajectory_planning")
        if tp_dir not in sys.path:
            sys.path.insert(0, tp_dir)
        import tp_core
        return tp_core

    def _build_model(self):
        tp = self._tp()
        self.max_obs = tp.MAX_OBS
        model = tp.build_model()
        self._kin = tp.ArmKinematics(model)
        self._cost = tp.TrajectoryCost(self._kin, tp.PlanConfig())
        self._obst_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{j}") for j in range(self.max_obs)]
        self._obst_geoms = [int(model.body_geomadr[b]) for b in self._obst_bodies]
        self._obst_contype = model.geom_contype[self._obst_geoms].copy()
        self._obst_conaff = model.geom_conaffinity[self._obst_geoms].copy()
        self._active_obst = set()
        self._pending_problem = None
        self.obstacles = np.zeros((self.max_obs, 4))
        return model

    @property
    def problem(self):
        """(q0, goal, obstacles) of the current episode (copies)."""
        return self._q0.copy(), self.goal.copy(), self.obstacles.copy()

    def reset(self, seed=None, options=None):
        self._pending_problem = options.get("problem") if options else None
        return super().reset(seed=seed, options=options)

    def _sample_problem(self):
        tp = self._tp()
        sampler = tp.ProblemSampler(self._kin, self._cost, seed=int(self.np_random.integers(2**31 - 1)))
        q0, goal, obs = sampler.sample(1)
        return q0[0].numpy().astype(np.float64), goal[0].numpy().astype(np.float64), obs[0].numpy().astype(np.float64)

    def _apply_obstacles(self, obs):
        m = self.model
        self._active_obst = set()
        self.obstacles = np.zeros((self.max_obs, 4))
        for j, (b, g) in enumerate(zip(self._obst_bodies, self._obst_geoms)):
            x, y, r, h = obs[j] if j < len(obs) else (0.0, 0.0, 0.0, 0.0)
            if h > 0:
                m.body_pos[b] = [x, y, h / 2]
                m.geom_size[g] = [r, h / 2, 0.0]
                m.geom_rbound[g] = np.hypot(r, h / 2)
                m.geom_contype[g], m.geom_conaffinity[g] = self._obst_contype[j], self._obst_conaff[j]
                m.geom_rgba[g, 3] = 1.0
                self._active_obst.add(g)
                self.obstacles[j] = [x, y, r, h]
            else:
                m.body_pos[b] = [2.0 + j, 2.0, -1.0]
                m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
                m.geom_rgba[g, 3] = 0.0

    def _task_reset(self):
        if self._pending_problem is not None:
            q0, goal, obs = (np.asarray(x, dtype=np.float64) for x in self._pending_problem)
            self._pending_problem = None
        else:
            q0, goal, obs = self._sample_problem()
        self._q0 = q0.copy()
        self._apply_obstacles(obs.reshape(-1, 4))
        self.data.qpos[self.arm_q] = q0
        self.data.qvel[self.arm_v] = 0.0
        self._set_goal(goal)
        mujoco.mj_forward(self.model, self.data)

    def _contacts(self):
        hit, grasp = super()._contacts()      # ground contact (keep_obstacle is False here)
        if not hit and self._active_obst:
            d = self.data
            for i in range(d.ncon):
                pair = {int(d.contact[i].geom1), int(d.contact[i].geom2)}
                if pair & self.moving_geoms and pair & self._active_obst:
                    hit = True
                    break
        return hit, grasp

    def _obs(self):
        return np.concatenate(self._base_obs() + [self.obstacles.reshape(-1)])


class ReachDREnv(ReachEnv):
    """Reach with domain randomisation: dynamics, action scale and observation noise."""
    randomize_dynamics = True

    def _obs(self):
        o = np.concatenate(self._base_obs())
        return o + self.np_random.normal(0, 0.005, size=o.shape)


class ReachMovingEnv(ReachEnv):
    """Goal moves on a horizontal circle; reward tracks the moving goal."""
    success_tol = 0.04

    def _task_reset(self):
        self.data.qpos[self.arm_q] = self._random_arm_q()
        mujoco.mj_forward(self.model, self.data)
        c = self._random_goal(z_lo=0.12, z_hi=0.35)
        self._center, self._radius = c, self.np_random.uniform(0.05, 0.10)
        self._omega = self.np_random.choice([-1, 1]) * self.np_random.uniform(1.0, 2.0)   # rad/s
        self._phase = self.np_random.uniform(0, 2 * np.pi)
        self._update_goal()

    def _update_goal(self):
        t = self._step * self.dt
        ang = self._phase + self._omega * t
        self._set_goal(self._center + self._radius * np.array([np.cos(ang), np.sin(ang), 0.0]))
        self._goal_vel = self._radius * self._omega * np.array([-np.sin(ang), np.cos(ang), 0.0])

    def _task_step(self):
        self._update_goal()

    def _obs(self):
        return np.concatenate(self._base_obs() + [self._goal_vel])

    def _success(self):
        return self._dist() < self.success_tol


class PushEnv(ManipulatorTaskEnv):
    """Push the box on the ground to a goal spot."""
    keep_box = True
    ground_is_hazard = False
    goal_marker_rgba = (0.9, 0.3, 0.9, 0.5)
    success_tol = 0.05

    def _task_reset(self):
        self.data.qpos[self.arm_q] = np.deg2rad([0, 20, 60, 0]) + self.np_random.normal(0, 0.05, 4)
        ang = self.np_random.uniform(-0.6, 0.6)
        r = self.np_random.uniform(0.32, 0.42)
        box = np.array([r * np.cos(ang), r * np.sin(ang), 0.03])
        self.data.qpos[self.box_q:self.box_q + 3] = box
        self.data.qpos[self.box_q + 3:self.box_q + 7] = [1, 0, 0, 0]
        self.data.qvel[self.box_v:self.box_v + 6] = 0
        g_ang = ang + self.np_random.uniform(-0.4, 0.4)
        g_r = np.clip(r + self.np_random.uniform(0.10, 0.18), 0.3, 0.55)
        self._set_goal([g_r * np.cos(g_ang), g_r * np.sin(g_ang), 0.03])
        mujoco.mj_forward(self.model, self.data)

    @property
    def box(self):
        return self.data.xpos[self.box_body]

    def _obs(self):
        bv = self.data.qvel[self.box_v:self.box_v + 3]
        return np.concatenate([self.q, self.qd / 10.0, self.ee, self.goal, self.box, bv, self.box - self.goal,
                               self.box - self.ee])

    def _dist(self):
        return float(np.linalg.norm(self.box[:2] - self.goal[:2]))

    def _reward(self, a):
        d_ee = np.linalg.norm(self.ee - self.box)
        d = self._dist()
        return -1.0 * d_ee + np.exp(-10 * d_ee) - 3.0 * d + 2.0 * (d < self.success_tol) - 0.01 * float(a[:4] @ a[:4])

    def _success(self):
        return self._dist() < self.success_tol


class LiftEnv(PushEnv):
    """Grasp the box and lift it above 15 cm (5th action drives the gripper)."""
    gripper_action = True
    lift_height = 0.15
    goal_marker_rgba = (0.9, 0.3, 0.9, 0.5)

    def _task_reset(self):
        super()._task_reset()
        self.data.ctrl[self.fing_act] = 1.0                  # start open
        self._set_goal(self.box + np.array([0, 0, self.lift_height + 0.05]))
        mujoco.mj_forward(self.model, self.data)

    def _obs(self):
        bv = self.data.qvel[self.box_v:self.box_v + 3]
        return np.concatenate([self.q, self.qd / 10.0, self.ee, self.box, bv, self.box - self.ee,
                               self.data.qpos[self.fing_q], [self.box[2] - 0.03]])

    def _dist(self):
        return float(max(self.lift_height - (self.box[2] - 0.03), 0.0))

    def _reward(self, a):
        d_ee = np.linalg.norm(self.ee - self.box)
        lift = np.clip(self.box[2] - 0.03, 0, self.lift_height)
        return np.exp(-10 * d_ee) - d_ee + 1.0 * self._grasp + 10.0 * lift + 2.0 * self._success() - 0.01 * float(a @ a)

    def _success(self):
        return self.box[2] - 0.03 > self.lift_height


class TrackTrajectoryEnv(ManipulatorTaskEnv):
    """Follow a random smooth joint-space reference q_ref(t) over the episode."""
    success_tol = 0.10   # rad, mean tracking error

    def _task_reset(self):
        q0, q1 = self._random_arm_q(), self._random_arm_q()
        qm = 0.5 * (q0 + q1) + self.np_random.normal(0, 0.3, 4)
        T = self.max_steps
        s = np.linspace(0, 1, T + 2)[:, None]
        # smooth 3-point curve (quadratic Bezier) with zero velocity at both ends via smoothstep time
        u = 3 * s ** 2 - 2 * s ** 3
        self._ref = (1 - u) ** 2 * q0 + 2 * (1 - u) * u * qm + u ** 2 * q1
        self._ref = np.clip(self._ref, self.q_lo + 0.05, self.q_hi - 0.05)
        self.data.qpos[self.arm_q] = self._ref[0]
        mujoco.mj_forward(self.model, self.data)
        self._set_goal(self._fk(self._ref[-1]))
        self._err_hist = []

    def _ref_at(self, k):
        return self._ref[min(k, len(self._ref) - 1)]

    def _task_step(self):
        self._err_hist.append(float(np.linalg.norm(self.q - self._ref_at(self._step))))

    def _obs(self):
        k = self._step
        r0, r1 = self._ref_at(k), self._ref_at(k + 1)
        return np.concatenate([self.q, self.qd / 10.0, r0, r1, r0 - self.q, [k / self.max_steps]])

    def _dist(self):
        return float(np.mean(self._err_hist)) if self._err_hist else float(np.linalg.norm(self.q - self._ref[0]))

    def _reward(self, a):
        e = np.linalg.norm(self.q - self._ref_at(self._step))
        return 1.0 - 5.0 * min(e, 1.0) - 0.01 * float(a[:4] @ a[:4])

    def _success(self):
        return self._dist() < self.success_tol


class HoldPoseEnv(ManipulatorTaskEnv):
    """Hold a target joint pose while random pushes hit the wrist every 0.5 s."""
    success_tol = 0.10

    def _task_reset(self):
        self._q_tgt = self._random_arm_q()
        self.data.qpos[self.arm_q] = self._q_tgt + self.np_random.normal(0, 0.05, 4)
        mujoco.mj_forward(self.model, self.data)
        self._set_goal(self._fk(self._q_tgt))
        self._force = np.zeros(3)
        self._err_hist = []
        self._wrist = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist")

    def _task_step(self):
        if self._step % 50 == 1:
            self._force = self.np_random.normal(0, 1, 3)
            self._force *= self.np_random.uniform(5.0, 15.0) / max(np.linalg.norm(self._force), 1e-6)
        if self._step % 50 == 26:
            self._force = np.zeros(3)
        self.data.xfrc_applied[self._wrist, :3] = self._force
        self._err_hist.append(float(np.linalg.norm(self.q - self._q_tgt)))

    def _obs(self):
        return np.concatenate([self.q, self.qd / 10.0, self._q_tgt, self._q_tgt - self.q, self._force / 15.0])

    def _dist(self):
        return float(np.mean(self._err_hist[-50:])) if self._err_hist else float(np.linalg.norm(self.q - self._q_tgt))

    def _reward(self, a):
        return 1.0 - 5.0 * min(np.linalg.norm(self.q - self._q_tgt), 1.0) - 0.01 * float(a[:4] @ a[:4])

    def _success(self):
        return self._dist() < self.success_tol


class ApproachObjectsEnv(gym.Wrapper):
    """The original DomainRandomizedEnv (two free objects, 6 torque actions, 1000 steps) with the
    same info keys as the other tasks. Episodes are truncated at 200 steps here to match."""

    def __init__(self, render_mode=None, randomize=True, max_steps=200, **kw):
        import sys
        sys.path.insert(0, str(ROOT))
        from manipulator_env import DomainRandomizedEnv
        env = DomainRandomizedEnv(randomize=randomize, render_mode=render_mode)
        super().__init__(env)
        self.max_steps = max_steps
        self.dt = float(env.model.opt.timestep)
        self._renderer = None

    def _dist(self):
        e = self.env
        g = e.data.xpos[e.gripper_body_id][:2]
        t = e.data.xpos[e.target_body_ids][:, :2]
        return float(np.min(np.linalg.norm(t - g, axis=1)))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._k = 0
        d = self._dist()
        return obs, dict(success=d < 0.05, dist=d, collision=False)

    def step(self, action):
        obs, r, term, trunc, _ = self.env.step(action)
        self._k += 1
        d = self._dist()
        return obs, r, term, trunc or self._k >= self.max_steps, dict(success=d < 0.05, dist=d, collision=False)

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.env.model, 400, 480)
                self._cam = _default_camera()
            self._renderer.update_scene(self.env.data, self._cam)
            return self._renderer.render()
        return self.env.render()


TASKS = {
    "Reach-v0": ReachEnv,
    "ReachSparse-v0": ReachSparseEnv,
    "ReachObstacle-v0": ReachObstacleEnv,
    "ReachMultiObstacle-v0": ReachMultiObstacleEnv,
    "ReachDR-v0": ReachDREnv,
    "ReachMoving-v0": ReachMovingEnv,
    "Push-v0": PushEnv,
    "Lift-v0": LiftEnv,
    "TrackTrajectory-v0": TrackTrajectoryEnv,
    "HoldPose-v0": HoldPoseEnv,
    "ApproachObjects-v0": ApproachObjectsEnv,
}

TASK_DESCRIPTIONS = {
    "Reach-v0": "무작위 목표 위치에 EE 도달 (dense 보상, 성공 3 cm)",
    "ReachSparse-v0": "Reach와 같은 과제, 희소 보상 (3 cm 안에서만 +1)",
    "ReachObstacle-v0": "시작-목표 사이 기둥(4 cm × 30 cm)을 피해 도달, 접촉 시 패널티·실패",
    "ReachMultiObstacle-v0": "기둥 1~3개(반지름 3~7 cm, 높이 15~40 cm)를 피해 도달, 궤적 플래너와 같은 문제 분포, 접촉 시 패널티·실패",
    "ReachDR-v0": "Reach + 질량·댐핑·마찰 ±20~30 %, 행동 스케일 ±15 %, 관측 노이즈",
    "ReachMoving-v0": "반지름 5~10 cm 원을 1~2 rad/s로 도는 목표 추적 (성공 4 cm)",
    "Push-v0": "바닥의 상자(6 cm)를 10~18 cm 떨어진 목표로 밀기 (성공 5 cm)",
    "Lift-v0": "상자를 잡아 15 cm 이상 들기, 그리퍼 열기/닫기 포함 (행동 5)",
    "TrackTrajectory-v0": "무작위 2초 관절 궤적 추종 (성공: 평균 오차 < 0.1 rad)",
    "HoldPose-v0": "0.5초마다 손목에 5~15 N 외력, 목표 자세 유지 (성공: 마지막 0.5초 평균 오차 < 0.1 rad)",
    "ApproachObjects-v0": "기존 DomainRandomizedEnv: 무작위 물체 2개에 접근, 토크 6, 동역학 랜덤화",
}


def make_env(task_id: str, **kwargs) -> gym.Env:
    return TASKS[task_id](**kwargs)


if __name__ == "__main__":
    for tid, cls in TASKS.items():
        env = make_env(tid)
        obs, info = env.reset(seed=0)
        ret = 0.0
        for _ in range(env.max_steps if hasattr(env, "max_steps") else 200):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            ret += r
            if term or trunc:
                break
        print(f"{tid:22s} obs {obs.shape} act {env.action_space.shape} | random-policy return {ret:8.1f} "
              f"success={info['success']} dist={info['dist']:.3f} collision={info['collision']}")
        env.close()
