"""
MuJoCo 실행/평가: 계획된 B-spline 궤적을 computed-torque 제어기로 실제 동역학 시뮬레이션에서 추종하고
충돌(접촉), 최종 목표 오차, 장애물 최소 거리, 추종 오차, 토크 포화 등을 측정한다. 오프스크린 렌더링 헬퍼 포함.
"""
from __future__ import annotations

import mujoco
import numpy as np

from tp_core import ArmKinematics, SplineBasis, build_model

METHOD_COLORS = {  # categorical slots in fixed order (validated palette)
    "learned": "#2a78d6",
    "learned+refine": "#eb6834",
    "trajopt": "#1baf7a",
    "linear": "#eda100",
}


def hex_rgba(h, a=1.0):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a], dtype=np.float32)


class SimScene:
    def __init__(self, model: mujoco.MjModel | None = None, kp=400.0, kd=40.0):
        self.model = model if model is not None else build_model()
        self.data = mujoco.MjData(self.model)
        self.kin = ArmKinematics(self.model)
        m = self.model
        self.kp, self.kd = kp, kd
        self.qadr, self.dadr = self.kin.qpos_adr, self.kin.dof_adr
        self.ee_body = self.kin.ee_body
        self.obstacle_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
        self.obstacle_geom = int(m.body_geomadr[self.obstacle_body])
        self.goal_mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "goal")])
        ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        base_geoms = [g for g in self.kin.robot_geoms if g not in self.kin.moving_geoms]
        self.moving = set(self.kin.moving_geoms)
        self.hazards = {self.obstacle_geom: "obstacle", ground: "ground", **{g: "base" for g in base_geoms}}

        # actuators driving the arm joints / the fingers
        trn = m.actuator_trnid[:, 0]
        self.arm_act = np.array([int(np.where(trn == j)[0][0]) for j in self.kin.joint_ids])
        self.finger_act = np.array([a for a in range(m.nu) if a not in self.arm_act])
        self.finger_dof = np.array([m.jnt_dofadr[trn[a]] for a in self.finger_act])
        self.gear = m.actuator_gear[:, 0].copy()
        self._M = np.zeros((m.nv, m.nv))

    def set_problem(self, q0, goal, obs_xy):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        m.body_pos[self.obstacle_body, :2] = obs_xy
        d.qpos[self.qadr] = q0
        d.mocap_pos[self.goal_mocap] = goal
        mujoco.mj_forward(m, d)

    def _control(self, q_d, qd_d, qdd_d):
        m, d = self.model, self.data
        q, qd = d.qpos[self.qadr], d.qvel[self.dadr]
        mujoco.mj_fullM(m, d, self._M)
        Ma = self._M[np.ix_(self.dadr, self.dadr)]
        acc = qdd_d + self.kp * (q_d - q) + self.kd * (qd_d - qd)
        tau = Ma @ acc + d.qfrc_bias[self.dadr] - d.qfrc_passive[self.dadr]
        u = tau / self.gear[self.arm_act]
        d.ctrl[self.arm_act] = np.clip(u, -1, 1)
        # fingers: hold closed
        d.ctrl[self.finger_act] = np.clip((-50 * d.qpos[self.finger_dof] - 2 * d.qvel[self.finger_dof])
                                          / self.gear[self.finger_act], -1, 1)
        return np.abs(u) > 1.0

    def _hazard_contacts(self):
        d = self.data
        hits = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.moving and g2 in self.hazards:
                hits.add(self.hazards[g2])
            elif g2 in self.moving and g1 in self.hazards:
                hits.add(self.hazards[g1])
        return hits

    def _obstacle_distance(self):
        fromto = np.zeros(6)
        return min(mujoco.mj_geomDistance(self.model, self.data, g, self.obstacle_geom, 0.5, fromto)
                   for g in self.moving)

    def rollout(self, ctrl_pts: np.ndarray, basis: SplineBasis, q0, goal, obs_xy, settle=0.5,
                frame_cb=None, frame_every=17, success_tol=0.02):
        """Track the spline defined by ctrl_pts (K,4). Returns (metrics dict, log dict)."""
        self.set_problem(q0, goal, obs_xy)
        m, d = self.model, self.data
        dt = m.opt.timestep
        n_steps = int(round((basis.duration + settle) / dt))
        ts = np.arange(n_steps) * dt
        s = np.clip(ts / basis.duration, 0, 1)
        moving = (ts < basis.duration)[:, None]
        Q = basis.matrix(s, 0) @ ctrl_pts
        QD = basis.matrix(s, 1) @ ctrl_pts * moving
        QDD = basis.matrix(s, 2) @ ctrl_pts * moving

        log_q, log_ee = np.zeros((n_steps, 4)), np.zeros((n_steps, 3))
        coll_steps, coll_kinds, sat_steps = 0, set(), 0
        min_dist = np.inf
        for k in range(n_steps):
            sat = self._control(Q[k], QD[k], QDD[k])
            sat_steps += bool(sat.any())
            mujoco.mj_step(m, d)
            log_q[k] = d.qpos[self.qadr]
            log_ee[k] = d.xpos[self.ee_body]
            hits = self._hazard_contacts()
            if hits:
                coll_steps += 1
                coll_kinds |= hits
            if k % 10 == 0:
                min_dist = min(min_dist, self._obstacle_distance())
            if frame_cb is not None and k % frame_every == 0:
                frame_cb(k * dt, log_ee[: k + 1])

        final_err = float(np.linalg.norm(log_ee[-1] - goal))
        collided = coll_steps > 0
        track = log_q - Q
        metrics = dict(
            success=bool(final_err < success_tol and not collided),
            final_err_mm=final_err * 1000,
            collision=collided,
            collision_kinds="|".join(sorted(coll_kinds)),
            collision_time_s=coll_steps * dt,
            min_obstacle_dist_cm=float(min_dist * 100),
            track_rmse_deg=float(np.rad2deg(np.sqrt((track ** 2).mean()))),
            track_max_deg=float(np.rad2deg(np.abs(track).max())),
            ee_path_len_m=float(np.linalg.norm(np.diff(log_ee, axis=0), axis=1).sum()),
            torque_sat_frac=sat_steps / n_steps,
        )
        log = dict(t=ts, q=log_q, q_des=Q, ee=log_ee)
        return metrics, log


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def default_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.15, 0.0, 0.15]
    cam.distance = 1.35
    cam.azimuth = 150
    cam.elevation = -28
    return cam


def add_polyline(scene: mujoco.MjvScene, pts: np.ndarray, rgba, width=0.004, stride=1):
    pts = np.asarray(pts, dtype=np.float64)[::stride]
    for a, b in zip(pts[:-1], pts[1:]):
        if scene.ngeom >= scene.maxgeom:
            return
        if np.linalg.norm(b - a) < 1e-6:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.ngeom += 1
