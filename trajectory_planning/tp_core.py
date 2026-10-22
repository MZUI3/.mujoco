"""
Trajectory planning core: MuJoCo scene, differentiable kinematics, cost, planners.

문제 정의
    시작 관절각 q0, 목표 엔드이펙터 위치 goal(xyz), 원기둥 장애물 위치 obstacle(xy)가 주어지면
    T초 동안 장애물/바닥/베이스와 충돌하지 않고 goal에 도달하는 부드러운 관절 궤적 q(t)를 만든다.

궤적 표현
    clamped cubic B-spline (K개 제어점). c0=c1=q0, c_{K-2}=c_{K-1}=q_end 로 고정하여
    시작/끝 속도가 0이 되도록 한다. 플래너는 q_end와 내부 제어점만 결정한다.

학습 방식
    전문가 데모 없이, MuJoCo 모델에서 추출한 미분 가능한 순기구학(torch)으로 계산한
    궤적 비용(목표 오차 + 충돌 + 관절 한계 + 가속도/속도)을 직접 최소화한다 (self-supervised).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parent.parent
ROBOT_XML = ROOT / "mujoco200_win64" / "model" / "yang" / "yang_manipulator.xml"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll"]
EE_BODY = "gripper_base"
# bodies that never move relative to the base (excluded from collision spheres)
STATIC_ROBOT_BODIES = {"base", "link1"}

OBST_RADIUS = 0.04
OBST_HALF_HEIGHT = 0.15
BASE_CYL = dict(xy=(0.0, 0.0), radius=0.12, z_lo=0.04, z_hi=0.17)


@dataclass
class PlanConfig:
    n_ctrl: int = 10          # B-spline control points
    n_samples: int = 64       # time samples used by the cost
    duration: float = 2.0     # seconds
    safety_margin: float = 0.02
    w_goal_lin: float = 50.0
    w_goal_sq: float = 500.0
    w_collision: float = 3000.0
    w_terminal_collision: float = 3000.0  # final pose is held after arrival -> penalise it on its own
    w_limits: float = 200.0
    w_acc: float = 2e-3
    w_vel: float = 2e-2


# --------------------------------------------------------------------------- #
# MuJoCo scene
# --------------------------------------------------------------------------- #
def build_model() -> mujoco.MjModel:
    """Robot from the existing XML, free objects removed, tall pole obstacle + goal marker."""
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    for name in ("target_object1", "target_object2"):
        body = spec.body(name)
        if body is not None:
            spec.delete(body)

    obstacle = spec.body("obstacle")
    obstacle.pos = [0.35, 0.0, OBST_HALF_HEIGHT]
    geom = obstacle.geoms[0]
    geom.size = [OBST_RADIUS, OBST_HALF_HEIGHT, 0.0]
    geom.rgba = [0.85, 0.35, 0.25, 1.0]

    goal = spec.worldbody.add_body(name="goal", mocap=True)
    goal.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0],
                  contype=0, conaffinity=0, rgba=[0.2, 0.9, 0.3, 0.7])
    return spec.compile()


# --------------------------------------------------------------------------- #
# Differentiable kinematics (extracted from the MuJoCo model)
# --------------------------------------------------------------------------- #
def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def _axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues. axis (3,), angle (...) -> (..., 3, 3)."""
    x, y, z = axis
    K = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=angle.dtype)
    s = torch.sin(angle)[..., None, None]
    c = torch.cos(angle)[..., None, None]
    eye = torch.eye(3, dtype=angle.dtype)
    return eye + s * K + (1 - c) * (K @ K)


class ArmKinematics:
    """Batched forward kinematics + collision spheres for the arm joints."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        rng = model.jnt_range[self.joint_ids].copy()
        self.q_lo = torch.tensor(rng[:, 0], dtype=torch.float32)
        self.q_hi = torch.tensor(rng[:, 1], dtype=torch.float32)

        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)

        # bodies in the robot subtree, parent-before-child (MuJoCo order guarantees this)
        self.bodies = [b for b in range(model.nbody) if self._is_descendant(b, base_id)]
        self.root_R = torch.tensor(data.xmat[base_id].reshape(3, 3), dtype=torch.float32)
        self.root_p = torch.tensor(data.xpos[base_id], dtype=torch.float32)

        self.body_info = {}
        for b in self.bodies:
            if b == base_id:
                continue
            joint = None
            for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
                if j in self.joint_ids:
                    joint = (self.joint_ids.index(j),
                             torch.tensor(model.jnt_axis[j], dtype=torch.float32),
                             torch.tensor(model.jnt_pos[j], dtype=torch.float32))
                # finger slide joints are held at 0 -> treated as fixed
            self.body_info[b] = dict(
                parent=int(model.body_parentid[b]),
                R=torch.tensor(_quat_to_mat(model.body_quat[b]), dtype=torch.float32),
                p=torch.tensor(model.body_pos[b], dtype=torch.float32),
                joint=joint,
            )
        self.base_id = base_id

        # collision spheres from geoms of moving bodies
        sph_body, sph_pos, sph_rad = [], [], []
        self.robot_geoms = []
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            if b not in self.bodies:
                continue
            self.robot_geoms.append(g)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name in STATIC_ROBOT_BODIES:
                continue
            for p, r in self._geom_spheres(g):
                sph_body.append(b)
                sph_pos.append(p)
                sph_rad.append(r)
        self.sph_body = np.array(sph_body)
        self.sph_pos = torch.tensor(np.array(sph_pos), dtype=torch.float32)
        self.sph_rad = torch.tensor(np.array(sph_rad), dtype=torch.float32)
        self.moving_geoms = [g for g in self.robot_geoms
                             if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                  int(model.geom_bodyid[g])) not in STATIC_ROBOT_BODIES]

    def _is_descendant(self, b, root):
        while b != 0:
            if b == root:
                return True
            b = int(self.model.body_parentid[b])
        return root == 0

    def _geom_spheres(self, g):
        m = self.model
        gtype, size = m.geom_type[g], m.geom_size[g]
        R = _quat_to_mat(m.geom_quat[g])
        c = m.geom_pos[g]
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r, h = size[0], size[1]
            n = max(2, int(np.ceil(2 * h / r)) + 1)
            return [(c + R @ np.array([0, 0, s]), r) for s in np.linspace(-h, h, n)]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            ax = int(np.argmax(size))
            others = [i for i in range(3) if i != ax]
            r = float(np.linalg.norm(size[others]))
            h = max(size[ax] - r * 0.5, 0.0)
            n = max(1, int(np.ceil(2 * h / r)) + 1)
            pts = []
            for s in np.linspace(-h, h, n):
                off = np.zeros(3)
                off[ax] = s
                pts.append((c + R @ off, r))
            return pts
        return [(c, float(m.geom_rbound[g]))]

    def forward(self, q: torch.Tensor):
        """q (..., 4) -> dict body_id -> (R (...,3,3), p (...,3))."""
        batch = q.shape[:-1]
        frames = {self.base_id: (self.root_R.expand(*batch, 3, 3), self.root_p.expand(*batch, 3))}
        for b, info in self.body_info.items():
            Rp, pp = frames[info["parent"]]
            R = Rp @ info["R"]
            p = pp + (Rp @ info["p"])
            if info["joint"] is not None:
                idx, axis, jpos = info["joint"]
                Rj = _axis_angle(axis, q[..., idx])
                R_new = R @ Rj
                p = p + (R @ jpos) - (R_new @ jpos)
                R = R_new
            frames[b] = (R, p)
        return frames

    def ee_pos(self, q):
        return self.forward(q)[self.ee_body][1]

    def ee_jacobian(self, q):
        """Analytic positional Jacobian. q (...,4) -> ee (...,3), J (...,3,4)."""
        frames = self.forward(q)
        p_ee = frames[self.ee_body][1]
        cols = [None] * len(self.joint_ids)
        for b, info in self.body_info.items():
            if info["joint"] is None:
                continue
            idx, axis, jpos = info["joint"]
            R, p = frames[b]
            cols[idx] = torch.cross(R @ axis, p_ee - (p + R @ jpos), dim=-1)
        return p_ee, torch.stack(cols, dim=-1)

    def dls_step(self, q, goal, damping=0.03, max_step=0.3):
        """One damped-least-squares IK step towards goal (differentiable)."""
        ee, J = self.ee_jacobian(q)
        e = goal - ee
        JJt = J @ J.transpose(-1, -2) + damping ** 2 * torch.eye(3)
        dq = (J.transpose(-1, -2) @ torch.linalg.solve(JJt, e[..., None]))[..., 0]
        return torch.clamp(q + dq.clamp(-max_step, max_step), self.q_lo + 0.02, self.q_hi - 0.02)

    def spheres(self, q):
        """q (..., 4) -> sphere centers (..., S, 3), radii (S,)."""
        frames = self.forward(q)
        centers = []
        for i, b in enumerate(self.sph_body):
            R, p = frames[int(b)]
            centers.append(p + R @ self.sph_pos[i])
        return torch.stack(centers, dim=-2), self.sph_rad


# --------------------------------------------------------------------------- #
# B-spline trajectory parametrisation
# --------------------------------------------------------------------------- #
class SplineBasis:
    def __init__(self, n_ctrl: int, duration: float, degree: int = 3):
        self.k = degree
        self.n_ctrl = n_ctrl
        self.duration = duration
        n_inner = n_ctrl - degree - 1
        inner = np.linspace(0, 1, n_inner + 2)[1:-1]
        self.knots = np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])
        # Greville abscissae: where each control point "lives" in time
        self.greville = np.array([self.knots[i + 1:i + 1 + degree].mean() for i in range(n_ctrl)])

    def matrix(self, ts: np.ndarray, deriv: int = 0) -> np.ndarray:
        """Basis matrix (len(ts), n_ctrl) for normalised time ts in [0,1]; derivatives w.r.t. seconds."""
        ts = np.clip(ts, 0.0, 1.0)
        B = np.zeros((len(ts), self.n_ctrl))
        for i in range(self.n_ctrl):
            e = np.zeros(self.n_ctrl)
            e[i] = 1.0
            spl = BSpline(self.knots, e, self.k, extrapolate=False)
            if deriv:
                spl = spl.derivative(deriv)
            vals = spl(ts)
            B[:, i] = np.nan_to_num(vals)
        # BSpline evaluates the right end to 0 for the last basis; fix t==1 exactly
        if deriv == 0:
            B[ts >= 1.0] = 0.0
            B[ts >= 1.0, -1] = 1.0
        return B / (self.duration ** deriv)

    def evaluate(self, ctrl: np.ndarray, t: float):
        """ctrl (K, D), t seconds -> q, qd, qdd at t (each (D,))."""
        s = np.array([min(max(t / self.duration, 0.0), 1.0)])
        out = []
        for d in range(3):
            if t >= self.duration and d > 0:
                out.append(np.zeros(ctrl.shape[1]))
            else:
                out.append((self.matrix(s, d) @ ctrl)[0])
        return out


def assemble_ctrl(q0: torch.Tensor, q_end: torch.Tensor, inner_delta: torch.Tensor,
                  basis: SplineBasis) -> torch.Tensor:
    """q0,q_end (B,D); inner_delta (B,K-4,D) -> control points (B,K,D)."""
    g = torch.tensor(basis.greville[2:-2], dtype=q0.dtype)[None, :, None]
    inner = q0[:, None] + g * (q_end - q0)[:, None] + inner_delta
    return torch.cat([q0[:, None], q0[:, None], inner, q_end[:, None], q_end[:, None]], dim=1)


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def capped_cylinder_sdf(p: torch.Tensor, xy: torch.Tensor, radius: float, z_lo: float, z_hi: float):
    """Signed distance from points p (...,3) to a vertical cylinder; xy broadcastable to (...,2)."""
    d_r = torch.linalg.norm(p[..., :2] - xy, dim=-1) - radius
    zc, hh = 0.5 * (z_lo + z_hi), 0.5 * (z_hi - z_lo)
    d_z = torch.abs(p[..., 2] - zc) - hh
    outside = torch.sqrt(torch.clamp(d_r, min=0) ** 2 + torch.clamp(d_z, min=0) ** 2 + 1e-12)
    inside = torch.clamp(torch.maximum(d_r, d_z), max=0)
    return outside + inside


class TrajectoryCost:
    def __init__(self, kin: ArmKinematics, cfg: PlanConfig):
        self.kin, self.cfg = kin, cfg
        self.basis = SplineBasis(cfg.n_ctrl, cfg.duration)
        ts = np.linspace(0, 1, cfg.n_samples)
        self.B0 = torch.tensor(self.basis.matrix(ts, 0), dtype=torch.float32)
        self.B1 = torch.tensor(self.basis.matrix(ts, 1), dtype=torch.float32)
        self.B2 = torch.tensor(self.basis.matrix(ts, 2), dtype=torch.float32)
        self.base_xy = torch.tensor(BASE_CYL["xy"], dtype=torch.float32)

    def clearance(self, q: torch.Tensor, obs_xy: torch.Tensor):
        """q (B,T,D), obs_xy (B,2) -> signed clearance of every sphere (B,T,S) (min over obstacles)."""
        centers, rad = self.kin.spheres(q)
        d_obs = capped_cylinder_sdf(centers, obs_xy[:, None, None, :], OBST_RADIUS, 0.0, 2 * OBST_HALF_HEIGHT)
        d_base = capped_cylinder_sdf(centers, self.base_xy, BASE_CYL["radius"], BASE_CYL["z_lo"], BASE_CYL["z_hi"])
        d_ground = centers[..., 2]
        return torch.minimum(torch.minimum(d_obs, d_base), d_ground) - rad

    def __call__(self, ctrl: torch.Tensor, goal: torch.Tensor, obs_xy: torch.Tensor):
        cfg = self.cfg
        q = self.B0 @ ctrl                         # (B,T,D)
        qd = self.B1 @ ctrl
        qdd = self.B2 @ ctrl

        ee_T = self.kin.ee_pos(q[:, -1])
        goal_err = torch.linalg.norm(ee_T - goal, dim=-1)
        c_goal = cfg.w_goal_lin * goal_err + cfg.w_goal_sq * goal_err ** 2

        clr = self.clearance(q, obs_xy)
        pen = torch.clamp(cfg.safety_margin - clr, min=0)
        c_col = cfg.w_collision * (pen ** 2).sum(-1).mean(-1) + cfg.w_terminal_collision * (pen[:, -1] ** 2).sum(-1)

        lim_m = 0.05
        c_lim = cfg.w_limits * ((torch.clamp(q - (self.kin.q_hi - lim_m), min=0) ** 2
                                 + torch.clamp((self.kin.q_lo + lim_m) - q, min=0) ** 2).sum(-1).mean(-1))
        c_acc = cfg.w_acc * (qdd ** 2).sum(-1).mean(-1)
        c_vel = cfg.w_vel * (qd ** 2).sum(-1).mean(-1)

        total = c_goal + c_col + c_lim + c_acc + c_vel
        return total, dict(goal=c_goal, collision=c_col, limits=c_lim, acc=c_acc, vel=c_vel,
                           goal_err=goal_err, min_clearance=clr.amin(dim=(1, 2)))


# --------------------------------------------------------------------------- #
# Problem sampling
# --------------------------------------------------------------------------- #
class ProblemSampler:
    """Random (q0, goal, obstacle) triples where the start and a goal configuration are collision-free."""

    def __init__(self, kin: ArmKinematics, cost: TrajectoryCost, seed: int | None = None):
        self.kin, self.cost = kin, cost
        self.rng = np.random.default_rng(seed)
        lo = np.deg2rad([-170, -80, -140, -170])
        hi = np.deg2rad([170, 80, 140, 170])
        self.q_lo, self.q_hi = lo, hi

    def _rand_q(self, n):
        return self.rng.uniform(self.q_lo, self.q_hi, size=(n, 4))

    @torch.no_grad()
    def sample(self, n: int):
        out_q0, out_goal, out_obs = [], [], []
        got = 0
        while got < n:
            m = 256  # fixed chunk -> sample(n) is a prefix of sample(N) for the same seed
            q0 = torch.tensor(self._rand_q(m), dtype=torch.float32)
            qg = torch.tensor(self._rand_q(m), dtype=torch.float32)
            ee0 = self.kin.ee_pos(q0)
            goal = self.kin.ee_pos(qg)

            r_goal = torch.linalg.norm(goal[:, :2], dim=-1)
            ok = (goal[:, 2] > 0.05) & (goal[:, 2] < 0.45) & (r_goal > 0.22) & (r_goal < 0.62)
            ok &= (ee0[:, 2] > 0.10) & (torch.linalg.norm(ee0 - goal, dim=-1) > 0.20)

            # obstacle on the EE path of a straight joint-space motion q0 -> qg (so it actually blocks
            # the naive path), or half the time between start EE and goal in xy; pushed away from the base
            s = torch.tensor(self.rng.uniform(0.3, 0.7, size=(m, 1)), dtype=torch.float32)
            on_path = self.kin.ee_pos(q0 + s * (qg - q0))[:, :2]
            between = 0.5 * (ee0[:, :2] + goal[:, :2])
            pick = torch.tensor(self.rng.random(m) < 0.5)[:, None]
            mid = torch.where(pick, on_path, between)
            mid = mid + torch.tensor(self.rng.normal(0, 0.03, size=(m, 2)), dtype=torch.float32)
            r_mid = torch.linalg.norm(mid, dim=-1, keepdim=True).clamp(min=1e-6)
            obs = torch.where(r_mid < 0.24, mid / r_mid * 0.24, mid)

            # goal must be reachable without touching the pole
            d_goal_obs = torch.linalg.norm(goal[:, :2] - obs, dim=-1)
            ok &= (d_goal_obs > OBST_RADIUS + 0.09) | (goal[:, 2] > 2 * OBST_HALF_HEIGHT + 0.09)
            margin = self.cost.cfg.safety_margin + 0.01
            ok &= self.cost.clearance(q0[:, None], obs).amin(dim=(1, 2)) > margin
            ok &= self.cost.clearance(qg[:, None], obs).amin(dim=(1, 2)) > margin

            idx = torch.nonzero(ok).flatten()[: n - got]
            out_q0.append(q0[idx])
            out_goal.append(goal[idx])
            out_obs.append(obs[idx])
            got += len(idx)
        return torch.cat(out_q0), torch.cat(out_goal), torch.cat(out_obs)


# --------------------------------------------------------------------------- #
# Learned planner
# --------------------------------------------------------------------------- #
def problem_features(q0, goal, obs_xy, q_ik):
    ang_g = torch.atan2(goal[:, 1], goal[:, 0])
    ang_o = torch.atan2(obs_xy[:, 1], obs_xy[:, 0])
    rel = obs_xy - goal[:, :2]
    return torch.cat([
        q0, torch.sin(q0), torch.cos(q0),
        q_ik, torch.sin(q_ik), torch.cos(q_ik), q_ik - q0,
        goal, torch.sin(ang_g)[:, None], torch.cos(ang_g)[:, None], torch.linalg.norm(goal[:, :2], dim=-1, keepdim=True),
        obs_xy, torch.sin(ang_o)[:, None], torch.cos(ang_o)[:, None], torch.linalg.norm(obs_xy, dim=-1, keepdim=True),
        rel,
    ], dim=-1)


class PlannerNet(nn.Module):
    """MLP -> (q_end, interior control-point offsets).

    Conditioned on a collision-aware IK solution q_ik of the goal (the discrete IK-branch choice is
    hard for gradient descent, so it is solved analytically). The network predicts a correction of
    q_end around q_ik and the via-point offsets that bend the path around the obstacle. q_end is then
    passed through a few differentiable DLS-IK steps (analytic Jacobian) so the goal stays exact.
    With the zero-initialised head the untrained network equals the straight-line baseline."""

    def __init__(self, n_ctrl: int, n_joints: int = 4, hidden: int = 512, depth: int = 4,
                 q_lo=None, q_hi=None, kin: ArmKinematics | None = None, ik_steps: int = 5):
        super().__init__()
        self.n_ctrl, self.n_joints = n_ctrl, n_joints
        self.kin, self.ik_steps = kin, ik_steps
        in_dim = 7 * n_joints + 6 + 5 + 2
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_joints + (n_ctrl - 4) * n_joints)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("q_lo", torch.as_tensor(q_lo, dtype=torch.float32))
        self.register_buffer("q_hi", torch.as_tensor(q_hi, dtype=torch.float32))

    def forward(self, q0, goal, obs_xy, q_ik):
        h = self.trunk(problem_features(q0, goal, obs_xy, q_ik))
        out = self.head(h)
        q_end = torch.clamp(q_ik + 0.5 * torch.tanh(out[:, :self.n_joints]), self.q_lo + 0.02, self.q_hi - 0.02)
        if self.kin is not None:
            for _ in range(self.ik_steps):
                q_end = self.kin.dls_step(q_end, goal)
        inner = 0.5 * out[:, self.n_joints:].view(-1, self.n_ctrl - 4, self.n_joints)
        return q_end, inner


# --------------------------------------------------------------------------- #
# Planners (all return control points (B,K,D) and wall-clock planning time)
# --------------------------------------------------------------------------- #
def _optimize(cost: TrajectoryCost, q0, goal, obs, q_end, inner, iters, lr):
    q_end = q_end.detach().clone().requires_grad_(True)
    inner = inner.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([q_end, inner], lr=lr)
    for _ in range(iters):
        ctrl = assemble_ctrl(q0, q_end, inner, cost.basis)
        loss = cost(ctrl, goal, obs)[0].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return assemble_ctrl(q0, q_end, inner, cost.basis).detach()


def solve_ik(kin: ArmKinematics, q_init, goal, iters=60, damping=0.05, n_restarts=24, seed=0,
             clearance_fn=None):
    """Batched damped-least-squares IK. Tries q_init plus a fixed set of restart configurations
    (the same for every problem, so the result does not depend on batch size) and keeps, among
    converged (and, with clearance_fn, collision-free) solutions, the one closest to q_init."""
    gen = torch.Generator().manual_seed(seed)
    B, D = q_init.shape
    lo, hi = kin.q_lo + 0.05, kin.q_hi - 0.05
    restarts = lo + (hi - lo) * torch.rand(n_restarts, D, generator=gen)
    starts = [q_init] + [r.expand(B, D) for r in restarts]
    q = torch.cat(starts)                                  # (R*B, D)
    tgt = goal.repeat(len(starts), 1)
    with torch.no_grad():
        for _ in range(iters):
            q = torch.clamp(kin.dls_step(q, tgt, damping), lo, hi)
        err = torch.linalg.norm(kin.ee_pos(q) - tgt, dim=-1).view(len(starts), B)
        dist = torch.linalg.norm(q.view(len(starts), B, D) - q_init[None], dim=-1)
        score = torch.where(err < 2e-3, dist, 100 + err)    # prefer converged, then nearest
        if clearance_fn is not None:
            clr = clearance_fn(q[:, None]).amin(dim=(1, 2)).view(len(starts), B)
            score = score + 10 * (clr < 0.01)
        best = score.argmin(0)
        return q.view(len(starts), B, D)[best, torch.arange(B)]


def goal_ik(cost, q0, goal, obs):
    """Collision-free IK solution of the goal closest to q0 (shared by all planners)."""
    return solve_ik(cost.kin, q0, goal, clearance_fn=lambda q: cost.clearance(q, obs.repeat(len(q) // len(obs), 1)))


def plan_linear(cost, q0, goal, obs):
    """Baseline: IK for the goal, straight line in joint space (ignores the obstacle along the way)."""
    t0 = time.perf_counter()
    q_end = goal_ik(cost, q0, goal, obs)
    inner = torch.zeros(q0.shape[0], cost.cfg.n_ctrl - 4, q0.shape[1])
    ctrl = assemble_ctrl(q0, q_end, inner, cost.basis)
    return ctrl, time.perf_counter() - t0


def plan_trajopt(cost, q0, goal, obs, iters=400, n_inits=4, seed=0):
    """Baseline: per-problem gradient trajectory optimisation of the same cost.
    Several initialisations (straight line, 'lift over', random bends) are optimised in one batch
    and the cheapest result per problem is kept."""
    t0 = time.perf_counter()
    lin, _ = plan_linear(cost, q0, goal, obs)
    B, K, D = lin.shape
    q_end = lin[:, -1]
    gen = torch.Generator().manual_seed(seed)
    bump = torch.sin(torch.pi * torch.tensor(cost.basis.greville[2:-2], dtype=q0.dtype))[None, :, None]
    inits = [torch.zeros(B, K - 4, D)]
    over = torch.zeros(1, 1, D)
    over[..., 1], over[..., 2] = -0.7, -0.5                # raise shoulder + elbow mid-way
    inits.append(bump * over.expand(B, 1, D))
    for _ in range(n_inits - 2):
        inits.append(bump * 0.6 * torch.randn(B, 1, D, generator=gen))
    R = len(inits)
    rep = lambda x: x.repeat(R, *([1] * (x.dim() - 1)))
    q0r, goalr, obsr = rep(q0), rep(goal), rep(obs)
    ctrl = _optimize(cost, q0r, goalr, obsr, rep(q_end), torch.cat(inits), iters, lr=0.02)
    with torch.no_grad():
        c = cost(ctrl, goalr, obsr)[0].view(R, B)
    best = c.argmin(0)
    ctrl = ctrl.view(R, B, K, D)[best, torch.arange(B)]
    return ctrl, time.perf_counter() - t0


@torch.no_grad()
def plan_learned(net, cost, q0, goal, obs):
    """Goal IK + one network forward (timing includes the IK)."""
    t0 = time.perf_counter()
    q_end, inner = net(q0, goal, obs, goal_ik(cost, q0, goal, obs))
    ctrl = assemble_ctrl(q0, q_end, inner, cost.basis)
    return ctrl, time.perf_counter() - t0


def plan_learned_refine(net, cost, q0, goal, obs, iters=30):
    """Learned plan as a warm start + a few optimisation steps."""
    t0 = time.perf_counter()
    with torch.no_grad():
        q_end, inner = net(q0, goal, obs, goal_ik(cost, q0, goal, obs))
    ctrl = _optimize(cost, q0, goal, obs, q_end, inner, iters, lr=0.01)
    return ctrl, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def load_planner(path, kin: ArmKinematics):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PlanConfig(**ckpt["plan_config"])
    net = PlannerNet(cfg.n_ctrl, hidden=ckpt["hidden"], depth=ckpt["depth"], q_lo=kin.q_lo, q_hi=kin.q_hi,
                     kin=kin, ik_steps=ckpt.get("ik_steps", 0))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, cfg


def save_planner(path, net: PlannerNet, cfg: PlanConfig, hidden, depth, extra=None):
    torch.save(dict(state_dict=net.state_dict(), plan_config=asdict(cfg), hidden=hidden, depth=depth,
                    ik_steps=net.ik_steps if net.kin is not None else 0, **(extra or {})), path)


if __name__ == "__main__":
    # self-test: torch FK must match MuJoCo
    model = build_model()
    kin = ArmKinematics(model)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(50):
        q = rng.uniform(kin.q_lo.numpy(), kin.q_hi.numpy())
        data.qpos[kin.qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        frames = kin.forward(torch.tensor(q, dtype=torch.float32))
        for b in kin.body_info:
            max_err = max(max_err, float(np.abs(frames[b][1].numpy() - data.xpos[b]).max()))
    print(f"FK max position error vs MuJoCo: {max_err:.2e} m")
    print(f"collision spheres: {len(kin.sph_rad)}, moving geoms: {len(kin.moving_geoms)}")
    cost = TrajectoryCost(kin, PlanConfig())
    q0, goal, obs = ProblemSampler(kin, cost, seed=0).sample(8)
    print("sampled problems:", q0.shape, goal.shape, obs.shape)
