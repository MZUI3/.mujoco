"""
플래너 데모로 정책 학습: 행동 복제(BC) + DAgger.

    python rl_zoo/train_bc.py                                   # data/planner_demos.npz -> models_zoo/ReachMultiObstacle-v0__BC/
    python rl_zoo/train_bc.py --dagger 0 --no-context --run ReachMultiObstacle-v0__BC_naive --algo-label BC_naive   # 순수 BC(obs -> action) 대조군
    python rl_zoo/train_bc.py --demos data/planner_demos.npz --epochs 20 --hidden 256 256 --dagger 8 --dagger-episodes 400

정책은 MLP(256×2, ReLU, tanh 출력)이고 정규화한 관측에서 토크 행동 4개를 MSE로 회귀한다. 에피소드 단위로 5 %를 검증에 떼어 둔다.

- `--context`(기본): 관측에 에피소드 문맥(정규화 시간 k/200, 시작 자세 q0; envs/episode_context_wrapper.py)을 붙인다.
  플래너 궤적은 (q0, goal, 장애물)의 함수이고 시계에 맞춰 추종되므로, 이 둘이 없으면 전문가 행동이 관측만으로 정해지지 않는다.
- `--dagger N`(기본 8, 회당 400 에피소드): BC 뒤에 DAgger를 N회 돈다. 매 회 새 문제에서 **현재 정책**으로 롤아웃하면서 방문한 상태마다
  전문가(플래너 궤적 computed-torque 추종기, rl_zoo/collect_planner_demos.ComputedTorqueTracker)의 행동을 라벨로 모으고,
  데이터를 합쳐 다시 학습한다. 순수 BC는 자기 오차가 누적되면 데모 분포 밖으로 벗어나 0 %에 머무는데, DAgger는 그 상태들을 라벨링한다.

출력: model.pt(가중치 + 관측 정규화 통계), config.json(algo "BC"), eval.json(20 에피소드), learning_curve.png, train_history.json.
evaluate_zoo.py / visualize_zoo.py 는 `load_bc()`가 돌려주는 BCPolicy(`predict(obs, deterministic=True)`)로 이 모델을 읽고,
config의 `episode_context`가 참이면 환경을 EpisodeContextWrapper로 감싼다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trajectory_planning"))

import numpy as np
import torch
import torch.nn as nn

from envs import make_env
from envs.episode_context_wrapper import EpisodeContextWrapper, context_features
from rl_zoo.zoo_config import MODELS_DIR, run_name

ENV_ID = "ReachMultiObstacle-v0"
DEMO_PATH = ROOT / "data" / "planner_demos.npz"


class BCNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def forward(self, obs):
        return self.net((obs - self.obs_mean) / self.obs_std)


class BCPolicy:
    """SB3-like interface: predict(obs, deterministic=True) -> (action, None)."""

    def __init__(self, net: BCNet):
        self.net = net.eval()

    @torch.no_grad()
    def predict(self, obs, deterministic=True):
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        a = self.net(o[None] if o.dim() == 1 else o).numpy()
        return (a[0] if o.dim() == 1 else a), None


def load_bc(run_dir) -> BCPolicy:
    ckpt = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    net = BCNet(ckpt["obs_dim"], ckpt["act_dim"], tuple(ckpt["hidden"]))
    net.load_state_dict(ckpt["state_dict"])
    return BCPolicy(net)


def wrap_env(env, context: bool):
    return EpisodeContextWrapper(env) if context else env


def split_episodes(starts, val_frac, rng):
    ep_id = np.cumsum(starts) - 1
    n_ep = int(ep_id[-1]) + 1
    val_eps = rng.choice(n_ep, size=max(1, int(round(val_frac * n_ep))), replace=False)
    is_val = np.isin(ep_id, val_eps)
    return ~is_val, is_val, n_ep


def fit(net, X, Y, Xv, Yv, epochs, batch, lr, weight_decay, log_prefix=""):
    """Train (or continue training) net on (X, Y); returns per-epoch history."""
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    steps = int(np.ceil(len(X) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps))
    X, Y, Xv, Yv = (torch.as_tensor(a) for a in (X, Y, Xv, Yv))
    hist, t0 = [], time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            val = float(((net(Xv) - Yv) ** 2).mean())
        hist.append(dict(epoch=ep + 1, train_mse=tot / len(perm), val_mse=val))
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"{log_prefix}epoch {ep + 1:3d}  train mse {tot / len(perm):.6f}  val mse {val:.6f}  ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    return hist


def dagger_rollouts(policy, n_episodes, seed, context, track="time", exec_noise=0.0, beta=0.0, env_id=ENV_ID,
                    kp=400.0, kd=40.0):
    """Roll out `policy` on fresh planner problems; label every visited state with the expert's action.

    Returns (X, Y, stats). beta: probability of executing the expert's action instead of the learner's."""
    from rl_zoo.collect_planner_demos import ComputedTorqueTracker, load_planner_stack, plan_batched
    env = wrap_env(make_env(env_id), context)
    base = env.unwrapped
    tp_core, kin, cost, net, cfg = load_planner_stack(env)
    tracker = ComputedTorqueTracker(env, kp=kp, kd=kd, mode=track)
    q0, goal, obs = tp_core.ProblemSampler(kin, cost, seed=seed).sample(n_episodes)
    ctrl = plan_batched(tp_core, net, cost, q0, goal, obs)
    q0, goal, obs = (t.numpy().astype(np.float64) for t in (q0, goal, obs))
    rng = np.random.default_rng(seed)
    X, Y, succ, coll, dist = [], [], [], [], []
    for i in range(n_episodes):
        tracker.set_reference(ctrl[i], cost.basis)
        o, info = env.reset(options=dict(problem=(q0[i], goal[i], obs[i])))
        for k in range(base.max_steps):
            a_exp = tracker.action(k)
            X.append(np.asarray(o, dtype=np.float32))
            Y.append(a_exp.astype(np.float32))
            a = a_exp if rng.random() < beta else policy.predict(o)[0]
            if exec_noise > 0:
                a = np.clip(a + rng.normal(0, exec_noise, size=a.shape), -1, 1)
            o, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        succ.append(bool(info["success"]))
        coll.append(bool(info["collision"]))
        dist.append(float(info["dist"]))
    env.close()
    stats = dict(episodes=n_episodes, success_rate=float(np.mean(succ)), collision_rate=float(np.mean(coll)),
                 mean_final_dist=float(np.mean(dist)))
    return np.asarray(X), np.asarray(Y), stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=str, default=str(DEMO_PATH))
    p.add_argument("--env", type=str, default=ENV_ID)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--epochs", type=int, default=20, help="BC epochs on the demos")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--context", dest="context", action="store_true", default=True,
                   help="append episode context (time, q0) to the observation (default)")
    p.add_argument("--no-context", dest="context", action="store_false")
    p.add_argument("--dagger", type=int, default=8, help="DAgger iterations after BC (0 = pure BC)")
    p.add_argument("--dagger-episodes", type=int, default=400, help="learner rollouts labelled per iteration")
    p.add_argument("--dagger-epochs", type=int, default=15, help="re-training epochs per iteration (warm start)")
    p.add_argument("--dagger-lr", type=float, default=None, help="learning rate for the DAgger re-training (default: --lr)")
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    p.add_argument("--dagger-noise", type=float, default=0.0, help="exec noise on the learner's action during DAgger rollouts")
    p.add_argument("--track", type=str, default="time", choices=["time", "path"], help="expert reference selection")
    p.add_argument("--kp", type=float, default=400.0, help="expert computed-torque gains (should match the demos)")
    p.add_argument("--kd", type=float, default=40.0)
    p.add_argument("--dagger-beta", type=float, default=0.0,
                   help="iteration i executes the expert's action with probability beta**(i+1) (0 = pure learner rollouts)")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--run", type=str, default=None, help="output dir name under models_zoo (default <env>__BC)")
    p.add_argument("--algo-label", type=str, default="BC", help='algo name written to config.json ("BC", "BC_naive", ...)')
    p.add_argument("--out", type=str, default=None, help="full output directory (overrides --run)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    data = np.load(args.demos, allow_pickle=False)
    O, A, starts = data["obs"].astype(np.float32), data["actions"].astype(np.float32), data["episode_starts"]
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    if args.context:
        O = np.concatenate([O, context_features(starts, data["q0"].astype(np.float32))], axis=1)
    tr, va, n_ep = split_episodes(starts, args.val_frac, rng)
    print(f"demos: {n_ep} episodes, {len(O)} transitions (train {tr.sum()}, val {va.sum()}), input {O.shape[1]} "
          f"({'obs + time + q0' if args.context else 'obs only'}), act {A.shape[1]}", flush=True)

    net = BCNet(O.shape[1], A.shape[1], tuple(args.hidden))
    net.obs_mean[:] = torch.tensor(O[tr].mean(0))
    net.obs_std[:] = torch.tensor(O[tr].std(0) + 1e-6)
    t0 = time.time()
    Xtr, Ytr = O[tr], A[tr]
    hist = fit(net, Xtr, Ytr, O[va], A[va], args.epochs, args.batch, args.lr, args.weight_decay, log_prefix="[BC] ")
    stages = [dict(stage="BC", n_train=int(len(Xtr)), hist=hist)]

    dagger_stats = []
    for it in range(args.dagger):
        policy = BCPolicy(net)
        beta = args.dagger_beta ** (it + 1) if args.dagger_beta > 0 else 0.0
        X_new, Y_new, st = dagger_rollouts(policy, args.dagger_episodes, seed=args.seed * 1000 + 10 + it, context=args.context,
                                           track=args.track, exec_noise=args.dagger_noise, beta=beta, kp=args.kp, kd=args.kd)
        print(f"[DAgger {it + 1}/{args.dagger}] learner rollouts: success {100 * st['success_rate']:.0f}%  "
              f"collision {100 * st['collision_rate']:.0f}%  final dist {100 * st['mean_final_dist']:.1f} cm  "
              f"(+{len(X_new)} labelled states, {(time.time() - t0) / 60:.1f} min)", flush=True)
        dagger_stats.append(dict(iteration=it + 1, **st))
        Xtr, Ytr = np.concatenate([Xtr, X_new]), np.concatenate([Ytr, Y_new])
        h = fit(net, Xtr, Ytr, O[va], A[va], args.dagger_epochs, args.batch, args.dagger_lr or args.lr, args.weight_decay,
                log_prefix=f"[DAgger {it + 1}] ")
        stages.append(dict(stage=f"DAgger {it + 1}", n_train=int(len(Xtr)), hist=h, rollouts=st))
    train_time = time.time() - t0

    out = Path(args.out) if args.out else MODELS_DIR / (args.run or run_name(args.env, "BC"))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=net.state_dict(), obs_dim=O.shape[1], act_dim=A.shape[1], hidden=list(args.hidden),
                    env_id=args.env, episode_context=args.context), out / "model.pt")

    from rl_zoo.evaluate_zoo import rollouts
    policy = BCPolicy(net)
    ev = rollouts(args.env, lambda obs, e: policy.predict(obs)[0], args.eval_episodes, 10_000, context_wrapper=args.context)
    cfg = dict(env_id=args.env, algo=args.algo_label, timesteps=int(len(Xtr)), seed=args.seed, train_time_s=train_time, goal_wrapper=False,
               episode_context=args.context, obs_dim=int(O.shape[1]), act_dim=int(A.shape[1]),
               algo_kwargs=dict(hidden=args.hidden, epochs=args.epochs, batch=args.batch, lr=args.lr, weight_decay=args.weight_decay,
                                val_frac=args.val_frac, dagger_iters=args.dagger, dagger_episodes=args.dagger_episodes,
                                dagger_epochs=args.dagger_epochs, dagger_lr=args.dagger_lr or args.lr, dagger_noise=args.dagger_noise,
                                dagger_beta=args.dagger_beta,
                                track=args.track, kp=args.kp, kd=args.kd,
                                demos=str(Path(args.demos)).replace("\\", "/"), n_demo_episodes=int(n_ep),
                                n_demo_transitions=int(len(O)), n_train_transitions=int(len(Xtr)),
                                final_train_mse=stages[-1]["hist"][-1]["train_mse"], final_val_mse=stages[-1]["hist"][-1]["val_mse"],
                                dagger_rollout_stats=dagger_stats, demo_meta=meta))
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
    (out / "eval.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
    (out / "train_history.json").write_text(json.dumps(stages, indent=1), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        x0 = 0
        for s in stages:
            xs = [x0 + h["epoch"] for h in s["hist"]]
            axes[0].plot(xs, [h["train_mse"] for h in s["hist"]], color="#2a78d6", lw=2, label="train" if x0 == 0 else None)
            axes[0].plot(xs, [h["val_mse"] for h in s["hist"]], color="#eb6834", lw=2, label="val (demo episodes)" if x0 == 0 else None)
            if x0:
                axes[0].axvline(x0, color="#9a9a9a", lw=0.8, ls="--")
            x0 = xs[-1]
        axes[0].set_yscale("log")
        axes[0].set_xlabel("epoch (dashed: DAgger iterations)")
        axes[0].set_ylabel("action MSE")
        axes[0].legend(frameon=False)
        if dagger_stats:
            axes[1].plot([d["iteration"] for d in dagger_stats], [100 * d["success_rate"] for d in dagger_stats], "o-", color="#1baf7a", lw=2)
        axes[1].plot([len(dagger_stats) + 1], [100 * ev["success_rate"]], "s", color="#c2185b", label="final (eval seeds)")
        axes[1].set_ylim(0, 100)
        axes[1].set_xlabel("DAgger iteration (learner rollouts before re-training)")
        axes[1].set_ylabel("success rate (%)")
        axes[1].legend(frameon=False)
        for a in axes:
            a.grid(alpha=0.25)
            a.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{args.env} / BC{' + DAgger' if args.dagger else ''} on planner demos")
        fig.tight_layout()
        fig.savefig(out / "learning_curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # plotting must never fail a run
        print(f"curve plot failed: {e}")
    print(f"[{args.env}/BC] demos {n_ep} ep / {len(O)} transitions, DAgger x{args.dagger} -> {len(Xtr)} training transitions, "
          f"{train_time / 60:.1f} min | val mse {stages[-1]['hist'][-1]['val_mse']:.6f} | "
          f"success {ev['success_rate'] * 100:.0f}%  return {ev['mean_return']:.1f}  collision {ev['collision_rate'] * 100:.0f}%")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
