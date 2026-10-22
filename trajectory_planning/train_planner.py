"""
Learned trajectory planner 학습.

(q0, goal, obstacle, goal-IK 해) -> B-spline 관절 궤적을 한 번의 forward로 출력하는 MLP를
미분 가능한 궤적 비용(목표 오차 + 충돌 + 관절 한계 + 가속도/속도)으로 직접 학습한다.
전문가 데모가 필요 없다. 무작위 문제와 그 충돌 인지 IK 해를 미리 계산한 문제 풀에서 배치를 뽑는다.

Usage:
    python trajectory_planning/train_planner.py                    # 기본 3000 iters (~15분, CPU)
    python trajectory_planning/train_planner.py --iters 300        # 빠른 동작 확인
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tp_core import (ROOT, ArmKinematics, PlanConfig, PlannerNet, ProblemSampler, TrajectoryCost,
                     assemble_ctrl, build_model, goal_ik, save_planner)


@torch.no_grad()
def build_pool(sampler, cost, n, chunk=2048):
    """Random problems + their collision-aware goal IK (the expensive part, done once)."""
    parts = []
    for start in range(0, n, chunk):
        q0, goal, obs = sampler.sample(min(chunk, n - start))
        parts.append((q0, goal, obs, goal_ik(cost, q0, goal, obs)))
    return tuple(torch.cat(x) for x in zip(*parts))


@torch.no_grad()
def validate(net, cost, val):
    q0, goal, obs, q_ik = val
    q_end, inner = net(q0, goal, obs, q_ik)
    c, info = cost(assemble_ctrl(q0, q_end, inner, cost.basis), goal, obs)
    reach = info["goal_err"] < 0.02
    free = info["min_clearance"] > 0
    return dict(cost=float(c.mean()), goal_err_mm=float(info["goal_err"].mean() * 1000),
                reach=float(reach.float().mean()), collision_free=float(free.float().mean()),
                success=float((reach & free).float().mean()))


def _smooth(x, k=10):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    head = c[1:k] / np.arange(1, k)
    return np.concatenate([head, (c[k:] - c[:-k]) / k])


def plot_history(hist, path):
    it = [h["iter"] for h in hist["train"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for key, color in [("loss", "#52514e"), ("collision", "#eb6834"), ("goal", "#2a78d6"), ("acc", "#1baf7a")]:
        ax.plot(it, np.maximum(_smooth([h[key] for h in hist["train"]]), 1e-6), color=color, lw=2,
                label="total" if key == "loss" else key)
    ax.set_yscale("log")
    ax.set_title("Training cost (batch mean, moving avg.)")
    ax.set_xlabel("iteration")
    ax.legend(frameon=False)

    vit = [h["iter"] for h in hist["val"]]
    ax = axes[1]
    for key, color, label in [("success", "#2a78d6", "success"), ("reach", "#1baf7a", "reach < 2 cm"),
                              ("collision_free", "#eb6834", "collision-free")]:
        ax.plot(vit, [h[key] * 100 for h in hist["val"]], color=color, lw=2, label=label)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title("Validation rates (kinematic model)")
    ax.set_xlabel("iteration")
    ax.legend(frameon=False, loc="lower right")

    ax.annotate("iter 0 = straight-line\n(untrained, zero head)", xy=(0, hist["val"][0]["success"] * 100),
                xytext=(0.25, 0.3), textcoords="axes fraction", fontsize=9, color="#52514e",
                arrowprops=dict(arrowstyle="->", color="#52514e"))

    ax = axes[2]
    ax.plot(vit, [h["cost"] for h in hist["val"]], color="#2a78d6", lw=2)
    ax.set_yscale("log")
    ax.set_title("Validation mean trajectory cost")
    ax.set_xlabel("iteration")
    for a in axes:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--ik-steps", type=int, default=5, help="differentiable DLS-IK steps on q_end (0 = pure MLP)")
    p.add_argument("--pool", type=int, default=100_000, help="number of pre-sampled training problems")
    p.add_argument("--curriculum", type=float, default=0.0,
                   help="fraction of training over which the collision weight ramps from 10%% to 100%%")
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--save-dir", type=str, default=str(ROOT / "models_tp"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model()
    kin = ArmKinematics(model)
    cfg = PlanConfig()
    cost = TrajectoryCost(kin, cfg)
    t0 = time.time()
    pool = build_pool(ProblemSampler(kin, cost, seed=args.seed), cost, args.pool)
    val = build_pool(ProblemSampler(kin, cost, seed=10_000 + args.seed), cost, 1024)
    print(f"problem pool: {len(pool[0])} train / {len(val[0])} val ({time.time() - t0:.0f}s)", flush=True)
    gen = torch.Generator().manual_seed(args.seed)

    net = PlannerNet(cfg.n_ctrl, hidden=args.hidden, depth=args.depth, q_lo=kin.q_lo, q_hi=kin.q_hi,
                     kin=kin if args.ik_steps > 0 else None, ik_steps=args.ik_steps)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)

    hist = dict(args=vars(args), plan_config=asdict(cfg), train=[], val=[])
    best = (-1.0, float("inf"))
    w_col, w_term = cfg.w_collision, cfg.w_terminal_collision
    t_start = time.time()
    v0 = validate(net, cost, val)  # untrained net == straight-line plan through q_ik
    v0["iter"] = 0
    hist["val"].append(v0)
    print(f"[    0/{args.iters}] val success={v0['success']*100:5.1f}% (untrained = straight line)", flush=True)
    for it in range(1, args.iters + 1):
        ramp = min(1.0, it / max(1, args.curriculum * args.iters))
        cost.cfg.w_collision = w_col * (0.1 + 0.9 * ramp)
        cost.cfg.w_terminal_collision = w_term * (0.1 + 0.9 * ramp)

        idx = torch.randint(len(pool[0]), (args.batch,), generator=gen)
        q0, goal, obs, q_ik = (x[idx] for x in pool)
        q_end, inner = net(q0, goal, obs, q_ik)
        c, info = cost(assemble_ctrl(q0, q_end, inner, cost.basis), goal, obs)
        loss = c.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % 10 == 0 or it == 1:
            hist["train"].append(dict(iter=it, loss=loss.item(), lr=sched.get_last_lr()[0],
                                      **{k: info[k].mean().item() for k in ("goal", "collision", "limits", "acc", "vel")}))

        if it % args.val_every == 0 or it == args.iters:
            cost.cfg.w_collision, cost.cfg.w_terminal_collision = w_col, w_term
            v = validate(net, cost, val)
            v["iter"] = it
            hist["val"].append(v)
            elapsed = time.time() - t_start
            print(f"[{it:5d}/{args.iters}] loss={loss.item():.3f} | val success={v['success']*100:5.1f}% "
                  f"reach={v['reach']*100:5.1f}% free={v['collision_free']*100:5.1f}% "
                  f"err={v['goal_err_mm']:.1f}mm cost={v['cost']:.3f} | {elapsed/60:.1f} min", flush=True)
            key = (v["success"], -v["cost"])
            if ramp >= 1.0 and key > (best[0], -best[1]):
                best = (v["success"], v["cost"])
                save_planner(save_dir / "planner.pt", net, cfg, args.hidden, args.depth, extra=dict(val=v))

    cost.cfg.w_collision, cost.cfg.w_terminal_collision = w_col, w_term
    if best[0] < 0:  # curriculum never finished (very short run): save the final net
        save_planner(save_dir / "planner.pt", net, cfg, args.hidden, args.depth, extra=dict(val=hist["val"][-1]))
    save_planner(save_dir / "planner_last.pt", net, cfg, args.hidden, args.depth, extra=dict(val=hist["val"][-1]))
    with open(save_dir / "train_history.json", "w") as f:
        json.dump(hist, f, indent=1)
    plot_history(hist, save_dir / "training_curves.png")
    print(f"saved: {save_dir / 'planner.pt'} (best val success {best[0]*100:.1f}%)")
    print(f"saved: {save_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
