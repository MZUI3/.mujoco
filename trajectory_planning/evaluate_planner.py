"""
학습된 trajectory planner 성능 평가 (MuJoCo 동역학 시뮬레이션) + 그래프.

비교 방법
    learned         : 신경망 1회 forward
    learned+refine  : 신경망 출력을 초기값으로 30 step 최적화
    trajopt         : 문제마다 같은 비용을 처음부터 최적화 (다중 초기값, 400 step) - 품질 기준선
    linear          : IK 후 관절공간 직선 보간 (장애물 무시) - 하한 기준선

Usage:
    python trajectory_planning/evaluate_planner.py --n 100                      # 1~3 poles, random size
    python trajectory_planning/evaluate_planner.py --obstacles 1 1 --fixed-size  # v1 distribution
Outputs (results_tp/):
    eval_results.csv, eval_summary.json, eval_summary.md,
    summary_rates.png, success_by_obstacles.png, distributions.png, example_paths_3d.png, tracking_example.png
"""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tp_core import (MAX_OBS, ROOT, ProblemSampler, TrajectoryCost, load_planner, obstacle_mask,
                     plan_learned, plan_learned_refine, plan_linear, plan_trajopt)
from tp_sim import METHOD_COLORS, SimScene

METHODS = ["learned", "learned+refine", "trajopt", "linear"]


def make_planners(net, cost):
    return {
        "learned": lambda q0, g, o: plan_learned(net, cost, q0, g, o),
        "learned+refine": lambda q0, g, o: plan_learned_refine(net, cost, q0, g, o),
        "trajopt": lambda q0, g, o: plan_trajopt(cost, q0, g, o),
        "linear": lambda q0, g, o: plan_linear(cost, q0, g, o),
    }


def planned_metrics(cost, ctrl, goal, obs):
    """Metrics of the plan itself (kinematic model)."""
    with torch.no_grad():
        _, info = cost(ctrl, goal, obs)
        ts = np.linspace(0, 1, 200)
        B3 = torch.tensor(cost.basis.matrix(ts, 3), dtype=torch.float32)
        jerk = ((B3 @ ctrl) ** 2).sum(-1).mean(-1) * cost.basis.duration  # integrated squared jerk
    return dict(plan_goal_err_mm=(info["goal_err"] * 1000).numpy(),
                plan_min_clearance_cm=(info["min_clearance"] * 100).numpy(),
                plan_jerk=jerk.numpy())


def measure_latency(planner, q0, goal, obs, n):
    """Single-problem planning latency (ms), the way it would be called online."""
    out = []
    for i in range(n):
        _, dt = planner(q0[i:i + 1], goal[i:i + 1], obs[i:i + 1])
        out.append(dt * 1000)
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _style(ax):
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)


def plot_rates(summary, methods, path):
    keys = [("success_rate", "Success"), ("reach_rate", "Reached goal (< 2 cm)"), ("collision_rate", "Collided")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (k, title) in zip(axes, keys):
        vals = [summary[m][k] * 100 for m in methods]
        bars = ax.bar(methods, vals, color=[METHOD_COLORS[m] for m in methods], width=0.6,
                      edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, color="#222")
        ax.set_ylim(0, 110)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=15)
        _style(ax)
    axes[0].set_ylabel("% of test problems (MuJoCo rollout)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_success_by_obstacles(rows, methods, path):
    counts = sorted({r["n_obs"] for r in rows})
    fig, ax = plt.subplots(figsize=(1.6 + 1.6 * len(counts), 4))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals, ns = [], []
        for k in counts:
            sel = [r["success"] for r in rows if r["method"] == m and r["n_obs"] == k]
            vals.append(100 * np.mean(sel) if sel else np.nan)
            ns.append(len(sel))
        x = np.arange(len(counts)) + (i - (len(methods) - 1) / 2) * w
        bars = ax.bar(x, vals, width=w * 0.95, color=METHOD_COLORS[m], label=m, edgecolor="white", linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color="#222")
    ax.set_xticks(range(len(counts)), [f"{k} pole{'s' if k > 1 else ''}\n(n={sum(1 for r in rows if r['n_obs'] == k and r['method'] == methods[0])})"
                                        for k in counts])
    ax.set_ylim(0, 112)
    ax.set_ylabel("success rate (%)")
    ax.set_title("Success rate by number of obstacles (MuJoCo rollout)")
    ax.legend(frameon=False, ncol=len(methods), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_distributions(rows, latency, methods, path):
    panels = [
        ("final_err_mm", "Final EE error (mm)", True),
        ("min_obstacle_dist_cm", "Min distance to obstacle (cm)", False),
        ("ee_path_len_m", "EE path length (m)", False),
        ("plan_jerk", "Integrated squared jerk (plan)", True),
        ("track_rmse_deg", "Tracking RMSE (deg)", True),
        ("latency_ms", "Planning latency, 1 problem (ms)", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, logy) in zip(axes.flat, panels):
        if key == "latency_ms":
            data = [latency[m] for m in methods]
        else:
            data = [[r[key] for r in rows if r["method"] == m] for m in methods]
        if logy:
            data = [np.maximum(d, 1e-3) for d in data]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                        medianprops=dict(color="#111", lw=1.5), flierprops=dict(markersize=3, alpha=0.5))
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(methods) + 1), methods, rotation=15)
        if logy:
            ax.set_yscale("log")
        if key == "min_obstacle_dist_cm":
            ax.axhline(0, color="#e34948", lw=1, ls="--")
            ax.text(0.55, 0, " contact", color="#e34948", va="bottom", fontsize=9)
        if key == "final_err_mm":
            ax.axhline(20, color="#52514e", lw=1, ls="--")
            ax.text(0.55, 20, " 2 cm tol.", color="#52514e", va="bottom", fontsize=9)
        ax.set_title(title)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _draw_cylinder(ax, xy, r, h, color="#e34948"):
    th = np.linspace(0, 2 * np.pi, 40)
    z = np.linspace(0, h, 2)
    TH, Z = np.meshgrid(th, z)
    ax.plot_surface(xy[0] + r * np.cos(TH), xy[1] + r * np.sin(TH), Z, color=color, alpha=0.35, linewidth=0)


def plot_examples_3d(examples, methods, path):
    n = len(examples)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, ex in enumerate(examples):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        for x, y, r, h in ex["obs"]:
            if h > 0:
                _draw_cylinder(ax, (x, y), r, h)
        _draw_cylinder(ax, (0, 0), 0.12, 0.17, color="#888780")
        for m in methods:
            ee = ex["logs"][m]["ee"]
            ok = ex["metrics"][m]["success"]
            ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=METHOD_COLORS[m], lw=2,
                    ls="-" if ok else ":", label=f"{m} ({'ok' if ok else 'fail'})")
        start = ex["logs"][methods[0]]["ee"][0]
        ax.scatter(*start, color="#111", s=30, label="start")
        ax.scatter(*ex["goal"], color="#008300", s=60, marker="*", label="goal")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(0, 0.7)
        ax.set_box_aspect((1.2, 1.2, 0.7))
        ax.view_init(elev=28, azim=-60)
        n_act = int((ex["obs"][:, 3] > 0).sum())
        ax.set_title(f"problem #{ex['idx']} ({n_act} pole{'s' if n_act > 1 else ''})")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tracking(log, method, path):
    names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(log["t"], np.rad2deg(log["q_des"][:, j]), color="#52514e", lw=2, ls="--", label="planned")
        ax.plot(log["t"], np.rad2deg(log["q"][:, j]), color=METHOD_COLORS[method], lw=2, label="MuJoCo")
        ax.set_title(names[j])
        ax.set_xlabel("time (s)")
        _style(ax)
    axes[0].set_ylabel("angle (deg)")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Joint tracking - {method}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=str(ROOT / "models_tp" / "planner.pt"))
    p.add_argument("--n", type=int, default=100, help="number of test problems")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--latency-n", type=int, default=10)
    p.add_argument("--obstacles", type=int, nargs=2, default=[1, MAX_OBS], metavar=("LO", "HI"),
                   help="number of poles per test problem is uniform in [LO, HI]")
    p.add_argument("--fixed-size", action="store_true", help="v1 poles (4 cm x 30 cm) instead of random sizes")
    p.add_argument("--out", type=str, default=str(ROOT / "results_tp"))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    sim = SimScene()
    net, cfg = load_planner(args.model, sim.kin)
    cost = TrajectoryCost(sim.kin, cfg)
    sampler = ProblemSampler(sim.kin, cost, seed=args.seed, n_obs=tuple(args.obstacles), fixed_size=args.fixed_size)
    q0, goal, obs = sampler.sample(args.n)
    n_obs = obstacle_mask(obs).sum(1).tolist()
    print(f"test problems: {args.n}, poles per problem: " +
          ", ".join(f"{k}: {n_obs.count(k)}" for k in sorted(set(n_obs))), flush=True)
    planners = make_planners(net, cost)
    methods = args.methods

    rows, logs, latency, plans = [], {m: [] for m in methods}, {}, {}
    for m in methods:
        print(f"[{m}] planning {args.n} problems ...", flush=True)
        ctrl, batch_time = planners[m](q0, goal, obs)
        plans[m] = ctrl
        latency[m] = measure_latency(planners[m], q0, goal, obs, args.latency_n)
        pm = planned_metrics(cost, ctrl, goal, obs)
        print(f"[{m}] MuJoCo rollouts ...", flush=True)
        t0 = time.time()
        for i in range(args.n):
            met, log = sim.rollout(ctrl[i].numpy(), cost.basis, q0[i].numpy(), goal[i].numpy(), obs[i].numpy())
            row = dict(problem=i, method=m, n_obs=n_obs[i], **met, **{k: float(v[i]) for k, v in pm.items()},
                       batch_plan_ms_per_problem=batch_time / args.n * 1000)
            rows.append(row)
            logs[m].append(log)
        print(f"[{m}] done ({time.time() - t0:.1f}s)", flush=True)

    # ---- summary ----
    summary = {}
    for m in methods:
        r = [x for x in rows if x["method"] == m]
        f = lambda k: np.array([x[k] for x in r], dtype=float)
        summary[m] = dict(
            success_rate=f("success").mean(),
            reach_rate=(f("final_err_mm") < 20).mean(),
            collision_rate=f("collision").mean(),
            final_err_mm_median=float(np.median(f("final_err_mm"))),
            min_obstacle_dist_cm_median=float(np.median(f("min_obstacle_dist_cm"))),
            ee_path_len_m_mean=f("ee_path_len_m").mean(),
            plan_jerk_median=float(np.median(f("plan_jerk"))),
            track_rmse_deg_mean=f("track_rmse_deg").mean(),
            torque_sat_frac_mean=f("torque_sat_frac").mean(),
            latency_ms_median=float(np.median(latency[m])),
            batch_plan_ms_per_problem=r[0]["batch_plan_ms_per_problem"],
        )
        summary[m] = {k: float(v) for k, v in summary[m].items()}
        summary[m]["success_by_n_obs"] = {int(k): float(np.mean([x["success"] for x in r if x["n_obs"] == k]))
                                          for k in sorted(set(n_obs))}

    with open(out / "eval_results.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "eval_summary.json", "w") as fj:
        json.dump(dict(n_problems=args.n, seed=args.seed, model=args.model,
                       obstacles=dict(n_obs=args.obstacles, fixed_size=args.fixed_size,
                                      count={int(k): n_obs.count(k) for k in sorted(set(n_obs))}),
                       summary=summary), fj, indent=2)

    header = ("| method | success | reach<2cm | collision | median err (mm) | median clearance (cm) "
              "| EE path (m) | median jerk | track RMSE (deg) | latency (ms) |")
    lines = [header, "|" + "---|" * 10]
    for m in methods:
        s = summary[m]
        lines.append(f"| {m} | {s['success_rate']*100:.1f}% | {s['reach_rate']*100:.1f}% | "
                     f"{s['collision_rate']*100:.1f}% | {s['final_err_mm_median']:.2f} | "
                     f"{s['min_obstacle_dist_cm_median']:.2f} | {s['ee_path_len_m_mean']:.3f} | "
                     f"{s['plan_jerk_median']:.1f} | {s['track_rmse_deg_mean']:.2f} | {s['latency_ms_median']:.1f} |")
    table = "\n".join(lines)
    counts = sorted(set(n_obs))
    lines2 = ["| method | " + " | ".join(f"{k} pole{'s' if k > 1 else ''} (n={n_obs.count(k)})" for k in counts) + " |",
              "|" + "---|" * (len(counts) + 1)]
    for m in methods:
        lines2.append(f"| {m} | " + " | ".join(f"{summary[m]['success_by_n_obs'][k]*100:.1f}%" for k in counts) + " |")
    table2 = "\n".join(lines2)
    dist = (f"poles per problem uniform in [{args.obstacles[0]}, {args.obstacles[1]}], "
            f"{'fixed 4 cm x 30 cm' if args.fixed_size else 'radius 3-7 cm, height 15-40 cm'}")
    (out / "eval_summary.md").write_text(
        f"# Trajectory planner evaluation ({args.n} problems, seed {args.seed})\n\n{dist}\n\n"
        + table + "\n\nSuccess rate by number of obstacles\n\n" + table2 + "\n", encoding="utf-8")
    print("\n" + table + "\n\n" + table2 + "\n")

    # ---- plots ----
    plot_rates(summary, methods, out / "summary_rates.png")
    plot_success_by_obstacles(rows, methods, out / "success_by_obstacles.png")
    plot_distributions(rows, latency, methods, out / "distributions.png")

    # examples: prefer problems where the naive straight line fails (obstacle actually matters)
    by = {(r["problem"], r["method"]): r for r in rows}
    hard = [i for i in range(args.n) if "linear" in methods and not by[(i, "linear")]["success"]]
    hard.sort(key=lambda i: -n_obs[i])  # show the busiest scenes first
    picks = (hard + [i for i in range(args.n) if i not in hard])[:4]
    examples = [dict(idx=i, goal=goal[i].numpy(), obs=obs[i].numpy(),
                     logs={m: logs[m][i] for m in methods},
                     metrics={m: by[(i, m)] for m in methods}) for i in picks]
    plot_examples_3d(examples, methods, out / "example_paths_3d.png")
    main_m = "learned" if "learned" in methods else methods[0]
    plot_tracking(logs[main_m][picks[0]], main_m, out / "tracking_example.png")
    print(f"saved results to {out}")


if __name__ == "__main__":
    main()
