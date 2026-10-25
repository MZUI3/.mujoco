"""
모델 zoo 전체 평가: models_zoo/ 의 모든 SB3 모델을 결정적 정책으로 N 에피소드 실행하고, 환경별 무작위 정책 기준선과
함께 표/그래프로 정리한다. 기존 모델(PPO/Diffusion/궤적 플래너)의 기록도 표에 합친다.

    python rl_zoo/evaluate_zoo.py --episodes 20
출력: results_zoo/zoo_summary.md, zoo_summary.json, zoo_success.png, zoo_returns.png
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs import TASK_DESCRIPTIONS, TASKS, make_env
from rl_zoo.zoo_config import CORE, MODELS_DIR, RESULTS_DIR, base_algo

ALGO_COLORS = {"SAC": "#2a78d6", "SAC_HER": "#7c4dff", "TD3": "#1baf7a", "PPO": "#eb6834", "A2C": "#eda100",
               "BC": "#c2185b", "BC_naive": "#e58fb3", "random": "#9a9a9a"}


def model_file(run_dir: Path):
    """model.zip (SB3) or model.pt (BC); None if the directory holds no model."""
    for name in ("model.zip", "model.pt"):
        if (run_dir / name).exists():
            return run_dir / name
    return None


def load_model(run_dir: Path):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg["algo"].startswith("BC"):   # behaviour cloning on planner demos (rl_zoo/train_bc.py), torch checkpoint
        from rl_zoo.train_bc import load_bc
        return load_bc(run_dir), cfg
    import stable_baselines3 as sb3
    env = None
    if cfg.get("goal_wrapper"):   # HerReplayBuffer models need the (goal-conditioned) env at load time
        from envs.goal_wrapper import GoalObsWrapper
        env = GoalObsWrapper(make_env(cfg["env_id"]), sparse="Sparse" in cfg["env_id"])
    model = getattr(sb3, base_algo(cfg["algo"])).load(run_dir / "model.zip", env=env, device="cpu")
    return model, cfg


def rollouts(env_id, policy, episodes, seed, goal_wrapper=False, context_wrapper=False):
    env = make_env(env_id)
    if goal_wrapper:
        from envs.goal_wrapper import GoalObsWrapper
        env = GoalObsWrapper(env, sparse="Sparse" in env_id)
    if context_wrapper:               # BC policies trained with episode context (time, q0)
        from envs.episode_context_wrapper import EpisodeContextWrapper
        env = EpisodeContextWrapper(env)
    rets, succ, dists, cols = [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs, env))
            ret += r
            done = term or trunc
        rets.append(ret)
        succ.append(info["success"])
        dists.append(info["dist"])
        cols.append(info["collision"])
    env.close()
    return dict(mean_return=float(np.mean(rets)), std_return=float(np.std(rets)), success_rate=float(np.mean(succ)),
                mean_final_dist=float(np.mean(dists)), collision_rate=float(np.mean(cols)), episodes=episodes)


def legacy_rows():
    """Models that are not SB3 zoo runs, with the metrics recorded in the repo."""
    rows = []
    cmp_csv = ROOT / "results" / "compare_results.csv"
    if cmp_csv.exists():
        import csv
        data = list(csv.DictReader(open(cmp_csv)))
        for algo, name, path in [("ppo", "PPO (PyTorch 자체 구현)", "models/ppo_policy.pt"),
                                 ("diffusion", "Diffusion Policy (TemporalUNet+DDPM)", "models/diffusion_policy.pt")]:
            r = [float(x["return"]) for x in data if x["algorithm"] == algo]
            if r:
                rows.append(dict(env_id="ApproachObjects-v0", algo=name, path=path, timesteps=None,
                                 mean_return=float(np.mean(r)), std_return=float(np.std(r)), success_rate=None,
                                 mean_final_dist=None, collision_rate=None, episodes=len(r),
                                 note="results/compare_results.csv 기록 (200 step 에피소드)"))
    if (ROOT / "models_dr" / "diffusion_policy_dr.pt").exists():
        rows.append(dict(env_id="ApproachObjects-v0", algo="Diffusion Policy + Domain Randomization", path="models_dr/diffusion_policy_dr.pt",
                         timesteps=None, mean_return=None, std_return=None, success_rate=None, mean_final_dist=None,
                         collision_rate=None, episodes=0, note="평가 기록 없음"))
    tp = ROOT / "results_tp" / "eval_summary.json"
    if tp.exists():
        s = json.loads(tp.read_text(encoding="utf-8"))
        L = s["summary"]["learned"]
        rows.append(dict(env_id="TrajectoryPlanning (기둥 1~3개)", algo="Learned Trajectory Planner (self-supervised)",
                         path="models_tp/planner.pt", timesteps=None, mean_return=None, std_return=None,
                         success_rate=L["success_rate"], mean_final_dist=L["final_err_mm_median"] / 1000,
                         collision_rate=L["collision_rate"], episodes=s["n_problems"],
                         note=f"results_tp/eval_summary.json, {s['n_problems']}문제 MuJoCo 롤아웃"))
    return rows


def aggregate(rows):
    """Group per-seed rows by (env, algo): mean/std over seeds of the per-seed metrics."""
    groups = {}
    for r in rows:
        groups.setdefault((r["env_id"], r["algo"]), []).append(r)
    agg = []
    for (e, a), rs in groups.items():
        rs = sorted(rs, key=lambda r: r.get("seed", 0))
        s = np.array([r["success_rate"] for r in rs])
        ret = np.array([r["mean_return"] for r in rs])
        agg.append(dict(env_id=e, algo=a, n_seeds=len(rs), seeds=[r.get("seed", 0) for r in rs],
                        timesteps=rs[0]["timesteps"], path=rs[0].get("path"),
                        success_rate=float(s.mean()), success_std=float(s.std()),
                        success_min=float(s.min()), success_max=float(s.max()),
                        mean_return=float(ret.mean()), std_return=float(ret.std() if len(rs) > 1 else rs[0]["std_return"]),
                        mean_final_dist=float(np.mean([r["mean_final_dist"] for r in rs])),
                        collision_rate=float(np.mean([r["collision_rate"] for r in rs])),
                        train_time_s=float(np.mean([r.get("train_time_s") or 0 for r in rs])),
                        core=(e, a) in CORE))
    return agg


def plot_success(rows, path):
    envs_ = [e for e in TASKS if any(r["env_id"] == e for r in rows)]
    algos = ["random"] + sorted({r["algo"] for r in rows if r["env_id"] in TASKS and r["algo"] != "random"},
                                key=lambda a: list(ALGO_COLORS).index(a) if a in ALGO_COLORS else 99)
    fig, ax = plt.subplots(figsize=(1.2 + 1.5 * len(envs_), 4.2))
    w = 0.8 / len(algos)
    for i, a in enumerate(algos):
        xs, ys = [], []
        for j, e in enumerate(envs_):
            r = next((r for r in rows if r["env_id"] == e and r["algo"] == a), None)
            if r is not None and r["success_rate"] is not None:
                xs.append(j + (i - (len(algos) - 1) / 2) * w)
                ys.append(r["success_rate"] * 100)
        errs = [100 * next(r for r in rows if r["env_id"] == e and r["algo"] == a).get("success_std", 0)
                for e in envs_ if any(r["env_id"] == e and r["algo"] == a for r in rows)]
        bars = ax.bar(xs, ys, width=w * 0.95, color=ALGO_COLORS.get(a, "#555"), label=a, edgecolor="white",
                      yerr=errs if any(errs) else None, capsize=2, error_kw=dict(lw=0.8))
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7, color="#222")
    ax.set_xticks(range(len(envs_)), [e.replace("-v0", "") for e in envs_], rotation=20, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("success rate (%)")
    ax.set_title("Model zoo: success rate per task (deterministic policy, 20 episodes; error bars = std over seeds)")
    ax.legend(frameon=False, ncol=len(algos), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_returns(rows, path):
    envs_ = [e for e in TASKS if any(r["env_id"] == e for r in rows)]
    n_rows = max(1, int(np.ceil(len(envs_) / 5)))
    fig, axes = plt.subplots(n_rows, 5, figsize=(17, 3.25 * n_rows), squeeze=False)
    for ax, e in zip(axes.flat, envs_):
        rs = [r for r in rows if r["env_id"] == e and r["mean_return"] is not None]
        names = [r["algo"] for r in rs]
        vals = [r["mean_return"] for r in rs]
        errs = [r["std_return"] or 0 for r in rs]
        ax.bar(names, vals, yerr=errs, color=[ALGO_COLORS.get(n, "#555") for n in names], capsize=3, edgecolor="white")
        ax.set_title(e.replace("-v0", ""), fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(envs_):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("mean episode return")
    fig.suptitle("Model zoo: mean episode return (± std) per task")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out", type=str, default=str(RESULTS_DIR))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    run_dirs = sorted(d for d in MODELS_DIR.iterdir() if d.is_dir() and model_file(d) is not None)
    envs_used = sorted({json.loads((d / "config.json").read_text(encoding="utf-8"))["env_id"] for d in run_dirs},
                       key=list(TASKS).index)
    for e in envs_used:
        print(f"[{e}] random baseline", flush=True)
        r = rollouts(e, lambda obs, env: env.action_space.sample(), args.episodes, args.seed)
        rows.append(dict(env_id=e, algo="random", path=None, timesteps=0, note="무작위 행동 기준선", **r))
    for d in run_dirs:
        model, cfg = load_model(d)
        print(f"[{cfg['env_id']}] {cfg['algo']}", flush=True)
        r = rollouts(cfg["env_id"], lambda obs, env: model.predict(obs, deterministic=True)[0], args.episodes, args.seed,
                     goal_wrapper=cfg.get("goal_wrapper", False), context_wrapper=cfg.get("episode_context", False))
        rows.append(dict(env_id=cfg["env_id"], algo=cfg["algo"], seed=cfg.get("seed", 0),
                         path=str(model_file(d).relative_to(ROOT)).replace("\\", "/"),
                         timesteps=cfg["timesteps"], train_time_s=cfg.get("train_time_s"),
                         note=(f"{cfg['algo']} on {cfg['timesteps']//1000}k planner-demo transitions, seed {cfg.get('seed', 0)}" if cfg["algo"].startswith("BC")
                               else f"SB3 {cfg['algo']}, {cfg['timesteps']//1000}k steps, seed {cfg.get('seed', 0)}"), **r))
    legacy = legacy_rows()
    agg = aggregate(rows)

    (out / "zoo_summary.json").write_text(json.dumps(dict(episodes=args.episodes, seed=args.seed, runs=rows, zoo=agg,
                                                          legacy=legacy), indent=1, ensure_ascii=False), encoding="utf-8")
    plot_success(agg, out / "zoo_success.png")
    plot_returns(agg, out / "zoo_returns.png")

    def f(x, fmt):
        return "-" if x is None else format(x, fmt)
    lines = ["# RL model zoo evaluation", "",
             f"결정적 정책, 환경당 {args.episodes} 에피소드(200 step), 평가 seed {args.seed}. success는 마지막 step의 과제 성공 조건.",
             "학습 seed가 여러 개인 모델은 seed별 성공률의 평균 ± 표준편차(n = seed 수)로 표시합니다.", "",
             "## 환경", "", "| env id | 과제 | obs | act |", "|---|---|---|---|"]
    for e in TASKS:
        env = make_env(e)
        lines.append(f"| `{e}` | {TASK_DESCRIPTIONS[e]} | {env.observation_space.shape[0]} | {env.action_space.shape[0]} |")
        env.close()
    lines += ["", "## 모델", "",
              "| # | env | model | steps | seeds | success | mean return ± std | final dist (m) | collision | train (min) | 파일 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    k = 0
    for e in envs_used:
        for r in [r for r in agg if r["env_id"] == e]:
            if r["algo"] == "random":
                lines.append(f"|  | `{e}` | random (기준선) | - | - | {f(r['success_rate']*100, '.0f')}% | "
                             f"{r['mean_return']:.1f} ± {r['std_return']:.1f} | {r['mean_final_dist']:.3f} | "
                             f"{r['collision_rate']*100:.0f}% | - | - |")
            else:
                k += r["n_seeds"]
                succ = (f"**{r['success_rate']*100:.0f} ± {r['success_std']*100:.0f}%**" if r["n_seeds"] > 1
                        else f"**{r['success_rate']*100:.0f}%**")
                steps = f"{r['timesteps']//1000}k" + (" demo" if r["algo"].startswith("BC") else "")
                lines.append(f"| {k} | `{e}` | {r['algo']} | {steps} | {r['n_seeds']} | {succ} | "
                             f"{r['mean_return']:.1f} ± {r['std_return']:.1f} | {r['mean_final_dist']:.3f} | "
                             f"{r['collision_rate']*100:.0f}% | {f(r['train_time_s']/60, '.1f')} | `{r['path']}` |")
    for r in legacy:
        k += 1
        sr = "-" if r["success_rate"] is None else f"{r['success_rate']*100:.1f}%"
        mr = "-" if r["mean_return"] is None else f"{r['mean_return']:.1f} ± {r['std_return']:.1f}"
        fd = "-" if r["mean_final_dist"] is None else f"{r['mean_final_dist']:.4f}"
        cr = "-" if r["collision_rate"] is None else f"{r['collision_rate']*100:.1f}%"
        lines.append(f"| {k} | {r['env_id']} | {r['algo']} | - | - | {sr} | {mr} | {fd} | {cr} | - | `{r['path']}` ({r['note']}) |")
    n_zoo = sum(r['algo'] != 'random' for r in rows)
    lines += ["", f"총 {k}개 모델 (SB3 zoo {n_zoo}개 = {sum(r['algo'] != 'random' for r in agg)}개 (env, algo) 조합 × seed + 기존 {len(legacy)}개).", "",
              "![success](zoo_success.png)", "", "![returns](zoo_returns.png)", ""]
    (out / "zoo_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
