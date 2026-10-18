"""
GRPO (Group Relative Policy Optimization) agent for DroneNavEnv.

GRPO is a critic-free policy-gradient method. Instead of learning a value
function to get a baseline, it samples a *group* of G rollouts from the same
start state and scores each rollout relative to the rest of its group:

    A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

Every step of rollout i gets advantage A_i ("outcome supervision"). With
--advantage process, rewards are normalized per group instead and each step
gets the sum of the normalized rewards that follow it ("process supervision").

The policy is then updated with a PPO-style clipped surrogate over several
epochs, plus a KL penalty to a reference policy that is refreshed
periodically (as in DeepSeekMath's iterative GRPO).

The environment always resets to (0, 0). To give each group a different start
state, training places the drone on a random free cell after reset; the
default start is used for a share of groups and for evaluation.

Usage:
    uv run python grpo_agent.py train --iterations 300
    uv run python grpo_agent.py eval --episodes 10
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment import DroneNavEnv
from rule_based_agent import RuleBasedAgent, render

MODEL_PATH = "models/grpo_drone.pt"
RESULTS_DIR = "results"


class Policy(nn.Module):
    def __init__(self, obs_dim=8, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return Categorical(logits=self.net(obs))

    @torch.no_grad()
    def act(self, obs, greedy=True):
        dist = self(torch.as_tensor(obs, dtype=torch.float32))
        return int(dist.probs.argmax()) if greedy else int(dist.sample())


def free_cells(env):
    return [(r, c) for r in range(env.height) for c in range(env.width)
            if (r, c) not in env.obstacles and (r, c) != env.target_pos]


def reset_at(env, start):
    """Reset the env, then move the drone to `start` (the env has a fixed start)."""
    env.reset()
    env.drone_pos = start
    return env._get_obs()


def rollout(env, policy, start, greedy=False):
    obs = reset_at(env, start)
    observations, actions, rewards = [], [], []
    done, info = False, {}
    while not done:
        action = policy.act(obs, greedy=greedy)
        observations.append(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return {
        "obs": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "success": env.drone_pos == env.target_pos,
        "collision": info.get("collision", False),
    }


def group_advantages(group, mode, eps=1e-6):
    """Group-relative advantages, one array per rollout (one value per step)."""
    if mode == "outcome":
        returns = np.array([g["rewards"].sum() for g in group])
        norm = (returns - returns.mean()) / (returns.std() + eps)
        return [np.full(len(g["rewards"]), a, dtype=np.float32) for g, a in zip(group, norm)]
    # process: normalize every step reward against all step rewards in the group
    all_r = np.concatenate([g["rewards"] for g in group])
    mean, std = all_r.mean(), all_r.std() + eps
    return [np.cumsum(((g["rewards"] - mean) / std)[::-1])[::-1].astype(np.float32)
            for g in group]


def evaluate(env, policy, starts):
    """Greedy rollout from each start; returns per-start results."""
    out = []
    for s in starts:
        g = rollout(env, policy, s, greedy=True)
        out.append({"start": s, "success": g["success"], "collision": g["collision"],
                    "steps": len(g["rewards"]), "reward": float(g["rewards"].sum())})
    return out


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = DroneNavEnv()
    starts = free_cells(env)

    policy = Policy()
    ref = Policy()
    ref.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    history = []
    t0 = time.time()
    for it in range(1, args.iterations + 1):
        # 1. Sample groups: G rollouts per start state from the current (old) policy.
        batch_obs, batch_act, batch_adv = [], [], []
        returns, successes = [], []
        for _ in range(args.groups):
            start = env.start_pos if rng.random() < args.default_start_prob \
                else starts[rng.integers(len(starts))]
            group = [rollout(env, policy, start) for _ in range(args.group_size)]
            for g, adv in zip(group, group_advantages(group, args.advantage)):
                batch_obs.append(g["obs"])
                batch_act.append(g["actions"])
                batch_adv.append(adv)
                returns.append(g["rewards"].sum())
                successes.append(g["success"])

        obs = torch.as_tensor(np.concatenate(batch_obs))
        act = torch.as_tensor(np.concatenate(batch_act))
        adv = torch.as_tensor(np.concatenate(batch_adv))
        with torch.no_grad():
            old_logp = policy(obs).log_prob(act)
            ref_logp_all = torch.log(ref(obs).probs + 1e-8)

        # 2. Clipped surrogate + KL(pi || ref) + entropy, over several epochs.
        for _ in range(args.epochs):
            dist = policy(obs)
            logp = dist.log_prob(act)
            ratio = torch.exp(logp - old_logp)
            surr = torch.min(ratio * adv,
                             torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv)
            # k3 estimator of KL to the reference policy, as in GRPO.
            ref_logp = ref_logp_all.gather(1, act[:, None]).squeeze(1)
            kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
            loss = -(surr - args.beta * kl).mean() - args.ent_coef * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()

        if it % args.ref_update == 0:
            ref.load_state_dict(policy.state_dict())

        # 3. Log greedy performance from the default start and from all starts.
        if it % args.log_every == 0 or it == args.iterations:
            default = evaluate(env, policy, [env.start_pos])[0]
            everywhere = evaluate(env, policy, starts)
            rec = {
                "iteration": it,
                "episodes": it * args.groups * args.group_size,
                "train_mean_return": float(np.mean(returns)),
                "train_success_rate": float(np.mean(successes)),
                "default_start_reward": default["reward"],
                "default_start_steps": default["steps"],
                "default_start_success": default["success"],
                "all_starts_success_rate": float(np.mean([e["success"] for e in everywhere])),
                "kl": float(kl.mean().detach()),
            }
            history.append(rec)
            print(f"it {it:4d} | train R {rec['train_mean_return']:7.2f} "
                  f"succ {rec['train_success_rate']:.2f} | greedy default: "
                  f"R {default['reward']:6.1f} steps {default['steps']:3d} "
                  f"succ {default['success']} | all-starts succ "
                  f"{rec['all_starts_success_rate']:.2f} | {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "args": vars(args)}, MODEL_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "grpo_training_log.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history)
    print(f"Saved model to {MODEL_PATH}")


def plot_history(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [h["episodes"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(eps, [h["train_mean_return"] for h in history], label="mean training return")
    ax1.plot(eps, [h["default_start_reward"] for h in history], label="greedy, default start")
    ax1.axhline(66.2, ls="--", c="grey", label="rule-based (66.2)")
    ax1.set_xlabel("training episodes"); ax1.set_ylabel("episode reward"); ax1.legend()
    ax2.plot(eps, [h["train_success_rate"] for h in history], label="training rollouts")
    ax2.plot(eps, [h["all_starts_success_rate"] for h in history], label="greedy, all start cells")
    ax2.set_xlabel("training episodes"); ax2.set_ylabel("success rate"); ax2.set_ylim(0, 1.05)
    ax2.legend()
    fig.suptitle("GRPO on DroneNavEnv")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "grpo_training_curve.png"), dpi=120)


def rule_based_from(env, start):
    agent = RuleBasedAgent(grid_size=(env.height, env.width))
    obs = reset_at(env, start)
    total, steps, done, info = 0.0, 0, False, {}
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs))
        total += r
        steps += 1
        done = term or trunc
    return {"success": env.drone_pos == env.target_pos, "collision": info.get("collision", False),
            "steps": steps, "reward": total}


def summarize(results):
    return {
        "success_rate": float(np.mean([r["success"] for r in results])),
        "collision_rate": float(np.mean([r["collision"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_reward": float(np.mean([r["reward"] for r in results])),
    }


def run_eval(args):
    env = DroneNavEnv()
    ckpt = torch.load(MODEL_PATH)
    policy = Policy()
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    # Default start, as in rule_based_agent.py (the env is deterministic,
    # so every episode from the default start is identical under a greedy policy).
    print("GRPO agent, default start (0, 0):")
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total, path, done, info = 0.0, [env.drone_pos], False, {}
        while not done:
            obs, r, term, trunc, info = env.step(policy.act(obs))
            total += r
            path.append(env.drone_pos)
            done = term or trunc
        print(f"Episode {ep + 1}: success={env.drone_pos == env.target_pos} "
              f"steps={len(path) - 1} reward={total:.1f} collision={info.get('collision', False)}")
    print("\nPath taken:")
    print(render(env, path))

    # Generalization: every free cell as a start state, GRPO vs rule-based.
    starts = free_cells(env)
    grpo = summarize(evaluate(env, policy, starts))
    rule = summarize([rule_based_from(env, s) for s in starts])
    default_grpo = evaluate(env, policy, [env.start_pos])[0]
    default_rule = rule_based_from(env, env.start_pos)

    print(f"\nComparison ({len(starts)} start cells):")
    print(f"{'agent':<12}{'success':>9}{'collision':>11}{'mean steps':>12}{'mean reward':>13}"
          f"{'default R':>11}{'default steps':>15}")
    for name, s, d in (("GRPO", grpo, default_grpo), ("rule-based", rule, default_rule)):
        print(f"{name:<12}{s['success_rate']:>9.2f}{s['collision_rate']:>11.2f}"
              f"{s['mean_steps']:>12.1f}{s['mean_reward']:>13.1f}"
              f"{d['reward']:>11.1f}{d['steps']:>15d}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "grpo_eval.json"), "w") as f:
        json.dump({"default_start": {"grpo": default_grpo | {"start": list(env.start_pos)},
                                     "rule_based": default_rule},
                   "all_starts": {"n_starts": len(starts), "grpo": grpo, "rule_based": rule}},
                  f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else list(o))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--iterations", type=int, default=300)
    t.add_argument("--groups", type=int, default=8, help="start states per iteration")
    t.add_argument("--group-size", type=int, default=16, help="rollouts per start state (G)")
    t.add_argument("--advantage", choices=["outcome", "process"], default="outcome")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--clip", type=float, default=0.2)
    t.add_argument("--beta", type=float, default=0.01, help="KL penalty to reference policy")
    t.add_argument("--ref-update", type=int, default=10, help="iterations between ref refreshes")
    t.add_argument("--ent-coef", type=float, default=0.01)
    t.add_argument("--default-start-prob", type=float, default=0.25)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    train(args) if args.cmd == "train" else run_eval(args)
