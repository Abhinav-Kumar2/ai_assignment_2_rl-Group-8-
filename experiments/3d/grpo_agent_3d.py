import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from environment_3d import DroneNavEnv3D

MODEL_PATH = os.path.join(BASE_DIR, "models", "grpo_drone_3d.pt")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "grpo_3d_evaluation_summary.json")


class PolicyNetwork(nn.Module):
    """Critic-free policy network mapping 12-dim observation to 6 discrete action logits."""
    def __init__(self, obs_dim=12, action_dim=6, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return Categorical(logits=self.net(obs))

    @torch.no_grad()
    def act(self, obs, greedy=False):
        dist = self(torch.as_tensor(obs, dtype=torch.float32))
        return int(dist.probs.argmax()) if greedy else int(dist.sample())


def get_free_cells(env):
    return [
        (r, c, a)
        for r in range(env.height)
        for c in range(env.width)
        for a in range(env.depth)
        if (r, c, a) not in env.obstacles and (r, c, a) != env.target_pos
    ]


def rollout(env, policy, start=None, greedy=False):
    obs, _ = env.reset(options={"start_pos": start} if start is not None else None)
    observations, actions, rewards = [], [], []
    done, collision = False, False

    while not done:
        action = policy.act(obs, greedy=greedy)
        observations.append(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if info.get("collision", False):
            collision = True
        done = terminated or truncated

    success = env.drone_pos == env.target_pos
    return {
        "obs": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "success": success,
        "collision": collision,
        "truncated": done and not success and not collision,
    }


def compute_group_advantages(group, eps=1e-6):
    """
    GRPO advantage calculation:
    Scores each rollout trajectory against its group of sampled peer rollouts from the same start state:
        A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
    Assigns A_i as the constant advantage to each step of trajectory i (outcome supervision).
    """
    returns = np.array([g["rewards"].sum() for g in group], dtype=np.float32)
    norm = (returns - returns.mean()) / (returns.std() + eps)
    return [np.full(len(g["rewards"]), a, dtype=np.float32) for g, a in zip(group, norm)]


def train(iterations=100, groups=4, group_size=8, epochs=4, lr=3e-4, clip_eps=0.2,
          beta=0.01, ent_coef=0.01, ref_update_freq=10, seed=0,
          model_path=MODEL_PATH):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env = DroneNavEnv3D()
    free_cells = get_free_cells(env)

    policy = PolicyNetwork()
    ref_policy = PolicyNetwork()
    ref_policy.load_state_dict(policy.state_dict())
    ref_policy.eval()

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    print(f"Starting GRPO training for {iterations} iterations...")
    pbar = tqdm(range(1, iterations + 1), desc="GRPO Training", unit="iter")
    for it in pbar:
        batch_obs, batch_act, batch_adv = [], [], []
        returns, successes = [], []

        for _ in range(groups):
            # Mix default start (25%) and random free starts (75%)
            start = env.start_pos if rng.random() < 0.25 else free_cells[rng.integers(len(free_cells))]
            group = [rollout(env, policy, start=start, greedy=False) for _ in range(group_size)]
            advantages = compute_group_advantages(group)

            for g, adv in zip(group, advantages):
                batch_obs.append(g["obs"])
                batch_act.append(g["actions"])
                batch_adv.append(adv)
                returns.append(float(g["rewards"].sum()))
                successes.append(g["success"])

        if not batch_obs or all(len(o) == 0 for o in batch_obs):
            continue

        obs_t = torch.as_tensor(np.concatenate(batch_obs), dtype=torch.float32)
        act_t = torch.as_tensor(np.concatenate(batch_act), dtype=torch.int64)
        adv_t = torch.as_tensor(np.concatenate(batch_adv), dtype=torch.float32)

        with torch.no_grad():
            old_logp = policy(obs_t).log_prob(act_t)
            ref_logp_all = torch.log(ref_policy(obs_t).probs + 1e-8)

        # Optimization epochs on collected group trajectories
        for _ in range(epochs):
            dist = policy(obs_t)
            logp = dist.log_prob(act_t)
            ratio = torch.exp(logp - old_logp)

            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            surrogate = torch.min(surr1, surr2)

            ref_logp = ref_logp_all.gather(1, act_t.unsqueeze(1)).squeeze(1)
            # k3 estimator of KL divergence: exp(ref - pi) - (ref - pi) - 1
            kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0

            loss = -(surrogate - beta * kl).mean() - ent_coef * dist.entropy().mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        if it % ref_update_freq == 0:
            ref_policy.load_state_dict(policy.state_dict())

        if it % 10 == 0 or it == iterations:
            mean_ret = float(np.mean(returns)) if returns else 0.0
            succ_rate = float(np.mean(successes)) if successes else 0.0
            print(f"Iteration {it:3d}/{iterations}: mean_return={mean_ret:7.2f}, success_rate={succ_rate * 100:.1f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(policy.state_dict(), model_path)
    print(f"Saved trained GRPO model to {model_path}")
    return policy


def evaluate(model_path=MODEL_PATH, episodes=10, seed=0, output_path=RESULTS_PATH, verbose=True):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path}. Please train a model first using:\n"
            f"  python experiments/3d/grpo_agent_3d.py --train --iterations 100"
        )

    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    policy.eval()

    env = DroneNavEnv3D()
    successes, collisions, timeouts = 0, 0, 0
    all_rewards, all_steps = [], []

    for ep in tqdm(range(episodes), desc="Evaluating", unit="ep"):
        res = rollout(env, policy, start=env.start_pos, greedy=True)
        ep_reward = float(res["rewards"].sum())
        steps = len(res["rewards"])
        success = res["success"]
        collision = res["collision"]
        timeout = res["truncated"]

        if success:
            successes += 1
            reason = "target_reached"
        elif collision:
            collisions += 1
            reason = "collision"
        elif timeout:
            timeouts += 1
            reason = "time_limit"
        else:
            reason = "other"

        all_rewards.append(ep_reward)
        all_steps.append(steps)

        if verbose:
            print(f"Episode {ep + 1:3d}: success={success} steps={steps:3d} "
                  f"reward={ep_reward:7.2f} reason={reason}")

    n = len(all_rewards)
    summary = {
        "model_path": os.path.relpath(model_path, BASE_DIR),
        "episodes": n,
        "success_rate": round(successes / n, 4) if n else 0.0,
        "collision_rate": round(collisions / n, 4) if n else 0.0,
        "timeout_rate": round(timeouts / n, 4) if n else 0.0,
        "mean_episode_reward": round(float(np.mean(all_rewards)), 3) if n else 0.0,
        "mean_episode_length": round(float(np.mean(all_steps)), 2) if n else 0.0,
        "success_count": successes,
        "collision_count": collisions,
        "timeout_count": timeouts,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nGRPO Evaluation Summary ({n} episodes):")
    print(f"  Success rate:        {summary['success_rate'] * 100:.1f}% ({successes}/{n})")
    print(f"  Collision rate:      {summary['collision_rate'] * 100:.1f}% ({collisions}/{n})")
    print(f"  Timeout rate:        {summary['timeout_rate'] * 100:.1f}% ({timeouts}/{n})")
    print(f"  Mean episode length: {summary['mean_episode_length']:.2f}")
    print(f"  Mean episode reward: {summary['mean_episode_reward']:.2f}")
    print(f"Saved evaluation summary to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Critic-free GRPO Agent for DroneNavEnv3D")
    parser.add_argument("--train", action="store_true", help="train GRPO agent")
    parser.add_argument("--eval", action="store_true", help="evaluate trained GRPO agent")
    parser.add_argument("--iterations", type=int, default=100, help="training iterations")
    parser.add_argument("--episodes", type=int, default=10, help="evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="checkpoint path")
    args = parser.parse_args()

    if args.eval:
        evaluate(model_path=args.model_path, episodes=args.episodes, seed=args.seed)
    elif args.train:
        train(iterations=args.iterations, seed=args.seed, model_path=args.model_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
