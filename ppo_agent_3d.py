# ppo training script for 3d drone navigation in dronenavenv3d

import argparse
import csv
import json
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from agents_3d import Agent3D
from environment_3d import DroneNavEnv3D

# paths are relative to this file so the script works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_drone_3d.zip")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


class PPOAgent3D(Agent3D):
    """trained ppo policy wrapped in the Agent3D interface used by visualizer_3d.py"""

    name = "ppo"

    def __init__(self, model_path=MODEL_PATH, deterministic=True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model file not found at {model_path} (train one with: uv run python ppo_agent_3d.py)"
            )
        self.model = PPO.load(model_path, device="cpu")
        self.deterministic = deterministic

    def act(self, obs, env):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)


def train(timesteps=50_000, seed=0, n_envs=8, verbose=1, model_path=MODEL_PATH):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    # verify observation and action spaces before training
    check_env = DroneNavEnv3D()
    obs_space = check_env.observation_space
    act_space = check_env.action_space
    print(f"verified 3d environment spaces:")
    print(f"  observation space: {obs_space} (shape: {obs_space.shape})")
    print(f"  action space: {act_space} (n: {act_space.n})")

    print(f"\ntraining configuration:")
    print(f"  timesteps: {timesteps}")
    print(f"  seed: {seed}")
    print(f"  parallel envs: {n_envs}")
    print(f"  policy: MlpPolicy (64x64)")
    print(f"  model output: {model_path}")

    # vectorize environments for training without gui
    env = make_vec_env(DroneNavEnv3D, n_envs=n_envs, seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        device="cuda:0",
        verbose=verbose,
    )

    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(model_path)
    print(f"\nsaved trained model to: {model_path}")
    return model


def verify_saved_model(path=MODEL_PATH):
    print(f"\nverifying model loading on fresh DroneNavEnv3D instance...")
    agent = PPOAgent3D(path)
    env = DroneNavEnv3D()
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and steps < 20:
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"model verification successful:")
    print(f"  rollout steps tested: {steps}")
    print(f"  drone end pos: {env.drone_pos}")
    print(f"  total reward over test steps: {total_reward:.2f}")
    print(f"  terminated: {terminated}, truncated: {truncated}")


def evaluate(model_path=MODEL_PATH, episodes=10, seed=0, output_dir=RESULTS_DIR, verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    agent = PPOAgent3D(model_path)
    env = DroneNavEnv3D()

    per_episode = []
    for ep in range(episodes):
        ep_seed = seed + ep
        obs, _ = env.reset(seed=ep_seed)
        agent.reset(env)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        collision = False

        while not (terminated or truncated):
            action = agent.act(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if info.get("collision", False):
                collision = True

        target_reached = env.drone_pos == env.target_pos
        if target_reached:
            termination_reason = "target_reached"
        elif collision:
            termination_reason = "collision"
        elif truncated:
            termination_reason = "time_limit"
        else:
            termination_reason = "other"

        per_episode.append({
            "episode": ep + 1,
            "seed": ep_seed,
            "reward": round(total_reward, 3),
            "steps": steps,
            "success": target_reached,
            "collision": collision,
            "truncated": truncated,
            "termination_reason": termination_reason,
        })

        if verbose:
            print(f"Episode {ep + 1:3d}: success={target_reached} steps={steps:3d} "
                  f"reward={total_reward:7.2f} reason={termination_reason}")

    n = len(per_episode)
    successes = sum(1 for e in per_episode if e["success"])
    collisions = sum(1 for e in per_episode if e["collision"])
    truncations = sum(1 for e in per_episode if e["truncated"])
    mean_length = float(np.mean([e["steps"] for e in per_episode])) if n else 0.0
    mean_reward = float(np.mean([e["reward"] for e in per_episode])) if n else 0.0

    summary = {
        "model_path": os.path.relpath(model_path, BASE_DIR),
        "episodes": n,
        "success_rate": round(successes / n, 4) if n else 0.0,
        "collision_rate": round(collisions / n, 4) if n else 0.0,
        "truncation_rate": round(truncations / n, 4) if n else 0.0,
        "average_episode_length": round(mean_length, 2),
        "average_episode_reward": round(mean_reward, 3),
        "success_count": successes,
        "collision_count": collisions,
        "truncation_count": truncations,
    }

    # Save per-episode CSV
    csv_path = os.path.join(output_dir, "ppo_3d_evaluation_episodes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "seed", "reward", "steps", "success", "collision", "truncated", "termination_reason"
        ])
        writer.writeheader()
        writer.writerows(per_episode)

    # Save summary JSON
    json_path = os.path.join(output_dir, "ppo_3d_evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate graph
    plot_path = os.path.join(output_dir, "ppo_3d_evaluation_metrics.png")
    _plot_evaluation_metrics(per_episode, summary, plot_path)

    print(f"\nEvaluation Summary ({n} episodes):")
    print(f"  Success rate:           {summary['success_rate'] * 100:.1f}% ({successes}/{n})")
    print(f"  Collision rate:         {summary['collision_rate'] * 100:.1f}% ({collisions}/{n})")
    print(f"  Truncation rate:        {summary['truncation_rate'] * 100:.1f}% ({truncations}/{n})")
    print(f"  Average episode length: {summary['average_episode_length']:.2f}")
    print(f"  Average episode reward: {summary['average_episode_reward']:.2f}")
    print(f"Saved artifacts:")
    print(f"  CSV:   {csv_path}")
    print(f"  JSON:  {json_path}")
    print(f"  Plot:  {plot_path}")

    return summary


def _plot_evaluation_metrics(per_episode, summary, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Rates bar chart
    rates = [summary["success_rate"] * 100, summary["collision_rate"] * 100, summary["truncation_rate"] * 100]
    labels = ["Success", "Collision", "Truncated"]
    colors = ["#2ca02c", "#d62728", "#ff7f0e"]
    bars = ax1.bar(labels, rates, color=colors, alpha=0.85)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("3D PPO Outcome Rates")
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Per-episode reward plot
    rewards = [e["reward"] for e in per_episode]
    episodes = [e["episode"] for e in per_episode]
    ax2.plot(episodes, rewards, marker="o", color="#1f77b4", label="Episode Reward")
    ax2.axhline(summary["average_episode_reward"], color="gray", linestyle="--",
                label=f"Mean: {summary['average_episode_reward']:.1f}")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.set_title("Reward per Episode")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="PPO 3D Agent training and evaluation")
    parser.add_argument("--eval", action="store_true", help="evaluate trained model without training")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="path to model zip")
    parser.add_argument("--episodes", type=int, default=10, help="number of evaluation episodes")
    parser.add_argument("--timesteps", type=int, default=50_000, help="training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--smoke-test", action="store_true", help="run short smoke test (1024 timesteps)")
    args = parser.parse_args()

    if args.eval:
        evaluate(model_path=args.model_path, episodes=args.episodes, seed=args.seed)
        return

    steps = 1024 if args.smoke_test else args.timesteps
    train(timesteps=steps, seed=args.seed, n_envs=4 if args.smoke_test else 8, verbose=0,
          model_path=args.model_path)
    verify_saved_model(args.model_path)


if __name__ == "__main__":
    main()
