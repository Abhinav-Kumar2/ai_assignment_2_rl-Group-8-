# ppo training script for 3d drone navigation in dronenavenv3d

import argparse
import csv
import json
import os
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

from agents_3d import Agent3D
from environment_3d import DroneNavEnv3D

# paths are relative to this file so the script works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_drone_3d.zip")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEFAULT_GPS_NOISE = 0.5


class PPOAgent3D(Agent3D):
    """trained ppo policy wrapped in the Agent3D interface used by visualizer_3d.py"""

    name = "ppo"

    def __init__(self, model_path=MODEL_PATH, deterministic=True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model file not found at {model_path} (train one with: python ppo_agent_3d.py)"
            )
        self.model = PPO.load(model_path, device="cpu")
        self.deterministic = deterministic

    def act(self, obs, env):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)


class TrainingStartsWrapper(gym.Wrapper):
    """
    training-only wrapper: with probability random_start_prob an episode starts
    from a random free cell instead of the fixed start, and gps noise is drawn
    per episode from [0, max_gps_noise].

    the map needs a detour away from the target, so from the fixed start alone
    ppo only ever sees the "fly at the target and crash" outcome. random starts
    let it learn the way through each wall gap from nearby cells first, then
    chain those pieces back to the real start. varying the noise level makes
    the policy robust to any gps noise up to max_gps_noise. rewards and
    observations are the env's own, evaluation always uses the plain env.

    with geodesic_shaping the env's "closer to the target" bonus (manhattan
    distance) is swapped for the same bonus measured along the shortest free
    path (bfs over the map). the manhattan bonus punishes the westward detour
    that the map requires and is what traps ppo against the walls. both are
    potential-based shaping, so the best policy is unchanged; the policy still
    only sees the observation, the map is used for the training reward only.
    """

    def __init__(self, env, random_start_prob=0.5, max_gps_noise=0.0, geodesic_shaping=True,
                 start_pos=None, target_pos=None):
        super().__init__(env)
        base = env.unwrapped
        if start_pos is not None:
            base.start_pos = tuple(start_pos)
            base.drone_pos = base.start_pos
        if target_pos is not None:
            base.target_pos = tuple(target_pos)
            base.obstacles.discard(base.target_pos)

        self.random_start_prob = random_start_prob
        self.max_gps_noise = max_gps_noise
        self.geodesic_shaping = geodesic_shaping
        self.path_dist = self._bfs_from_target(base)
        # Only select free cells that are reachable from the target
        self.free_cells = [
            pos for pos in self.path_dist.keys()
            if pos != base.target_pos
        ]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        rng = base.np_random  # seeded by the reset above
        base.gps_noise_std = float(rng.uniform(0.0, self.max_gps_noise)) if self.max_gps_noise > 0 else 0.0
        if rng.random() < self.random_start_prob and self.free_cells:
            base.drone_pos = self.free_cells[rng.integers(len(self.free_cells))]
        return base._get_obs(), info

    @staticmethod
    def _bfs_from_target(base):
        dist = {base.target_pos: 0}
        queue = deque([base.target_pos])
        while queue:
            pos = queue.popleft()
            for dr, dc, da in base._action_to_delta.values():
                nxt = (pos[0] + dr, pos[1] + dc, pos[2] + da)
                if nxt not in dist and not base._is_blocked(nxt):
                    dist[nxt] = dist[pos] + 1
                    queue.append(nxt)
        return dist

    def step(self, action):
        base = self.env.unwrapped
        prev = base.drone_pos
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.geodesic_shaping:
            curr = base.drone_pos
            if prev in self.path_dist and curr in self.path_dist:
                manhattan_bonus = base._dist_to_target(prev) - base._dist_to_target(curr)
                path_bonus = self.path_dist[prev] - self.path_dist[curr]
                reward = reward - manhattan_bonus + path_bonus
        return obs, reward, terminated, truncated, info


def train(timesteps=50_000, seed=0, n_envs=8, verbose=1, model_path=MODEL_PATH,
          random_start_prob=0.5, max_gps_noise=0.0, geodesic_shaping=True,
          start_pos=(0, 0, 0), target_pos=None):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    # verify observation and action spaces before training
    check_env = DroneNavEnv3D(start_pos=start_pos, target_pos=target_pos)
    obs_space = check_env.observation_space
    act_space = check_env.action_space
    print(f"verified 3d environment spaces:")
    print(f"  observation space: {obs_space} (shape: {obs_space.shape})")
    print(f"  action space: {act_space} (n: {act_space.n})")
    print(f"  start pos: {check_env.start_pos}")
    print(f"  target pos: {check_env.target_pos}")

    print(f"\ntraining configuration:")
    print(f"  timesteps: {timesteps}")
    print(f"  seed: {seed}")
    print(f"  parallel envs: {n_envs}")
    print(f"  policy: MlpPolicy (64x64)")
    print(f"  random start prob: {random_start_prob}")
    print(f"  gps noise std per episode: uniform [0, {max_gps_noise}]")
    print(f"  shortest-path reward shaping: {geodesic_shaping}")
    print(f"  model output: {model_path}")

    # a 64x64 mlp is fastest on one cpu thread; extra threads only add overhead
    torch.set_num_threads(1)

    # vectorize environments for training without gui
    env_kwargs = dict(start_pos=start_pos, target_pos=target_pos)
    env = make_vec_env(
        DroneNavEnv3D, n_envs=n_envs, seed=seed,
        env_kwargs=env_kwargs,
        wrapper_class=TrainingStartsWrapper,
        wrapper_kwargs=dict(random_start_prob=random_start_prob, max_gps_noise=max_gps_noise,
                            geodesic_shaping=geodesic_shaping),
    )

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
        device="cpu",
        verbose=verbose,
    )

    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(model_path)
    print(f"\nsaved trained model to: {model_path}")
    return model


def verify_saved_model(path=MODEL_PATH, gps_noise=0.0):
    print(f"\nverifying model loading on fresh DroneNavEnv3D instance...")
    agent = PPOAgent3D(path)
    env = DroneNavEnv3D(gps_noise_std=gps_noise)
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


def evaluate(model_path=MODEL_PATH, episodes=10, seed=0, output_dir=RESULTS_DIR, verbose=True, gps_noise=0.0,
             random_starts=False, start_pos=(0, 0, 0), target_pos=None):
    os.makedirs(output_dir, exist_ok=True)
    agent = PPOAgent3D(model_path)
    env = DroneNavEnv3D(gps_noise_std=gps_noise, start_pos=start_pos, target_pos=target_pos)
    # random_starts: each episode starts from a random free cell instead of the fixed start
    starts = TrainingStartsWrapper(DroneNavEnv3D(start_pos=start_pos, target_pos=target_pos), geodesic_shaping=False).free_cells if random_starts else None
    start_rng = np.random.default_rng(seed)

    per_episode = []
    for ep in tqdm(range(episodes), desc="Evaluating", unit="ep"):
        ep_seed = seed + ep
        obs, _ = env.reset(seed=ep_seed)
        if starts is not None:
            env.drone_pos = starts[start_rng.integers(len(starts))]
            obs = env._get_obs()
        start_pos = env.drone_pos
        agent.reset(env)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        collision = False

        visited_history = [env.drone_pos]
        ineffective_steps = 0
        max_consec_ineffective = 0
        cur_consec_ineffective = 0
        oscillations = 0
        min_dist_to_goal = env._dist_to_target(env.drone_pos)
        steps_since_progress = 0
        max_steps_without_progress = 0

        while not (terminated or truncated):
            prev_pos = env.drone_pos
            action = agent.act(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            curr_pos = env.drone_pos
            total_reward += float(reward)
            steps += 1
            if info.get("collision", False):
                collision = True

            # Track stuck and oscillation diagnostics
            pos_changed = (curr_pos != prev_pos)
            if not pos_changed:
                ineffective_steps += 1
                cur_consec_ineffective += 1
                if cur_consec_ineffective > max_consec_ineffective:
                    max_consec_ineffective = cur_consec_ineffective
            else:
                cur_consec_ineffective = 0

            if len(visited_history) >= 2 and curr_pos == visited_history[-2] and pos_changed:
                oscillations += 1

            dist = env._dist_to_target(curr_pos)
            if dist < min_dist_to_goal:
                min_dist_to_goal = dist
                steps_since_progress = 0
            else:
                steps_since_progress += 1
                if steps_since_progress > max_steps_without_progress:
                    max_steps_without_progress = steps_since_progress

            visited_history.append(curr_pos)

        target_reached = env.drone_pos == env.target_pos
        if target_reached:
            termination_reason = "target_reached"
        elif collision:
            termination_reason = "collision"
        elif truncated:
            termination_reason = "time_limit"
        else:
            termination_reason = "other"

        final_dist = env._dist_to_target(env.drone_pos)
        per_episode.append({
            "episode": ep + 1,
            "seed": ep_seed,
            "start": str(start_pos),
            "reward": round(total_reward, 3),
            "steps": steps,
            "success": target_reached,
            "collision": collision,
            "truncated": truncated,
            "termination_reason": termination_reason,
            "ineffective_actions": ineffective_steps,
            "max_consecutive_ineffective": max_consec_ineffective,
            "oscillations": oscillations,
            "max_steps_without_progress": max_steps_without_progress,
            "final_distance_to_goal": final_dist,
        })

        if verbose:
            print(f"Episode {ep + 1:3d}: success={target_reached} steps={steps:3d} "
                  f"reward={total_reward:7.2f} reason={termination_reason} "
                  f"ineffective={ineffective_steps} osc={oscillations} final_dist={final_dist}")

    n = len(per_episode)
    successes = sum(1 for e in per_episode if e["success"])
    collisions = sum(1 for e in per_episode if e["collision"])
    truncations = sum(1 for e in per_episode if e["truncated"])
    mean_length = float(np.mean([e["steps"] for e in per_episode])) if n else 0.0
    mean_reward = float(np.mean([e["reward"] for e in per_episode])) if n else 0.0
    mean_ineffective = float(np.mean([e["ineffective_actions"] for e in per_episode])) if n else 0.0
    mean_oscillations = float(np.mean([e["oscillations"] for e in per_episode])) if n else 0.0
    mean_final_dist = float(np.mean([e["final_distance_to_goal"] for e in per_episode])) if n else 0.0

    summary = {
        "model_path": os.path.relpath(model_path, BASE_DIR),
        "gps_noise_std": gps_noise,
        "random_starts": random_starts,
        "episodes": n,
        "success_rate": round(successes / n, 4) if n else 0.0,
        "collision_rate": round(collisions / n, 4) if n else 0.0,
        "truncation_rate": round(truncations / n, 4) if n else 0.0,
        "average_episode_length": round(mean_length, 2),
        "average_episode_reward": round(mean_reward, 3),
        "average_ineffective_actions": round(mean_ineffective, 2),
        "average_oscillations": round(mean_oscillations, 2),
        "average_final_distance_to_goal": round(mean_final_dist, 2),
        "success_count": successes,
        "collision_count": collisions,
        "truncation_count": truncations,
    }

    # one set of files per noise level, e.g. ppo_3d_evaluation_noise0.5_summary.json
    prefix = os.path.join(output_dir, "ppo_3d_evaluation" + (f"_noise{gps_noise:g}" if gps_noise > 0 else "")
                          + ("_random_starts" if random_starts else ""))

    # Save per-episode CSV
    csv_path = prefix + "_episodes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "seed", "start", "reward", "steps", "success", "collision", "truncated", "termination_reason",
            "ineffective_actions", "max_consecutive_ineffective", "oscillations", "max_steps_without_progress", "final_distance_to_goal"
        ])
        writer.writeheader()
        writer.writerows(per_episode)

    # Save summary JSON
    json_path = prefix + "_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate graph
    plot_path = prefix + "_metrics.png"
    _plot_evaluation_metrics(per_episode, summary, plot_path)

    print(f"\nEvaluation Summary ({n} episodes, gps noise std {gps_noise:g}, "
          f"{'random' if random_starts else 'fixed'} starts):")
    print(f"  Success rate:           {summary['success_rate'] * 100:.1f}% ({successes}/{n})")
    print(f"  Collision rate:         {summary['collision_rate'] * 100:.1f}% ({collisions}/{n})")
    print(f"  Truncation rate:        {summary['truncation_rate'] * 100:.1f}% ({truncations}/{n})")
    print(f"  Average episode length: {summary['average_episode_length']:.2f}")
    print(f"  Average episode reward: {summary['average_episode_reward']:.2f}")
    print(f"  Average final distance: {summary['average_final_distance_to_goal']:.2f}")
    print(f"  Average oscillations:   {summary['average_oscillations']:.2f}")
    print(f"  Average ineffective:    {summary['average_ineffective_actions']:.2f}")
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
    ax1.set_title(f"3D PPO Outcome Rates (gps noise {summary['gps_noise_std']:g})")
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
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--gps-noise", type=float, default=DEFAULT_GPS_NOISE,
                        help="gps noise std (grid cells): evaluation noise, and the max noise sampled per training episode")
    parser.add_argument("--random-start-prob", type=float, default=0.5,
                        help="fraction of training episodes that start from a random free cell")
    parser.add_argument("--random-starts", action="store_true",
                        help="evaluate from random free start cells instead of the fixed start")
    parser.add_argument("--start", type=int, nargs=3, default=[0, 0, 0],
                        help="start position as row col alt (default: 0 0 0)")
    parser.add_argument("--target", type=int, nargs=3, default=None,
                        help="target position as row col alt (default: 9 9 9)")
    parser.add_argument("--smoke-test", action="store_true", help="run short smoke test (1024 timesteps)")
    args = parser.parse_args()

    start_pos = tuple(args.start)
    target_pos = tuple(args.target) if args.target is not None else None

    if args.eval:
        evaluate(model_path=args.model_path, episodes=args.episodes, seed=args.seed, gps_noise=args.gps_noise,
                 random_starts=args.random_starts, start_pos=start_pos, target_pos=target_pos)
        return

    steps = 1024 if args.smoke_test else args.timesteps
    train(timesteps=steps, seed=args.seed, n_envs=4 if args.smoke_test else 8, verbose=0,
          model_path=args.model_path, random_start_prob=args.random_start_prob, max_gps_noise=args.gps_noise,
          start_pos=start_pos, target_pos=target_pos)
    verify_saved_model(args.model_path, gps_noise=args.gps_noise)


if __name__ == "__main__":
    main()
