import argparse
import collections
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add repo root to sys.path so environment_3d can be imported directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from environment_3d import DroneNavEnv3D

MODEL_PATH = os.path.join(BASE_DIR, "models", "dqn_drone_3d.pt")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "dqn_3d_evaluation_summary.json")


class QNetwork(nn.Module):
    """Standard MLP Q-network mapping 12-dim observation to 6 discrete action Q-values."""
    def __init__(self, obs_dim=12, action_dim=6, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            torch.as_tensor(np.array(obs), dtype=torch.float32),
            torch.as_tensor(action, dtype=torch.int64),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(np.array(next_obs), dtype=torch.float32),
            torch.as_tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def select_action(q_net, obs, epsilon, action_dim=6):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_values = q_net(obs_t)
        return int(q_values.argmax(dim=1).item())


def train(timesteps=100000, seed=0, lr=1e-3, gamma=0.99, batch_size=64,
          buffer_capacity=50000, target_update_freq=500,
          epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_steps=40000,
          model_path=MODEL_PATH):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = DroneNavEnv3D()
    q_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    obs, _ = env.reset(seed=seed)
    epsilon = epsilon_start
    episodes = 0
    ep_reward = 0.0
    ep_steps = 0

    print(f"Starting DQN training for {timesteps} timesteps...")
    pbar = tqdm(range(1, timesteps + 1), desc="DQN Training", unit="step")
    for t in pbar:
        action = select_action(q_net, obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        ep_steps += 1

        # Truncation vs. Termination distinction:
        # Gymnasium: 'terminated' indicates true terminal state (collision or goal reached) -> done = True.
        # 'truncated' indicates artificial time-limit cutoff -> state is non-terminal, so bootstrap next Q (done = False).
        done = terminated

        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs

        # Linear decay of epsilon
        if t <= epsilon_decay_steps:
            epsilon = epsilon_start - (epsilon_start - epsilon_min) * (t / epsilon_decay_steps)
        else:
            epsilon = epsilon_min

        # Learning update
        if len(replay_buffer) >= batch_size:
            b_obs, b_act, b_rew, b_next_obs, b_done = replay_buffer.sample(batch_size)
            q_values = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)

            # Standard DQN target: target = r + gamma * (1 - done) * max_a Q_target(next_obs, a)
            with torch.no_grad():
                next_q = target_net(b_next_obs).max(dim=1)[0]
                target = b_rew + gamma * (1.0 - b_done) * next_q

            loss = loss_fn(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

        # Update target network periodically
        if t % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Handle episode reset on termination or truncation
        if terminated or truncated:
            episodes += 1
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(q_net.state_dict(), model_path)
    print(f"Saved trained DQN model to {model_path}")
    return q_net


def evaluate(model_path=MODEL_PATH, episodes=10, seed=0, output_path=RESULTS_PATH, verbose=True):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path}. Please train a model first using:\n"
            f"  python experiments/3d/dqn_agent_3d.py --train --timesteps 100000"
        )

    q_net = QNetwork()
    q_net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    q_net.eval()

    env = DroneNavEnv3D()
    successes, collisions, timeouts = 0, 0, 0
    all_rewards, all_steps = [], []

    for ep in tqdm(range(episodes), desc="Evaluating", unit="ep"):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        steps = 0
        terminated, truncated = False, False
        collision = False

        while not (terminated or truncated):
            action = select_action(q_net, obs, epsilon=0.0)  # greedy action
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1
            if info.get("collision", False):
                collision = True

        success = env.drone_pos == env.target_pos
        timeout = truncated and not success and not collision

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

    print(f"\nDQN Evaluation Summary ({n} episodes):")
    print(f"  Success rate:        {summary['success_rate'] * 100:.1f}% ({successes}/{n})")
    print(f"  Collision rate:      {summary['collision_rate'] * 100:.1f}% ({collisions}/{n})")
    print(f"  Timeout rate:        {summary['timeout_rate'] * 100:.1f}% ({timeouts}/{n})")
    print(f"  Mean episode length: {summary['mean_episode_length']:.2f}")
    print(f"  Mean episode reward: {summary['mean_episode_reward']:.2f}")
    print(f"Saved evaluation summary to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Standard DQN Agent for DroneNavEnv3D")
    parser.add_argument("--train", action="store_true", help="train DQN agent")
    parser.add_argument("--eval", action="store_true", help="evaluate trained DQN agent")
    parser.add_argument("--timesteps", type=int, default=100000, help="training timesteps")
    parser.add_argument("--episodes", type=int, default=10, help="evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="checkpoint path")
    args = parser.parse_args()

    if args.eval:
        evaluate(model_path=args.model_path, episodes=args.episodes, seed=args.seed)
    elif args.train:
        train(timesteps=args.timesteps, seed=args.seed, model_path=args.model_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
