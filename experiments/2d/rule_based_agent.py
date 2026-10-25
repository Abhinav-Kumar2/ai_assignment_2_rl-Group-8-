"""
Rule-based (non-learning) agent for DroneNavEnv.

The agent only uses the 8-dim observation the environment returns, plus a
short-term memory of the cells it has already visited:

    1. Never pick an action whose neighbour is flagged as blocked
       (obstacle or map edge), so the drone never collides.
    2. Among the safe actions, prefer the ones that reduce the Manhattan
       distance to the target.
    3. Break ties (and escape dead ends) by preferring the least-visited
       neighbour, so the drone does not oscillate between two cells.
    4. Among still-tied actions, prefer the axis with the larger remaining
       distance, which keeps the path close to a straight diagonal.

Usage:
    uv run python rule_based_agent.py --episodes 10
"""
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from environment import DroneNavEnv

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {UP: "Up", DOWN: "Down", LEFT: "Left", RIGHT: "Right"}
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class RuleBasedAgent:
    def __init__(self, grid_size=(10, 10)):
        self.height, self.width = grid_size
        self.visits = defaultdict(int)

    def reset(self):
        self.visits.clear()

    def _decode(self, obs):
        """Recover integer position, target offset and blocked flags from obs."""
        r = int(round(obs[0] * (self.height - 1)))
        c = int(round(obs[1] * (self.width - 1)))
        dr = int(round(obs[2] * (self.height - 1)))
        dc = int(round(obs[3] * (self.width - 1)))
        blocked = {UP: obs[4] > 0.5, DOWN: obs[5] > 0.5,
                   LEFT: obs[6] > 0.5, RIGHT: obs[7] > 0.5}
        return (r, c), (dr, dc), blocked

    def act(self, obs):
        (r, c), (dr, dc), blocked = self._decode(obs)
        self.visits[(r, c)] += 1

        safe = [a for a in ACTION_DELTAS if not blocked[a]]
        if not safe:
            # Fully boxed in; every move collides, so any choice is equivalent.
            return UP

        def score(a):
            ar, ac = ACTION_DELTAS[a]
            nxt = (r + ar, c + ac)
            progress = ar * np.sign(dr) + ac * np.sign(dc)  # +1 closer, -1 farther, 0 sideways
            axis_gap = abs(dr) if ar else abs(dc)
            # Lower tuple wins: fewest visits first, then most progress, then larger gap.
            return (self.visits[nxt], -progress, -axis_gap)

        return min(safe, key=score)


def run(episodes=10, verbose=True):
    env = DroneNavEnv()
    agent = RuleBasedAgent(grid_size=(env.height, env.width))
    results = []

    for ep in tqdm(range(episodes), desc="Running episodes", unit="ep"):
        obs, _ = env.reset(seed=ep)
        agent.reset()
        total, steps, path = 0.0, 0, [env.drone_pos]
        terminated = truncated = False
        info = {}

        while not (terminated or truncated):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            path.append(env.drone_pos)

        success = env.drone_pos == env.target_pos
        results.append((success, steps, total, info.get("collision", False)))
        if verbose:
            print(f"Episode {ep + 1}: success={success} steps={steps} "
                  f"reward={total:.1f} collision={info.get('collision', False)}")

    if verbose:
        print("\nPath taken in the last episode:")
        print(render(env, path))
        n = len(results)
        print(f"\nSuccess rate: {sum(r[0] for r in results)}/{n}, "
              f"mean steps: {np.mean([r[1] for r in results]):.1f}, "
              f"mean reward: {np.mean([r[2] for r in results]):.1f}, "
              f"collisions: {sum(r[3] for r in results)}")
    return results


def render(env, path):
    visited = set(path)
    rows = []
    for r in range(env.height):
        row = []
        for c in range(env.width):
            p = (r, c)
            if p == env.start_pos:
                row.append("S")
            elif p == env.target_pos:
                row.append("T")
            elif p in env.obstacles:
                row.append("#")
            elif p in visited:
                row.append("*")
            else:
                row.append(".")
        rows.append(" ".join(row))
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    run(args.episodes)
