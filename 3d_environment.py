# 3d grid environment for autonomous drone navigation in disaster areas
#
# discrete actions:
#   0: up      (row - 1, col, alt)   north
#   1: down    (row + 1, col, alt)   south
#   2: left    (row, col - 1, alt)   west
#   3: right   (row, col + 1, alt)   east
#   4: ascend  (row, col, alt + 1)
#   5: descend (row, col, alt - 1)
#
# observations:
#   normalized drone pos (r, c, a), normalized rel target (r, c, a),
#   and 6 binary blocked flags in action order

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneNavEnv3D(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_size=(10, 10, 10), max_steps=200):
        super().__init__()
        self.height, self.width, self.depth = grid_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self._action_to_delta = {
            0: (-1, 0, 0),  # up
            1: (1, 0, 0),   # down
            2: (0, -1, 0),  # left
            3: (0, 1, 0),   # right
            4: (0, 0, 1),   # ascend
            5: (0, 0, -1),  # descend
        }

        self.start_pos = (0, 0, 0)
        self.target_pos = (self.height - 1, self.width - 1, self.depth - 1)
        self.obstacles = self._build_obstacles()

        self.drone_pos = self.start_pos
        self.current_step = 0

    def _build_obstacles(self):
        # map has two wall barriers with gaps forcing a detour:
        # wall a (row 3): gap on east side near ground (cols 7-8, alt 0-1)
        # wall b (row 6): gap on west side high up (cols 1-2, alt 7-8)
        obstacles = set()

        def wall(row, gap_cols, gap_alts):
            for c in range(self.width):
                for a in range(self.depth):
                    if not (c in gap_cols and a in gap_alts):
                        obstacles.add((row, c, a))

        wall(3, gap_cols=(7, 8), gap_alts=(0, 1))
        wall(6, gap_cols=(1, 2), gap_alts=(7, 8))

        # buildings on the ground
        buildings = {
            (1, 2): 5, (2, 5): 7, (0, 7): 3,   # zone 1
            (4, 3): 6, (5, 6): 9, (4, 8): 4,   # zone 2
            (7, 5): 8, (8, 2): 4, (8, 7): 9,   # zone 3
        }
        for (r, c), h in buildings.items():
            for a in range(min(h, self.depth)):
                obstacles.add((r, c, a))

        # floating debris and smoke cells
        obstacles |= {
            (1, 5, 1), (2, 7, 0), (5, 2, 6), (4, 5, 3),
            (7, 2, 8), (8, 5, 9), (9, 8, 8), (8, 9, 8),
        }

        obstacles.discard(self.start_pos)
        obstacles.discard(self.target_pos)
        return obstacles

    def _is_blocked(self, pos):
        r, c, a = pos
        out_of_bounds = not (
            0 <= r < self.height and 0 <= c < self.width and 0 <= a < self.depth
        )
        return out_of_bounds or ((r, c, a) in self.obstacles)

    def _get_obs(self):
        r, c, a = self.drone_pos
        tr, tc, ta = self.target_pos

        norm_r = r / (self.height - 1)
        norm_c = c / (self.width - 1)
        norm_a = a / (self.depth - 1)
        rel_r = (tr - r) / (self.height - 1)
        rel_c = (tc - c) / (self.width - 1)
        rel_a = (ta - a) / (self.depth - 1)

        blocked = [
            1.0 if self._is_blocked((r + dr, c + dc, a + da)) else 0.0
            for dr, dc, da in self._action_to_delta.values()
        ]

        return np.array(
            [norm_r, norm_c, norm_a, rel_r, rel_c, rel_a, *blocked],
            dtype=np.float32,
        )

    def _dist_to_target(self, pos):
        return sum(abs(p - t) for p, t in zip(pos, self.target_pos))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = self.start_pos
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        dr, dc, da = self._action_to_delta[int(action)]
        new_pos = (self.drone_pos[0] + dr, self.drone_pos[1] + dc, self.drone_pos[2] + da)

        # collision with obstacle or boundary
        if self._is_blocked(new_pos):
            reward = -10.0
            terminated = True
            truncated = False
            return self._get_obs(), reward, terminated, truncated, {"collision": True}

        # step penalty and distance progress
        prev_dist = self._dist_to_target(self.drone_pos)
        curr_dist = self._dist_to_target(new_pos)
        self.drone_pos = new_pos

        reward = -0.1 + (prev_dist - curr_dist) * 1.0

        if self.drone_pos == self.target_pos:
            reward += 50.0
            terminated = True
            truncated = False
        else:
            terminated = False
            truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}


if __name__ == "__main__":
    from collections import deque

    env = DroneNavEnv3D()

    # bfs check from start over free cells
    dist = {env.start_pos: 0}
    queue = deque([env.start_pos])
    while queue:
        pos = queue.popleft()
        for dr, dc, da in env._action_to_delta.values():
            nxt = (pos[0] + dr, pos[1] + dc, pos[2] + da)
            if nxt not in dist and not env._is_blocked(nxt):
                dist[nxt] = dist[pos] + 1
                queue.append(nxt)

    assert env.target_pos in dist, "target unreachable"
    n_free = env.height * env.width * env.depth - len(env.obstacles)
    assert len(dist) == n_free, f"{n_free - len(dist)} free cells unreachable"
    shortest = dist[env.target_pos]
    manhattan = env._dist_to_target(env.start_pos)
    assert shortest > manhattan, "expected forced detour"

    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs), f"invalid obs: {obs}"

    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, (float, int))
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    print(f"{len(env.obstacles)} obstacle cells, shortest path {shortest} steps "
          f"(straight-line Manhattan distance {manhattan}).")
    print("smoke check passed successfully")
