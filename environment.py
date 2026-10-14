import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneNavEnv(gym.Env):
    """
    2D Grid Environment for Autonomous Drone Navigation in Disaster-Affected Areas.

    Actions (Discrete(4)):
        0: Up    (row - 1, col)
        1: Down  (row + 1, col)
        2: Left  (row, col - 1)
        3: Right (row, col + 1)

    Observation (Box(8,), float32):
        [norm_drone_r, norm_drone_c,
         norm_rel_target_r, norm_rel_target_c,
         obs_up, obs_down, obs_left, obs_right]
    """
    metadata = {"render_modes": []}

    def __init__(self, grid_size=(10, 10), max_steps=100):
        super().__init__()
        self.height, self.width = grid_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self._action_to_delta = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }

        self.start_pos = (0, 0)
        self.target_pos = (self.height - 1, self.width - 1)
        self.obstacles = {
            (2, 2), (2, 3), (2, 4),
            (4, 6), (5, 6), (6, 6),
            (7, 2), (7, 3), (8, 3)
        }

        self.drone_pos = self.start_pos
        self.current_step = 0

    def _is_blocked(self, pos):
        r, c = pos
        out_of_bounds = not (0 <= r < self.height and 0 <= c < self.width)
        return out_of_bounds or ((r, c) in self.obstacles)

    def _get_obs(self):
        r, c = self.drone_pos
        tr, tc = self.target_pos

        norm_r = r / (self.height - 1)
        norm_c = c / (self.width - 1)
        rel_r = (tr - r) / (self.height - 1)
        rel_c = (tc - c) / (self.width - 1)

        obs_up = 1.0 if self._is_blocked((r - 1, c)) else 0.0
        obs_down = 1.0 if self._is_blocked((r + 1, c)) else 0.0
        obs_left = 1.0 if self._is_blocked((r, c - 1)) else 0.0
        obs_right = 1.0 if self._is_blocked((r, c + 1)) else 0.0

        return np.array(
            [norm_r, norm_c, rel_r, rel_c, obs_up, obs_down, obs_left, obs_right],
            dtype=np.float32,
        )

    def _dist_to_target(self, pos):
        return abs(pos[0] - self.target_pos[0]) + abs(pos[1] - self.target_pos[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = self.start_pos
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        dr, dc = self._action_to_delta[int(action)]
        new_pos = (self.drone_pos[0] + dr, self.drone_pos[1] + dc)

        # Collision with obstacle or map boundary
        if self._is_blocked(new_pos):
            reward = -10.0
            terminated = True
            truncated = False
            return self._get_obs(), reward, terminated, truncated, {"collision": True}

        # Step penalty and distance-progress reward
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
    env = DroneNavEnv()
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs), f"Invalid initial obs: {obs}"

    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"Obs outside space: {obs}"
        assert isinstance(reward, (float, int)), f"Reward not float/int: {type(reward)}"
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    print("Smoke check passed successfully.")
