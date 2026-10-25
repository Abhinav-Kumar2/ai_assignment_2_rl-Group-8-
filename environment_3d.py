import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneNavEnv3D(gym.Env):
    """
    3D Grid Environment for Autonomous Drone Navigation in Disaster-Affected Areas.

    Same task as DroneNavEnv (environment.py), extended with altitude:
    fly from the start cell to the target cell without hitting obstacles
    (collapsed buildings, debris) or leaving the map.

    Positions are (row, col, alt). Actions 0-3 match the 2D environment.

    Actions (Discrete(6)):
        0: Up      (row - 1, col, alt)   north
        1: Down    (row + 1, col, alt)   south
        2: Left    (row, col - 1, alt)   west
        3: Right   (row, col + 1, alt)   east
        4: Ascend  (row, col, alt + 1)
        5: Descend (row, col, alt - 1)

    Observation (Box(12,), float32):
        [norm_drone_r, norm_drone_c, norm_drone_a,
         norm_rel_target_r, norm_rel_target_c, norm_rel_target_a,
         obs_up, obs_down, obs_left, obs_right, obs_ascend, obs_descend]

    GPS-derived indices: 0-5. Indices 0-2 are the drone's position and 3-5
    are the offset to the target computed from that position, so both come
    from the GPS reading. Indices 6-11 are local obstacle sensors and always
    use the true position.

    gps_noise_std (grid cells, default 0.0): standard deviation of Gaussian
    noise added to the observed (row, col, alt) before indices 0-5 are
    computed. The noisy position is clipped to the grid, so observations stay
    inside the observation space. The noise is drawn from self.np_random
    (seeded by reset(seed=...)) and never changes the true drone_pos used for
    physics, collisions, rewards and termination. With 0.0 no noise is drawn
    and observations are exactly as without GPS noise.
    """
    metadata = {"render_modes": []}

    def __init__(self, grid_size=(10, 10, 10), max_steps=200, gps_noise_std=0.0,
                 start_pos=(0, 0, 0), target_pos=None):
        super().__init__()
        if gps_noise_std < 0:
            raise ValueError(f"gps_noise_std must be >= 0, got {gps_noise_std}")
        self.height, self.width, self.depth = grid_size
        self.max_steps = max_steps
        self.gps_noise_std = float(gps_noise_std)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self._action_to_delta = {
            0: (-1, 0, 0),  # Up
            1: (1, 0, 0),   # Down
            2: (0, -1, 0),  # Left
            3: (0, 1, 0),   # Right
            4: (0, 0, 1),   # Ascend
            5: (0, 0, -1),  # Descend
        }

        self.start_pos = tuple(start_pos)
        self.target_pos = tuple(target_pos) if target_pos is not None else (self.height - 1, self.width - 1, self.depth - 1)
        self.obstacles = self._build_obstacles()

        self.drone_pos = self.start_pos
        self.current_step = 0

    def _build_obstacles(self):
        """
        The map is cut into three zones by two collapsed-building walls that
        span the full width and height of the grid, each with one 2x2 gap:

          wall A (row 3): gap at the east side, near the ground (cols 7-8, alt 0-1)
          wall B (row 6): gap at the west side, high up       (cols 1-2, alt 7-8)

        Getting from gap A to gap B means flying west, away from the target,
        so no path that only reduces the distance to the target exists and a
        greedy "move toward the target" policy gets stuck against wall B.
        """
        obstacles = set()

        def wall(row, gap_cols, gap_alts):
            for c in range(self.width):
                for a in range(self.depth):
                    if not (c in gap_cols and a in gap_alts):
                        obstacles.add((row, c, a))

        wall(3, gap_cols=(7, 8), gap_alts=(0, 1))
        wall(6, gap_cols=(1, 2), gap_alts=(7, 8))

        # Damaged buildings standing on the ground, each to its own height.
        buildings = {
            (1, 2): 5, (2, 5): 7, (0, 7): 3,   # zone 1
            (4, 3): 6, (5, 6): 9, (4, 8): 4,   # zone 2
            (7, 5): 8, (8, 2): 4, (8, 7): 9,   # zone 3
        }
        for (r, c), h in buildings.items():
            for a in range(min(h, self.depth)):
                obstacles.add((r, c, a))

        # Floating debris and smoke-filled cells.
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

    def _gps_position(self):
        """Position as reported by GPS: the true position plus Gaussian noise, clipped to the grid."""
        if self.gps_noise_std == 0.0:
            return self.drone_pos
        noisy = np.asarray(self.drone_pos, dtype=np.float64) + self.np_random.normal(
            0.0, self.gps_noise_std, size=3
        )
        upper = np.array([self.height - 1, self.width - 1, self.depth - 1], dtype=np.float64)
        return tuple(np.clip(noisy, 0.0, upper))

    def _get_obs(self):
        r, c, a = self.drone_pos
        tr, tc, ta = self.target_pos
        gr, gc, ga = self._gps_position()

        # GPS-derived: indices 0-5
        norm_r = gr / (self.height - 1)
        norm_c = gc / (self.width - 1)
        norm_a = ga / (self.depth - 1)
        rel_r = (tr - gr) / (self.height - 1)
        rel_c = (tc - gc) / (self.width - 1)
        rel_a = (ta - ga) / (self.depth - 1)

        # Local obstacle sensors (true position): indices 6-11
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
        if options and "start_pos" in options:
            self.start_pos = tuple(options["start_pos"])
        if options and "target_pos" in options:
            self.target_pos = tuple(options["target_pos"])
            self.obstacles.discard(self.target_pos)
        self.drone_pos = self.start_pos
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        dr, dc, da = self._action_to_delta[int(action)]
        new_pos = (self.drone_pos[0] + dr, self.drone_pos[1] + dc, self.drone_pos[2] + da)

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
    from collections import deque

    env = DroneNavEnv3D()

    # BFS from the start over free cells.
    dist = {env.start_pos: 0}
    queue = deque([env.start_pos])
    while queue:
        pos = queue.popleft()
        for dr, dc, da in env._action_to_delta.values():
            nxt = (pos[0] + dr, pos[1] + dc, pos[2] + da)
            if nxt not in dist and not env._is_blocked(nxt):
                dist[nxt] = dist[pos] + 1
                queue.append(nxt)

    # The target must be reachable, and there must be no sealed-off pockets
    # (agents may be started from any free cell during training).
    assert env.target_pos in dist, "Target is unreachable from start"
    n_free = env.height * env.width * env.depth - len(env.obstacles)
    assert len(dist) == n_free, f"{n_free - len(dist)} free cells are unreachable"
    shortest = dist[env.target_pos]
    manhattan = env._dist_to_target(env.start_pos)
    # The obstacles must force a detour away from the target.
    assert shortest > manhattan, "A distance-reducing path to the target exists"

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

    print(f"{len(env.obstacles)} obstacle cells, shortest path {shortest} steps "
          f"(straight-line Manhattan distance {manhattan}).")
    print("Smoke check passed successfully.")
