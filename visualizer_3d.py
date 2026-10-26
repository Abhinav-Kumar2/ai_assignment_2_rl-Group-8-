# 3d drone env visualizer for dronenavenv3d using matplotlib
#
# supports both realistic disaster-city rendering and basic/minimal rendering.
# toggle between modes dynamically with the 'm' key or via --mode [realistic|basic].
#
# works with any agent that follows the Agent3D interface in agents_3d.py,
# so a new agent can be plugged in without touching this file:
#   python visualizer_3d.py                                # manual control (realistic mode)
#   python visualizer_3d.py --mode basic                   # manual control (basic mode)
#   python visualizer_3d.py --agent ppo                    # trained 3d ppo policy
#   python visualizer_3d.py --agent my_module:MyAgent      # your own agent
#   python visualizer_3d.py --agent ppo --save results/ppo_3d_rollout.gif   # headless
#
# keys:
#   up arrow: move north (row - 1)
#   down arrow: move south (row + 1)
#   left arrow: move west (col - 1)
#   right arrow: move east (col + 1)
#   pageup: ascend (alt + 1)
#   pagedown: descend (alt - 1)
#   m: toggle visual mode (realistic disaster city <-> basic minimal)
#   space: play / pause the agent
#   n: let the agent take one step
#   backspace: reset
#   q / escape: quit
#
# manual keys and agent steps can be mixed: take the drone somewhere by hand,
# then press n or space to see what the agent does from there.

import argparse
import sys

import numpy as np
import matplotlib

# free every default shortcut (s for save, f for fullscreen, arrows for
# back/forward, p for pan, ...) so they cannot clash with the controls above
for key in list(matplotlib.rcParams):
    if key.startswith("keymap."):
        matplotlib.rcParams[key] = []

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from agents_3d import load_agent
from environment_3d import DroneNavEnv3D

action_keymap = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "pageup": 4,
    "pagedown": 5,
}

action_names = {
    0: "up (north)",
    1: "down (south)",
    2: "left (west)",
    3: "right (east)",
    4: "ascend",
    5: "descend",
}


def use_interactive_backend():
    """switch to the first interactive backend that actually loads; returns False if none does"""
    for backend in ["QtAgg", "TkAgg", "GTK4Agg", "GTK3Agg", "MacOSX"]:
        try:
            plt.switch_backend(backend)
            return True
        except Exception:
            pass
    return False


def build_disaster_city_mesh(obstacles, width, height, depth):
    """
    Constructs 3D geometry representing a disaster-affected urban environment
    matching the environment's obstacle cells.
    """
    polys, colors, edges, widths = [], [], [], []
    lines, line_colors, line_widths = [], [], []

    # Ground plane
    polys.append([
        [-0.5, -0.5, -0.5],
        [width - 0.5, -0.5, -0.5],
        [width - 0.5, height - 0.5, -0.5],
        [-0.5, height - 0.5, -0.5],
    ])
    colors.append("#201e1b")
    edges.append("#38322c")
    widths.append(1.0)

    # Subtle ground navigation voxel grid lines
    for c in range(width + 1):
        lines.append([[-0.5 + c, -0.5, -0.49], [-0.5 + c, height - 0.5, -0.49]])
        line_colors.append("#302b26")
        line_widths.append(0.8)
    for r in range(height + 1):
        lines.append([[-0.5, -0.5 + r, -0.49], [width - 0.5, -0.5 + r, -0.49]])
        line_colors.append("#302b26")
        line_widths.append(0.8)

    # Road / rubble crack lines on the ground
    if width >= 5 and height >= 5:
        lines.append([[2.5, -0.5, -0.48], [2.5, height - 0.5, -0.48]])
        line_colors.append("#423c34")
        line_widths.append(1.2)
        lines.append([[-0.5, 4.5, -0.48], [width - 0.5, 4.5, -0.48]])
        line_colors.append("#423c34")
        line_widths.append(1.2)

    if not obstacles:
        return polys, colors, edges, widths, lines, line_colors, line_widths

    by_rc = {}
    for (r, c, a) in obstacles:
        by_rc.setdefault((r, c), []).append(a)

    spans = []
    for (r, c), alts in by_rc.items():
        alts.sort()
        start_a = alts[0]
        prev_a = alts[0]
        for a in alts[1:]:
            if a == prev_a + 1:
                prev_a = a
            else:
                spans.append((r, c, start_a, prev_a))
                start_a = a
                prev_a = a
        spans.append((r, c, start_a, prev_a))

    b_colors = ["#5a5650", "#69635b", "#4e4a44", "#756d62", "#58534c", "#635e56"]
    roof_colors = ["#383430", "#423d38", "#322f2b"]
    rubble_colors = ["#847665", "#968875", "#736657", "#8c7d6b"]

    for (r, c, a0, a1) in spans:
        span_len = a1 - a0 + 1
        # Isolated single rubble/debris cell not on ground and not in primary walls
        if span_len == 1 and a0 > 0 and r not in (3, 6):
            seed = (r * 17 + c * 31 + a0 * 7) % 100
            dx = 0.36 + 0.04 * (seed % 3)
            dy = 0.36 + 0.04 * ((seed // 3) % 3)
            dz = 0.36
            x0, x1 = c - dx, c + dx
            y0, y1 = r - dy, r + dy
            z0, z1 = a0 - dz, a0 + dz
            r_col = rubble_colors[seed % len(rubble_colors)]

            top_f = [
                [x0 + 0.08, y0 + 0.08, z1],
                [x1, y0 + 0.1, z1 - 0.08],
                [x1 - 0.08, y1, z1 + 0.04],
                [x0, y1 - 0.08, z1 - 0.04],
            ]
            bot_f = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]
            for i in range(4):
                ni = (i + 1) % 4
                polys.append([bot_f[i], bot_f[ni], top_f[ni], top_f[i]])
                colors.append(r_col)
                edges.append("#24201c")
                widths.append(0.6)
            polys.append(top_f)
            colors.append("#a89b8a")
            edges.append("#24201c")
            widths.append(0.6)
            polys.append(bot_f)
            colors.append(r_col)
            edges.append("#24201c")
            widths.append(0.6)
        else:
            x0, x1 = c - 0.44, c + 0.44
            y0, y1 = r - 0.44, r + 0.44
            z0 = a0 - 0.5
            z1 = a1 + 0.44
            seed = (r * 13 + c * 29 + a0 * 11) % 100
            base_col = b_colors[seed % len(b_colors)]
            roof_col = roof_colors[seed % len(roof_colors)]

            is_damaged = (span_len >= 4 and seed % 2 == 0)
            if is_damaged:
                top_f = [
                    [x0, y0, z1],
                    [x1 - 0.12, y0, z1 - 0.08],
                    [x1, y1 - 0.15, z1 - 0.15],
                    [x0, y1, z1],
                ]
            else:
                top_f = [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]

            polys.append(top_f)
            colors.append(roof_col)
            edges.append("#1c1917")
            widths.append(0.7)

            if a0 > 0:
                polys.append([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])
                colors.append(base_col)
                edges.append("#282522")
                widths.append(0.5)

            # 4 building / wall facade sides
            polys.append([[x0, y0, z0], [x1, y0, z0], top_f[1], top_f[0]])
            colors.append(base_col)
            edges.append("#282522")
            widths.append(0.5)

            polys.append([top_f[3], top_f[2], [x1, y1, z0], [x0, y1, z0]])
            colors.append(base_col)
            edges.append("#282522")
            widths.append(0.5)

            polys.append([[x0, y0, z0], top_f[0], top_f[3], [x0, y1, z0]])
            colors.append(base_col)
            edges.append("#282522")
            widths.append(0.5)

            polys.append([top_f[1], [x1, y0, z0], [x1, y1, z0], top_f[2]])
            colors.append(base_col)
            edges.append("#282522")
            widths.append(0.5)

            if span_len >= 2:
                for floor_a in range(a0, a1 + 1):
                    fz = floor_a
                    lines.append([[x0, y0, fz - 0.4], [x1, y0, fz - 0.4]])
                    line_colors.append("#2b2723")
                    line_widths.append(0.6)
                    lines.append([[x0, y1, fz - 0.4], [x1, y1, fz - 0.4]])
                    line_colors.append("#2b2723")
                    line_widths.append(0.6)
                    if (floor_a + seed) % 2 == 0:
                        lines.append([[c - 0.22, y0 - 0.01, fz], [c + 0.22, y0 - 0.01, fz]])
                        line_colors.append("#111518")
                        line_widths.append(1.5)
                        lines.append([[c - 0.22, y1 + 0.01, fz], [c + 0.22, y1 + 0.01, fz]])
                        line_colors.append("#111518")
                        line_widths.append(1.5)

    return polys, colors, edges, widths, lines, line_colors, line_widths


def draw_3d_quadcopter(ax, drone_pos, collided=False, rotor_phase=0.0):
    """Draws a 3D quadcopter drone with rotating rotors and heading marker."""
    dr, dc, da = drone_pos
    cx, cy, cz = float(dc), float(dr), float(da)
    arm_len = 0.38
    body_hw = 0.14
    body_h = 0.05
    rotor_r = 0.13

    accent_col = "#ff1744" if collided else "#00e5ff"
    body_col = "#21070a" if collided else "#121415"
    nose_col = "#ff1744" if collided else "#ffea00"

    x0, x1 = cx - body_hw, cx + body_hw
    y0, y1 = cy - body_hw, cy + body_hw
    z0, z1 = cz - body_h, cz + body_h

    body_faces = [
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
        [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
    ]
    nose = [
        [cx - 0.07, cy - 0.04, z1 + 0.01],
        [cx + 0.07, cy - 0.04, z1 + 0.01],
        [cx, cy - 0.15, z1 + 0.01],
    ]
    body_faces.append(nose)

    facecolors = [body_col] * 6 + [nose_col]
    edgecolors = [accent_col] * 6 + ["#ffffff"]

    body_poly = Poly3DCollection(
        body_faces, facecolors=facecolors, edgecolors=edgecolors, linewidths=1.2, alpha=1.0
    )
    ax.add_collection3d(body_poly)

    arm_lines = [
        [[cx - arm_len, cy - arm_len, cz], [cx + arm_len, cy + arm_len, cz]],
        [[cx - arm_len, cy + arm_len, cz], [cx + arm_len, cy - arm_len, cz]],
    ]

    rotor_lines = []
    offsets = [
        (-arm_len, -arm_len),
        (arm_len, -arm_len),
        (arm_len, arm_len),
        (-arm_len, arm_len),
    ]
    for i, (dx, dy) in enumerate(offsets):
        rx = cx + dx
        ry = cy + dy
        rz = cz + body_h + 0.02
        arm_lines.append([[rx, ry, cz], [rx, ry, rz]])
        blade_ang = rotor_phase + i * (np.pi / 2)
        bx0 = rx + rotor_r * np.cos(blade_ang)
        by0 = ry + rotor_r * np.sin(blade_ang)
        bx1 = rx - rotor_r * np.cos(blade_ang)
        by1 = ry - rotor_r * np.sin(blade_ang)
        rotor_lines.append([[bx0, by0, rz], [bx1, by1, rz]])
        th = np.linspace(0, 2 * np.pi, 12)
        rotor_lines.append(
            np.column_stack([rx + rotor_r * np.cos(th), ry + rotor_r * np.sin(th), np.full_like(th, rz)])
        )

    ax.add_collection3d(Line3DCollection(arm_lines, colors="#222629", linewidths=3.0))
    ax.add_collection3d(Line3DCollection(rotor_lines, colors=accent_col, linewidths=1.5, alpha=1.0))

    ax.scatter(
        [cx], [cy], [cz],
        c=accent_col, marker="o", s=85, edgecolor="#ffffff", linewidth=1.2, depthshade=False, label="Drone"
    )
    ax.plot([cx, cx], [cy, cy], [-0.5, cz], color=accent_col, linestyle=":", linewidth=1.1, alpha=0.55)


class Visualizer3D:
    """
    Draws a DroneNavEnv3D-style env and steps it from keys or from an agent.

    mode: 'realistic' (disaster city, 3D buildings & quadcopter) or
          'basic' (minimal grid blocks & clean markers)
    """

    def __init__(self, env=None, agent=None, interval_ms=250, seed=None, mode="realistic"):
        self.env = env if env is not None else DroneNavEnv3D()
        self.agent = agent
        self.interval_ms = interval_ms
        self.seed = seed
        self.mode = mode.lower() if mode else "realistic"
        self.playing = False
        self.episode = 0
        self.rotor_phase = 0.0

        self.fig = plt.figure(figsize=(10, 7.5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=24, azim=-60)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self._on_timer)

        self.obs_coords = np.array(sorted(self.env.obstacles)) if self.env.obstacles else np.empty((0, 3))

        (
            self.city_polys,
            self.city_colors,
            self.city_edges,
            self.city_widths,
            self.city_lines,
            self.city_line_colors,
            self.city_line_widths,
        ) = build_disaster_city_mesh(
            self.env.obstacles, self.env.width, self.env.height, self.env.depth
        )

        self.reset()

    @property
    def done(self):
        return self.terminated or self.truncated

    def reset(self):
        seed = self.seed if self.episode == 0 else None
        self.obs, _ = self.env.reset(seed=seed)
        if self.agent is not None and hasattr(self.agent, "reset"):
            self.agent.reset(self.env)
        self.episode += 1
        self.trajectory = [self.env.drone_pos]
        self.terminated = False
        self.truncated = False
        self.collided = False
        self.last_action_desc = "reset"
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.status_msg = "running"
        self.render()

    def step(self, action, source="manual"):
        if self.done:
            return

        action = int(action)
        self.last_action_desc = f"{action_names.get(action, str(action))} [{source}]"
        self.obs, reward, self.terminated, self.truncated, info = self.env.step(action)
        self.last_reward = float(reward)
        self.total_reward += float(reward)
        self.trajectory.append(self.env.drone_pos)
        self.collided = bool(info.get("collision", False))

        if self.terminated:
            if self.collided:
                self.status_msg = "collision! episode terminated"
            elif self.env.drone_pos == self.env.target_pos:
                self.status_msg = "success! target reached"
            else:
                self.status_msg = "episode terminated"
        elif self.truncated:
            self.status_msg = "truncated! max steps reached"
        else:
            self.status_msg = "running"

        self.render()

    def agent_step(self):
        if self.agent is None or self.done:
            return
        self.step(self.agent.act(self.obs, self.env), source=getattr(self.agent, "name", "agent"))

    def run_episode(self, max_steps=None):
        """play the agent until the episode ends (no gui needed); returns the episode return"""
        limit = max_steps if max_steps is not None else self.env.max_steps
        for _ in range(limit):
            if self.done:
                break
            self.agent_step()
        return self.total_reward

    def gps_reading(self):
        """(row, col, alt) the agent observed: DroneNavEnv3D obs indices 0-2 are the normalized gps position"""
        try:
            dims = np.array([self.env.height, self.env.width, self.env.depth], dtype=float) - 1
            return np.asarray(self.obs[:3], dtype=float) * dims
        except (TypeError, IndexError, AttributeError):
            return None

    def render(self):
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.clear()
        self.ax.view_init(elev=elev, azim=azim)

        if self.mode == "basic":
            self._render_basic()
        else:
            self._render_realistic()

        self.fig.canvas.draw_idle()

    def _render_basic(self):
        """Minimal and basic visualization: clean grid, cube blocks, classic markers."""
        self.fig.patch.set_facecolor("#ffffff")
        self.ax.set_facecolor("#ffffff")
        self.ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        self.ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        self.ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        self.ax.grid(color="#cccccc", linestyle="-", linewidth=0.5, alpha=0.7)

        # Basic obstacle blocks
        if len(self.obs_coords) > 0:
            self.ax.scatter(
                self.obs_coords[:, 1],
                self.obs_coords[:, 0],
                self.obs_coords[:, 2],
                c="#555555",
                marker="s",
                s=45,
                alpha=0.35,
                label="obstacles",
            )

        start = self.env.start_pos
        target = self.env.target_pos
        self.ax.scatter(
            [start[1]], [start[0]], [start[2]],
            c="blue", marker="^", s=90, label="start", depthshade=False
        )
        self.ax.scatter(
            [target[1]], [target[0]], [target[2]],
            c="gold", marker="*", s=200, edgecolor="black", label="target", depthshade=False
        )

        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(
                traj[:, 1], traj[:, 0], traj[:, 2],
                color="tab:cyan", linestyle="-", marker=".", linewidth=2, alpha=0.9, label="path"
            )

        drone = self.env.drone_pos
        drone_color = "red" if self.collided else "lime"
        self.ax.scatter(
            [drone[1]], [drone[0]], [drone[2]],
            c=drone_color, marker="o", s=130, edgecolor="black", label="drone", depthshade=False
        )

        gps = self.gps_reading()
        gps_noisy = gps is not None and getattr(self.env, "gps_noise_std", 0.0) > 0
        if gps_noisy:
            self.ax.scatter(
                [gps[1]], [gps[0]], [gps[2]],
                facecolors="none", edgecolors="magenta", marker="o", s=160, linewidths=1.5,
                label="gps reading", depthshade=False
            )
            self.ax.plot(
                [drone[1], gps[1]], [drone[0], gps[0]], [drone[2], gps[2]],
                color="magenta", linestyle=":", linewidth=1
            )

        self.ax.set_xlim(-0.5, self.env.width - 0.5)
        self.ax.set_ylim(-0.5, self.env.height - 0.5)
        self.ax.set_zlim(-0.5, self.env.depth - 0.5)
        self.ax.set_xlabel("col (x / east)", color="#222222")
        self.ax.set_ylabel("row (y / south)", color="#222222")
        self.ax.set_zlabel("alt (z / up)", color="#222222")
        self.ax.tick_params(colors="#333333", labelsize=8)

        gps_text = f" (gps: {tuple(round(float(g), 1) for g in gps)})" if gps_noisy else ""
        controller = "manual"
        if self.agent is not None:
            controller = f"{getattr(self.agent, 'name', 'agent')} ({'playing' if self.playing else 'paused'})"

        self.ax.set_title(
            f"[Basic Mode] Ep {self.episode} | step: {self.env.current_step}/{self.env.max_steps} | "
            f"status: {self.status_msg} | return: {self.total_reward:.1f}\n"
            f"pos: {drone}{gps_text} | target: {target} | last: {self.last_action_desc} (r={self.last_reward:.1f}) | "
            f"ctrl: {controller} | [press 'm' for realistic]",
            fontsize=8.5,
            color="#111111",
        )
        self.ax.legend(loc="upper right", fontsize=8)
        self.fig.tight_layout()

    def _render_realistic(self):
        """Realistic disaster-affected urban environment with 3D buildings and quadcopter drone."""
        self.fig.patch.set_facecolor("#111214")
        self.ax.set_facecolor("#151618")
        self.ax.xaxis.set_pane_color((0.09, 0.09, 0.10, 1.0))
        self.ax.yaxis.set_pane_color((0.09, 0.09, 0.10, 1.0))
        self.ax.zaxis.set_pane_color((0.09, 0.09, 0.10, 1.0))
        self.ax.grid(color="#26282b", linestyle=":", linewidth=0.5, alpha=0.5)

        # 1. Render disaster city mesh
        if self.city_polys:
            poly_coll = Poly3DCollection(
                self.city_polys,
                facecolors=self.city_colors,
                edgecolors=self.city_edges,
                linewidths=self.city_widths,
                alpha=0.65,
            )
            self.ax.add_collection3d(poly_coll)
        if self.city_lines:
            line_coll = Line3DCollection(
                self.city_lines,
                colors=self.city_line_colors,
                linewidths=self.city_line_widths,
                alpha=0.65,
            )
            self.ax.add_collection3d(line_coll)

        # 2. Start base marker
        start = self.env.start_pos
        sx, sy, sz = float(start[1]), float(start[0]), float(start[2])
        self.ax.scatter(
            [sx], [sy], [sz],
            c="#00e676", marker="^", s=170, edgecolor="#ffffff", linewidth=1.2,
            depthshade=False, label="Start Base"
        )

        # 3. Target emergency beacon with vertical beacon beam
        target = self.env.target_pos
        tx, ty, tz = float(target[1]), float(target[0]), float(target[2])
        self.ax.plot(
            [tx, tx], [ty, ty], [-0.5, tz],
            color="#ff9100", linestyle="--", linewidth=1.8, alpha=0.85
        )
        self.ax.scatter(
            [tx], [ty], [tz],
            c="#ff3d00", marker="*", s=280, edgecolor="#ffffff", linewidth=1.4,
            depthshade=False, label="Target Beacon"
        )

        # 4. Flight trajectory line
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(
                traj[:, 1], traj[:, 0], traj[:, 2],
                color="#00e5ff", linestyle="-", linewidth=2.4, marker="o", markersize=3.5,
                alpha=0.95, label="Flight Path"
            )

        # 5. Recognizable 3D quadcopter drone
        drone = self.env.drone_pos
        self.rotor_phase = (self.rotor_phase + np.pi / 4) % (2 * np.pi)
        draw_3d_quadcopter(
            self.ax, drone, collided=self.collided, rotor_phase=self.rotor_phase
        )

        # 6. Direction to target line
        cx, cy, cz = float(drone[1]), float(drone[0]), float(drone[2])
        self.ax.plot(
            [cx, tx], [cy, ty], [cz, tz],
            color="#ffab40", linestyle=":", linewidth=1.0, alpha=0.45, label="Target Vector"
        )

        # 7. GPS reading visualization (if noisy)
        gps = self.gps_reading()
        gps_noisy = gps is not None and getattr(self.env, "gps_noise_std", 0.0) > 0
        if gps_noisy:
            self.ax.scatter(
                [gps[1]], [gps[0]], [gps[2]],
                facecolors="none", edgecolors="#e040fb", marker="o", s=170, linewidths=1.6,
                label="GPS Estimate", depthshade=False
            )
            self.ax.plot(
                [cx, gps[1]], [cy, gps[0]], [cz, gps[2]],
                color="#e040fb", linestyle=":", linewidth=1.3, alpha=0.85
            )

        self.ax.set_xlim(-0.5, self.env.width - 0.5)
        self.ax.set_ylim(-0.5, self.env.height - 0.5)
        self.ax.set_zlim(-0.5, self.env.depth - 0.5)
        self.ax.set_xlabel("East (col / x)", color="#cfd8dc", labelpad=8)
        self.ax.set_ylabel("South (row / y)", color="#cfd8dc", labelpad=8)
        self.ax.set_zlabel("Altitude (alt / z)", color="#cfd8dc", labelpad=8)
        self.ax.tick_params(colors="#90a4ae", labelsize=8)

        gps_text = f" | GPS: {tuple(round(float(g), 1) for g in gps)}" if gps_noisy else ""
        controller = "manual"
        if self.agent is not None:
            controller = f"{getattr(self.agent, 'name', 'agent')} ({'playing' if self.playing else 'paused'})"

        status_color = "#ff5252" if self.collided else ("#69f0ae" if "success" in self.status_msg else "#e0e0e0")
        self.ax.set_title(
            f"Autonomous 3D Drone Navigation - Disaster Response Simulation [Realistic]\n"
            f"Ep {self.episode} | Step: {self.env.current_step}/{self.env.max_steps} | "
            f"Status: {self.status_msg} | Return: {self.total_reward:.1f}\n"
            f"Pos: {drone}{gps_text} | Target: {target} | Last: {self.last_action_desc} (r={self.last_reward:.1f}) | "
            f"Ctrl: {controller} | [press 'm' for basic]",
            fontsize=8.5,
            color=status_color,
            pad=10,
        )
        self.ax.legend(
            loc="upper right",
            facecolor="#1e2022",
            edgecolor="#37474f",
            labelcolor="#eceff1",
            fontsize=7.5,
        )
        self.fig.tight_layout()

    def set_playing(self, playing):
        self.playing = playing and self.agent is not None
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.render()

    def _on_timer(self):
        if not self.playing:
            return
        if self.done:
            self.set_playing(False)
            return
        self.agent_step()

    def on_key(self, event):
        if event.key in ("q", "escape"):
            self.timer.stop()
            plt.close(self.fig)
            return
        if event.key == "m":
            self.mode = "basic" if self.mode == "realistic" else "realistic"
            self.render()
            return
        if event.key == "backspace":
            self.reset()
            return
        if event.key == " ":
            if self.done:
                self.reset()
            self.set_playing(not self.playing)
            return
        if event.key == "n":
            self.agent_step()
            return
        if event.key in action_keymap:
            self.step(action_keymap[event.key])


def save_rollouts(vis, path, episodes=1, fps=6, hold_frames=6):
    """headless: play the agent for some episodes and save every step as a gif (or a png of the final frame)"""
    from matplotlib.animation import PillowWriter

    if path.lower().endswith(".png"):
        vis.run_episode()
        vis.fig.savefig(path, dpi=110)
        return [vis.total_reward]

    returns = []
    writer = PillowWriter(fps=fps)
    with writer.saving(vis.fig, path, dpi=85):
        for ep in range(episodes):
            if ep > 0:
                vis.reset()
            writer.grab_frame()
            while not vis.done:
                vis.agent_step()
                writer.grab_frame()
            for _ in range(hold_frames):
                writer.grab_frame()
            returns.append(vis.total_reward)
    return returns


def main():
    parser = argparse.ArgumentParser(description="3d drone env visualizer with manual and agent control")
    parser.add_argument("--agent", type=str, default=None,
                        help="agent to play: a name from agents_3d.AGENT_REGISTRY (ppo, random) or module:Class")
    parser.add_argument("--model-path", type=str, default=None,
                        help="model file passed to the agent as model_path (e.g. for ppo)")
    parser.add_argument("--gps-noise", type=float, default=0.0,
                        help="gps_noise_std for the environment (grid cells); the noisy reading is drawn in magenta")
    parser.add_argument("--seed", type=int, default=None, help="seed for the first env.reset()")
    parser.add_argument("--interval", type=int, default=250, help="ms between agent steps when playing")
    parser.add_argument("--autoplay", action="store_true", help="start playing the agent right away")
    parser.add_argument("--mode", type=str, choices=["realistic", "basic"], default="realistic",
                        help="visual style: 'realistic' (disaster urban scene) or 'basic' (minimal grid blocks); toggle with 'm'")
    parser.add_argument("--save", type=str, default=None,
                        help="headless: write the agent's rollout(s) to a .gif (or final frame to a .png) and exit")
    parser.add_argument("--episodes", type=int, default=1, help="episodes to record with --save")
    args = parser.parse_args()

    agent = None
    if args.agent:
        kwargs = {"model_path": args.model_path} if args.model_path else {}
        agent = load_agent(args.agent, **kwargs)
    elif args.save:
        parser.error("--save needs --agent")

    if args.save:
        plt.switch_backend("Agg")
    elif not use_interactive_backend():
        print("no interactive matplotlib backend found (install PyQt or tkinter); "
              "use --save to record rollouts instead", file=sys.stderr)
        sys.exit(1)

    env = DroneNavEnv3D(gps_noise_std=args.gps_noise)
    vis = Visualizer3D(env=env, agent=agent, interval_ms=args.interval, seed=args.seed, mode=args.mode)

    if args.save:
        returns = save_rollouts(vis, args.save, episodes=args.episodes)
        for i, r in enumerate(returns, 1):
            print(f"episode {i}: return {r:.2f}")
        print(f"saved: {args.save}")
        return

    print("3d environment visualizer initialized")
    print(f"current mode: {vis.mode} (press 'm' anytime to toggle between realistic and basic)")
    print("controls:")
    print("  up arrow: up (row - 1 / north)")
    print("  down arrow: down (row + 1 / south)")
    print("  left arrow: left (col - 1 / west)")
    print("  right arrow: right (col + 1 / east)")
    print("  pageup: ascend (alt + 1)")
    print("  pagedown: descend (alt - 1)")
    print("  m: toggle visual mode (realistic <-> basic)")
    if agent is not None:
        print(f"  space: play / pause agent ({getattr(agent, 'name', 'agent')})")
        print("  n: agent takes one step")
    print("  backspace: reset")
    print("  q / esc: quit")
    if args.autoplay:
        vis.set_playing(True)
    plt.show()


if __name__ == "__main__":
    main()
