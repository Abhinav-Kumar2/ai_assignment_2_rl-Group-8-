# 3d grid visualizer for dronenavenv3d using matplotlib
#
# keys:
#   up arrow: move north (row - 1)
#   down arrow: move south (row + 1)
#   left arrow: move west (col - 1)
#   right arrow: move east (col + 1)
#   pageup: ascend (alt + 1)
#   pagedown: descend (alt - 1)
#   backspace: reset
#   q / escape: quit

import importlib
import numpy as np
import matplotlib

# disable conflicting default shortcuts (like s for save and f for fullscreen)
for key in ["keymap.save", "keymap.fullscreen", "keymap.quit", "keymap.back", "keymap.forward"]:
    if key in matplotlib.rcParams:
        matplotlib.rcParams[key] = []

# try interactive backends before importing pyplot
for backend in ["QtAgg", "TkAgg", "GTK3Agg"]:
    try:
        matplotlib.use(backend)
        break
    except Exception:
        pass

import matplotlib.pyplot as plt

env_mod = importlib.import_module("3d_environment")
DroneNavEnv3D = env_mod.DroneNavEnv3D

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


class Visualizer3D:
    def __init__(self, env=None):
        self.env = env if env is not None else DroneNavEnv3D()
        self.obs, _ = self.env.reset()
        self.trajectory = [self.env.drone_pos]
        self.terminated = False
        self.truncated = False
        self.last_action_desc = "none"
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.status_msg = "running"

        self.fig = plt.figure(figsize=(9, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.obs_coords = np.array(list(self.env.obstacles))
        self.render()

    def step(self, action):
        if self.terminated or self.truncated:
            return

        self.last_action_desc = action_names.get(action, str(action))
        self.obs, reward, self.terminated, self.truncated, info = self.env.step(action)
        self.last_reward = reward
        self.total_reward += reward
        self.trajectory.append(self.env.drone_pos)

        if self.terminated:
            if info.get("collision", False):
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

    def reset(self):
        self.obs, _ = self.env.reset()
        self.trajectory = [self.env.drone_pos]
        self.terminated = False
        self.truncated = False
        self.last_action_desc = "reset"
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.status_msg = "running"
        self.render()

    def render(self):
        self.ax.clear()

        # obstacles
        if len(self.obs_coords) > 0:
            self.ax.scatter(
                self.obs_coords[:, 1],
                self.obs_coords[:, 0],
                self.obs_coords[:, 2],
                c="dimgray",
                marker="s",
                s=35,
                alpha=0.25,
                label="obstacles",
            )

        # start and target
        start = self.env.start_pos
        target = self.env.target_pos
        self.ax.scatter(
            [start[1]], [start[0]], [start[2]],
            c="blue", marker="^", s=90, label="start"
        )
        self.ax.scatter(
            [target[1]], [target[0]], [target[2]],
            c="gold", marker="*", s=160, edgecolor="black", label="target"
        )

        # path taken
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(
                traj[:, 1], traj[:, 0], traj[:, 2],
                color="cyan", linestyle="--", linewidth=1.5, alpha=0.8
            )

        # current drone position
        drone = self.env.drone_pos
        drone_color = "red" if (self.terminated and self.status_msg.startswith("collision")) else "lime"
        self.ax.scatter(
            [drone[1]], [drone[0]], [drone[2]],
            c=drone_color, marker="o", s=130, edgecolor="black", label="drone"
        )

        self.ax.set_xlim(0, self.env.width - 1)
        self.ax.set_ylim(self.env.height - 1, 0)
        self.ax.set_zlim(0, self.env.depth - 1)
        self.ax.set_xlabel("col (x / east)")
        self.ax.set_ylabel("row (y / south)")
        self.ax.set_zlabel("alt (z / up)")
        self.ax.set_title(
            f"step: {self.env.current_step}/{self.env.max_steps} | status: {self.status_msg} | return: {self.total_reward:.1f}\n"
            f"pos: {drone} | target: {target} | last: {self.last_action_desc} (r={self.last_reward:.1f})",
            fontsize=9,
        )
        self.ax.legend(loc="upper right", fontsize=8)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
            return
        if event.key == "backspace":
            self.reset()
            return
        if event.key in action_keymap:
            self.step(action_keymap[event.key])


def main():
    vis = Visualizer3D()
    print("3d environment visualizer initialized")
    print("controls:")
    print("  up arrow: up (row - 1)")
    print("  down arrow: down (row + 1)")
    print("  left arrow: left (col - 1)")
    print("  right arrow: right (col + 1)")
    print("  pageup: ascend (alt + 1)")
    print("  pagedown: descend (alt - 1)")
    print("  backspace: reset")
    print("  q / esc: quit")
    plt.show()


if __name__ == "__main__":
    main()
