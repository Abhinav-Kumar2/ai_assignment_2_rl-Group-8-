# 3d grid visualizer for dronenavenv3d using matplotlib
#
# works with any agent that follows the Agent3D interface in agents_3d.py,
# so a new agent can be plugged in without touching this file:
#   uv run python visualizer_3d.py                                # manual control only
#   uv run python visualizer_3d.py --agent ppo                    # trained 3d ppo policy
#   uv run python visualizer_3d.py --agent my_module:MyAgent      # your own agent
#   uv run python visualizer_3d.py --agent ppo --save results/ppo_3d_rollout.gif   # headless
#
# keys:
#   up arrow: move north (row - 1)
#   down arrow: move south (row + 1)
#   left arrow: move west (col - 1)
#   right arrow: move east (col + 1)
#   pageup: ascend (alt + 1)
#   pagedown: descend (alt - 1)
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


class Visualizer3D:
    """
    draws a DroneNavEnv3D-style env and steps it from keys or from an agent.

    env: anything with the DroneNavEnv3D api (reset/step, drone_pos, start_pos,
         target_pos, obstacles, height/width/depth, current_step, max_steps)
    agent: optional object with act(obs, env) -> int and reset(env), see agents_3d.py
    """

    def __init__(self, env=None, agent=None, interval_ms=250, seed=None):
        self.env = env if env is not None else DroneNavEnv3D()
        self.agent = agent
        self.interval_ms = interval_ms
        self.seed = seed
        self.playing = False
        self.episode = 0

        self.fig = plt.figure(figsize=(9, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=25, azim=-60)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self._on_timer)

        self.obs_coords = np.array(sorted(self.env.obstacles)) if self.env.obstacles else np.empty((0, 3))
        self.reset()

    @property
    def done(self):
        return self.terminated or self.truncated

    def reset(self):
        # first episode uses the given seed, later ones continue the env's rng
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

    def render(self):
        # ax.clear() would also reset the camera, so keep the user's view
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.clear()
        self.ax.view_init(elev=elev, azim=azim)

        # axes are x = col (east), y = row (south), z = alt; env positions are (row, col, alt)
        if len(self.obs_coords) > 0:
            self.ax.scatter(
                self.obs_coords[:, 1],
                self.obs_coords[:, 0],
                self.obs_coords[:, 2],
                c="#444444",
                marker="s",
                s=40,
                alpha=0.3,
                label="obstacles",
            )

        # start and target
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

        # path taken
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(
                traj[:, 1], traj[:, 0], traj[:, 2],
                color="tab:cyan", linestyle="-", marker=".", linewidth=2, alpha=0.9, label="path"
            )

        # current drone position
        drone = self.env.drone_pos
        drone_color = "red" if self.collided else "lime"
        self.ax.scatter(
            [drone[1]], [drone[0]], [drone[2]],
            c=drone_color, marker="o", s=130, edgecolor="black", label="drone", depthshade=False
        )

        # half-cell margin so markers on the border are not clipped
        self.ax.set_xlim(-0.5, self.env.width - 0.5)
        self.ax.set_ylim(self.env.height - 0.5, -0.5)
        self.ax.set_zlim(-0.5, self.env.depth - 0.5)
        self.ax.set_xlabel("col (x / east)")
        self.ax.set_ylabel("row (y / south)")
        self.ax.set_zlabel("alt (z / up)")

        controller = "manual"
        if self.agent is not None:
            controller = f"{getattr(self.agent, 'name', 'agent')} ({'playing' if self.playing else 'paused'})"
        self.ax.set_title(
            f"episode {self.episode} | step: {self.env.current_step}/{self.env.max_steps} | "
            f"status: {self.status_msg} | return: {self.total_reward:.1f}\n"
            f"pos: {drone} | target: {target} | last: {self.last_action_desc} (r={self.last_reward:.1f}) | "
            f"control: {controller}",
            fontsize=9,
        )
        self.ax.legend(loc="upper right", fontsize=8)
        self.fig.canvas.draw_idle()

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
    with writer.saving(vis.fig, path, dpi=80):
        for ep in range(episodes):
            if ep > 0:
                vis.reset()
            writer.grab_frame()
            while not vis.done:
                vis.agent_step()
                writer.grab_frame()
            for _ in range(hold_frames):  # pause on the final frame
                writer.grab_frame()
            returns.append(vis.total_reward)
    return returns


def main():
    parser = argparse.ArgumentParser(description="3d drone env visualizer with manual and agent control")
    parser.add_argument("--agent", type=str, default=None,
                        help="agent to play: a name from agents_3d.AGENT_REGISTRY (ppo, random) or module:Class")
    parser.add_argument("--model-path", type=str, default=None,
                        help="model file passed to the agent as model_path (e.g. for ppo)")
    parser.add_argument("--gps-noise", type=float, default=0.0, help="gps_noise_std for the environment")
    parser.add_argument("--seed", type=int, default=None, help="seed for the first env.reset()")
    parser.add_argument("--interval", type=int, default=250, help="ms between agent steps when playing")
    parser.add_argument("--autoplay", action="store_true", help="start playing the agent right away")
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
    vis = Visualizer3D(env=env, agent=agent, interval_ms=args.interval, seed=args.seed)

    if args.save:
        returns = save_rollouts(vis, args.save, episodes=args.episodes)
        for i, r in enumerate(returns, 1):
            print(f"episode {i}: return {r:.2f}")
        print(f"saved: {args.save}")
        return

    print("3d environment visualizer initialized")
    print("controls:")
    print("  up arrow: up (row - 1)")
    print("  down arrow: down (row + 1)")
    print("  left arrow: left (col - 1)")
    print("  right arrow: right (col + 1)")
    print("  pageup: ascend (alt + 1)")
    print("  pagedown: descend (alt - 1)")
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
