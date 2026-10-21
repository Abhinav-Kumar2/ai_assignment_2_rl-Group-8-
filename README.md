# NavRL: Autonomous Drone Navigation in Disaster-Affected Environments

Autonomous drone navigation in complex, 3D disaster-affected environments using reinforcement learning.

The core pipeline is focused on a 3D PPO agent navigating a 10x10x10 voxel disaster environment featuring collapsed building walls, altitude obstacles, debris, and forced detours.

---

## 3D Environment Setup & Verification

### Dependencies Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Smoke Checks
Verify the 3D environment graph connectivity, reachability, and Gymnasium API protocol:
```bash
python 3d_environment.py
```

---

## 3D Environment Visualization

`visualize_3d.py` provides an interactive 3D Matplotlib visualization for `DroneNavEnv3D`.

### Launching the Visualizer
```bash
python visualize_3d.py
```

### Interactive Keyboard Controls
When the Matplotlib window is focused, use the following keys to manually step the drone:
- **`Up Arrow`**: Move North / Up (`row - 1`)
- **`Down Arrow`**: Move South / Down (`row + 1`)
- **`Left Arrow`**: Move West / Left (`col - 1`)
- **`Right Arrow`**: Move East / Right (`col + 1`)
- **`PageUp`**: Ascend (`alt + 1`)
- **`PageDown`**: Descend (`alt - 1`)
- **`Backspace`**: Reset environment to initial state `(0, 0, 0)`
- **`q`** or **`Escape`**: Close visualization cleanly

The display tracks:
- **Obstacles**: Grey cubes representing collapsed walls, damaged buildings, and debris.
- **Start & Target**: Blue triangle at `(0, 0, 0)` and Gold star at `(9, 9, 9)`.
- **Drone & Trajectory**: Lime circle (turns red on collision) with a cyan path trail.
- **HUD Title**: Displays current step count, episode status (Running, Collision, Target Reached), cumulative return, drone coordinates, and last action taken.

---

## Project Structure

- `3d_environment.py`: `DroneNavEnv3D`, the 10x10x10 voxel grid environment with altitude and disaster barriers.
- `visualize_3d.py`: Interactive 3D Matplotlib visualizer for manual simulation control and inspection.
- `models/`: Saved model checkpoints and policy weights.
- `results/`: Evaluation logs, metrics, plots, and trajectories.
- `experiments/2d/`: Archive of preliminary 2D exploration experiments (`environment.py`, `rule_based_agent.py`, `ppo_agent.py`, `grpo_agent.py`).
