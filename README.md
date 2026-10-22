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

## 3D PPO Agent

`ppo_3d_agent.py` trains a Stable-Baselines3 PPO agent (`MlpPolicy` 64x64) directly on `DroneNavEnv3D` without launching any visualization window.

### Training Command
```bash
python ppo_3d_agent.py --timesteps 50000 --seed 0
```
This runs headless PPO training across 8 vectorized environments, verifies the 12-dimensional observation and 6-discrete action spaces, and saves the trained policy to `models/ppo_drone_3d.zip`.

### Smoke Test Mode
Run a quick end-to-end check (1,024 timesteps, model save, and reload verification on a fresh environment):
```bash
python ppo_3d_agent.py --smoke-test
```

### Evaluation Command
Evaluate the trained 3D policy deterministically without re-training:
```bash
python ppo_3d_agent.py --eval --episodes 10
```
Optional flags:
- `--model-path`: Custom checkpoint path (defaults to `models/ppo_drone_3d.zip`).
- `--episodes`: Number of evaluation episodes (defaults to `10`).
- `--seed`: Initial evaluation seed (defaults to `0`).

Artifacts generated under `results/`:
- `results/ppo_3d_evaluation_episodes.csv`: Per-episode returns, step count, and termination status.
- `results/ppo_3d_evaluation_summary.json`: Success rate, collision rate, truncation rate, mean length, and mean reward.
- `results/ppo_3d_evaluation_metrics.png`: Visual plot of outcome rates and per-episode rewards.

---

## Project Structure

- `3d_environment.py`: `DroneNavEnv3D`, the 10x10x10 voxel grid environment with altitude and disaster barriers.
- `ppo_3d_agent.py`: PPO training and verification pipeline for the 3D environment.
- `visualize_3d.py`: Interactive 3D Matplotlib visualizer for manual simulation control and inspection.
- `models/`: Saved model checkpoints and policy weights.
- `results/`: Evaluation logs, metrics, plots, and trajectories.
- `experiments/2d/`: Archive of preliminary 2D exploration experiments (`environment.py`, `rule_based_agent.py`, `ppo_agent.py`, `grpo_agent.py`).
