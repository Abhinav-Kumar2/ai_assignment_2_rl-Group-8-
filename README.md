# NavRL: Autonomous Drone Navigation in Disaster-Affected Environments

Autonomous drone navigation in complex, disaster-affected 3D environments using reinforcement learning.

The **primary agent** of this repository is the **3D PPO Agent** (`ppo_agent_3d.py`), which navigates a 10x10x10 voxel environment with collapsed building walls, altitude obstacles, debris, forced detours, and realistic GPS sensor noise.

Preliminary 2D implementations (rule-based agent, 2D PPO, and GRPO) are preserved as reference baselines and early experimentations under `experiments/2d/`.

---

## Environment Setup

### 1. Virtual Environment Setup
Ensure Python 3.11+ is installed, then create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(On Windows PowerShell: `.venv\Scripts\Activate.ps1`)*

### 2. Smoke Checks
Verify that the 3D environment graph connectivity and reachability work as expected:
```bash
python environment_3d.py
```

Verify the GPS sensor noise implementation and bounds:
```bash
python check_gps_noise.py
```

---

## Primary Agent: 3D PPO Agent

The primary model is an on-policy Proximal Policy Optimization (PPO) agent (`MlpPolicy` 64x64, Stable-Baselines3) adapted to `DroneNavEnv3D`.

### 1. Training Command
To train the 3D PPO agent with geodesic shortest-path potential shaping, random-start curriculum, and Gaussian GPS noise:
```bash
python ppo_agent_3d.py --timesteps 50000 --seed 0
```
- Trains across 8 parallel vectorized environments.
- Saves the model weights to `models/ppo_drone_3d.zip`.

For a quick verification check (1,024 timesteps):
```bash
python ppo_agent_3d.py --smoke-test
```

### 2. Evaluation Command
Evaluate the trained 3D policy deterministically without retraining:
```bash
python ppo_agent_3d.py --eval --episodes 10
```

Optional evaluation flags:
- `--model-path`: Checkpoint file to evaluate (defaults to `models/ppo_drone_3d.zip`).
- `--episodes`: Number of evaluation episodes (e.g. `--episodes 100`).
- `--gps-noise`: GPS noise standard deviation in grid cells (defaults to `0.5`, use `0.0` for noise-free).
- `--random-starts`: Evaluate from random free start cells across the map instead of the fixed start `(0, 0, 0)`.

Artifacts generated under `results/`:
- `results/ppo_3d_evaluation_episodes.csv`: Per-episode returns, step count, and termination status.
- `results/ppo_3d_evaluation_summary.json`: Success rate, collision rate, truncation rate, mean length, and mean reward.
- `results/ppo_3d_evaluation_metrics.png`: Visual plot of outcome rates and per-episode rewards.

---

## 3D Environment Visualization

`visualizer_3d.py` provides an interactive 3D Matplotlib visualization for `DroneNavEnv3D` that displays obstacles, start/target points, drone trajectory, and observed noisy GPS positions.

### Launching the Visualizer
- **Manual Keyboard Control**:
  ```bash
  python visualizer_3d.py
  ```
- **Autonomous Playback with Trained 3D PPO Agent**:
  ```bash
  python visualizer_3d.py --agent ppo --gps-noise 0.5
  ```
- **Headless GIF Export**:
  ```bash
  python visualizer_3d.py --agent ppo --gps-noise 0.5 --save results/ppo_3d_rollout.gif
  ```

### Interactive Controls
When the visualizer window is focused:
- **`Up Arrow`**: Move North (`row - 1`)
- **`Down Arrow`**: Move South (`row + 1`)
- **`Left Arrow`**: Move West (`col - 1`)
- **`Right Arrow`**: Move East (`col + 1`)
- **`PageUp`**: Ascend (`alt + 1`)
- **`PageDown`**: Descend (`alt - 1`)
- **`Space`**: Play / Pause agent stepping
- **`n`**: Let the agent take one single step
- **`Backspace`**: Reset to initial state
- **`q`** or **`Escape`**: Quit visualizer

---

## 3D Environment Specification (`environment_3d.py`)

- **State / Observation Space**: `gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)`
  - Indices 0–2: Normalized observed position `(r, c, a)`
  - Indices 3–5: Normalized offset vector to target
  - Indices 6–11: 6 local binary obstacle/boundary detection flags in action order
- **Action Space**: `gymnasium.spaces.Discrete(6)`
  - `0: Up`, `1: Down`, `2: Left`, `3: Right`, `4: Ascend`, `5: Descend`
- **Reward Function**:
  - Step penalty: `-0.1` per step
  - Distance progress bonus: `+1.0` per unit of progress toward target
  - Collision penalty: `-10.0` (terminates episode)
  - Target reached bonus: `+50.0` (terminates episode)
- **Obstacle Map**:
  - Two full-grid partition walls at row 3 (gap east near ground) and row 6 (gap west high up).
  - Forces an intentional westward detour away from the target, requiring a 37-step optimal path (Manhattan distance 27).
- **GPS Noise**:
  - Configurable zero-mean Gaussian position noise (`gps_noise_std`) applied to observed position and target offset. Ground truth state for collisions and physics remains unaffected.

---

## 2D Experimentations and Baselines (`experiments/2d/`)

Prior to the 3D pipeline, several 2D baselines were developed on a 10x10 grid:

### 1. Rule-Based Baseline
Non-learning greedy baseline with obstacle avoidance and visit-frequency memory:
```bash
python experiments/2d/rule_based_agent.py --episodes 10
```
Optimal path on default 2D map: 18 steps, reward 66.2, 0 collisions.

### 2. 2D PPO Baseline
Standard Stable-Baselines3 PPO on the 2D task:
```bash
python experiments/2d/ppo_agent.py train --timesteps 50000
python experiments/2d/ppo_agent.py eval --episodes 10
```

### 3. 2D GRPO Baseline
Group Relative Policy Optimization from scratch in PyTorch (without a value critic):
```bash
python experiments/2d/grpo_agent.py train --iterations 300
python experiments/2d/grpo_agent.py eval --episodes 10
```

---

## Project Structure

```
├── environment_3d.py       # 10x10x10 3D Gymnasium environment with GPS noise
├── ppo_agent_3d.py         # Primary 3D PPO agent: training, geodesic shaping & evaluation
├── visualizer_3d.py        # 3D Matplotlib visualizer (manual, agent playback, GIF recording)
├── agents_3d.py            # Base Agent3D interface and agent registry
├── check_gps_noise.py      # Verification suite for GPS sensor noise behavior
├── requirements.txt        # Python package dependencies
├── models/
│   └── ppo_drone_3d.zip    # Saved 3D PPO policy checkpoint
├── results/                # Evaluation summaries (JSON), logs (CSV), plots (PNG), and GIFs
└── experiments/
    └── 2d/                 # Historical 2D exploratory baselines
        ├── environment.py
        ├── rule_based_agent.py
        ├── ppo_agent.py
        └── grpo_agent.py
```
