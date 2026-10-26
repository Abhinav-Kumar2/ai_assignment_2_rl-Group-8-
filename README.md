# NavRL: Autonomous Drone Navigation in Disaster-Affected Environments

NavRL investigates autonomous drone navigation in complex, obstacle-filled disaster zones using reinforcement learning.

A simulated quadcopter drone learns to navigate a three-dimensional grid environment representing collapsed buildings, suspended debris, and narrow corridors to reach a designated rescue target. The environment features forced non-greedy detours, sparse reward barriers, and Gaussian GPS sensor noise.

The primary agent of this repository is the **3D Proximal Policy Optimization (PPO) Agent** (`ppo_agent_3d.py`). Secondary baseline and exploratory algorithms (3D DQN, 3D GRPO, and historical 2D models) are documented in the [Experiments](#experiments) section.

> **Note on Simulation Fidelity:** This project uses a discrete grid-based simulation (`gymnasium.Env`) to evaluate spatial reinforcement learning, planning under partial/noisy observation, and detour handling. It does not model continuous aerodynamics, battery physics, or wind dynamics.

---

## Environment Setup

### 1. Prerequisites
- **Python:** 3.10 or 3.11 recommended.
- **Operating System:** Linux, macOS, or Windows.

### 2. Clone and Enter the Repository
```bash
git clone https://github.com/Abhinav-Kumar2/ai_assignment_2_rl.git
cd ai_assignment_2_rl
```

### 3. Virtual Environment Creation & Activation

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
Install all required libraries from `requirements.txt`:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*Key packages:* `gymnasium`, `stable-baselines3`, `torch`, `matplotlib`, `PyQt5`, `numpy`, `tqdm`, `rich`.

---

## Main 3D Environment

The primary environment is implemented in `environment_3d.py` as `DroneNavEnv3D`.

### Running the Environment Smoke Test
Verify map connectivity, reachability, action steps, and observation bounds:
```bash
python environment_3d.py
```
*What this verifies:*
- A Breadth-First Search (BFS) confirms all 745 free cells are reachable from the start position.
- Confirms the shortest path to the target requires 37 steps, which exceeds the straight-line Manhattan distance (27), enforcing an intentional detour.
- Executes 50 random action steps to ensure rewards, transitions, and observations stay strictly inside the Gymnasium spaces.

### Core Environment Specifications
- **Grid Dimensions:** 10 x 10 x 10 voxels (`height=10`, `width=10`, `depth=10`).
- **Coordinate System:** Positions are indexed as `(row, col, alt)`:
  - `row`: North / South axis (index 0 to 9)
  - `col`: West / East axis (index 0 to 9)
  - `alt`: Altitude axis (index 0 to 9)
- **Start and Target:** Default start is `(0, 0, 0)` (ground level, northwest corner); default target is `(9, 9, 9)` (high altitude, southeast corner). Configurable via `start_pos` and `target_pos`.
- **Action Space (`Discrete(6)`):**
  - `0`: Up / North (`row - 1`)
  - `1`: Down / South (`row + 1`)
  - `2`: Left / West (`row - 1`)
  - `3`: Right / East (`row + 1`)
  - `4`: Ascend (`alt + 1`)
  - `5`: Descend (`alt - 1`)
- **Observation Space (`Box(12,)`, float32 in `[-1.0, 1.0]`):**
  - Indices 0–2: Observed drone position normalized by grid dimensions.
  - Indices 3–5: Observed relative target vector normalized by grid dimensions.
  - Indices 6–11: 6 local binary obstacle/boundary detection flags for each action direction (derived from true physics position).
- **Obstacle Layout (255 blocked voxels):**
  - **Wall A (row 3):** Full-grid barrier with a 2x2 ground-level gap at the east side (`cols 7-8`, `alt 0-1`).
  - **Wall B (row 6):** Full-grid barrier with a 2x2 high-altitude gap at the west side (`cols 1-2`, `alt 7-8`).
  - **Collapsed Buildings:** Vertical columns of varying heights in zones 1, 2, and 3.
  - **Floating Debris:** Suspended obstruction cells in mid-air.
- **Reward Structure:**
  - Step penalty: `-0.1` per movement step.
  - Distance progress bonus: `+1.0` per unit reduction in distance to target.
  - Collision penalty: `-10.0` (terminates episode).
  - Target reached bonus: `+50.0` (terminates episode).
- **Termination Conditions:**
  - `terminated=True`: Drone collides with an obstacle/boundary, or reaches the target cell.
  - `truncated=True`: Episode step count reaches `max_steps` (default: 200).
- **GPS Noise Simulation:**
  - Configurable standard deviation `gps_noise_std` (default 0.0).
  - Adds Gaussian noise to the drone's perceived position (affecting observation indices 0–5).
  - Physical collision checking and local proximity sensors (indices 6–11) always use true ground truth coordinates.

---

## Primary Agent: PPO

The main agent uses Proximal Policy Optimization (PPO) via Stable-Baselines3 (`ppo_agent_3d.py`).

### Key Design & Training Details
- **Architecture:** `MlpPolicy` with two hidden layers of 64 units each.
- **Geodesic Potential Shaping:** Replaces naive Manhattan distance shaping with BFS shortest-path distance shaping along free corridors during training, preventing the agent from getting stuck against full-width barrier walls.
- **Random-Start Curriculum:** Training starts episodes with probability `random_start_prob=0.5` from random reachable cells, allowing the policy to solve intermediate wall passages before chaining full flights from `(0, 0, 0)`.
- **Parallel Rollouts:** Uses 8 vectorized environments (`n_envs=8`).
- **Model Checkpoints:** Saved to `models/ppo_drone_3d.zip`.

### Training the PPO Agent
Train the policy from scratch with the recommended 500,000 timesteps to ensure full convergence through the dual-wall maze (achieving a 100% success rate):
```bash
python ppo_agent_3d.py --timesteps 500000 --seed 0
```
*(Training takes ~1 minute on modern CPUs across the 8 vectorized parallel environments).*

*Fast verification smoke test (1,024 timesteps on 4 parallel envs):*
```bash
python ppo_agent_3d.py --smoke-test
```

### Evaluating the PPO Agent
Evaluate a saved checkpoint deterministically across multiple episodes:
```bash
python ppo_agent_3d.py --eval --episodes 10
```

### Evaluation with GPS Sensor Noise
Evaluate policy robustness under simulated GPS sensor drift:
```bash
python ppo_agent_3d.py --eval --episodes 10 --gps-noise 0.5
```

### Supported PPO CLI Arguments
| Flag | Default | Description |
|---|---|---|
| `--eval` | `False` | Run deterministic evaluation without training |
| `--model-path` | `models/ppo_drone_3d.zip` | Path to load/save model checkpoint |
| `--episodes` | `10` | Number of evaluation episodes |
| `--timesteps` | `1000000` | Total training timesteps |
| `--seed` | `0` | Random seed for environments and models |
| `--gps-noise` | `0.5` | GPS noise standard deviation (grid cells) |
| `--random-start-prob` | `0.5` | Training probability of starting from a random free cell |
| `--random-starts` | `False` | Evaluate from random start cells instead of `(0,0,0)` |
| `--start` | `0 0 0` | Custom start coordinate (`row col alt`) |
| `--target` | `9 9 9` | Custom target coordinate (`row col alt`) |
| `--smoke-test` | `False` | Run lightweight 1,024-step verification |

*Evaluation outputs:* Summaries and plots are written to `results/ppo_3d_evaluation_noise<std>_summary.json`, `results/ppo_3d_evaluation_noise<std>_episodes.csv`, and `results/ppo_3d_evaluation_noise<std>_metrics.png`.

---

## 3D Visualization

`visualizer_3d.py` provides an interactive 3D Matplotlib renderer for inspecting manual navigation, agent rollouts, and GPS noise.

### Modes of Visualization
- **Realistic Mode (Default):** Disaster-affected urban environment featuring 3D building facades, damaged rooftops, multi-faceted rubble, navigation ground terrain, and a 3D quadcopter drone with rotating rotors.
- **Basic Mode:** Clean, lightweight visualization showing minimalist obstacle cube blocks and standard markers.

Toggle between modes at runtime by pressing the **`m`** key or passing `--mode basic`.

### Visualization Commands

**Manual Keyboard Flight:**
```bash
python visualizer_3d.py
```

**Run in Basic Minimal Grid Mode:**
```bash
python visualizer_3d.py --mode basic
```

**Autonomous Playback with Trained 3D PPO Agent:**
```bash
python visualizer_3d.py --agent ppo
```

**Playback with Random Agent:**
```bash
python visualizer_3d.py --agent random
```

**Headless Rollout Recording (GIF Export):**
```bash
python visualizer_3d.py --agent ppo --save results/ppo_3d_rollout.gif
```
*(You can also export a single PNG of the final frame by passing a `.png` filename: `--save results/ppo_3d_rollout.png`)*

### Interactive Keyboard Controls
When the visualizer window is focused:
- **`Up Arrow`**: Move North (`row - 1`)
- **`Down Arrow`**: Move South (`row + 1`)
- **`Left Arrow`**: Move West (`col - 1`)
- **`Right Arrow`**: Move East (`col + 1`)
- **`PageUp`**: Ascend (`alt + 1`)
- **`PageDown`**: Descend (`alt - 1`)
- **`m`**: Toggle visual style between **Realistic** and **Basic**
- **`Space`**: Play / Pause autonomous agent stepping
- **`n`**: Step the agent forward by one single timestep
- **`Backspace`**: Reset episode to start position
- **`q`** or **`Escape`**: Close and exit

*Display Requirements:* Interactive display requires a GUI backend (e.g. PyQt5). On headless machines or remote servers without an X11/Wayland display, run with `--save <path.gif>` to generate recordings headlessly.

---

## GPS Noise Verification

`check_gps_noise.py` contains a self-contained test suite verifying sensor noise isolation:
```bash
python check_gps_noise.py
```
*What this verifies:*
1. **Zero Noise Consistency:** Confirms that `gps_noise_std=0.0` exactly matches the deterministic observation formulation.
2. **Channel Isolation:** Confirms that noise only perturbs GPS channels (indices 0–5) and never leaks into local obstacle proximity sensors (indices 6–11).
3. **Reproducibility:** Identical seeds generate identical noisy observations.
4. **Physical Invariance:** Drone transitions, collisions, terminations, and rewards are identical regardless of noise level.
5. **Observation Space Compliance:** All noisy observations remain within valid `Box(12,)` boundaries.

---

## Results and Model Artifacts

- **`models/`**:
  - `models/ppo_drone_3d.zip`: Primary trained 3D PPO checkpoint.
  - `models/ppo_drone.zip` / `models/ppo_drone_best.zip`: Checkpoints from historical 2D experiments.
- **`results/`**:
  - `results/ppo_3d_evaluation_noise0.5_summary.json`: Aggregated metrics (success, collision, truncation rates).
  - `results/ppo_3d_evaluation_noise0.5_episodes.csv`: Step-by-step episode logs.
  - `results/ppo_3d_evaluation_noise0.5_metrics.png`: Evaluation graphs.
  - `results/ppo_3d_rollout.gif`: Recorded 3D animation of agent flight path.

---

# Experiments

Secondary baselines, exploratory algorithms, and legacy 2D experiments. These are standalone studies and not required for the main 3D PPO pipeline.

### 1. 3D DQN Experiment (`experiments/3d/dqn_agent_3d.py`)
A standard Deep Q-Network implementation in PyTorch (without Double DQN modifications) for `DroneNavEnv3D`.
- Features an experience replay buffer (`capacity=50,000`) and target network updates every 1,000 steps.
- **Train:**
  ```bash
  python experiments/3d/dqn_agent_3d.py --train --timesteps 100000
  ```
- **Evaluate:**
  ```bash
  python experiments/3d/dqn_agent_3d.py --eval --episodes 10
  ```
- *Default Checkpoint:* `models/dqn_drone_3d.pt`
- *Evaluation Output:* `results/dqn_3d_evaluation_summary.json`

### 2. 3D GRPO Experiment (`experiments/3d/grpo_agent_3d.py`)
A critic-free Group Relative Policy Optimization (GRPO) implementation in PyTorch for `DroneNavEnv3D`.
- Samples groups of $G=4$ rollouts per reference state, normalizing rewards within each group to compute policy advantages without a value network.
- **Train:**
  ```bash
  python experiments/3d/grpo_agent_3d.py --train --iterations 100
  ```
- **Evaluate:**
  ```bash
  python experiments/3d/grpo_agent_3d.py --eval --episodes 10
  ```
- *Default Checkpoint:* `models/grpo_drone_3d.pt`
- *Evaluation Output:* `results/grpo_3d_evaluation_summary.json`

### 3. Historical 2D Experiments (`experiments/2d/`)
The initial 2D navigation prototypes on a 10x10 grid with 4 movement actions (`Up`, `Down`, `Left`, `Right`):
- **Rule-Based Agent:** Heuristic baseline with obstacle avoidance and visit-frequency memory:
  ```bash
  python experiments/2d/rule_based_agent.py --episodes 10
  ```
- **2D PPO Agent:** Baseline PPO policy using Stable-Baselines3:
  ```bash
  python experiments/2d/ppo_agent.py train --timesteps 50000
  python experiments/2d/ppo_agent.py eval --episodes 10
  ```
- **2D GRPO Agent:** Critic-free policy gradient on 2D grid:
  ```bash
  python experiments/2d/grpo_agent.py train --iterations 300
  python experiments/2d/grpo_agent.py eval --episodes 10
  ```

---

## Project Structure

```
ai_assignment_2_rl/
├── environment_3d.py            # Main 10x10x10 3D Gymnasium environment
├── ppo_agent_3d.py              # Primary 3D PPO agent (training & evaluation)
├── visualizer_3d.py             # 3D visualizer (realistic & basic modes, GIF export)
├── agents_3d.py                 # Agent3D interface and registry (PPO, Random)
├── check_gps_noise.py           # Verification test suite for GPS sensor noise
├── requirements.txt             # Verified Python package dependencies
├── models/                      # Saved neural network checkpoints
│   ├── ppo_drone_3d.zip         # Primary 3D PPO trained checkpoint
│   ├── ppo_drone.zip            # Historical 2D PPO checkpoint
│   └── ppo_drone_best.zip       # Historical 2D best evaluation checkpoint
├── results/                     # Metric summaries, CSV logs, plots, and GIFs
└── experiments/                 # Secondary and historical experiments
    ├── 3d/                      # 3D experimental algorithms
    │   ├── dqn_agent_3d.py      # Standard 3D Deep Q-Network
    │   └── grpo_agent_3d.py     # Critic-free 3D Group Relative Policy Optimization
    └── 2d/                      # Historical 2D baselines
        ├── environment.py       # 2D grid environment
        ├── rule_based_agent.py  # 2D non-learning rule-based agent
        ├── ppo_agent.py         # 2D Stable-Baselines3 PPO agent
        └── grpo_agent.py        # 2D GRPO agent
```

---

## Troubleshooting

- **GUI / Matplotlib Window Fails to Open:**
  - If running in headless environments (e.g. SSH, Docker, WSL without X-server), interactive Qt/Tk windows will not display. Use headless recording via `--save <path.gif>`:
    ```bash
    python visualizer_3d.py --agent ppo --save results/ppo_3d_rollout.gif
    ```
- **Model Checkpoint Not Found During Evaluation:**
  - If running `ppo_agent_3d.py --eval` without `models/ppo_drone_3d.zip`, train a model first using `python ppo_agent_3d.py --timesteps 500000` or specify a custom path via `--model-path <path>`.
- **Import Errors / Missing Packages:**
  - Ensure the virtual environment is active (`which python` points to `.venv/bin/python`), then run:
    ```bash
    pip install -r requirements.txt
    ```
- **Working Directory Context:**
  - All commands and scripts expect to be executed from the repository root directory (`ai_assignment_2_rl/`).
