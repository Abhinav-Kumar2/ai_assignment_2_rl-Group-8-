# NavRL: Autonomous Drone Navigation in Disaster-Affected Environments

Autonomous drone navigation in complex, disaster-affected environments using reinforcement learning.

## Environment Setup (uv)

```bash
uv sync                                   # creates .venv from pyproject.toml / uv.lock
uv run python environment.py              # smoke check
uv run python rule_based_agent.py --episodes 10   # rule-based baseline
uv run python ppo_agent.py train-eval             # train PPO, then compare with the baseline
uv run python grpo_agent.py train && uv run python grpo_agent.py eval   # train GRPO, then compare
```

## Environment Setup (pip, legacy)

### Linux / macOS
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Rule-Based Agent

`rule_based_agent.py` is a non-learning baseline to compare RL agents against. It uses only the
environment's 8-value observation (position, offset to target, and the four "blocked" flags) plus a
count of how often it has visited each cell. At every step it:

1. Ignores any move whose neighbouring cell is blocked (obstacle or map edge), so it never collides.
2. Among the safe moves, picks the cell it has visited least, which stops it oscillating or getting stuck in dead ends.
3. Breaks ties by preferring moves that reduce the Manhattan distance to the target, then the axis with the larger remaining gap.

On the default map it reaches the target in 18 steps (the shortest possible path) with a total
reward of 66.2 and no collisions.

## PPO Agent

`ppo_agent.py` trains a stable-baselines3 PPO agent (MLP 64x64, 8 parallel envs, 50k timesteps,
about 30 seconds on CPU) on the same 8-value observation. An `EvalCallback` keeps the best greedy
policy in `models/ppo_drone_best.zip`, and the learning curve goes to `results/ppo_learning_curve.png`.

```bash
uv run python ppo_agent.py train --timesteps 50000
uv run python ppo_agent.py eval --episodes 10
```

It also reaches the target in 18 steps with reward 66.2 and no collisions, matching the rule-based
agent. It finds this path within the first ~5k timesteps, because on the default map the greedy
"down, then right" route is already obstacle-free and optimal.

## GRPO Agent

`grpo_agent.py` implements Group Relative Policy Optimization from scratch in PyTorch. It has no value
critic: each iteration picks 8 start cells, samples a group of 16 rollouts from each, and scores every
rollout against its own group (`(R - mean) / std`). The policy (8-64-64-4 MLP) is updated with a
PPO-style clipped objective plus a KL penalty to a reference policy refreshed every 10 iterations.
Because the env always starts at (0, 0), training moves the drone to a random free cell after reset
(25% of groups use the default start), which gives the groups different start states.

```bash
uv run python grpo_agent.py train --iterations 300   # ~70 s on CPU, saves models/grpo_drone.pt
uv run python grpo_agent.py eval                      # default start + all 90 start cells vs rule-based
```

Result: from (0, 0) the greedy policy takes the optimal 18-step path (reward 66.2), and it reaches the
target with no collisions from all 90 free start cells (mean 9.1 steps, mean reward 58.2), matching the
rule-based agent. The training curve is in `results/grpo_training_curve.png`.

## Project Structure

- `environment.py`: `DroneNavEnv`, the 10x10 grid environment.
- `rule_based_agent.py`: non-learning baseline agent (greedy + obstacle avoidance + visit memory).
- `ppo_agent.py`: PPO agent (stable-baselines3): training, evaluation and comparison with the baseline.
- `grpo_agent.py`: GRPO agent (PyTorch, no critic): training, evaluation and comparison with the baseline.

- `experiments/`: Scripts and configurations for running training and evaluation runs.
- `models/`: Saved model checkpoints and policy weights.
- `results/`: Evaluation logs, metrics, plots, and trajectories.
