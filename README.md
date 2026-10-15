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
 
