# NavRL: Autonomous Drone Navigation in Disaster-Affected Environments

Autonomous drone navigation in complex, disaster-affected environments using reinforcement learning.

## Environment Setup

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

## Project Structure

- `experiments/`: Scripts and configurations for running training and evaluation runs.
- `models/`: Saved model checkpoints and policy weights.
- `results/`: Evaluation logs, metrics, plots, and trajectories.
