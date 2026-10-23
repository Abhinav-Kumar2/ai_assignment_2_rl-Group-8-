# agent interface for dronenavenv3d, shared by the visualizer and evaluation code
#
# to plug a new agent into visualizer_3d.py:
#   1. subclass Agent3D (or write any class with the same act/reset methods)
#   2. implement act(obs, env) -> action int in 0..5
#   3. run: uv run python visualizer_3d.py --agent my_module:MyAgent
#      or register a short name in AGENT_REGISTRY below and use --agent <name>
#
# agents get the env as well as the observation so that non-learning agents
# (rule-based, planners) can read env state; learned policies should only use obs.

import importlib


class Agent3D:
    """base class for agents that act in DroneNavEnv3D"""

    name = "agent"

    def reset(self, env):
        """called after every env.reset(), before the first act()"""

    def act(self, obs, env):
        """return the action (int 0..5) for the current observation"""
        raise NotImplementedError


class RandomAgent(Agent3D):
    """uniform random actions, useful as a baseline and for checking the plumbing"""

    name = "random"

    def __init__(self, seed=None):
        self.seed = seed

    def reset(self, env):
        if self.seed is not None:
            env.action_space.seed(self.seed)

    def act(self, obs, env):
        return int(env.action_space.sample())


# short names for --agent; values are "module:Class" so heavy imports
# (torch, stable-baselines3) only happen when that agent is used
AGENT_REGISTRY = {
    "random": "agents_3d:RandomAgent",
    "ppo": "ppo_agent_3d:PPOAgent3D",
}


def load_agent(spec, **kwargs):
    """build an agent from a registry name or a "module:Class" string; kwargs go to the constructor"""
    target = AGENT_REGISTRY.get(spec, spec)
    if ":" not in target:
        known = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(f"unknown agent '{spec}': use one of [{known}] or 'module:Class'")
    module_name, class_name = target.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    agent = cls(**kwargs)
    if not callable(getattr(agent, "act", None)):
        raise TypeError(f"{target} has no act(obs, env) method")
    return agent
