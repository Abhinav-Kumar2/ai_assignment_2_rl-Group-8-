# ppo training script for 3d drone navigation in dronenavenv3d

import argparse
import importlib
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

env_mod = importlib.import_module("3d_environment")
DroneNavEnv3D = env_mod.DroneNavEnv3D

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_drone_3d.zip")


def train(timesteps=50_000, seed=0, n_envs=8, verbose=1):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # verify observation and action spaces before training
    check_env = DroneNavEnv3D()
    obs_space = check_env.observation_space
    act_space = check_env.action_space
    print(f"verified 3d environment spaces:")
    print(f"  observation space: {obs_space} (shape: {obs_space.shape})")
    print(f"  action space: {act_space} (n: {act_space.n})")

    print(f"\ntraining configuration:")
    print(f"  timesteps: {timesteps}")
    print(f"  seed: {seed}")
    print(f"  parallel envs: {n_envs}")
    print(f"  policy: MlpPolicy (64x64)")
    print(f"  model output: {MODEL_PATH}")

    # vectorize environments for training without gui
    env = make_vec_env(DroneNavEnv3D, n_envs=n_envs, seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        device="cpu",
        verbose=verbose,
    )

    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(MODEL_PATH)
    print(f"\nsaved trained model to: {MODEL_PATH}")
    return model


def verify_saved_model(path=MODEL_PATH):
    print(f"\nverifying model loading on fresh DroneNavEnv3D instance...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"model file not found at {path}")

    model = PPO.load(path)
    env = DroneNavEnv3D()
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and steps < 20:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"model verification successful:")
    print(f"  rollout steps tested: {steps}")
    print(f"  drone end pos: {env.drone_pos}")
    print(f"  total reward over test steps: {total_reward:.2f}")
    print(f"  terminated: {terminated}, truncated: {truncated}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000, help="training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--smoke-test", action="store_true", help="run short smoke test (1024 timesteps)")
    args = parser.parse_args()

    steps = 1024 if args.smoke_test else args.timesteps
    train(timesteps=steps, seed=args.seed, n_envs=4 if args.smoke_test else 8, verbose=0)
    verify_saved_model()


if __name__ == "__main__":
    main()
