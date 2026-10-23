"""Smoke check for GPS position noise in DroneNavEnv3D (environment_3d.py)."""
import numpy as np

from environment_3d import DroneNavEnv3D

GPS_IDX = slice(0, 6)  # position (0-2) and offset to target (3-5)
SENSOR_IDX = slice(6, 12)  # blocked flags, always from the true position


def noise_free_obs(env):
    """The observation as computed before GPS noise existed, from the true position."""
    r, c, a = env.drone_pos
    tr, tc, ta = env.target_pos
    h, w, d = env.height - 1, env.width - 1, env.depth - 1
    blocked = [
        1.0 if env._is_blocked((r + dr, c + dc, a + da)) else 0.0
        for dr, dc, da in env._action_to_delta.values()
    ]
    return np.array(
        [r / h, c / w, a / d, (tr - r) / h, (tc - c) / w, (ta - a) / d, *blocked],
        dtype=np.float32,
    )


def rollout(env, actions, seed):
    """Run the actions (resetting on episode end) and record everything the env produces."""
    obs, _ = env.reset(seed=seed)
    records = [(obs, env.drone_pos, None, None, None, noise_free_obs(env))]
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        records.append((obs, env.drone_pos, reward, terminated, truncated, noise_free_obs(env)))
        if terminated or truncated:
            obs, _ = env.reset()
            records.append((obs, env.drone_pos, None, None, None, noise_free_obs(env)))
    return records


actions = np.random.default_rng(0).integers(0, 6, size=300)

default = rollout(DroneNavEnv3D(), actions, seed=123)
zero = rollout(DroneNavEnv3D(gps_noise_std=0.0), actions, seed=123)
noisy = rollout(DroneNavEnv3D(gps_noise_std=0.5), actions, seed=123)
noisy_again = rollout(DroneNavEnv3D(gps_noise_std=0.5), actions, seed=123)

# 1. Zero noise reproduces the original position observations exactly.
for (obs_d, *_), (obs_z, *_, ref) in zip(default, zero):
    assert np.array_equal(obs_z, ref), f"zero-noise obs {obs_z} != original {ref}"
    assert np.array_equal(obs_d, obs_z)
print(f"[1] zero noise: {len(zero)} observations identical to the original formula")

# 2. Nonzero noise changes the observed position, and only the GPS indices.
n_changed = sum(not np.allclose(o[GPS_IDX], ref[GPS_IDX]) for o, *_, ref in noisy)
assert n_changed > 0, "noise never changed the observed position"
for o, *_, ref in noisy:
    assert np.array_equal(o[SENSOR_IDX], ref[SENSOR_IDX]), "noise leaked into obstacle flags"
max_err = max(np.abs(o[0:3] - ref[0:3]).max() for o, *_, ref in noisy)
print(f"[2] std=0.5: {n_changed}/{len(noisy)} observations have a changed GPS part "
      f"(max |error| {max_err:.3f} normalized); indices 6-11 unchanged")

# Same seed -> same noise.
assert all(np.array_equal(a[0], b[0]) for a, b in zip(noisy, noisy_again))
print("    same seed gives identical noisy observations")

# 3. True position, rewards and termination are unaffected by the noise.
for z, n in zip(zero, noisy):
    assert z[1:5] == n[1:5], f"true trajectory diverged: {z[1:5]} vs {n[1:5]}"
episodes = sum(1 for rec in zero if rec[2] is None)
print(f"[3] true positions, rewards, terminated/truncated identical over "
      f"{len(actions)} steps ({episodes} episodes)")

# 4. Shape and bounds stay valid, including under very large noise.
big = rollout(DroneNavEnv3D(gps_noise_std=50.0), actions, seed=7)
space = DroneNavEnv3D().observation_space
for o, *_ in default + noisy + big:
    assert o.shape == (12,) and o.dtype == np.float32
    assert space.contains(o), f"obs outside observation space: {o}"
    assert 0.0 <= o[0:3].min() and o[0:3].max() <= 1.0, "position outside [0, 1]"
print(f"[4] shape (12,), float32, inside observation space for std 0, 0.5 and 50")

try:
    DroneNavEnv3D(gps_noise_std=-1.0)
    raise AssertionError("negative std accepted")
except ValueError:
    pass

print("GPS noise smoke check passed.")
