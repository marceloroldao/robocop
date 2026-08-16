from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .field import FieldState


@dataclass
class HumanoidObservation:
    q: np.ndarray
    qd: np.ndarray
    q_target: np.ndarray
    field_state: FieldState


def quaternion_vertical(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return 1.0
    w, x, y, z = q / n
    return float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))


def extract_humanoid_state(env, q_target: Optional[np.ndarray] = None) -> HumanoidObservation:
    data = env.unwrapped.data
    action_dim = int(env.action_space.shape[0])
    qpos = np.asarray(data.qpos, dtype=float)
    qvel = np.asarray(data.qvel, dtype=float)

    q = np.array(qpos[7:7 + action_dim], copy=True)
    qd = np.array(qvel[6:6 + action_dim], copy=True)
    if q_target is None:
        q_target = np.array(q, copy=True)

    state = FieldState(
        height=float(qpos[2]),
        vertical=quaternion_vertical(qpos[3:7]),
        omega=float(np.linalg.norm(qvel[3:6])),
        vel_z=float(qvel[2]),
    )
    return HumanoidObservation(q=q, qd=qd, q_target=np.asarray(q_target, dtype=float), field_state=state)


def make_humanoid_env(render_mode=None):
    import gymnasium as gym

    for env_id in ("Humanoid-v5", "Humanoid-v4"):
        try:
            return gym.make(env_id, render_mode=render_mode)
        except Exception:
            pass
    raise RuntimeError("Could not create Humanoid-v5 or Humanoid-v4. Install the 'sim' extra.")
