from __future__ import annotations

import numpy as np

from .field import ResolutiveField
from .mujoco_env import extract_humanoid_state


def _mujoco_module():
    import mujoco
    return mujoco


def capture_state(env):
    mujoco = _mujoco_module()
    model = env.unwrapped.model
    data = env.unwrapped.data
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    size = mujoco.mj_stateSize(model, spec)
    state = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    ctrl = np.array(data.ctrl, copy=True)
    return state, ctrl


def restore_state(env, state, ctrl):
    mujoco = _mujoco_module()
    model = env.unwrapped.model
    data = env.unwrapped.data
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    mujoco.mj_setState(model, data, state, spec)
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)


def finite_difference_gradient(env, field: ResolutiveField, action: np.ndarray, epsilon: float = 0.035, indices=None):
    action = np.asarray(action, dtype=float)
    if indices is None:
        indices = range(action.size)

    state, ctrl = capture_state(env)
    grad = np.zeros_like(action)
    sims = 0

    try:
        for j in indices:
            restore_state(env, state, ctrl)
            plus = action.copy()
            plus[j] += epsilon
            plus = np.clip(plus, -0.4, 0.4)
            env.unwrapped.do_simulation(plus, env.unwrapped.frame_skip)
            rp = field.score(extract_humanoid_state(env).field_state)
            sims += 1

            restore_state(env, state, ctrl)
            minus = action.copy()
            minus[j] -= epsilon
            minus = np.clip(minus, -0.4, 0.4)
            env.unwrapped.do_simulation(minus, env.unwrapped.frame_skip)
            rm = field.score(extract_humanoid_state(env).field_state)
            sims += 1

            grad[j] = (rp - rm) / (2.0 * epsilon)
    finally:
        restore_state(env, state, ctrl)

    return grad, sims
