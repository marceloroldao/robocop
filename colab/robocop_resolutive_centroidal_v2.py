# RoboCOP — Resolutive Centroidal V2
# Google Colab / Gymnasium Humanoid-v5
# Two-stage experiment:
#   1) Bootstrap: a simple physical PD stabilizer + tiny deterministic exploration
#      generates valid transitions.
#   2) Evaluation: resolutive memory selects a future joint/body state; the PD
#      layer only executes that desired state. No neural network is used.

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

SEED = 42
np.random.seed(SEED)


@dataclass
class Record:
    state: np.ndarray
    target_q: np.ndarray
    gain: float
    quality: float


class ResolutiveFutureStateMemory:
    def __init__(self, max_records=5000):
        self.records = []
        self.max_records = int(max_records)
        self.scales = None

    def fit_scales(self, states):
        x = np.asarray(states, float)
        s = np.std(x, axis=0)
        self.scales = np.where(s > 1e-5, s, 1.0)

    def add(self, state, target_q, gain, quality):
        self.records.append(Record(np.asarray(state, float).copy(),
                                   np.asarray(target_q, float).copy(),
                                   float(gain), float(quality)))
        if len(self.records) > self.max_records:
            self.records.sort(key=lambda r: (r.gain, r.quality), reverse=True)
            del self.records[self.max_records:]

    def recall(self, state, max_distance=0.90):
        if not self.records or self.scales is None:
            return None
        x = np.asarray(state, float)
        best = None
        for r in self.records:
            z = (x - r.state) / self.scales
            d = float(np.sqrt(np.mean(z*z)))
            if d > max_distance:
                continue
            # Prefer close states, but reward historically useful transitions.
            score = (1.0 + 4.0*r.gain + 0.25*r.quality) / (1.0 + d)
            if best is None or score > best[0]:
                best = (score, d, r)
        return best


def quaternion_tilt(q):
    # For the MuJoCo free-joint quaternion, a simple upright deviation proxy.
    q = np.asarray(q, float)
    q = q / max(np.linalg.norm(q), 1e-9)
    return float(2.0 * np.arccos(np.clip(abs(q[0]), 0.0, 1.0)))


def body_quality(data, target_z):
    z = float(data.qpos[2])
    tilt = quaternion_tilt(data.qpos[3:7])
    ang = float(np.linalg.norm(data.qvel[3:6]))
    vz = float(abs(data.qvel[2]))
    # Smooth bounded score; higher is better.
    qz = np.exp(-((z-target_z)/0.22)**2)
    qt = np.exp(-(tilt/0.45)**2)
    qa = np.exp(-(ang/3.0)**2)
    qv = np.exp(-(vz/1.5)**2)
    return float(0.45*qz + 0.30*qt + 0.15*qa + 0.10*qv)


def state_vector(data, q_ref, target_z):
    qj = np.asarray(data.qpos[7:], float)
    vj = np.asarray(data.qvel[6:], float)
    qerr = qj - q_ref
    z = float(data.qpos[2])
    tilt = quaternion_tilt(data.qpos[3:7])
    torso_v = np.asarray(data.qvel[:6], float)
    # Keep body configuration and derivative explicitly in the address.
    return np.concatenate([
        np.array([(z-target_z)/0.25, tilt/0.5], float),
        torso_v / np.array([2,2,2,4,4,4], float),
        qerr / 1.0,
        vj / 5.0,
    ])


def pd_action(env, q_target, kp=0.55, kd=0.08):
    data = env.unwrapped.data
    qj = np.asarray(data.qpos[7:], float)
    vj = np.asarray(data.qvel[6:], float)
    raw = kp*(np.asarray(q_target)-qj) - kd*vj
    return np.clip(raw, env.action_space.low, env.action_space.high)


def run_episode(env, q_ref, target_z, memory=None, seed=42,
                max_steps=1000, explore=False, learn=False,
                horizon=6, recall_distance=0.90):
    obs, _ = env.reset(seed=seed)
    # Gym's reset adds small noise; reference posture is the model's nominal pose,
    # not the noisy reset pose, so all episodes share the same physical target.
    history = []
    state_buffer = []
    recalls = 0
    misses = 0
    learned = 0
    initial_xy = np.asarray(env.unwrapped.data.qpos[:2], float).copy()

    for t in range(max_steps):
        data = env.unwrapped.data
        x = state_vector(data, q_ref, target_z)
        q_now = body_quality(data, target_z)
        desired_q = q_ref.copy()
        recalled = None

        if memory is not None and len(memory.records):
            recalled = memory.recall(x, max_distance=recall_distance)
            if recalled is not None:
                recalls += 1
                desired_q = recalled[2].target_q.copy()
            else:
                misses += 1

        action = pd_action(env, desired_q)

        if explore:
            # Tiny deterministic excitation; enough to create alternative physical
            # transitions without turning bootstrap into random control.
            phase = 2*np.pi*(t % 80)/80.0
            excitation = 0.025*np.sin(phase + np.arange(action.size)*0.37)
            action = np.clip(action + excitation, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)

        next_data = env.unwrapped.data
        q_next = body_quality(next_data, target_z)
        next_qj = np.asarray(next_data.qpos[7:], float).copy()

        if learn:
            # Store the current address, then after a short horizon compare it with
            # the future physical state. This avoids requiring monotonic one-step gain.
            state_buffer.append((x.copy(), q_now, next_qj.copy()))
            if len(state_buffer) > horizon:
                old_x, old_q, _ = state_buffer.pop(0)
                gain = q_next - old_q
                if gain > 0.005 and q_next > 0.45:
                    memory.add(old_x, next_qj, gain, q_next)
                    learned += 1

        history.append({
            'step': t,
            'quality': q_now,
            'height': float(data.qpos[2]),
            'recall': recalled is not None,
            'memory': 0 if memory is None else len(memory.records),
        })

        if terminated or truncated:
            break

    final_xy = np.asarray(env.unwrapped.data.qpos[:2], float)
    return {
        'steps': len(history),
        'mean_quality': float(np.mean([h['quality'] for h in history])) if history else 0.0,
        'min_quality': float(np.min([h['quality'] for h in history])) if history else 0.0,
        'recalls': recalls,
        'misses': misses,
        'recall_rate': recalls/max(1, len(history)),
        'learned': learned,
        'displacement_xy': float(np.linalg.norm(final_xy-initial_xy)),
        'history': history,
    }


def main():
    env = gym.make('Humanoid-v5')
    env.reset(seed=SEED)
    data = env.unwrapped.data
    q_ref = np.asarray(env.unwrapped.model.qpos0[7:], float).copy()
    target_z = float(env.unwrapped.model.qpos0[2])
    memory = ResolutiveFutureStateMemory()

    # Baseline: pure local physical controller.
    baseline = run_episode(env, q_ref, target_z, memory=None,
                           seed=SEED+100, max_steps=1000,
                           explore=False, learn=False)

    # Bootstrap several episodes. We retain all observed state vectors to estimate
    # a normalized geometry only from bootstrap data.
    bootstrap_states = []
    bootstrap_results = []
    for ep in range(8):
        # Temporary permissive scale during collection; learning does not need recall.
        if memory.scales is None:
            sample_state = state_vector(env.unwrapped.data, q_ref, target_z)
            memory.scales = np.ones_like(sample_state)
        r = run_episode(env, q_ref, target_z, memory=memory,
                        seed=SEED+ep, max_steps=1000,
                        explore=True, learn=True)
        bootstrap_results.append(r)
        bootstrap_states.extend([np.asarray(h_dummy, float) for h_dummy in []])

    # Reconstruct scales from stored memory addresses (training only).
    if memory.records:
        memory.fit_scales([r.state for r in memory.records])

    # Evaluation with memory frozen.
    before = len(memory.records)
    resolutive = run_episode(env, q_ref, target_z, memory=memory,
                             seed=SEED+100, max_steps=1000,
                             explore=False, learn=False,
                             recall_distance=0.90)
    frozen = (before == len(memory.records))
    env.close()

    print('\n' + '='*72)
    print('RoboCOP — RESOLUTIVE CENTROIDAL V2')
    print('='*72)
    print(f"baseline steps/quality : {baseline['steps']} / {baseline['mean_quality']:.4f}")
    print(f"bootstrap episodes     : {len(bootstrap_results)}")
    print(f"bootstrap mean steps   : {np.mean([r['steps'] for r in bootstrap_results]):.1f}")
    print(f"memory records         : {len(memory.records)}")
    print(f"resolutive steps       : {resolutive['steps']}")
    print(f"resolutive quality     : {resolutive['mean_quality']:.4f}")
    print(f"recalls / rate         : {resolutive['recalls']} / {100*resolutive['recall_rate']:.2f}%")
    print(f"misses                 : {resolutive['misses']}")
    print(f"displacement           : {resolutive['displacement_xy']:.4f}")
    print(f"memory frozen eval     : {'PASS' if frozen else 'FAIL'}")
    print('='*72)

    # Plot baseline and resolutive quality for direct visual comparison.
    plt.figure(figsize=(12,4))
    plt.plot([h['step'] for h in baseline['history']], [h['quality'] for h in baseline['history']], label='PD baseline')
    plt.plot([h['step'] for h in resolutive['history']], [h['quality'] for h in resolutive['history']], label='Resolutive + PD')
    plt.xlabel('step'); plt.ylabel('quality'); plt.title('Baseline vs Resolutive future-state control'); plt.legend(); plt.grid(); plt.show()

    return baseline, bootstrap_results, resolutive, memory


if __name__ == '__main__':
    BASELINE, BOOTSTRAP, RESOLUTIVE, MEMORY = main()
