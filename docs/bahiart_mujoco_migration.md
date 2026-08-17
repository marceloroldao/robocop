# BahiaRT MuJoCo migration path

## Decision

RoboCOP will evaluate the 2025 BahiaRT MuJoCo base as the primary external competition baseline for the next integration phase.

The upstream code remains an external dependency. No BahiaRT source is copied into the RoboCOP repository until its license and redistribution conditions are explicitly verified.

## Why this base

- It targets the MuJoCo-based RoboCup Soccer Simulation 3D stack.
- It is Python-based, reducing adapter friction with RoboCOP's current experimental code.
- It lets RoboCOP compare a conventional competition baseline with the resolutive memory/control layer without rebuilding locomotion from zero.

## Integration boundary

The intended boundary is:

```text
BahiaRT / MuJoCo state
        |
        v
sensor/observation adapter
        |
        +--> baseline BahiaRT action --------------------+
        |                                                |
        +--> RoboCOP memory / field proposal             |
                                                         v
                                             A/B action selector
                                                         |
                                                         v
                                                   MuJoCo robot
```

The first experiments MUST be pass-through: RoboCOP observes but does not alter the BahiaRT action. Only after a reproducible baseline is recorded should modulation be enabled.

## Phase 0 — provenance and license audit

Run:

```bash
bash scripts/fetch_bahiart_mujoco_external.sh
bash scripts/inspect_bahiart_mujoco.sh
```

Record the upstream commit and license result. If no explicit compatible license is found, keep the upstream checkout external and do not redistribute its source or derived source inside RoboCOP.

## Phase 1 — interface inventory

Identify, without modifying upstream code:

1. observation/state construction;
2. policy/controller action output;
3. MuJoCo `data.ctrl` write boundary;
4. reset/episode boundary;
5. fall and stability signals;
6. simulation timestep and actuator ordering.

The target trace schema is:

```text
(state_t, action_t, state_t+1, reward/stability, terminated, metadata)
```

## Phase 2 — passive recorder

Create an external runtime hook or wrapper that records the baseline while returning the original action unchanged. Validation criterion: bitwise/numerical identity of the original action before and after instrumentation.

## Phase 3 — transition memory

Index transitions by hierarchical state resolution and learn recovery motifs:

```text
unstable state -> action -> more stable next state
```

The memory must preserve directionality. Similar sensor snapshots with opposite trends are not treated as the same transition.

## Phase 4 — A/B controller

Compare:

- BahiaRT baseline;
- BahiaRT + passive RoboCOP recorder;
- BahiaRT + resolutive memory suggestion;
- BahiaRT + bounded resolutive action modulation.

Primary metrics: completion/survival, fall rate, task reward, control energy, torque proxy, CPU time, intervention rate, and recovery success.

## Reproducibility

- Keep upstream commit hashes in experiment metadata.
- Keep training and evaluation seeds separate.
- Do not use GitHub Actions for benchmark claims; run reproducible local/server scripts and commit summarized results.
- Negative results are retained.
