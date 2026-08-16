# RoboCOP Roadmap

## Phase 0 — Reproducible baseline

- Reproduce PD/PID Humanoid baselines with fixed seeds.
- Port the full field controller as the reference implementation.
- Record survival, reward, reward/step, torque proxy, energy proxy, CPU time, and virtual simulations per step.

## Phase 1 — Hierarchical field control

- Port Z1/Z2/Z3 progressive-resolution controllers.
- Escalate computation from coarse sensing to deeper field evaluation based on uncertainty.
- Track per-layer utilization and Pareto trade-offs between performance, energy, and compute.

## Phase 2 — Descriptive field memory

- Persist field/trajectory memory across episodes.
- Store direction, dispersion, confidence, energy, reward, visit count, and trajectory outcome.
- Resolve states progressively through coarse-to-fine descriptions.
- Compare frozen-memory evaluation on unseen seeds against full field evaluation.

## Phase 3 — Trajectory memory

- Replace single averaged directions with multiple trajectory prototypes per descriptive region.
- Select trajectories using compatibility and measured quality instead of only mean gradient.
- Optimize stability + reward - energy under controlled benchmarks.

## Phase 4 — Locomotion skills

- Standing balance.
- Recovery from perturbations.
- Walk to target.
- Stop and hold.
- Turn in place and while walking.
- Fall detection and recovery.

## Phase 5 — Robot-soccer adaptation

- Introduce a simulation adapter compatible with a RoboCup-oriented humanoid environment.
- Add localization/perception interfaces.
- Add ball approach, kick, dribble, and recovery skills.
- Build behavior selection above the low-level controller.

## Scientific discipline

Every performance claim must identify the environment version, seed set, parameters, baseline, raw outputs, aggregate statistics, and whether the memory/controller was trained or tuned on the evaluation seeds. Negative results remain part of the experimental record.
