# RoboCOP

**RoboCOP** is an experimental robotics-control research project focused on efficient humanoid control, hierarchical sensing, field-guided control, and persistent descriptive memory.

The project is currently a research prototype. Its initial benchmark environment is MuJoCo/Gymnasium Humanoid. A longer-term research goal is to evaluate whether the architecture can be adapted to humanoid robot soccer simulation environments, including RoboCup 3D-style tasks.

## Research direction

RoboCOP investigates a controller stack built from:

- conventional PD/PID baselines;
- a continuous field-guided control layer;
- hierarchical sensing and progressive resolution (Z1/Z2/Z3);
- uncertainty-driven escalation to deeper computation;
- persistent descriptive memory of previously explored state regions;
- trajectory reuse to reduce repeated virtual simulations;
- performance/energy/compute Pareto optimization.

The core experimental question is whether expensive local field evaluation can be progressively replaced by structured memory while preserving stability and control quality.

## FC Portugal competitive baseline

RoboCOP now includes a license-neutral bridge protocol for experiments against the public FC Portugal RoboCup 3D codebase. FC Portugal remains an external GPL-3.0 dependency and is not copied into the RoboCOP core. The first integration target is the upstream Walk/Step locomotion stack: preserve its trained walking policy as the baseline and evaluate resolutive transition memory/reflexes only as measured overlays.

See `docs/fcportugal_integration_audit.md` and `scripts/fetch_fcportugal_external.sh`.

## Current experimental milestones

Early MuJoCo experiments have compared PD, full field evaluation, hierarchical approximations, and persistent descriptive memory. Results must be treated as experimental benchmarks rather than claims of new physical laws or real-robot performance.

## Repository structure

```text
controllers/     Control algorithms and baselines
memory/          Descriptive and trajectory-memory systems
environments/    MuJoCo and future simulation adapters
experiments/     Reproducible experiment suites
benchmarks/      Benchmark definitions and comparison tools
results/         Machine-readable experimental outputs
docs/            Design notes, methodology, and roadmap
```

## Reproducibility policy

Experiments should record fixed seeds, environment/version information, controller parameters, raw CSV results, aggregate statistics, and negative results. Claims should be tied to reproducible experiments.

## Licensing

This repository uses a dual-purpose research/commercial licensing model:

- academic, educational, and non-commercial research use is permitted under the terms in `LICENSE`;
- commercial use, commercial deployment, paid products/services, or proprietary integration requires a separate commercial license; see `COMMERCIAL_LICENSE.md`.

This licensing model is **source-available and is not intended to be an OSI-approved open-source license**.

External dependencies retain their own licenses. In particular, the FC Portugal codebase is GPL-3.0 and is intentionally kept outside the RoboCOP core.

## Status

Research preview. Interfaces, algorithms, benchmarks, and file formats may change substantially before a stable release.

Copyright (c) 2026 Marcelo Roldão Matos.