# FC Portugal integration audit

## Decision

Use FC Portugal as an **external GPL-3.0 baseline** and keep RoboCOP's source-available core legally and technically separate. Do not copy FC Portugal source files, trained walk models, or modified FC Portugal modules into the RoboCOP core.

The integration boundary is a small wire protocol (`robocop.fcp_protocol`) carrying sensor frames from the FC Portugal side to RoboCOP and optional resolutive directives back.

## Upstream audited

- Repository: `m-abr/FCPCodebase`
- Language: mainly Python, with C++ modules
- License: GNU GPL v3.0
- Official RoboCup 3D release: FC Portugal 2023 codebase

## Architecture map

### 1. Agent/world layer

`agent/Base_Agent.py` builds the world model, parser, server communication, inverse kinematics, behavior manager, path manager and radio. This is the best place for a GPL-side adapter to **read** a coherent sensor/world snapshot.

### 2. High-level soccer decision

`agent/Agent.py::think_and_send()` performs preprocessing, chooses high-level behavior, broadcasts and sends commands. RoboCOP should not replace this first. It is a later experimental hook for resolutive skill selection and team strategy.

### 3. Behavior router

`behaviors/Behavior.py::execute()` routes and resets skills. This is a clean later hook for a `behavior_guard` directive, for example preventing a risky transition or selecting a previously validated skill transition.

### 4. Walk / Step locomotion

`behaviors/custom/Walk/Walk.py` converts target/orientation into an observation, evaluates a trained MLP policy, then executes the action. `Step` is a lower-level primitive used by Walk and Dribble.

**First integration target:** keep the upstream Walk policy untouched and let RoboCOP observe transitions around it. Only when a high-confidence memory match exists should RoboCOP return a small `reflex_blend` correction. The baseline action remains authoritative.

### 5. Fall / Get_Up

The codebase already has `Fall` and `Get_Up`. RoboCOP should use these initially as ground-truth outcome labels: a transition leading into a fall becomes a negative memory; a transition that avoids the fall becomes a candidate reflex.

## Resolutive mapping

The first FC Portugal experiment should map the existing RoboCOP ideas as follows:

- **address**: signed torso orientation + signed angular velocity + joint configuration + joint velocity + foot/contact information + current skill;
- **trajectory**: `(state_t, baseline_action_t, state_t+1)`;
- **good rail**: transition that improves balance while preserving the requested walk intent;
- **bad rail**: transition that approaches a fall or forces Get_Up;
- **Z1**: torso direction/rates + active behavior;
- **Z2**: Z1 + selected leg/hip/knee/ankle joint state;
- **Z3**: full joint state + contacts + walk target;
- **resolution escalation**: deeper lookup only when the coarse trajectory is ambiguous;
- **reflex**: a small correction blended onto the upstream policy action, never a blind replacement in phase 1.

## Licensing boundary

FC Portugal is GPL-3.0. RoboCOP currently uses a custom research/commercial source-available license. To preserve RoboCOP's independent licensing, the recommended architecture is process-level separation:

```text
FCPCodebase (GPL-3.0)
    |
    | JSON/IPC sensor frame
    v
RoboCOP resolutive service (RoboCOP license)
    |
    | JSON/IPC directive
    v
FCP-side GPL adapter applies directive to its own action
```

A GPL-side adapter that imports or modifies FC Portugal should live with the FC Portugal checkout (or in a separately GPL-licensed adapter project), not inside the RoboCOP core.

This document is an engineering/licensing design note, not legal advice.

## Experiment sequence

1. **Pass-through validation** — FC Portugal action goes through unchanged; verify zero behavioral difference.
2. **Recorder only** — store signed sensor transitions and fall/get-up outcomes; no control modification.
3. **Offline memory** — measure whether good/bad transition clusters separate and estimate hit rate at Z1/Z2/Z3.
4. **Reflex shadow mode** — compute what RoboCOP *would* do but do not apply it; measure disagreement and confidence.
5. **Reflex blend A/B test** — apply only high-confidence corrections and compare against untouched FC Portugal on identical seeds/match scenarios.
6. Only after locomotion wins should resolutive logic be tested at behavior selection or team strategy level.

## Required metrics

- fall rate / Get_Up count;
- distance walked before fall;
- target tracking error;
- torso roll/pitch RMS and peak;
- energy/action proxy;
- bridge latency and CPU cost;
- memory hit rate by Z-level;
- fraction of upstream actions modified;
- performance with the bridge in pure pass-through mode (must match baseline).

## Scientific rule

FC Portugal remains the baseline. A resolutive modification is retained only if it beats or preserves the upstream controller under the same scenarios while adding an explicit measurable advantage (stability, compute, energy, transition quality, or match performance).
