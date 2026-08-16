# V6.1 — Realized Outcome Credit

## Problem found in V6

The first V6 benchmark implementation stored the PD baseline energy before the field correction and used the instantaneous field score as a proxy for reward/survival. Therefore the trajectory memory could not reliably learn which remembered branch actually produced low-energy behavior.

## V6.1 correction

V6.1 stores a pending `(state, gradient, executed action)` tuple and assigns the next MuJoCo transition outcome back to that trajectory after `env.step()`.

The memory now receives:

- actual executed-action energy `mean(action**2)`;
- actual environment reward from the next transition;
- realized one-step survival (`1` when the transition continues, `0` on termination/truncation);
- the gradient/trajectory that generated that action.

Trajectory quality also normalizes reward and bounds the energy penalty so the raw reward scale does not dominate energy selection.

## Local core validation

Three isolated tests pass:

1. realized energy is computed from the executed action, not the PD baseline;
2. terminal outcomes receive zero survival credit;
3. for equal reward and survival, energy `0.004` is preferred over `0.020`.

The full physical effect must still be measured in a MuJoCo-capable Colab/runner using `benchmarks/humanoid_balance.py`.
