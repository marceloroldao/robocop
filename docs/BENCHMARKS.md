# Benchmark protocol

## Humanoid balance suite

Run locally or in Colab with:

```bash
pip install -e '.[sim]'
python benchmarks/humanoid_balance.py --seeds 20 --train-seeds 30 --max-steps 500
```

The benchmark compares four controller families on identical held-out seeds:

1. PD baseline.
2. Full finite-difference resolutive field (34 virtual simulations per decision for 17 actuators).
3. Hierarchical confidence controller derived from V3.
4. Persistent descriptive-memory controller derived from V5.

Reported metrics are survival steps, total reward, reward per step, mean absolute torque, energy proxy, controller CPU time, virtual simulations per step, and memory hit rate.

## Reproducibility rules

- Training seeds and test seeds must not overlap.
- Memory is frozen before blind evaluation.
- All controllers in an evaluation batch receive the same test seeds.
- Negative results are retained.
- Do not compare CPU figures across materially different machines as if they were absolute performance measurements; use them mainly for within-run relative comparisons.

## Current validation status

The simulator-independent core and adapter math are covered by unit tests. Full MuJoCo benchmark execution requires the optional `sim` dependencies and must be run on a MuJoCo-capable runner or Colab environment. Results should be committed under `results/` only after the exact benchmark command and environment metadata are recorded.
