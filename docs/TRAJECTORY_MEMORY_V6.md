# V6 — Hierarchical Trajectory Memory

V6 replaces the single averaged gradient per descriptive region with multiple trajectory prototypes.

## Motivation

V5 achieved a very high memory hit rate and a large reduction in finite-difference simulations, but coarse Z1 regions could collapse physically different situations into one averaged gradient. That preserved survival while degrading torque/energy efficiency.

## Design

Each Z1/Z2/Z3 node stores up to four trajectory prototypes. A new observation is merged only with a prototype whose gradient cosine similarity exceeds the merge threshold. Otherwise a new branch is created.

The controller uses a coarse node only when its directional ambiguity is below a configured limit. Ambiguous Z1 nodes force resolution into Z2; ambiguous Z2 nodes force Z3.

Each trajectory prototype stores gradient, visits, confidence, mean energy, mean reward and mean survival. Selection uses a quality score that rewards survival/reward and penalizes energy.

## Expected benchmark behavior

Compared with V5, V6 should trade some Z1 cache hits for more Z2/Z3 hits while reducing torque and energy. The target is to preserve most of V5's computational saving without its energy regression.

## Validation

Core isolated tests verify:

1. Opposing trajectories in the same Z1 region are stored as separate prototypes and force deeper resolution.
2. When multiple eligible prototypes share a region, the lower-energy trajectory wins when reward and survival are equal.
3. Frozen memory is immutable.

The physical MuJoCo benchmark is available in `benchmarks/humanoid_balance.py` and must be run in a MuJoCo-capable environment.