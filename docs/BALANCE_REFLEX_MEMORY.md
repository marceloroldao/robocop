# Balance Reflex Memory

A balance reflex is not a static posture. It is a mapping from body configuration plus motion tendency to a corrective action that historically improved balance.

We represent a reflex as:

`R = (S_t, dS_t, a_t, S_{t+1}, dB, E)`

where `S_t` is the sensor state, `dS_t` captures the direction and rate of motion, `a_t` is the corrective action, `dB` is the observed balance improvement, and `E` is the action energy proxy.

Lookup therefore distinguishes states that look similar geometrically but are moving in different directions. A forward-falling state and a backward-falling state should not share the same reflex even if their joint positions are similar.

The initial implementation indexes coarse body state `(height, vertical, omega, vel_z)` together with joint position/velocity summaries and returns a stored corrective action only when historical balance gain is positive and the query is sufficiently close.

This is treated as an experimental control primitive and is benchmarked against V6.1.