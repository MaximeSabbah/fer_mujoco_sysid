# Friction campaign review summary

Generated deterministically (2026-07-27T00:00:00Z, revision r1). Conditioning numbers come from simulated position-tracked playback on the nominal model: the 14-parameter friction block (frictionloss + damping, all joints) must be identifiable (ratio >= 1e-06) with every per-joint frictionloss/damping correlation below 0.98.

| protocol | duration [s] | samples | conditioning ratio | worst fl/damping corr | content sha256 |
| --- | --- | --- | --- | --- | --- |
| fer-friction-a | 41.9 | 4191 |  |  | `cc8761971aed97f2...` |
| fer-friction-b | 41.6 | 4160 |  |  | `7e87f88317cf8ca3...` |
| fer-friction-holdout | 32.1 | 3210 |  |  | `00a29fb37db05996...` |
| fer-inertial-a | 21.0 | 2100 |  |  | `a695fc49835a3086...` |
| fer-inertial-b | 21.0 | 2100 |  |  | `74b76bf1a18627ea...` |
| fer-inertial-holdout | 19.2 | 1918 |  |  | `f0019cb2a20c4468...` |
