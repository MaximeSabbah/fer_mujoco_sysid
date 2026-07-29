# Campaign review summary

Generated deterministically (2026-07-27T00:00:00Z). Each canonical bundle is identified by its content SHA-256. Conditioning numbers come from simulated position-tracked playback on the nominal model: the 14-parameter friction block (frictionloss + damping, all joints) must be identifiable (ratio >= 1e-06) with every per-joint frictionloss/damping correlation below 0.98.

| protocol | duration [s] | samples | conditioning ratio | worst fl/damping corr | content sha256 |
| --- | --- | --- | --- | --- | --- |
| fer-friction-a | 41.9 | 4191 | 2.38e-02 | 0.728 | `8fd5b52d6c4d4c6d...` |
| fer-friction-b | 41.6 | 4160 | 2.65e-02 | 0.733 | `fa4089b515d1a657...` |
| fer-friction-holdout | 32.1 | 3210 | 3.00e-02 | 0.832 | `fcb243ebe006319c...` |
| fer-inertial-a | 21.0 | 2101 | 7.45e-02 | 0.805 | `c4d834bf7d9e0961...` |
| fer-inertial-b | 21.0 | 2101 | 6.69e-02 | 0.802 | `e196760e33fa45c5...` |
| fer-inertial-holdout | 21.0 | 2101 | 8.91e-02 | 0.866 | `e53a12f8da71a81c...` |
