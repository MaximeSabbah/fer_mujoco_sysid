# Campaign review summary

Generated deterministically (2026-07-27T00:00:00Z, revision r2). Conditioning numbers come from simulated position-tracked playback on the nominal model: the 14-parameter friction block (frictionloss + damping, all joints) must be identifiable (ratio >= 1e-06) with every per-joint frictionloss/damping correlation below 0.98.

| protocol | duration [s] | samples | conditioning ratio | worst fl/damping corr | content sha256 |
| --- | --- | --- | --- | --- | --- |
| fer-friction-a | 41.9 | 4191 | 2.38e-02 | 0.728 | `20c6983cce21a326...` |
| fer-friction-b | 41.6 | 4160 | 2.65e-02 | 0.733 | `b5f1725def58fbad...` |
| fer-friction-holdout | 32.1 | 3210 | 3.00e-02 | 0.832 | `048a61c1b0ccfd60...` |
| fer-inertial-a | 21.0 | 2101 | 7.45e-02 | 0.805 | `8461a18c2e9fe6c3...` |
| fer-inertial-b | 21.0 | 2101 | 6.69e-02 | 0.802 | `c9f5209d038c9501...` |
| fer-inertial-holdout | 21.0 | 2101 | 8.91e-02 | 0.866 | `f37365ef584fbfbb...` |
