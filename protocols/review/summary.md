# Campaign review summary

Generated deterministically (2026-07-27T00:00:00Z). Each protocol is identified by its content SHA-256. Conditioning numbers come from simulated position-tracked playback on the nominal model: the 14-parameter friction block (frictionloss + damping, all joints) must be identifiable (ratio >= 1e-06) with every per-joint frictionloss/damping correlation below 0.98.

| protocol | duration [s] | samples | conditioning ratio | worst fl/damping corr | content sha256 |
| --- | --- | --- | --- | --- | --- |
| fer-friction-a | 63.3 | 6332 | 2.41e-02 | 0.808 | `513414de362775fc...` |
| fer-friction-b | 62.8 | 6282 | 2.47e-02 | 0.816 | `11ce877687e43c0b...` |
| fer-friction-holdout | 37.7 | 3772 | 2.85e-02 | 0.833 | `ec3bd11e2447ec21...` |
| fer-inertial-a | 21.0 | 2101 | 1.15e-01 | 0.812 | `754dbee1dd3c2e9b...` |
| fer-inertial-b | 21.0 | 2101 | 7.16e-02 | 0.807 | `1fd7e298b6529038...` |
| fer-inertial-holdout | 21.0 | 2101 | 1.06e-01 | 0.867 | `394bdab92ec4af71...` |
