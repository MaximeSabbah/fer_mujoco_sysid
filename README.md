# FER MuJoCo system identification

Identify the dynamics of a Franka Research 3 (FER) — joint friction,
damping, armature and link inertias — directly as MuJoCo model parameters,
and hand the result to the controllers that consume the model.


## How it works

The fitting backend is MuJoCo's own `mujoco.sysid` toolbox, which does **not**
use the classical regressor least-squares formulation. It simulates short
windows with candidate parameters and minimizes the difference between
simulated and measured joint trajectories. That avoids differentiating
measured positions twice, but it also means the classical diagnostics do not
come for free — so this project builds them explicitly
(`diagnostics.py`): regressor condition numbers, relative standard
deviations, and torque reconstruction.

```
excitation.py   generate exciting motions (friction + inertial families)
     |          validated against FER limits before anything is written
     v
protocols/      committed, immutable, content-hashed motion bundles
     |
     v          play in MuJoCo, ROS simulation, or on the robot
preprocessing.py  zero-phase Butterworth filtering of the recordings
     |
     v
fitting.py      staged parameter fit through mujoco.sysid
     |
     v
diagnostics.py  cond(Y), sigma%, torque residuals
     |
     v
report.py       plots, videos, and the written result
```

## Layout

| Path | What it holds |
| --- | --- |
| `src/fer_mujoco_sysid/model.py` | builds the 7-axis identification model from the hydrax MJCF and verifies its hash |
| `src/fer_mujoco_sysid/excitation.py` | motion generation and limit validation for both families |
| `src/fer_mujoco_sysid/campaign.py` | the committed campaign: which protocols exist, and their seeds |
| `src/fer_mujoco_sysid/preprocessing.py` | filtering and differentiation of measured signals |
| `src/fer_mujoco_sysid/fitting.py` | the `mujoco.sysid` adapter: parameter groups, staged fits, conditioning |
| `src/fer_mujoco_sysid/diagnostics.py` | classical identification metrics |
| `src/fer_mujoco_sysid/report.py` | the end-to-end simulated run, its plots and videos |
| `src/fer_mujoco_sysid/io.py` | JSON/NPZ artifacts, content hashes, checksums |
| `protocols/` | committed motion bundles (manifest + arrays + checksums) |
| `contracts/nominal_sources.toml` | pins which hydrax/`sbmpc_ros` MJCF is the baseline, by commit and SHA-256 |
| `docs/` | the implementation plan and the generated review material |

## Excitation families

Different parameters need different motion, and one family cannot serve both:

| Family | Motion | Identifies |
| --- | --- | --- |
| **friction** | jerk-limited trapezoids: constant-velocity cruises at several low speeds, both directions, holds at the stops, all joints on one shared schedule | `frictionloss`, `damping` — friction is read directly off the cruise plateaus, where inertia contributes nothing |
| **inertial** | per-joint Fourier series at *different* base frequencies, scaled to the kinematic limits, selected for regressor conditioning | `armature`, link inertias — these need joints moving **independently** and accelerating hard, which the friction family deliberately does not do |

## Usage

Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-groups
./scripts/test                # fast suite (~20 s)
./scripts/test-all            # every gate, including multi-minute fits
./scripts/generate-protocols  # rebuild protocols/ and the review plots
./scripts/generate-protocols --check   # verify committed bundles still regenerate
./scripts/identification-demo # videos + the full simulated identification
```

## Status

Everything runs in simulation. The robot has not moved yet.

| Stage | State |
| --- | --- |
| Fitting engine, proven on synthetic data with known hidden values | done |
| Excitation protocols for both families | done |
| End-to-end simulated identification with full diagnostics | done — see `docs/identification_demo/result.md` |
| Playback in ROS simulation | not started |
| Playback on the real robot | not started |
| Identified model delivered to hydrax / `sbmpc_ros` | not started |

The [implementation plan](docs/implementation_plan.md) holds the roadmap,
the gates, and the decision log.
