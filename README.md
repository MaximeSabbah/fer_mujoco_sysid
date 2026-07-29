# FER MuJoCo system identification

Identify the dynamics of a Franka Research 3 (FER) — joint friction,
damping, armature and link inertias — directly as MuJoCo model parameters,
and hand the result to the controllers that consume the model.

Python 3.12 and [`uv`](https://docs.astral.sh/uv/); playback additionally
needs a sourced ROS 2 Jazzy environment.

```bash
uv sync --locked --all-groups
```

---

# Walkthrough

The stages in the order they make sense to look at. Everything here runs in
simulation — the robot has not moved yet — but every stage runs the same code
that will run on the robot; only the hardware behind the interface differs.

```
1. watch the motions        ./scripts/run-protocol <id> --watch
2. record a campaign        ./scripts/run-protocol <id> --record
3. identify from it         ./scripts/identify
4. read the result          output/mujoco_ros/identification/
```

Stages 2–4 are the chain a real campaign runs; stage 1 is the same command
without `--record`. Adding `--backend real --robot-ip <ip>` is the only
difference on hardware.

**Everything generated lands under `output/`, partitioned by where the data
came from** — and nothing in it is committed, because it is all reproducible:

| | |
| --- | --- |
| `output/mujoco/` | the standalone in-process pipeline, no ROS involved |
| `output/mujoco_ros/` | this ROS stack driving a simulated plant |
| `output/real/` | the robot |

Protocol review material lives with the protocols it describes, in
`protocols/review/`.

`identify` **refuses to fit across more than one of them**. A model built
from a mix of real and simulated recordings would describe no particular
machine, and that is not a mistake worth being able to make.

## 1. Watch the exciting motions

```bash
./scripts/run-protocol fer-friction-a --watch
```

`--watch` opens **both** views of the same run: the **MuJoCo viewer**, which
is the physics actually being simulated, and **RViz**, which draws the robot
description from the measured joint states — the view you also get on the
real robot. `--viewer` or `--rviz` for just one.

Six protocols to look at:

| Protocol | What it is | Duration |
| --- | --- | --- |
| `fer-friction-a`, `-b` | slow bidirectional sweeps: constant-velocity cruises at 0.05 / 0.15 / 0.4 rad/s, holds at the stops | ~42 s |
| `fer-friction-holdout` | same idea, different amplitudes and speeds — never used for fitting | ~32 s |
| `fer-inertial-a`, `-b` | per-joint Fourier series at distinct frequencies, up to 2.0 rad/s and 16 rad/s² | ~21 s |
| `fer-inertial-holdout` | different frequency mix — never used for fitting | ~21 s |

Every run begins with a slow **move to the protocol start** (0.2 rad/s cap),
so the first two seconds are the approach, not the excitation.

What is worth judging, and only you can judge it:

* does the motion fit your cell — does the arm stay where you expect;
* the inertial motions are much livelier than the friction ones. Acceptable?
* the slowest friction cruise, 0.05 rad/s, is the near-standstill regime
  that produced the 62 mm pregrasp error. Slow enough?

Changing any of this is still cheap: edit the specs in
[`src/fer_mujoco_sysid/campaign.py`](src/fer_mujoco_sysid/campaign.py), bump
`CAMPAIGN_REVISION`, regenerate. Once a hardware campaign has been recorded
against them, it is not.
[`protocols/README.md`](protocols/README.md) explains
*why* each family exists and why one family cannot do both jobs.

## 2. Record a campaign

```bash
./scripts/run-protocol fer-friction-a       --record
./scripts/run-protocol fer-friction-b       --record
./scripts/run-protocol fer-friction-holdout --record
```

Each run lands in `output/mujoco_ros/<protocol>/`:

| | |
| --- | --- |
| `raw/` | the MCAP bag, immutable — everything else is derived from it |
| `recording/` | a checksummed NumPy bundle, which is all the fit ever reads |

Recording is a plain `ros2 bag record` subscriber; it never touches the
control path. What lands in the bundle, and why:

| Channel | From | Used for |
| --- | --- | --- |
| `tau_cmd_Nm` | the controller's own `output.effort` | **the fit** — what actually drove the joint, known exactly rather than inferred |
| `q_rad`, `dq_rad_s` | the same message's `feedback` | the fit, on one clock with the torque |
| `q_desired_rad` | the same message's `reference` | tracking quality — what the controller was *aiming* at |
| `tau_J`, `tau_J_d`, `theta`, `dtheta` | Franka telemetry (hardware only) | not fitted; recorded because re-running the robot to recover a channel we chose not to record is the expensive mistake |

Torque and state come from **one message**, so the torque is never
interpolated against the velocity — that relationship is exactly what
friction identification reads. The controller publishes at 500 Hz, which is
the model's own timestep.

Every conversion prints a health verdict and **refuses** a run that did not
complete, has a gap too long to interpolate across, or sat at a torque limit.

For hardware: add `--backend real --robot-ip <ip>`, and `--speed-scale 0.5`
for the first run of each family.

## 3. Identify from the recordings

```bash
./scripts/identify                    # fits output/mujoco_ros/
./scripts/identify --backend real     # fits output/real/
```

Two stages, and the report shows both:

1. **Classical regressor solve** — `tau - tau_rigid = frictionloss*sign(dq) +
   damping*dq`, solved directly. Milliseconds. Gives cond(Y) and the relative
   standard deviations, and seeds stage 2 close enough to converge in one
   iteration instead of dozens.
2. **Rollout refinement** — matches simulated to measured trajectories, which
   is the objective that matters once the model class stops being exact.

Their **disagreement is reported as a diagnostic**: on data whose friction is
inside MuJoCo's model class they agree to about 1%. A large gap on the real
robot means the class is wrong (Stribeck, temperature, transmission) and is a
reason to stop rather than a number to publish.

### What to look at, in order

Everything lands in `output/mujoco_ros/identification/`.

1. **`rollout_vs_measurement.png`** — start here. The simulator started from a
   measured state, driven with the recorded torque, run open loop against what
   the arm did. Where the coloured line leaves the black one, the model is
   wrong. This is the acceptance metric, and the only one that still exists on
   hardware where no true parameter value does.
2. **`error_vs_horizon.png`** — how fast that agreement decays. A model can be
   excellent over 0.1 s and useless over 2 s; a planner cares about the latter.
3. **`result.md`** — the numbers: cond(Y), per-joint parameters with σ%, the
   held-out table at three horizons for nominal / classical / refined, and the
   export diff. Read the parameters against the breakaway torques you measured
   on the real robot — that is the sanity check no metric replaces.
4. **`friction_curves.png`** — the classical picture: the step at zero
   velocity is Coulomb friction, the slope is viscous. The nominal model has
   no step at all, which is the whole problem.
5. **`torque_tracking.png`** — recorded commanded torque against what each
   model reconstructs.
6. **`../<protocol>/replay.mp4`** — what the arm actually did, with a
   translucent ghost of what the controller asked for. Where the ghost pulls
   ahead, the arm did not follow.
7. **`../<protocol>/tracking.png`** — the same as numbers. This judges the
   *run*, not the model: if the arm did not follow the protocol, the data
   describes a motion nobody designed.

## 4. Read the standalone demo

```bash
./scripts/identification-demo                            # ~10 min: videos, fit, report
./scripts/identification-demo --reuse-fit --skip-videos  # report only, seconds
```

A self-contained run that never touches ROS: it plays the protocols in one
process against a hidden-truth model and fits from arrays in memory. Useful
as the reference the recorded chain is judged against, and much faster to
iterate on. The result is
`output/mujoco/result.md`,
in the order a classical identification is normally judged:

1. **Excitation quality** — condition number of the regressor per joint,
   measured *before* fitting. 1 is ideal; above ~100 the motion cannot
   separate the parameters. Currently 2.43 (friction), 2.75 (inertial).
2. **Preprocessing** — zero-phase 4th-order Butterworth at 8 Hz, applied
   identically to positions, velocities and torques.
3. **Identified parameters** with their **relative standard deviations**.
   The classical acceptance rule: a parameter whose σ% exceeds 20 % is not
   identified by this data and should be frozen rather than reported.
4. **Held-out validation** — a protocol the fit never saw, in physical
   units: rad per joint, millimetres at the gripper.
5. **The identified model** and the checks it had to pass (stage 5).
6. **Plots**:
   * `torque_tracking_friction.png` — measured joint torque against what
     each model reconstructs. This is where friction is visible;
   * `torque_tracking_inertial.png` — the same on an inertial protocol,
     where only friction has been fitted; the residual left there is what
     armature and inertial identification still has to remove;
   * `torque_residuals.png` — per-joint reconstruction error;
   * `friction_curves.png` — friction torque against velocity at the
     cruises: the dry-friction step at zero, and the viscous slope.

**What this proves and does not prove** is stated at the bottom of the
report, and it matters: the hidden truth differs from the model only in
parameters MuJoCo can express, and the plant is the same simulator the fit
uses. It demonstrates the machinery works. It says nothing yet about your
robot.

## 5. Take the identified model

The demo writes two files:

| File | What it is |
| --- | --- |
| `output/mujoco/fer_identified.xml` | the nominal MJCF with the identified parameters written in |
| `output/mujoco/fer_identified.json` | manifest: both files by SHA-256, which fields were exported, and their values |

The model is not written unless it passes two checks
([`export.py`](src/fer_mujoco_sysid/export.py)):

* **only exportable fields changed.** Joint friction, damping and armature
  are identification outputs. Masses, inertias, link poses, joint ranges,
  actuator limits and geometry are compared against the nominal model and
  must be identical — a change there is a different robot, not a better
  estimate of this one. The test suite proves the check bites, by tampering
  with a link mass and asserting the export is rejected;
* **the written file behaves like the fitted model.** Reloaded from disk and
  run open-loop over the held-out protocol, it reproduces the in-memory
  model's per-joint RMSE to 2.8e-07 rad.

`source_model.sha256` in the manifest pins which nominal model was
identified, so a candidate model can always be traced back to the fit that
produced it.

**Validating it.** The report's held-out section *is* the validation, in the
terms the acceptance rule fixes: open-loop prediction on a protocol the fit
never saw, in rad and millimetres, identified against nominal. On the
simulated run the gripper prediction improves from 68.8 mm RMSE to 0.52 mm.
The final validation is the original problem — the pregrasp task with the
identified model against the 62 mm baseline — which belongs to the consumer
repositories and is a separate reviewed change.

## What is not here yet

This chain identifies a **simulated** robot. Two things stand between it and
identifying yours:

1. **Recording.** Stage 1 plays protocols but does not save them. The
   recorder and the converter to a fitting dataset are the next slice.
2. **Real data.** Once recordings exist, stages 2 and 3 take them unchanged
   — which is the point of running the same code path in simulation first.

One consequence is already known and constrains the recorder: under a torque
command the robot compensates its own gravity, so **commanded effort cannot
identify link masses** — the gravity torque that reveals them is not in it.
Mass and centre of mass must come from the measured link-side `tau_J`
channel, which the Franka telemetry broadcaster publishes. Friction, damping
and armature are unaffected.

---

# Reference

## How it works

The fitting backend is MuJoCo's own `mujoco.sysid` toolbox, which does **not**
use the classical regressor least-squares formulation. It simulates short
windows with candidate parameters and minimizes the difference between
simulated and measured joint trajectories. That avoids differentiating
measured positions twice, but it also means the classical diagnostics do not
come for free — so this project builds them explicitly (`diagnostics.py`):
regressor condition numbers, relative standard deviations, and torque
reconstruction.

```
excitation.py   generate exciting motions (friction + inertial families)
     |          validated against FER limits before anything is written
     v
protocols/      committed, immutable, content-hashed motion bundles
     |
     v
playback.py     time scaling, move-to-start, preflight  --> ros/player_node.py
     |                                                      plays them in ROS
     v                                                      simulation or on
preprocessing.py  zero-phase Butterworth filtering          the robot
     |
     v
fitting.py      staged parameter fit through mujoco.sysid
     |
     v
diagnostics.py  cond(Y), sigma%, torque residuals
     |
     v
report.py       plots, videos, the written result
     |
     v
export.py       the identified MJCF, with its whitelist and provenance checks
```

## Excitation families

Different parameters need different motion, and one family cannot serve both:

| Family | Motion | Identifies |
| --- | --- | --- |
| **friction** | jerk-limited trapezoids: constant-velocity cruises at several low speeds, both directions, holds at the stops, all joints on one shared schedule | `frictionloss`, `damping` — friction is read directly off the cruise plateaus, where inertia contributes nothing |
| **inertial** | per-joint Fourier series at *different* base frequencies, scaled to the kinematic limits, selected for regressor conditioning | `armature`, link inertias — these need joints moving **independently** and accelerating hard, which the friction family deliberately does not do |

## Playing a protocol

`run-protocol` sources ROS and generates the simulation plant itself.

```bash
./scripts/run-protocol fer-friction-a                      # simulated
./scripts/run-protocol fer-friction-a --watch              # MuJoCo viewer + RViz
./scripts/run-protocol fer-friction-a \
    --backend real --robot-ip 172.16.0.2 --dry-run         # checks only, no motion
./scripts/run-protocol fer-friction-a \
    --backend real --robot-ip 172.16.0.2 --speed-scale 0.5 # half speed
```

`--backend` changes the hardware plugin and whether the Franka telemetry
broadcaster is spawned; everything else is the same code. The player verifies
the bundle's checksums and content hash, checks that the trajectory
controller is active and owns all seven joints **in effort mode**, checks
robot mode and error flags on hardware, moves to the protocol start under a
slow speed cap, verifies it arrived, and only then plays. Ctrl-C cancels the
trajectory.

`--speed-scale s` replays the identical path over `1/s` times as long:
velocities scale by `s`, accelerations by `s²`, positions are untouched, so
the workspace clearance validated at generation time still holds exactly.
Values above 1 are refused.

Effort mode is a requirement, not a convenience: friction is exactly the term
that changes with actuation mode, and the deployment stack drives this robot
in effort mode. Identifying under the robot's internal position controller
and deploying under torque control would measure the wrong friction.

## Layout

| Path | What it holds |
| --- | --- |
| `src/fer_mujoco_sysid/model.py` | builds the 7-axis identification model from the hydrax MJCF and verifies its hash |
| `src/fer_mujoco_sysid/excitation.py` | motion generation and limit validation for both families |
| `src/fer_mujoco_sysid/protocol.py` | the bundle format and its reader — NumPy only, so it runs under ROS |
| `src/fer_mujoco_sysid/playback.py` | time scaling, the move to the protocol start, and the preflight checks |
| `src/fer_mujoco_sysid/ros/` | the ROS player and the generated simulation plant |
| `src/fer_mujoco_sysid/campaign.py` | the committed campaign: which protocols exist, and their seeds |
| `src/fer_mujoco_sysid/preprocessing.py` | filtering and differentiation of measured signals |
| `src/fer_mujoco_sysid/fitting.py` | the `mujoco.sysid` adapter: parameter groups, staged fits, conditioning |
| `src/fer_mujoco_sysid/diagnostics.py` | classical identification metrics |
| `src/fer_mujoco_sysid/export.py` | writing identified parameters back into a model, with its checks |
| `src/fer_mujoco_sysid/report.py` | the end-to-end simulated run, its plots and videos |
| `src/fer_mujoco_sysid/io.py` | JSON/NPZ artifacts, content hashes, checksums |
| `protocols/` | committed motion bundles (manifest + arrays + checksums) |
| `config/`, `launch/`, `urdf/` | the FER controller configuration and the playback bringup |
| `contracts/nominal_sources.toml` | pins the baseline MJCF by commit and SHA-256 |
| `docs/` | the implementation plan and the generated review material |

## Every command

```bash
./scripts/test                # fast suite (~20 s)
./scripts/test-all            # every gate, including the multi-minute fits
./scripts/generate-protocols  # rebuild protocols/ and the review plots
./scripts/generate-protocols --check   # verify committed bundles still regenerate
./scripts/identification-demo # videos, the simulated identification, the model
./scripts/run-protocol <id>   # play one protocol, simulated or on the robot
```

## Status

The robot has not moved yet. Everything below runs in simulation.

| Stage | State |
| --- | --- |
| Fitting engine, proven on synthetic data with known hidden values | done |
| Excitation protocols for both families | done — r2 |
| End-to-end simulated identification with full diagnostics | done |
| Identified model exported, whitelist- and reload-checked | done |
| Playback: controller config, player, preflight, bringup | done |
| Playback rehearsed in ROS simulation | done — all six protocols |
| Recording and conversion to a dataset | not started |
| Playback on the real robot | not started |
| Identified model delivered to hydrax / `sbmpc_ros` | not started |

The [implementation plan](docs/implementation_plan.md) holds the roadmap, the
gates, and the decision log.
