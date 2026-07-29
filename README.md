# FER MuJoCo system identification

This repository identifies a Franka Research 3 model for MPPI. Its release
criterion is not merely a plausible parameter table: the identified MuJoCo
simulator must reproduce quantities measured from the plant over
MPPI-relevant rollout horizons.

The project is simulation-first. A complete campaign is run through both:

- `mujoco`: direct, in-process acquisition, which isolates the numerical
  identification path; and
- `mujoco_ros`: the same motions through the ROS controller, recording,
  conversion, timing, and identification path. This is the critical rehearsal
  for the real robot.

The real robot is the next milestone only after both simulation backends pass.

## Run the simulation gate

The project requires Python 3.12 and
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --locked --all-groups
./scripts/validate-simulation
```

The default command runs all six protocols on both backends, identifies each
backend independently, generates the operator evidence, validates against one
shared hidden truth, compares the backends, and writes a fresh timestamped run
under `output/simulation/`.

```text
./scripts/validate-simulation
  [--output NEW_DIRECTORY]
  [--backend all|mujoco|mujoco_ros]
  [--max-iters N]
  [--dynamic-starts N]
  [--no-media]
```

- The default `--backend all` is the release run. A single-backend run is
  useful for diagnosis but is not the complete handoff.
- `--output` must name a path that does not exist. Existing runs are never
  overwritten or silently reused.
- `--max-iters` and `--dynamic-starts` control optimization effort.
- `--no-media` is for fast diagnosis. Omit it for a deliverable run because
  videos and plots are required evidence.

The ROS backend expects ROS 2 Jazzy and the FER simulation dependencies. By
default `run-protocol` sources `/opt/ros/jazzy/setup.bash` and
`/opt/sbmpc_deps_ws/install/setup.bash`; override those locations with
`FER_SYSID_ROS_SETUP` and `FER_SYSID_WORKSPACE_SETUP`.

## What a run proves

One simulation run follows this chain:

```text
one immutable simulation truth
        |
        +-- direct MuJoCo: six recordings ----+
        |                                      |
        +-- MuJoCo through ROS: six bags ------+--> separate family fits
                                                       |
                                                       v
                                                held-out rollouts
                                                       |
                                                       v
                                           gravity-enabled consumer MJCF
                                                       |
                                                       v
                                      truth and cross-backend validation
```

The hidden plant differs from the nominal model in expressible friction,
armature, mass, centre-of-mass, and inertia coordinates. Both backends receive
the exact same checksummed `simulation_truth.json`; the fitter never reads
that truth. It is used only after fitting to answer two different questions:

1. Were the observable parameters recovered from known simulated data?
2. More importantly, does the exported simulator reproduce held-out
   positions, velocities, end-effector motion, and torque-equation behavior?

Simulation should close to near numerical precision. A fit that merely
improves over the nominal model is not enough for this simulation gate.
Parameter recovery is available only in simulation; held-out behavioral
reproduction is the metric that transfers to real data.

The aggregate run is release-ready only when both backends pass their strict
validation and agree with each other. `mujoco_ros` is the primary model path
for handoff because it exercises the acquisition route that will be used on
the robot.

## The two protocol families

There are exactly six canonical protocol IDs. Each lives directly at
`protocols/<protocol-id>/`; there are no `r2`, `r3`, or “latest” directories.
The bundle content hash and Git history provide identity and revision history.

| Protocol ID | Family | Role |
| --- | --- | --- |
| `fer-friction-a` | friction | training |
| `fer-friction-b` | friction | training |
| `fer-friction-holdout` | friction | validation only |
| `fer-inertial-a` | inertial | training |
| `fer-inertial-b` | inertial | training |
| `fer-inertial-holdout` | inertial | validation only |

The families are deliberately different and are not pooled into one generic
fit:

- The friction protocols provide low-speed, bidirectional,
  constant-velocity cruises. Their marked cruise windows identify Coulomb
  `frictionloss` and viscous `damping`; ramps, reversals, holds, and approach
  motion do not enter the cruise solve.
- The inertial protocols use independently excited Fourier motions with
  useful acceleration. They identify armature and the observable,
  CAD-bounded link mass, centre-of-mass, and inertia corrections.

Every candidate rigid-body coordinate is checked for sensitivity,
conditioning, correlation, and uncertainty. Coordinates the motion cannot
separate stay exactly at the CAD prior and are named in the report. The
pipeline does not pretend that a serial arm identifies every raw inertial
coordinate independently.

Friction and inertial data remain separate solves. The fitter may alternate
between those solves after the initial stages because an incorrect inertial
model can otherwise be absorbed into friction, but each update continues to
use only its correct protocol family. Neither holdout is used for fitting.

## Timing and torque data

The Franka robot-state broadcaster remains at the proven **100 Hz** collection
rate. Increasing it previously caused incomplete data collection, so this
pipeline does not change it.

Native timestamps and samples are preserved. When a lower-rate torque channel
must be placed on another clock, the converter uses causal previous-sample
hold and records sample age and new-sample masks. It never linearly
interpolates torque, uses a future value, or claims that a held sample is a
new 100 Hz measurement. Simulation diagnostics are thinned to the same rate
where a fair comparison requires it.

The real torque channel is intentionally fail-closed. During the first
hardware campaign its limiter and gravity composition must be established
from recorded telemetry; identification will not silently guess that an
ambiguous channel is the applied model input.

## Gravity convention

The delivered `fer_identified.xml` is a **gravity-enabled full consumer
model**. Hydrax/MPPI therefore predicts gravity in inverse dynamics and in its
rollouts.

- In MuJoCo simulation, send the full modeled joint torque.
- At the real Franka command boundary, subtract the model gravity contribution
  exactly once because the Franka supplies that compensation internally.

Gravity must not be removed from the consumer model, and it must not be
subtracted a second time elsewhere. Internally, identification uses the
gravity-free effective-effort projection that corresponds to torque on top of
the Franka compensation; export restores the identified physical parameters
to the gravity-enabled full model and verifies the consumer load paths.

## Deliverables and where to inspect them

A normal all-backend run has this shape:

```text
output/simulation/<run-id>/
  simulation_truth.json
  simulation_pipeline_status.json
  simulation_pipeline.json
  simulation_pipeline.md
  cross_backend_parameter_recovery.png
  mujoco/
    <protocol-id>/recording/...
    identification/
      identification.json
      identification_status.json
      result.md
      simulation_validation.json
      simulation_validation.md
      fer_identified.xml
      fer_identified.json
      media/<protocol-id>/replay.mp4
      media/<protocol-id>/tracking.png
      rollout_vs_measurement_<holdout>.png
      error_vs_horizon_<holdout>.png
      end_effector_<holdout>.png
      torque_tracking_<holdout>.png
      friction_curves.png
      parameter_recovery.png
      torque_equation_closure.png
  mujoco_ros/
    <protocol-id>/raw/...
    <protocol-id>/recording/...
    identification/...
```

Start with `simulation_pipeline.md` and
`simulation_pipeline_status.json`. They state whether the run is complete,
whether each backend passed, whether cross-backend agreement passed, and which
model—if any—is eligible for handoff. A failed or interrupted attempt keeps
its diagnostics but does not publish an accepted primary model.

For each backend:

- `result.md` explains the selected/frozen parameters, fit stages, recording
  lineage, and held-out reproduction verdict.
- `replay.mp4` shows the measured motion with the desired motion as a ghost;
  `tracking.png` judges whether the acquisition followed the intended
  protocol.
- The tagged holdout plots show joint, velocity, Cartesian, horizon-dependent,
  and torque reconstruction behavior separately for the friction and inertial
  families.
- `parameter_recovery.png` and `simulation_validation.md` compare the estimate
  with hidden simulation truth.
- `fer_identified.xml` is the usable gravity-enabled MuJoCo model;
  `fer_identified.json` records its hashes, source model, parameters, and
  verification.

Raw ROS bags, normalized recordings, truth, protocol hashes, reports, and
models are lineage-bound. Treat an output root as immutable. If a protocol,
truth model, source model, or setting changes, create a new run.

## Inspect or run one protocol

To watch the critical ROS simulation path without collecting a complete
campaign:

```bash
./scripts/run-protocol fer-friction-a --watch
```

To record one ROS simulation protocol into a new campaign root:

```bash
./scripts/run-protocol fer-friction-a \
  --backend mujoco_ros \
  --record /absolute/path/to/new-campaign
```

`--watch` opens the MuJoCo view and RViz; `--viewer` or `--rviz` selects only
one. Simulation performs a slow move to protocol start. The raw bag retains
that approach, while protocol markers keep it out of the fitting windows.

The standalone `mujoco` backend is an in-process acquisition path driven by
the validation pipeline. `run-protocol --backend mujoco_ros` is the ROS
simulation path.

## Transfer to the real robot

Do not start this phase until a fresh, full `validate-simulation` run is
release-ready and its videos, plots, reports, and `mujoco_ros` model have been
reviewed.

The real command changes only the backend and robot connection:

```bash
./scripts/run-protocol fer-friction-a \
  --backend real \
  --robot-ip <robot-ip> \
  --dry-run
```

On hardware, automatic move-to-start is disabled by default. The arm must
already be at rest at the reviewed start pose. `--allow-real-approach` is a
separate, watched operator opt-in, and `--speed-scale 0.5` is available for
the first run of each family.

Record all six protocol IDs into a new real campaign root. Preserve the raw
bags, verify the 100 Hz telemetry and torque semantics, then run the same
family separation, held-out evaluation, reporting, and export path. There is
no parameter-truth comparison on hardware: promotion depends on measured
held-out simulator reproduction and the final MPPI task-level comparison.

## Useful development commands

```bash
./scripts/test
./scripts/test-all
./scripts/generate-protocols --check
./scripts/identify <recording-root> --output <new-identification-directory>
```

`generate-protocols` rebuilds the canonical bundles only when protocol design
is intentionally being changed. Normal collection and identification consume
the committed, checksummed bundles.
