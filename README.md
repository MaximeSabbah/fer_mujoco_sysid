# FER MuJoCo system identification

Identify a Franka Research 3 as MuJoCo model parameters — joint friction,
damping, armature and the observable link inertials — and hand the result to
MPPI.

The release criterion is behavioural, not a plausible parameter table: the
identified simulator must reproduce a protocol it never saw, open loop, over
MPPI-relevant horizons. The same code path runs in-process, through ROS, and
on the robot; only the backend changes.

## Setup

Python 3.12 and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --locked --all-groups
```

The ROS backend additionally needs ROS 2 Jazzy and this project's dependency
overlay (the Franka description, hardware interface and
`mujoco_ros2_control`). `run-protocol` sources `/opt/ros/jazzy/setup.bash` and
`/opt/sbmpc_deps_ws/install/setup.bash`; override with `FER_SYSID_ROS_SETUP`
and `FER_SYSID_WORKSPACE_SETUP`. No consumer repository is required.

## Running a campaign

Every campaign is: play six protocols → record → identify → read the report.
Recordings are immutable, so a destination that already exists is refused.

**In-process MuJoCo** — no ROS, fastest, hidden truth known:

```bash
./scripts/simulate-campaign                 # -> output/mujoco/
./scripts/identify --backend mujoco         # -> output/mujoco/identification/
```

Add `--hardware-like-noise` (with `--output output/mujoco_noise`) for the
robustness variant; the default is noise-free and should close to numerical
precision.

**Through ROS** — the rehearsal for hardware, same motions through the real
controller, recorder and converter:

```bash
for p in fer-friction-a fer-friction-b fer-friction-holdout \
         fer-inertial-a fer-inertial-b fer-inertial-holdout; do
  ./scripts/run-protocol "$p" --record \
      --simulation-truth output/mujoco/simulation_truth.json
done
./scripts/identify --backend mujoco_ros
```

Passing the same `--simulation-truth` makes the two backends directly
comparable. To only watch a protocol, `./scripts/run-protocol <id> --watch`
(MuJoCo viewer + RViz; `--viewer` or `--rviz` for one).

`./scripts/validate-simulation` does both backends against one truth and
cross-compares them in a single command, into its own run root.

**On the robot** — only after the simulated runs have been reviewed:

```bash
./scripts/run-protocol fer-friction-a --backend real --robot-ip <ip> --dry-run
./scripts/run-protocol fer-friction-a --backend real --robot-ip <ip> --speed-scale 0.5
```

Automatic move-to-start is disabled on hardware: the arm must already be at
rest at the reviewed start pose, and `--allow-real-approach` is an explicit,
watched opt-in. Identification of real data is fail-closed on the torque
channel until its limiter and gravity composition are established from
recorded telemetry.

### What a run leaves behind

Artifacts are written into the output directory **as they are produced** —
metrics first, then each plot and video, then the report — so a run that fails
late still leaves everything it obtained, and a partial directory is readable
while the rest is still being made. The previous run's artifacts are cleared
when a new one starts; the fit checkpoint, the superseded-model archive and any
file you put there yourself are kept.

Only `fer_identified.xml`/`.json` are withheld until the end: releasing a model
requires passing the held-out gate *and* producing the full evidence, so a
rejected or incomplete run publishes diagnostics but never a model.

### Fitting takes tens of minutes, so it is cached

`identify` writes `fit_checkpoint.pickle` into its output directory as soon as
the solves finish, and any later run reuses it — the report, plots and videos
are then rebuilt in seconds. The checkpoint is keyed on the recording digests,
the source model, the fit knobs and the contents of every source file that
decides a fit, so it is discarded automatically (with the reason printed) when
any of those change. `--refit` forces a fresh solve.

Before spending a full fit on new data, prove the whole path in minutes:

```bash
./scripts/identify <campaign> --output /tmp/smoke --max-iters 1 --dynamic-starts 1
```

The fit is meaningless at one iteration, but every downstream step — held-out
evaluation, media, report, publish — runs exactly as it will on the real thing.

## The six protocols

| Protocol | Family | Role | Identifies |
| --- | --- | --- | --- |
| `fer-friction-a`, `-b` | friction | training | `frictionloss`, `damping`, from the marked constant-velocity cruises |
| `fer-friction-holdout` | friction | validation only | — |
| `fer-inertial-a`, `-b` | inertial | training | armature and the observable link mass / COM / inertia corrections |
| `fer-inertial-holdout` | inertial | validation only | — |

Family and role come from the bundle manifest, never from the identifier. The
families are fitted separately: one motion cannot serve both. Holdouts are
never fitted.

## Where results land

```text
output/<backend>/
  <protocol-id>/raw/           # the MCAP bag (ROS backends), immutable
  <protocol-id>/recording/     # checksummed arrays, all the fit reads
  identification/
    result.md                  # start here
    rollout_vs_measurement_<holdout>.png
    error_vs_horizon_<holdout>.png
    end_effector_<holdout>.png
    torque_tracking_<holdout>.png
    friction_curves.png
    media/<protocol-id>/replay.mp4     # measured motion, desired as a ghost
    media/<protocol-id>/tracking.png   # did the run follow the protocol
    fer_identified.xml / .json         # the model and its provenance
    identification.json / identification_status.json
```

Reading order: `result.md` for the numbers and the verdict,
`rollout_vs_measurement` for whether the model reproduces the robot,
`error_vs_horizon` for how fast that decays, then the videos to judge the
motion itself. Everything under `output/` is generated and git-ignored; a run
reproduces it.

`fer_identified.xml` is the **gravity-enabled** consumer model: MPPI predicts
gravity in its rollouts, and the model gravity contribution is subtracted
exactly once, at the real command boundary, because the Franka supplies that
compensation internally.

## What keeps a result honest

- A model is published only if it beats nominal on held-out reproduction in
  physical units; a rejected fit keeps its diagnostics and publishes no model.
- Inertial coordinates the motion cannot separate stay exactly at their CAD
  prior and are named in the report.
- Franka telemetry stays at its proven 100 Hz; a lower-rate channel placed on
  another clock uses causal previous-sample hold, never interpolation.
- Recordings bind the protocol content hash and the source model hash, so a
  changed motion can never be mistaken for the reviewed one.

## Development

```bash
./scripts/test                        # fast suite
./scripts/test-all                    # everything, including the slow fits
./scripts/generate-protocols --check  # verify the committed bundles regenerate
```

`generate-protocols` without `--check` rebuilds the canonical bundles, and is
only for deliberate protocol design changes. The
[implementation plan](docs/implementation_plan.md) holds the roadmap and the
decision log.
