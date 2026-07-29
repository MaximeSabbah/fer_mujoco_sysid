# Protocol review — what these motions are for

These are the six canonical motions used to identify a MuJoCo model whose
rollouts reproduce the robot. Review the protocol plots in `review/`, watch
the motions through `scripts/run-protocol`, and use
`./scripts/validate-simulation` for the complete videos, plots, reports, and
identified models from both simulation backends.

Regenerate the committed bundles and review plots only when intentionally
changing the protocol design:

```bash
./scripts/generate-protocols
./scripts/generate-protocols --check
```

`--check` regenerates in a temporary location and verifies that the committed
content hashes are reproducible.

## Why there are two families

Friction and rigid-body dynamics require different excitation. A clean
constant-velocity sweep makes friction legible but does not independently
excite the links. A dynamic multi-frequency motion separates inertial effects
but does not provide the clean cruise plateaus needed to distinguish dry from
viscous friction.

The identification pipeline therefore keeps `fer-friction` and
`fer-inertial` as separate scientific protocols. It does not pool them into
one generic fit. After the initial stages it may alternate between the two
family-specific solves, because an incorrect inertial model can otherwise be
absorbed into friction, but each update still uses only its intended family.

## Friction protocols

The joint-friction model has two terms:

```text
friction torque = frictionloss * sign(velocity) + damping * velocity
                  \___________________________/   \________________/
                     dry / Coulomb friction          viscous friction
```

At only one speed, the constant dry-friction offset and the
velocity-dependent slope cannot be separated. The training protocols
therefore sweep in both directions at three constant speeds: 0.05, 0.15, and
0.4 rad/s, with smooth transitions and stops between regimes.

| Motion part | Identification purpose |
| --- | --- |
| constant-speed cruise | one point on the friction curve after the modeled rigid-body terms and gravity convention are accounted for |
| several speeds | separates viscous slope from dry-friction offset |
| both directions | exposes the dry-friction sign change and directional asymmetry |
| slowest cruise | excites near-standstill behavior |
| holds | standstill diagnostic, retained in the recording but excluded from the sliding-friction fit |

The velocity profile is a jerk-limited trapezoid, or S-curve. Its flat cruise
sections are deliberate: only explicitly marked, zero-acceleration cruise
windows enter the classical friction solve. Approach, ramps, reversals,
holds, returns, and filter edges remain in the recording and media but do not
contaminate that solve.

## Inertial protocols

The friction motions put every joint on a shared schedule. That makes their
plateaus readable, but it does not provide enough independent acceleration to
separate armature from link mass, centre-of-mass, and inertia effects.

The inertial family instead uses a Fourier series per joint with distinct base
frequencies. Candidate motions are scaled against the FER position, velocity,
acceleration, jerk, torque, and scene-clearance constraints, then selected for
useful conditioning. The identification stage releases only parameter
directions that the recorded motion makes sensitive, conditioned, and
sufficiently independent. Unsupported coordinates stay at the CAD prior.

## The six canonical protocols

| Protocol ID | Family | Role | Approximate duration |
| --- | --- | --- | ---: |
| `fer-friction-a` | friction | training | 42 s |
| `fer-friction-b` | friction | training, different seeded amplitudes | 42 s |
| `fer-friction-holdout` | friction | validation only | 32 s |
| `fer-inertial-a` | inertial | training | 21 s |
| `fer-inertial-b` | inertial | training, different seed | 21 s |
| `fer-inertial-holdout` | inertial | validation only | 21 s |

The holdouts are never used to fit parameters. They test whether the
identified simulator reproduces measured joint position, joint velocity,
end-effector motion, and torque behavior on motions it did not optimize
against.

Each protocol starts and ends at rest at the reviewed home pose. The bundles
live directly under `protocols/<protocol-id>/`; there are no parallel
revision directories. Each bundle declares its `family`, `role`, limits,
segments, immutable `analysis_windows`, source model, payload, and
`content_sha256`. Git history and that content hash identify revisions.

Before a bundle is written, generation checks:

- FER position, velocity, acceleration, jerk, and torque limits with the
  configured safety margin;
- clearance against the modeled table and scene;
- consistency between position, velocity, and acceleration arrays;
- rest and continuity at the start and end; and
- commensurate inertial frequencies, so the trajectory returns continuously
  to its terminal pose.

## The 100 Hz analysis contract

Canonical protocol knots are on a 100 Hz grid. Analysis keeps an additional
0.1 s guard—ten samples—inside each declared window so filtering and timing
uncertainty cannot leak a ramp, reversal, or settle segment into an eligible
interval.

The Franka robot-state broadcaster also remains at its proven **100 Hz**
collection rate. Higher publication rates caused incomplete data collection.
Native samples and timestamps are preserved; lower-rate torque telemetry is
causally held when aligned to another clock, with sample age and new-sample
masks retained. The pipeline does not manufacture extra measurements through
linear interpolation.

These masks affect analysis only. Raw bags, complete normalized recordings,
plots, videos, holds, ramps, returns, and settle samples remain intact.

## What to review

For every protocol:

- Does the whole motion fit safely in the real cell?
- Does the arm remain near the intended home region and return to rest?
- Are the amplitudes large enough to excite the model without approaching
  joint, torque, or workspace limits?
- Are the slow friction cruises representative of the near-standstill regime
  relevant to the controller?
- Are the livelier inertial motions and their run lengths acceptable on the
  real robot?
- Does the tracking plot show that the simulated plant actually followed the
  requested excitation?

Inspect one motion through the critical ROS simulation route with:

```bash
./scripts/run-protocol fer-friction-a --watch
```

The complete evidence is generated by:

```bash
./scripts/validate-simulation
```

That command must run all six protocols through both direct `mujoco` and
`mujoco_ros`, using one shared immutable simulation truth. Its generated
status and reports—not this document—state whether the current code and
protocols pass the simulation gate. Do not transfer the campaign to the real
robot until a fresh all-backend run is accepted and its videos, plots,
identified model, and `mujoco_ros` acquisition evidence have been reviewed.
