# FER MuJoCo system-identification implementation plan

Last updated: 2026-07-23

## Purpose

This document is the shared implementation and validation roadmap for the
project. It records what will be built, the order in which it will be built,
and the evidence required before moving to the next stage.

The plan is deliberately gate-driven:

- each milestone is implemented as a small, reviewable change;
- automated validation is run before requesting review;
- milestone status changes only after its evidence has been reviewed; and
- decisions that change scope, model semantics, data semantics, or hardware
  behavior are recorded here before implementation continues.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| Planned | Scope is described, but implementation has not started |
| In progress | The current approved milestone is being implemented |
| Implemented | The code and automated evidence exist, but review is open |
| Validated | Automated gates pass and the milestone has been reviewed |
| Blocked | A named external decision or dependency prevents progress |

Only one milestone should normally be in progress at a time.

## Project success criteria

The project is complete when all of the following are true:

1. A fresh environment can reproduce an identification result from a versioned
   dataset and the committed dependency lock.
2. Identified inertial parameters are physically consistent.
3. Unidentifiable or weakly identifiable parameters are detected and reported,
   not presented as reliable estimates.
4. The identified model predicts held-out FER motion better than the nominal
   model using metrics fixed before inspecting the held-out result.
5. One canonical parameter manifest produces compatible Hydrax and
   `sbmpc_ros` MJCFs without changing their kinematics or interfaces.
6. The exported model loads and rolls out under the MuJoCo/MJX versions used by
   its consumers.
7. The same motion protocol, dataset schema, conversion, diagnostics, and
   validation code work in standalone simulation, ROS simulation, and on the
   robot.
8. Reusable motion protocols and publishable datasets include sufficient
   provenance for another user to reproduce the experiment.

## Primary scientific hypothesis

The real-robot pregrasp experiments make unmodeled joint friction the highest
priority physical hypothesis. The first model-improvement campaign therefore
targets low-speed, direction-dependent tracking and friction before attempting
a broad inertial refinement.

This is a priority, not a predetermined conclusion. The experiments must still
demonstrate that friction explains the observed residual. MuJoCo's native joint
`damping` and `frictionloss` are tested first. If they cannot explain residual
dependence on velocity magnitude, direction, reversal, and temperature, the
model class must be reconsidered rather than allowing inertial parameters to
absorb the error.

## System architecture

```text
motion protocol
      |
      v
limits / collision / excitation validation
      |
      v
playback backend: standalone MuJoCo | ROS MuJoCo | Agimus FER
      |
      v
immutable raw recording + manifest + hashes
      |
      v
time alignment / signal conversion / train-validation split
      |
      v
MuJoCo ModelSequences + staged sysid
      |
      v
canonical parameter manifest + uncertainty + conditioning
      |
      +--> Hydrax-compatible MJCF
      +--> sbmpc_ros-compatible MJCF
      |
      v
held-out prediction and task-level validation report
```

There is one physical parameter estimate. Target-specific MJCFs are rendered
from it; they are never fitted independently.

## Non-negotiable engineering rules

- MuJoCo `mujoco.sysid` public APIs are used through a project-owned adapter.
  Private `_src` APIs are not imported.
- The exact MuJoCo version is pinned for fitting.
- The existing Hydrax MJCF is the nominal identification baseline.
- The ROS MJCF is a compatibility/deployment wrapper, not a second model to
  identify.
- Source MJCFs are immutable inputs during identification. Candidate exports
  remain run artifacts until a separate integration change is approved.
- The deprecated standalone `sbmpc` repository is out of scope.
- This repository owns the required standalone, ROS MuJoCo, and Agimus FER
  playback/recording/conversion workflow. ROS support is not delegated to or
  treated as an optional integration with another source repository.
- Agimus FER packages are the hardware-interface and signal-semantics
  authority for the real robot.
- Kinematics remain fixed unless external pose metrology is deliberately added.
  Joint encoders alone are not treated as sufficient evidence for geometric
  calibration.
- Raw recordings are immutable. Conversion creates new derived artifacts.
- Training and held-out validation are separated by complete motion protocols,
  not by neighboring samples from the same trajectory. The same executable
  command fingerprint cannot occur in two scientific partitions, even after
  copying or renaming its protocol artifact.
- Commanded, desired, simulated-actuator, and measured torque retain distinct
  names and semantics throughout the data pipeline. A fit selects one explicit
  torque channel plus a documented composition/transformation only after its
  semantics have been validated. No measured or desired channel is
  authoritative by name alone, and a position reference is never silently
  treated as torque.
- Robot time, ROS header time, bag-receive time, and simulation `/clock` remain
  distinct until an explicit conversion step aligns them.
- Logged `q` and `dq` drive short forward-rollout residuals. Numerically
  differentiated `qdd` may be a diagnostic, but it is not treated as ground
  truth for the primary fit.
- Full per-link XML parameter recovery is not the definition of success:
  serial-chain inertial parameters contain structurally dependent directions.
  Physical validity and held-out dynamic equivalence are the final criteria.
- Friction is the first physical parameter family evaluated. Gravity-matched
  bidirectional motions and low-speed reversals are used to isolate it before
  dynamic inertial parameters are released.
- Playback does not stream time-critical commands from an ordinary Python
  sleep loop. The real robot uses an Agimus-compatible ROS 2 controller.
- Standalone and ROS playback consume the same compiled motion arrays and an
  explicitly fixed interpolation policy. Backend adapters may map names and
  transport, but they do not regenerate the trajectory.
- Recording and plotting must not add work to the robot's real-time control
  path.
- No external controller or ROS repository is modified without a separately
  reviewed integration diff.
- A reusable motion protocol is revalidated against the local model, payload,
  start state, and experiment cell before every hardware campaign.

## Milestone overview

| ID | Milestone | Status | Primary evidence |
| --- | --- | --- | --- |
| M0 | Reproducible foundation and model contract | Validated | Pinned environment and reviewed model parity |
| M1 | Protocol, dataset, and artifact schemas | In progress | Schema fixtures and round-trip tests |
| M2 | Parameterization and synthetic unit recovery | Planned | Known-parameter recovery and physical-validity report |
| M3 | FER excitation design and protocol validation | Planned | Conditioned, constrained train/validation protocols |
| M4 | Standalone end-to-end simulation pipeline | Planned | Dataset-to-fit-to-report synthetic run |
| M5 | ROS 2 MuJoCo end-to-end pipeline | Planned | Recorded ROS simulation dataset and equivalent fit |
| M6 | Agimus FER robot-ready playback and recording | Planned | Simulation dry run and reviewed hardware protocol |
| M7 | Real-robot identification dataset | Planned | Immutable, quality-checked, versioned dataset |
| M8 | Real-model fitting and held-out validation | Planned | Pre-registered validation report and uncertainty |
| M9 | Model integration, task validation, and release | Planned | Compatible MJCFs, regression tests, reproducible release |

## M0 — Reproducible foundation and model contract

### Scope

- Pin Python 3.12 and `mujoco[sysid]==3.10.0`.
- Record nominal model revisions and content hashes.
- Define the Hydrax-to-ROS semantic name mapping.
- Derive the contact-free seven-axis identification model from Hydrax while
  retaining the attached hand and finger inertia.
- Verify physical and rollout parity between the current wrappers.

### Exit gate

- The dependency lock is current.
- The public sysid API, including conditioning diagnostics, imports.
- Source hashes match the reviewed nominal models.
- Body inertials, joint dynamics, gripper frame, and mapped arm dynamics agree.
- The seven-axis projection has `nq = nv = nu = 7`, disables contacts, and
  preserves the hand subtree mass.

### Evidence

- `contracts/nominal_sources.toml`
- `docs/model_contract.md`
- `tests/test_model_contract.py`
- `tests/test_nominal_provenance.py`
- `tests/test_sysid_api.py`

## M1 — Protocol, dataset, and artifact schemas

### Scope

Define stable, versioned contracts before producing data:

- a reusable motion protocol;
- an immutable acquisition/run manifest;
- a canonical normalized numeric trajectory;
- a fit/export result manifest;
- joint order, units, sample times, and desired `q/dq/ddq`;
- commanded effort, desired effort, simulated actuator effort, total measured
  joint effort, and external effort as separate optional signals;
- source clock domains and explicit alignment metadata;
- fit, development, held-out-test, diagnostic-only, and excluded roles,
  separate from acquisition outcome;
- a canonical identified-parameter manifest; and
- optimizer, uncertainty, and report metadata;
- source, protocol, dataset, and model hashes; and
- directory conventions for reusable datasets.

Shareable motion protocols live in-repository. Robot recordings are local by
default; the schema also supports deliberately curated compact datasets and
externally stored public recordings.

### Required implementation

- Explicit schema version in every manifest.
- Strict joint-name and unit validation.
- Monotonic timestamp validation.
- Safe array serialization without Python pickles.
- Lossless association from every derived sequence back to the raw recording.
- A small committed example dataset.
- A command that validates a dataset without running identification.
- A dataset-level license independent from the code license where appropriate.
- A migration policy for future schema versions.

### Exit gate

- Valid examples round-trip without numerical or metadata loss.
- Invalid joint order, units, timestamps, missing signals, or changed hashes
  fail with actionable errors.
- Train and validation roles cannot overlap accidentally.
- A new user can understand every signal from the dataset alone.
- Repeating conversion with the same inputs produces the same artifact hash.
- Missing or ambiguous torque semantics are rejected rather than inferred.

### Review checkpoint

Approve the schemas and storage formats before implementing the fitting core.

### Implementation slices

- **M1a — contract foundation (current review):** packaged schemas, strict
  JSON/pickle-free NPZ utilities, deterministic content hashes, semantic
  validation for motion and normalized trajectories, and a MuJoCo 3.10
  public-API compatibility test.
- **M1b — complete artifact workflow (in progress):** sealed-bundle
  finalization, intrinsic validation, dataset-wide local reference and lineage
  closure, exact protocol-interval binding through recorded position,
  velocity, acceleration, and conditional effort reference traces,
  execution-interval scientific-partition fingerprinting, backend-aware signal
  provenance, physical-context checks, conservative cross-artifact torque-input
  eligibility, resolver-to-fit validation, and the sealed `validate-dataset`
  command are implemented. A committed compact simulation dataset remains
  before the M1 exit gate.

## M2 — Parameterization and synthetic unit recovery

### Scope

Build the project-owned adapter around MuJoCo sysid and establish that each
parameter group can be applied, recovered, and exported correctly.

### Parameter groups

1. Timing, sensor bias, and torque scale as nuisance parameters.
2. Joint friction loss and viscous damping.
3. Joint armature.
4. Body mass and center of mass.
5. Full physically consistent body inertia using MuJoCo's pseudo-inertia
   parameterization.

Kinematics, geometry, names, limits, and actuator ordering remain fixed.
`link0` is parity-checked but not estimated because it is fixed to the world.

Rigidly connected bodies are not exposed as independently observable
parameters. In the seven-axis projection, `hand`, `left_finger`, and
`right_finger` are rigidly attached downstream of joint 7; their effect is
represented through a canonical composite tool/payload inertia. A synthetic
test must demonstrate that redistributing inertia inside an unobservable rigid
composite does not create a falsely identifiable parameter.

### Required implementation

- Canonical SI-valued parameter manifest.
- Bounds and transformations with documented physical meaning.
- Nominal apply/export/reload round trip.
- Deterministic hidden-truth perturbation generator.
- Single-parameter recovery tests.
- Friction-first recovery in the order `frictionloss`, damping, then armature
  before any multi-body inertial block.
- Small well-conditioned multi-parameter recovery tests.
- Multiple initial guesses and bound-hit diagnostics.
- Conditioning and covariance/uncertainty report.
- Scaled residual Jacobian, singular values, parameter correlations, and
  multistart spread for every retained parameter block.

### Exit gate

- Applying nominal parameters changes no compiled physical quantity.
- Every generated inertia remains physically valid.
- Noise-free single-parameter cases recover the hidden value within the
  numerical tolerance agreed in the milestone review. The initial target is
  at most 1% relative error for a nonzero scalar and at least a 99% objective
  reduction.
- Multi-parameter cases either recover identifiable parameters or explicitly
  report the expected ambiguity.
- A retained parameter block initially requires
  `sigma_min / sigma_max >= 1e-6`; pairs with absolute correlation above 0.98
  are frozen, grouped, or reparameterized.
- Multiple starts produce held-out prediction scores within 5% for retained
  parameter blocks.
- The hidden truth is never available to the fitting code.
- No private MuJoCo sysid API is used.
- Synthetic low-speed reversals distinguish damping from friction loss, and an
  intentionally inadequate friction model leaves a detectable structured
  residual rather than corrupting inertial estimates.

### Failure response

If a parameter group is ill-conditioned, freeze or reparameterize it before
adding more parameters. Do not compensate by widening bounds until the result
appears plausible.

## M3 — FER excitation design and protocol validation

### Scope

Generate several complementary, FER-specific motion families:

- paired slow bidirectional sweeps, stops, and reversals at matched
  configurations for friction;
- static configurations and holds for gravity consistency and friction
  isolation;
- coupled multisine/Fourier motion for inertial excitation;
- smooth rest-to-rest quintic segments for workspace coverage; and
- separate motion seeds and configurations for held-out validation.

Friend-provided FR3 planners may inform the algorithms, but their trajectories,
limits, geometry, hardware code, and identified parameters are not reused
unchanged.

### Required validation

- FER joint position, velocity, and acceleration limits.
- Configurable margins to physical limits.
- Continuity of position, velocity, acceleration, and commanded effort.
- Jerk, predicted torque, and torque-rate checks.
- Collision checking against the canonical FER model and experiment cell.
- Defined start/finish state and deterministic return or hold behavior.
- Finite-difference residual sensitivity and singular-value/Fisher-information
  diagnostics.
- Separate train and validation campaigns.
- Validation across a bounded ensemble of plausible dynamic models, not only
  the nominal model.
- A machine-generated simulation-validation record tied to the protocol hash.
- Full plant-scene hashes and a fixed interpolation rule shared by standalone
  MuJoCo and the ROS trajectory controller.

### Exit gate

- All protocols are deterministic from a versioned specification and seed.
- No protocol violates its predeclared limits or margins in dense simulation.
- The selected campaign materially improves conditioning over an unoptimized
  baseline.
- Removing any selected motion family has a documented effect on observable
  parameter groups.
- Validation protocols were not used to choose fitted parameter values.
- The complete approach, excitation, and return/hold motion is validated, not
  only the nominal excitation interval.

### Review checkpoint

Review plots, conditioning, duration, limits, and the exact robot motion
campaign before any ROS or hardware player is implemented.

## M4 — Standalone end-to-end simulation pipeline

### Scope

Exercise the complete data path without ROS:

1. Create a perturbed, physically valid hidden-truth FER model.
2. Play approved protocols in native MuJoCo.
3. Produce realistic measured signals.
4. Add controlled noise, timing jitter, bias, and delay variants.
5. Save the same raw and normalized dataset formats intended for the robot.
6. Fit from the nominal model.
7. Validate on unseen protocols.
8. Generate an automatic report.

The converter builds short, optionally overlapping rollout windows initialized
from measured `q/dq`. Dataset splitting remains by complete source trajectory,
so overlapping windows can never cross the train/validation boundary.

The low-level open-loop effort primitive is not the controller-equivalence
gate. Before generating the M4 dataset, standalone playback must implement the
same interpolated effort-controller law and hashed controller configuration
used by the ROS effort-mode `JointTrajectoryController`. Desired effort
feedforward, total controller request, and realized MuJoCo actuator effort
remain separate signals.

### Required diagnostics

- Per-joint `q`, `dq`, applied torque, and residual plots.
- Timestamp and sampling-rate diagnostics.
- One-step and windowed open-loop prediction.
- End-effector position and orientation prediction.
- Train-versus-validation metrics.
- Parameter values, bounds, bound hits, uncertainty, and conditioning.
- Nominal-versus-identified comparison.
- Residual and prediction error versus velocity magnitude, velocity sign,
  reversal, configuration, and simulated temperature profile where applicable.
- Reproducibility information and artifact hashes.

### Exit gate

- A no-mismatch dataset yields only numerical residual.
- The on-disk pipeline recovers the same result as the in-memory unit test.
- Known timing offsets and signal-order changes are detected or recovered.
- The identified model materially reduces held-out prediction error relative
  to nominal under thresholds fixed before running the held-out evaluation.
- The result is repeatable from a clean environment and deterministic seed.
- With realistic corruption across a fixed Monte Carlo suite, at least 90% of
  runs improve held-out prediction and the median aggregate normalized error
  improves by at least 30%. These initial targets may only be changed before
  examining the corresponding validation results.
- Uncertainty combines local Jacobian information with trajectory/block
  resampling because time samples are autocorrelated.

## M5 — ROS 2 MuJoCo end-to-end pipeline

### Scope

Add ROS 2 packages in this repository for protocol playback, recording, and
conversion while using the existing FER MuJoCo plant.

### Required implementation

- Standard ROS 2 trajectory action playback rather than Python-rate streaming.
- One complete `FollowJointTrajectory` goal per protocol, leaving interpolation
  to the real-time controller.
- A dedicated effort-mode trajectory controller and launch path in this
  repository; the MPPI bridge and linear-feedback controller are not part of
  identification playback.
- One protocol description shared with standalone simulation.
- Recording to MCAP as the immutable source.
- Controller reference, measured state, actual/desired effort, robot state,
  diagnostics, timing, and action status where available.
- A converter that produces the same normalized dataset contract as M4.
- Automatic post-run validation and report generation.
- Separate controller-output and MuJoCo realized-actuator signals; neither is
  inferred from the other.
- A self-contained asset path for the ROS MJCF before the deprecated checkout
  can be removed.
- A cross-version rollout gate covering the sysid, Hydrax, and ROS MuJoCo
  runtimes. Numerical parity is not claimed while their versions differ
  without evidence.
- No modification of Hydrax or `sbmpc_ros` during this milestone.

### Exit gate

- The ROS simulation completes each protocol with correct start, stop, hold,
  cancellation, and failure behavior.
- Required topics exist at expected rates and have nonzero counts.
- Joint ordering and torque semantics agree with the standalone dataset.
- The same protocol knots and interpolation contract are demonstrated in both
  simulation paths.
- Cross-version rollout differences are below predeclared tolerances, or all
  three runtimes use the same accepted MuJoCo version.
- No recording or reporting work executes in the real-time controller path.
- Fitting the ROS-simulation dataset recovers the hidden simulation parameters
  and passes the same held-out gates as M4.
- Three repeated ROS simulation runs produce held-out scores within 5%.

### Review checkpoint

Visually inspect representative ROS simulations and review the generated
dataset/report before work on the real backend begins.

## M6 — Agimus FER robot-ready playback and recording

### Scope

Port the validated protocol player and recorder to the real FER using only the
Agimus hardware architecture.

### Required implementation

- Agimus-compatible ROS 2 trajectory controller and action interface.
- Exact mapping of FER joint names and effort/state signals.
- Source-backed documentation of `tau_J`, `tau_J_d`, gravity compensation,
  rate limiting, and any internal torque conventions.
- Preflight validation of protocol/model hashes, payload and end-effector,
  current state, controller exclusivity, robot mode, recorder readiness, and
  required state topics.
- Start-state validation and a controlled transition to the protocol start.
- Position, velocity, acceleration, jerk, predicted torque, torque-rate, and
  collision checks before a goal can be sent.
- Explicit hold/cancel/failure behavior.
- Comprehensive recording without blocking the real-time loop.
- Simulation mode exercising the exact player and recorder code.
- Hardware mode changes only the backend launch, controller endpoint, and
  Agimus-specific signal mapping; it does not regenerate the protocol.

### Exit gate

- All M5 tests pass through the hardware-facing adapter in simulation.
- The protocol compiler rejects intentionally invalid motions.
- State and torque semantics are demonstrated with a small deterministic test.
- The exact first robot protocol and expected motion are reviewed.
- Physical execution requires an explicit go-ahead with the user at the robot.

The human operator retains responsibility for enabling the robot and using the
emergency stop.

## M7 — Real-robot identification dataset

### Scope

Collect a structured campaign rather than a single long trajectory:

- slow bidirectional friction dataset as the first physical-model campaign;
- matched static/hold dataset;
- dynamic inertial dataset;
- independent held-out validation dataset; and
- optional known-payload dataset if approved.

Friction-sensitive protocols should be repeated across documented cold/warm
conditions where practical; temperature dependence must not be mistaken for a
single constant friction parameter.

### Required acquisition metadata

- robot and end-effector identity;
- software/model/protocol revisions and hashes;
- controller configuration;
- sample clocks and observed rates;
- protocol-knot-zero timestamp, clock, and recorded source;
- measured and desired joint states and efforts;
- robot mode, errors/reflexes, rate-limiter state where available;
- relevant temperature or current signals where available;
- payload and environment configuration; and
- operator notes and run acceptance/rejection reason.

### Exit gate

- Every accepted run passes automatic completeness, rate, timestamp, joint
  order, saturation, clipping, dropout, and bound checks.
- Rejected runs remain traceable but are excluded by manifest rather than
  silently deleted.
- Failed, reflex-limited, contact-affected, or saturated intervals are
  quarantined as whole runs unless a predeclared segmentation rule applies.
- Raw files are immutable and checksummed.
- Train and held-out roles are locked before fitting.
- The curated dataset can be replayed through the converter on another machine.

## M8 — Real-model fitting and held-out validation

### Scope

Fit the real model in stages:

1. timing, bias, and scale nuisance terms;
2. damping and friction from paired low-speed/reversal protocols;
3. inertial parameters and armature;
4. joint refinement of only well-conditioned groups; and
5. optional payload validation.

Each stage starts from the previous accepted result and records why parameters
were added, frozen, or removed.

### Validation metrics

- Per-joint and aggregate `q` and `dq` prediction errors.
- Applied-torque and residual diagnostics.
- End-effector position/orientation error derived from fixed kinematics.
- Short- and longer-window open-loop prediction.
- Error versus speed, direction, configuration, and temperature when present.
- Low-speed steady tracking offset and behavior around velocity reversal,
  including the pregrasp-relevant operating region.
- Residual autocorrelation and systematic bias.
- Parameter uncertainty, correlations, singular values, and bound proximity.
- Trajectory-level bootstrap confidence intervals for validation improvement.

### Exit gate

- Validation thresholds are written before evaluating the held-out dataset.
- The identified model improves the aggregate held-out metrics over nominal.
- The bootstrap 95% lower confidence bound on aggregate held-out improvement is
  above zero.
- No joint metric worsens by more than 10% without an explicitly reviewed
  explanation.
- Parameters are physically valid and stable across reasonable initial guesses.
- Bound-hitting or strongly correlated parameters are reported as uncertain and
  frozen or omitted from the exported model.
- The friction-first model materially improves low-speed and reversal residuals.
  If native damping plus friction loss cannot do so, inertial fitting pauses
  until the friction model-class decision is reviewed.
- A nominal-versus-identified report is reviewed before integration.

### Failure response

If held-out prediction does not improve, do not integrate the model. Diagnose
signal semantics, timing, excitation, unmodeled actuator behavior, and parameter
conditioning; then return to the earliest failed milestone.

## M9 — Model integration, task validation, and release

### Scope

- Export the canonical SI-valued parameter manifest.
- Render Hydrax- and ROS-compatible MJCFs while preserving wrapper interfaces.
- Apply only whitelisted mass, center-of-mass, inertia, armature, damping, and
  friction fields; a source-model hash mismatch is fatal.
- Reload and compare the exported models.
- Test the Hydrax output under its deployed MuJoCo/MJX runtime.
- Test the ROS output in the full ROS simulation.
- Update external model files only through separate reviewed diffs.
- Update nominal-parity tests so fixed Agimus facts remain fixed while
  identified dynamics are checked against the identified manifest.
- Validate representative FER tasks first in simulation and then, if approved,
  on the robot.
- Publish reproducible datasets, reports, limitations, and model provenance.

Generated XML candidates and structural diffs live in this repository first.
Only after their validation are separate changes proposed to consumer
repositories.

### Exit gate

- Export/reload produces identical realized parameters and rollouts.
- Fixed kinematics, names, limits, dimensions, gripper frame, and interfaces are
  unchanged.
- Both target runtimes pass their regression suites.
- Task-level tracking improves or remains acceptable without controller
  retuning being used to conceal a model regression.
- The previous nominal model remains available as a clear rollback.
- A clean clone can reproduce the published report and model.

## Cross-cutting artifact contract

Every identification run should produce a self-contained run directory:

```text
run_id/
  manifest.json
  checksums.sha256
  protocol/
  raw_source_reference/
  normalized/
  optimizer/
  identified_parameters.json
  models/
  metrics.json
  report.html
  plots/
```

The manifest records at least:

- schema and tool versions;
- MuJoCo and Python versions;
- source-model revision and hash;
- dataset and protocol identifiers/hashes;
- joint order and units;
- parameter groups and bounds;
- optimizer/backend/settings and seed;
- train/validation roles;
- timestamps and commands used; and
- generated artifact hashes.

A reusable published dataset should follow this logical layout:

```text
protocols/<protocol_id>/<revision>/
  protocol.json
  desired.npz
  checksums.sha256

datasets/<dataset_id>/<version>/
  README.md
  LICENSE
  CITATION.cff
  dataset.json
  checksums.sha256
  splits.json
  runs/<run_id>/
    run.json
    raw/
    processed/
  reports/
```

Acquisition runs normally live outside the source repository. A large MCAP is
published only deliberately and may be referenced through a versioned release;
its immutable URL, size, hash, metadata, and exact conversion command remain
tracked.

## Review procedure for each milestone

Before implementation:

1. State the exact milestone and files expected to change.
2. Surface any open decision that would materially alter the design.
3. Obtain approval before modifying existing external repositories.

At implementation handoff:

1. Present the exact diff scope.
2. Report automated tests and their commands.
3. Provide generated plots/artifacts where applicable.
4. State known limitations and failures.
5. Wait for review before marking the milestone validated or starting the next
   milestone.

## Stop and rollback rules

- Do not proceed to hardware because a simulation failure is merely
  inconvenient.
- Do not proceed to real fitting when signal semantics or timestamps are
  ambiguous.
- Do not add parameters to compensate for failed identifiability.
- Do not tune on held-out validation data.
- Do not integrate a model that fails held-out improvement.
- Keep the nominal model and every accepted parameter manifest reproducible.
- If a downstream gate fails, return to the earliest upstream assumption that
  the evidence contradicts.

## Scientific failure conditions

The following are failures even if an optimizer returns `success`:

- fitting the two MJCF wrappers independently;
- splitting neighboring samples or overlapping windows across train and
  validation;
- judging a full inertial fit only by literal XML-value recovery;
- accepting bound-hugging, rank-deficient, or multistart-dependent parameters;
- estimating delay, friction, and inertia simultaneously from one motion
  family without staged evidence;
- using differentiated acceleration as the primary measurement;
- silently exporting estimated delay, torque gain, or sensor bias into MJCF;
- reporting local least-squares covariance as calibrated uncertainty without
  trajectory-level resampling;
- improving training error while held-out residuals remain structured; or
- forcing an inadequate damping/friction model to fit through implausible
  inertial parameters.

## Decision log

| ID | Date | Decision |
| --- | --- | --- |
| D001 | 2026-07-23 | Use a standalone `fer_mujoco_sysid` repository. |
| D002 | 2026-07-23 | Pin fitting to Python 3.12 and `mujoco[sysid]==3.10.0`; do not upgrade consumer repositories for sysid development. |
| D003 | 2026-07-23 | Start from the existing Hydrax and `sbmpc_ros` MuJoCo models rather than reconstructing an MJCF from URDF. |
| D004 | 2026-07-23 | Treat Hydrax as the nominal physical template and render both wrappers from one identified parameter set. |
| D005 | 2026-07-23 | Exclude the deprecated standalone `sbmpc` repository. |
| D006 | 2026-07-23 | Validate the complete pipeline in simulation before playing excitation motions on the robot. |
| D007 | 2026-07-23 | Use Agimus packages as the only hardware architecture/reference for the FER. |
| D008 | 2026-07-23 | Keep reusable motion protocols and curated datasets as versionable repository assets. |
| D009 | 2026-07-23 | Keep kinematics fixed unless an external metrology experiment is added. |
| D010 | 2026-07-23 | Use controller-managed ROS 2 trajectory playback, not a Python timing loop, on hardware. |
| D011 | 2026-07-23 | Prioritize friction identification because the real pregrasp experiments exposed low-speed tracking error consistent with missing friction. |
| D012 | 2026-07-23 | Use strict JSON metadata plus safe numeric NPZ as the portable artifact boundary; adapt these artifacts to public MuJoCo sysid containers in memory. |
| D013 | 2026-07-23 | Use `fer_joint1` through `fer_joint7` as the canonical portable joint order and map Hydrax names at the adapter boundary. |
| D014 | 2026-07-23 | Keep reusable compiled motions in `protocols/`; keep per-robot recordings external by default and publish datasets only through deliberate curation. |
| D015 | 2026-07-23 | Use a dedicated effort-mode ROS trajectory-controller path in this repository; do not use the MPPI/LFC command path for identification playback. |
| D016 | 2026-07-23 | Treat MuJoCo 3.10/3.8/3.4 runtime skew as an explicit simulation gate before claiming standalone, ROS, and robot-ready parity. |
| D017 | 2026-07-23 | Keep standalone MuJoCo, ROS MuJoCo, Agimus playback, recording, conversion, fitting, and reporting in this one repository; consumer repositories only receive explicit model exports. |
| D018 | 2026-07-23 | Treat direct feedforward-only MuJoCo stepping as a low-level plant primitive, not ROS parity; parity requires the same interpolated effort feedforward plus feedback law and distinct requested/realized torque channels. |
| D019 | 2026-07-23 | Bind every normalized trajectory to an exact protocol interval using exact recorded desired position, velocity, acceleration, and conditional effort references, cross-signal canonical-time agreement, and a recorded protocol-knot-zero clock anchor; fingerprint the selected executable interval, excluding non-executed terminal effort and normalizing floating signed zero, and forbid that fingerprint from crossing scientific partitions independently of artifact naming or enclosing protocol length. |
| D020 | 2026-07-23 | Until structured transformation and delay fitting exists, admit only identity-scaled, zero-shifted, causal effort inputs to fitting; reject backend/source and protocol physical-context contradictions. |

## Open decisions

These decisions are intentionally deferred to the named milestone:

| Decision | Due |
| --- | --- |
| Initial parameter bounds and scaling | M2 |
| Numerical synthetic-recovery tolerances | M2 |
| Whether native MuJoCo damping plus friction loss is an adequate FER friction model | M2/M3 |
| Exact excitation families, margins, and campaign duration | M3 |
| ROS topic/action contract and MCAP topic set | M5 |
| Exact Agimus torque/control signal semantics | M6 |
| Whether to include a known-payload campaign | M6/M7 |
| Pre-registered real held-out improvement thresholds | Before M8 |
| External-repository model integration mechanism | M9 |

## Next review unit

After the current M1b and low-level rollout diff is reviewed, the next proposed
unit is the self-contained nominal FER model snapshot with complete license,
source-lock, and checksum provenance. It removes runtime/test dependence on
sibling Hydrax and `sbmpc_ros` checkouts before synthetic recovery or ROS
playback is built on top of the model.
