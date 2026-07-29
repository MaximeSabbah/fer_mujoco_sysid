# FER MuJoCo system-identification implementation plan (v2)

Last updated: 2026-07-28. Supersedes
[`archive/implementation_plan_v1.md`](archive/implementation_plan_v1.md);
the v1 decision log D001–D020 remains valid except where amended below.

## Purpose

Produce an identified MuJoCo model of the lab's FER robot (inertials,
armature, damping, joint friction) that the downstream controllers
(hydrax F-MPPI planner and `sbmpc_ros`) consume, replacing the
Menagerie-default dynamics.

Motivating evidence (run `sbmpc_runs/pregrasp_real_20260722T132356Z`): the
real pregrasp parks 62 mm short at a stable torque equilibrium while the
same stack reaches 7–8 mm in simulation. The planner holds constant
gravity-free torques (j1 0.62, j6 0.67, j7 0.43 Nm) against joints that do
not move; joint 1 (vertical axis, no gravity component) held 1.37 Nm without
motion. Unmodeled static friction is therefore the first-priority parameter
family, and the identified model must materially improve exactly this
low-speed / reversal / standstill regime.

Empirical friction lower bounds from that run (max |tau_J_d| while the joint
was still): j1 ≥ 1.37, j3 ≥ 1.11, j4 ≥ 1.50, j6 ≈ 1.1, j7 ≥ 0.55 Nm; j5 was
never pushed above 0.21 Nm and never moved. Any identified friction result
inconsistent with these bounds is wrong.

## What changed from v1

- **Playback uses the stock `JointTrajectoryController` (D021, refined by
  D027).** Excitation protocols are played by the Agimus stack's
  `joint_trajectory_controller`, which runs in **effort mode with an
  internal PID**; the fit consumes measured `q`, `dq` and the controller's
  **commanded effort**. No bespoke real-time controller is written, and the
  v1 controller-parity gates (D015/D018) are dropped.
- **The artifact/contract layer is deleted (D024, reversing D022).** It was
  about three quarters of the repository and served no identification
  purpose. Artifacts are a JSON manifest, an NPZ, a content hash and a
  `checksums.sha256`.
- **Milestones compressed.** The pipeline is built physics-first in the
  phases below; process artifacts (reports, provenance) are produced by the
  phases that need them rather than designed up front.

## Ground rules (unchanged in substance from v1)

- Friction first: `frictionloss` and `damping` are fitted from dedicated
  low-speed bidirectional protocols before any inertial parameter is
  released; armature next; inertials last, using MuJoCo's physically
  consistent pseudo-inertia parameterization.
- Staged fitting only; no single joint fit of delay+friction+inertia from
  one motion family. Ill-conditioned groups are frozen or reparameterized,
  never bound-widened until plausible.
- Raw recordings are immutable; conversion derives new artifacts. Train and
  held-out roles split by complete protocol, locked before fitting, with
  held-out thresholds written down before evaluation.
- Logged `q`/`dq` drive short-window forward-rollout residuals
  (`mujoco.sysid` `ModelSequences` intervals); differentiated `qdd` is a
  diagnostic, not the primary measurement.
- Public `mujoco.sysid` API only, through a project-owned adapter.
- Kinematics, names, limits, actuator ordering stay fixed. The hydrax
  `panda.xml` is the nominal baseline; a consumer's `fer_ros2_control.xml` is
  a deployment wrapper rendered from the same identified parameter manifest,
  never fitted independently. Consumer checkouts are **optional** (D032):
  identification, protocol generation, playback and simulation all run from
  the nominal model alone.
- Version skew is explicit: fitting runs on MuJoCo 3.10 (sysid toolbox),
  hydrax consumes via MJX 3.8, ROS sim via `mujoco_ros2_control`. Physical
  parameters transfer; a cross-version rollout comparison gates the export.
- No consumer repository (hydrax, `sbmpc_ros`, Agimus) is modified except
  through a separate reviewed diff. Recording never adds work to the
  real-time path. Hardware execution requires simulation validation of the
  identical protocol first and an explicit user go-ahead at the robot.
- One phase in progress at a time; each phase lands as a small reviewed
  diff with its automated evidence; status changes only after review.

## Phases

| ID | Phase | Status |
| --- | --- | --- |
| P0 | Foundation and model contract (old M0) | Validated |
| P1 | Artifact contracts | Deleted (D024); replaced by `io.py` |
| P2 | Sysid engine and synthetic parameter recovery | Implemented (review open) |
| P3 | Excitation protocol generation | Both families implemented (campaign awaiting user review) |
| P4 | Standalone end-to-end pipeline | Implemented — `report.py`, see `output/mujoco/` |
| P5 | ROS-sim end-to-end pipeline | Playback done (slice A); recording/conversion open (slice B) |
| P6 | Real-robot campaign readiness and data collection | Planned |
| P7 | Real-model fitting and held-out validation | Planned |
| P8 | Export and consumer integration | Export implemented (`export.py`, D033); consumer integration planned |

### P2 — Sysid engine and synthetic parameter recovery

Project-owned adapter around `mujoco.sysid`: parameter groups
(`frictionloss`, `damping`, `armature`, then body pseudo-inertia for the
moving links plus one lumped tool composite), staged fit driver, bounds,
multistart, and the conditioning/uncertainty report (singular values,
correlations, bound proximity). Synthetic recovery on the 7-axis model:
apply a hidden physically valid perturbation, generate motion, fit from
nominal, recover.

Exit gate: noise-free single-parameter recovery ≤ 1% relative error;
friction vs damping separated on synthetic low-speed reversals; nominal
apply/export/reload changes no compiled quantity; redistributing inertia
inside the rigid tool composite is reported unidentifiable, not "recovered";
hidden truth never readable by fitting code.

### P3 — Excitation protocol generation

Two protocol families compiled into `protocols/` with provenance: (a)
friction family — paired slow bidirectional sweeps, matched-configuration
reversals, holds, covering the pregrasp workspace region; (b) inertial
family — multisine/Fourier and rest-to-rest segments for inertial
excitation. Validation: FER position/velocity/acceleration limits with
margins, self/environment collision on the nominal model, predicted torque
and torque-rate bounds, deterministic seed→protocol reproducibility, and a
conditioning (Fisher/singular-value) improvement check over an unoptimized
baseline. Separate seeds/configurations reserved for held-out validation.
The friend repositories inform the algorithms only; no trajectory, limit,
or parameter is copied as truth.

Exit gate: protocols deterministic from spec+seed; no limit violation in
dense simulation of approach+excitation+return; conditioning materially
better than baseline; train and held-out campaigns disjoint.

### Model-acceptance rule (user requirement, 2026-07-27)

The `mujoco.sysid` optimizer objective is a normalized internal residual and
is never the basis for accepting an identified model. Acceptance is judged
by evidence that maps to the real robot: (a) held-out open-loop prediction
error in physical units — per-joint rad / rad·s⁻¹ and millimetres at the
end effector — identified vs nominal, on protocols the fit never saw;
(b) the same errors resolved by velocity regime, especially near-standstill
and reversals (the pregrasp-relevant region), checked for consistency with
the recorded breakaway lower bounds; (c) ultimately the pregrasp task
itself, sim then real, against the 62 mm baseline. The physical-units
evaluator implementing (a)+(b) is built in P4 with the pipeline that feeds
it, and reused unchanged for the ROS simulation and the real campaign.

### P4 — Standalone end-to-end pipeline

Without ROS: play P3 protocols on a hidden-truth-perturbed model under a
position-tracking law, record `q/dq/tau` at the model timestep with
controlled noise/jitter/bias variants, write the P1 dataset format, convert
to `ModelSequences`, run the P2 staged fit from nominal, evaluate held-out
protocols, emit the run-directory report (manifest, parameters,
uncertainty, plots).

Exit gate: no-mismatch dataset yields numerical-only residual; hidden
friction and released inertial groups recovered within P2 tolerances from
on-disk data; held-out prediction materially better than nominal under
thresholds fixed in advance; repeatable from a clean environment.

### P5 — ROS-sim end-to-end pipeline

Play the same compiled protocols through the existing
`mujoco_ros2_control` FER plant with the standard position-mode
`JointTrajectoryController` (one `FollowJointTrajectory` goal per
protocol), record MCAP (joint states, controller state, applied effort,
clock), convert through the same contract into the same fit.

Exit gate: identical protocol executes with correct start/hold/abort
behavior; converted dataset passes `validate-dataset`; fit on ROS-sim data
matches the P4 result within a stated tolerance; recording verified off the
control path.

### P6 — Real-robot campaign readiness and data collection

Agimus position-mode playback of the identical player code path: preflight
(protocol/model hashes, start-state check, controller exclusivity, robot
mode, recorder readiness), controlled move to protocol start, explicit
hold/cancel behavior, and the recording set (`q/dq/tau_J`, `tau_J_d`,
desired states, robot mode/errors, timing). Campaign order: friction family
(repeated cold/warm where practical), holds, inertial family, held-out
protocols. Every run auto-checked (completeness, rates, saturation,
dropouts) and accepted/rejected by manifest.

Exit gate: full campaign rehearsed in ROS sim through the hardware-facing
code path; first hardware protocol reviewed motion-by-motion with the user;
collected dataset immutable, checksummed, roles locked.

### P7 — Real-model fitting and held-out validation

Staged fit on the real dataset: nuisance (timing/bias/scale, not exported),
friction+damping, armature, then only well-conditioned inertial groups.
Metrics fixed before held-out evaluation, including the pregrasp-relevant
low-speed/standstill regime and consistency with the recorded friction
lower bounds above.

Exit gate: aggregate held-out improvement with bootstrap 95% lower bound
above zero; no joint metric worse by >10% without reviewed explanation;
friction stage alone materially improves low-speed/reversal residuals —
if `frictionloss`+`damping` cannot, inertial fitting pauses for a
model-class review (Stribeck/temperature) rather than absorbing the error.

### P8 — Export and consumer integration

Render the canonical parameter manifest into candidate hydrax and
`sbmpc_ros` MJCFs (whitelisted fields only; source-hash mismatch fatal);
reload/compare; cross-version rollout gate (3.10 vs MJX 3.8 vs ROS
runtime); consumer regression suites; then separate reviewed diffs to the
consumer repositories, keeping the previous nominal model as rollback. The
final acceptance test is the original problem: the pregrasp task in the
consumer stack, sim then (on approval) real, with terminal EE error
compared against the 62 mm baseline.

## Decision log (amendments)

| ID | Date | Decision |
| --- | --- | --- |
| D021 | 2026-07-23 | Position-mode `JointTrajectoryController` playback with measured `tau_J` as the fit torque input, in sim and on hardware. Supersedes D015 and the D018 controller-parity requirement; the fit is controller-agnostic. |
| D022 | 2026-07-23 | Freeze the P1 artifact/contract layer as complete. New contract features only when a phase needs them, inside that phase's diff. |
| D023 | 2026-07-23 | Adopt this v2 plan; v1 archived at `docs/archive/implementation_plan_v1.md`. v1 stop/rollback rules and scientific failure conditions remain binding. |
| D024 | 2026-07-28 | Delete the artifact/governance layer (`artifacts/`, sealed bundles, dataset catalog, schema registry) and the superseded standalone effort plant; flatten the package. Reverses D022's freeze: the layer was ~75 % of the repository and served no identification purpose. Artifacts keep a JSON manifest, an NPZ, a content hash and `checksums.sha256` — nothing more. |
| D025 | 2026-07-28 | Report identification the classical way, built on top of `mujoco.sysid` rather than expecting it from the toolbox: regressor condition number for excitation quality, zero-phase 4th-order Butterworth preprocessing, relative standard deviations with a 20 % freeze rule, and measured-vs-reconstructed torque plots (`diagnostics.py`, `preprocessing.py`). |
| D026 | 2026-07-28 | Two excitation families, because one cannot serve both: friction (shared-schedule constant-velocity cruises) and inertial (per-joint Fourier series at distinct base frequencies, limit-scaled, selected by `cond(Y)`). The friction family is provably unusable for inertia — its joint velocities are perfectly correlated. |
| D027 | 2026-07-28 | Hardware playback targets `joint_trajectory_controller/JointTrajectoryController`, already registered in the Agimus stack. It runs **effort-mode with an internal PID**, so the torque driving each joint is the JTC's commanded effort — a known quantity, which is what the fit consumes. Supersedes D021's "measured `tau_J` as the fit torque input". Open: the FR3 compensates gravity internally, so commanded effort excludes gravity; harmless for friction, must be handled explicitly before inertial identification. |
| D028 | 2026-07-28 | The *reason* effort mode is required, settled in slice A: friction is precisely the term that changes with actuation mode, and the deployment stack drives this robot in effort mode (`linear_feedback_controller`). Whatever the robot does internally under a torque command must be identical during identification and during deployment, or the friction identified here is not the friction the planner meets. Identifying under the robot's internal position controller and deploying under torque control would measure the wrong thing. Effort mode is therefore a requirement, not a convenience. |
| D029 | 2026-07-28 | **The ROS-simulation plant runs gravity-free.** Under a torque command the FER compensates its own weight, so the effort a controller sends is the effort *on top of* gravity compensation; `sbmpc_ros` encodes the same fact from the other side (`remove_gravity_compensation_effort`: true on the robot, false in simulation). Measured, not assumed: a gravity-enabled simulated plant left 36.9 mrad of standing error on joint 4 under these gains and failed the player's start-state check for a reason hardware does not have. Consequence for P7, now explicit rather than open: commanded effort cannot identify link masses, because the gravity torque that reveals them is not in it. Mass and COM must come from the measured link-side `tau_J` channel instead. |
| D030 | 2026-07-28 | **Campaign revision r2.** The r1 inertial protocols were not executable: per-joint base frequencies were incommensurate with the protocol duration, so each joint stopped mid-cycle and the trailing settle segment stepped back to the home pose within a single 10 ms sample — up to 0.59 rad. Through the trajectory controller that saturated four joints and aborted the goal on a path-tolerance violation, at full speed and again at half speed. Base frequencies are now integer multiples of the `1/duration` fundamental and the series is sampled through its endpoint, so it lands on home exactly; the trajectory-consistency check that catches this class of defect is enabled for the inertial family, where it had been disabled. Found by the slice A rehearsal in ROS simulation, never on hardware. r1 was never approved and is not kept. |
| D031 | 2026-07-28 | **Nothing that runs inside a ROS process imports MuJoCo.** A ROS environment carries whatever `mujoco` its own packages put on the path — in this workspace a `build/` data directory shadows the real package as an attribute-less namespace package. The bundle *reader* is therefore split into `protocol.py` (NumPy only) from the bundle *writer* in `excitation.py` (needs the model), the package `__init__` imports nothing, and the simulation scene is generated ahead of the launch in the project's own environment. Gated by a test that imports the robot-side modules in a subprocess and asserts MuJoCo and SciPy stayed out of `sys.modules`. |
| D032 | 2026-07-28 | **No consumer repository is required to use this one** (user requirement). The ROS-simulation plant is built from the hydrax nominal model — the model this project identifies — renamed to the ROS joint convention, rather than borrowed from `sbmpc_ros`. `ModelPaths.require()` now demands only the nominal model; the deployment-target MJCF became optional (`has_ros_overlay`), and the checks that compare against it skip when it is absent. Gated by a test that points the deployment-target path at nothing and asserts the contract, the nominal model and the simulation plant all still work. Side benefit: the simulated plant is now literally the model being identified, so it cannot drift from it. |
| D033 | 2026-07-28 | **An identified model is exported through a whitelist, or not at all** (`export.py`). Joint friction, damping and armature are identification outputs; masses, inertias, link poses, joint ranges, actuator limits and geometry are compared against the nominal model and must be identical — a change there is a different robot, not a better estimate of this one. The exported file is reloaded and must reproduce the in-memory model's held-out prediction, and a manifest pins both files by SHA-256 so a candidate model traces back to the fit that produced it. `export_identified_model` returns the verification, so a caller cannot obtain a model without its evidence. Gated including the negative case: a tampered link mass is rejected. |

## Next review unit

P2 slice 1 (implemented, review open): `fer_mujoco_sysid.sysid` adapter with
the friction parameter group, measured-run windowing, and the fit driver;
gates in `tests/test_synthetic_recovery.py` (nominal no-op, ≤1%
single-parameter frictionloss and damping recovery, frictionloss/damping
separation on slow reversals with |correlation| < 0.98).

Two conventions fixed during the slice and now binding on every future data
converter (P4/P5/P6): (a) long open-loop arm rollouts diverge — residuals
must use short windows re-initialized from measured states
(`measurement_sequences(..., window_s=...)`); (b) `mujoco.rollout` emits the
pre-step sensor row stamped with the post-step time, and measured data must
mirror that skew (see `MeasuredRun`), never re-stamp one side alone.

P2 slice 2 (implemented, review open): `armature_parameters` (fitted only
after friction, on dynamic excitation), `fit_staged` (friction → armature →
friction-refit ordering; each stage's result persisted into every stage's
spec), `fit_multistart` (deterministic in-bounds restarts, value-spread
report), and conditioning evidence on `FitResult` (scaled-Jacobian singular
values, conditioning ratio vs the 1e-6 threshold, bound proximity/hits,
correlations vs the 0.98 freeze limit). Gate evidence in
`tests/test_synthetic_recovery.py`: armature ≤1% on dynamic data; staged
fit measured 13%/19% friction bias at stage 1 (wrong armature), armature
5% at stage 2, friction ≤2% after the stage-3 refit with strict
improvement asserted; multistart starts agree ≤1% with clean conditioning.
`default_report` integration deliberately deferred to the P4 run-directory
report (it generates a full artifact bundle).

P2 slice 3 (implemented, review open — completes P2): `inertial_parameters`
(per-body inertia via the toolbox's physically consistent pseudo-inertia
parameterization; `MOVING_LINK_BODIES`/`TOOL_COMPOSITE_BODIES` encode the
convention that the rigid tool composite is represented by link7 with
hand/fingers frozen; includes a MuJoCo 3.10 workaround for the scalar
`MjsBody.mass` setter in the Mass path) and `conditioning_report` — a
pre-fit finite-difference identifiability check, the plan's freeze-before-
fitting workflow, reusable for P3 protocol conditioning. Gate evidence:
link3 mass recovered to 0.02%; estimating hand alongside link7 is rejected
with conditioning ratio ~5e-25 vs ~3e-4 for the link7-composite convention
(a 21-order-of-magnitude nullspace signature); a hidden quadratic-drag
truth (outside the damping+frictionloss class) stalls the friction stage at
a floor ~178x the in-class control across two amplitude regimes, and the
subsequent armature stage stays within 7% of nominal while explaining <1%
— detected as structured residual, not absorbed. Test suite split
(2026-07-27): `./scripts/test` = fast development set (~3 min, excludes the
`slow`-marked recovery fits); `./scripts/test-all` = complete 303-test gate
suite (~12 min) required green before review handoff.

P3 slice 1 (implemented, review open): `sysid/excitation.py` — friction-
family protocols as jerk-limited trapezoids (S-curves; user decision
2026-07-27, preferred over min-jerk because constant-velocity cruises make
friction directly readable and plottable per speed): one bidirectional
pass per cruise speed (default 0.05/0.15/0.4 rad/s, low speeds
over-represented), holds at the stops, matched reversal configurations,
seeded per-joint amplitude jitter, all-joint shared schedule. Fail-closed
validation before anything is written: position margins, FER
velocity/acceleration/jerk limits, array self-consistency, and an
inverse-dynamics predicted-torque check. Bundles follow the frozen P1
motion-protocol contract (`command_interface: joint_trajectory` per D021,
provenance-verified source-model hash from `contracts/nominal_sources.toml`,
checksums) and must pass `validate_motion_protocol` +
`verify_checksum_manifest` before the writer returns; revisions are
immutable. Gates in `tests/test_excitation.py`: determinism (identical
`content_sha256` for identical spec), cruise-plateau presence, limit
rejection, bundle round-trip, and (slow) a `conditioning_report` check
that simulated playback identifies the full 14-parameter friction block
with every per-joint frictionloss/damping correlation below the freeze
limit. Collision checking against the experiment cell is deliberately
deferred (needs the cell geometry; small home-centred amplitudes are safe
by construction, and the P3-final gate still requires it before hardware).

P3 slice 2 (implemented — **campaign awaiting user review**):
`sysid/campaign.py` defines the committed campaign as fixed seeded specs —
`fer-friction-a` (seed 101) and `fer-friction-b` (seed 102) as canonical
protocols, `fer-friction-holdout` (seed 901, different amplitudes and
cruise speeds) reserved for held-out validation (roles themselves are
assigned later in dataset splits, per the P1 contract). Bundles live under
`protocols/<id>/<revision>/` (100 Hz knot grid, compressed, ~250 KB each,
byte-reproducible via the fixed campaign timestamp);
`./scripts/generate-protocols --check` regenerates everything and compares
content hashes, and a fast test runs the same drift guard. Review material
in `protocols/review/`: per-joint position/velocity plots (holds
shaded, cruise speeds marked) and `summary.md` with simulated-playback
conditioning evidence (ratios 2.4e-2/2.6e-2/3.0e-2, worst per-joint
frictionloss/damping correlation 0.73/0.73/0.83 — all clean).

**Review checkpoint (open): the user reviews the motion in
`protocols/review/` before these motions are treated as the approved
campaign.** Changes are cheap now (edit the specs, regenerate as r2);
after P4/P5 build on them they are not.

Visual/demonstration layer added 2026-07-27 after user feedback that static
joint plots are not reviewable and that simulated identification
performance is the thing worth communicating (`sysid/demo.py`,
`./scripts/identification-demo`):

- an MP4 per protocol showing the arm executing the motion in MuJoCo
  (regenerable, not committed), plus `protocols/README.md`
  explaining why several cruise speeds exist (one speed cannot separate the
  Coulomb offset from the viscous slope) and what is and is not done yet;
- a full simulated identification round in `output/mujoco/`:
  hidden-truth friction of realistic magnitude (from the real breakaway
  bounds) → play the canonical protocols → fit the 14-parameter block from
  the frictionless nominal → judge on the **held-out** protocol in physical
  units, with per-joint friction-curve plots (measured cruise samples,
  truth, identified, nominal).

Result of that round: **all 14 parameters recovered to three decimals**;
held-out gripper prediction improved from **68.8 mm RMSE / 199 mm max
(nominal) to 0.02 mm / 0.15 mm (identified)**, worst joint 0.127 → 0.0001
rad. The nominal model's 68.8 mm held-out error landing near the real
robot's 62 mm standoff is a useful sanity signal on the chosen truth
magnitude — not evidence about the real robot.

### Hardware readiness (settled 2026-07-28)

* **Controller**: `joint_trajectory_controller/JointTrajectoryController` is
  registered in `agimus_franka_bringup/config/controllers.yaml`. The Agimus
  FR3 MoveIt config shows the intended Franka configuration —
  `command_interfaces: [effort]`, `state_interfaces: [position, velocity]`,
  gains p=600/d=30 (joints 1-4), 250/10, 150/10, 50/5. That file is
  `fr3_*`-named; this robot is a **FER**, so an equivalent FER config
  (`fer_joint1..7`, `arm_id: fer`) must be written here.
* **Torque channel**: resolved. Effort-mode JTC computes the commanded
  effort itself, so it is known exactly and needs no inference from
  `tau_J`/`tau_J_d` semantics.
* **Workspace**: the robot is bolted to a table and otherwise unobstructed,
  so collision checking reduces to a table plane below the base plus the
  existing joint limits.

Remaining before hardware, two reviewed slices:

* **Slice A — player**: *implemented, review open.* `config/fer_sysid_controllers.yaml`
  (effort-mode JTC on `fer_joint1..7`, Agimus FR3 gains, per-joint torque
  clamps at the FER limits, path/goal tolerances that abort a runaway),
  `playback.py` (time scaling, quintic move-to-start, preflight decisions —
  all ROS-free and unit-gated), `ros/player_node.py`, `ros/scene.py`,
  `urdf/fer_sysid_{real,mujoco}.urdf.xacro`,
  `launch/play_protocol.launch.py` and `scripts/run-protocol`.
  Evidence: all six r2 protocols played end to end through
  `mujoco_ros2_control` at full speed, plus a `--dry-run` path that runs
  every check and commands nothing. Gates in `tests/test_playback.py` and
  `tests/test_ros_scene.py`.
* **Slice B — recorder and converter**: MCAP recording of the JTC commanded
  effort and the broadcaster's measured state, converted into the
  `MeasuredRun` the existing pipeline already consumes.

Both are exercised in ROS simulation through the identical code path first;
the hardware step then changes only the launch target.

What the slice A rehearsal caught, none of it on hardware: the r1 inertial
protocols were unplayable (D030); the simulated plant needed gravity
disabled to represent the robot (D029); and no ROS process may import
MuJoCo (D031). This is the argument for the rehearsal step existing.

Two things are deliberately **not** settled by slice A and belong to the
first hardware session:

* **Speed scaling.** In simulation every protocol tracks at full speed with
  peak error ≤ 37 mrad and no torque saturation. The real arm carries
  friction the simulated one does not, so the first hardware run of each
  family should still go at `--speed-scale 0.5` and be compared against the
  simulated tracking before running at 1.0.
* **Whether the JTC's PD is stiff enough on hardware.** With `i: 0` the
  standing error is friction over gain — from the recorded breakaway bounds,
  roughly 2 mrad on joints 1–4 and about 11 mrad on joint 7. That is inside
  the 20 mrad start-state tolerance, but it is an estimate from simulation
  and the first run should confirm it.

Proposed next unit — either P3 slice 3 (inertial multisine family) or,
preferably, start P4 (standalone end-to-end pipeline) on the friction
campaign first so the friction goal keeps moving; the inertial family can
land while P4 is under review.
