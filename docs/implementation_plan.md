# FER MuJoCo system-identification implementation plan (v2)

Last updated: 2026-07-23. Supersedes
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

- **Playback is position-mode (D021).** Excitation protocols are played by
  the standard position-mode `JointTrajectoryController` (ROS sim and
  Agimus hardware alike); the fit consumes measured `q`, `dq`, and the
  measured link-side torque `tau_J`. The fit is controller-agnostic, so the
  v1 effort-mode trajectory controller and its controller-parity gates
  (D015/D018 machinery, old M4 parity work) are dropped. `tau_J` is the
  same link-side interface the MPC commands through, so friction identified
  from it is the friction the planner needs.
- **The artifact/contract layer is frozen (D022).** The existing schemas,
  sealed bundles, and `validate-dataset` command are the storage boundary
  as-is. M1 is declared complete. New contract features are added only when
  a pipeline phase concretely needs them, as part of that phase's diff.
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
  `panda.xml` is the nominal baseline; the ROS `fer_ros2_control.xml` is a
  deployment wrapper rendered from the same identified parameter manifest,
  never fitted independently.
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
| P1 | Artifact contracts (old M1, frozen per D022) | Complete-frozen |
| P2 | Sysid engine and synthetic parameter recovery | Slice 1 implemented (review open) |
| P3 | Excitation protocol generation | Planned |
| P4 | Standalone end-to-end pipeline | Planned |
| P5 | ROS-sim end-to-end pipeline | Planned |
| P6 | Real-robot campaign readiness and data collection | Planned |
| P7 | Real-model fitting and held-out validation | Planned |
| P8 | Export and consumer integration | Planned |

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

Proposed P2 slice 2: armature parameter group, multistart, and the
conditioning/uncertainty report (singular values, bound proximity) feeding
`default_report`.
