# Artifact contracts

This document defines the portable data boundary of `fer_mujoco_sysid`.
Runtime adapters may read ROS messages, MCAP recordings, or native MuJoCo
arrays, but they must convert those inputs into these contracts before fitting.

The contracts have four goals:

1. preserve the meaning and timing of every signal;
2. make every derived value traceable to immutable source data;
3. reject ambiguous data instead of silently guessing; and
4. keep the public artifacts independent of Python pickles, ROS installations,
   and machine-local paths.

The normative schemas are the versioned JSON Schema resources in
[`src/fer_mujoco_sysid/schemas/`](../src/fer_mujoco_sysid/schemas). They ship
with the Python package so validation behaves the same in a source checkout
and an installed wheel. This document explains their intended use.

## Current implementation boundary

M1a provides strict JSON/NPZ I/O, versioned structural schemas,
scientific-content hashing, file-reference validation, and full semantic
validators for motion protocols and normalized trajectories. M1b now also
provides deterministic bundle sealing, intrinsic validation of acquisition
runs, split assignments, identified parameters, and fit results, plus
fail-closed local dataset closure and selected-torque eligibility checks.

`fer-mujoco-sysid validate-dataset` verifies a sealed dataset and all of its
sealed local protocol, run, trajectory, and split references without running
identification or accessing the network. A compact committed simulation
dataset remains in M1b. Optional `dataset.supersedes` and
`acquisition-run.protocol_validation` references are rejected by the current
closure validator until those referenced artifact contracts are implemented;
they are never silently omitted from validation.

## Formats

Metadata is strict UTF-8 JSON. Dense numerical data is stored in NumPy NPZ
archives containing only explicitly named numeric arrays.

- JSON is parsed with non-standard constants such as `NaN` and `Infinity`
  disabled.
- Unknown manifest fields are rejected.
- NPZ is loaded with `allow_pickle=False`.
- Object arrays are rejected.
- Canonical numerical arrays use fixed little-endian dtypes.
- Units, joint ordering, shapes, clock domains, and signal semantics live in
  the JSON manifest rather than being inferred from an array filename.

The nominal-source contract remains TOML because it is a small, hand-maintained
project configuration. TOML is not a second experiment-artifact format.

## Canonical arm convention

All portable FER arm artifacts use this joint order:

```text
fer_joint1
fer_joint2
fer_joint3
fer_joint4
fer_joint5
fer_joint6
fer_joint7
```

The Hydrax adapter maps these semantic identities to `joint1` through
`joint7`. Source-specific names are recorded at the acquisition boundary, but
they never change the canonical array ordering.

Unless a schema says otherwise, physical quantities use SI:

| Quantity | Unit |
| --- | --- |
| Time | `s` or exact integer `ns` |
| Joint position | `rad` |
| Joint velocity | `rad/s` |
| Joint acceleration | `rad/s^2` |
| Joint torque | `N*m` |
| Length | `m` |
| Mass | `kg` |
| Inertia | `kg*m^2` |
| Temperature | `K` |

## 1. Motion protocol

A motion protocol is the reusable asset intended to be shared in this
repository:

```text
protocols/<protocol_id>/<revision>/
  protocol.json
  desired.npz
  README.md                 # optional
  checksums.sha256
```

The compiled arrays are authoritative for playback. Generator parameters alone
are not sufficient because a generator can change between software versions.
The protocol records both the compiled motion and the generator provenance.

`desired.npz` contains:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `time_from_start_ns` | `(N,)` | `<i8` | Exact trajectory knot time |
| `q_rad` | `(N, 7)` | `<f8` | Desired joint position |
| `dq_rad_s` | `(N, 7)` | `<f8` | Desired joint velocity |
| `ddq_rad_s2` | `(N, 7)` | `<f8` | Desired joint acceleration |

Time begins at zero and is strictly increasing. The manifest records the
command interface, exact sample period, segment boundaries, start/end
requirements, source model and scene, end effector and payload, generator
version and deterministic seed. The continuous interpolation/playback contract
is finalized with the player in M3; it is not inferred by the M1a validator.

Fit, development, and held-out roles do not belong to a protocol. They belong
to a versioned dataset split.

A future machine-generated simulation record will bind the protocol to the
model, scene, payload, and constraint-profile digests that were checked. The
normal command-line workflow will create and consume this record automatically;
the operator will not manage it manually. A motion checked in one environment
is not implicitly valid in every environment.

## 2. Acquisition run

An acquisition run is the immutable record of executing exactly one protocol
instance:

```text
<external-run-root>/<run_id>/
  run.json
  raw/
    rosbag/                  # ROS run, normally MCAP
    # or native simulation payloads
  notes.md                   # optional, finalized before sealing
  checksums.sha256
  seal.json
```

The normal output root is outside the source repository. Real-robot data is
local by default and is published only through an explicit curation step.

The run manifest pins:

- the exact compiled protocol digest;
- backend: standalone MuJoCo, ROS MuJoCo, or Agimus FER;
- source model, robot, hand, payload, controller, software and container
  revisions;
- start/end times and observed outcome;
- raw files, topics or streams, message counts and expected rates;
- all available clock domains;
- the timestamp, clock, and recorded source of protocol knot zero; and
- complete signal-source and torque semantics.

Signal provenance is backend-aware. Standalone plant signals may originate
from MuJoCo or an artifact-local file, while scheduled references originate
from the protocol player and name the exact protocol artifact. ROS MuJoCo and
Agimus FER signals must name recorded ROS messages; a hardware run cannot be
relabelled from direct MuJoCo fields. Dataset closure also requires the
executed end effector and payload to equal the referenced protocol context.
The source model used to generate a deliberately perturbed synthetic run may
differ from the nominal protocol model, so that identity is recorded rather
than forced equal.

`protocol_timing` is the acquisition-time anchor for the compiled motion. It
records the timestamp of protocol knot zero in one declared run clock and names
the scheduled-position signal and source from which that timestamp was
obtained. The anchor must share that signal's clock. ROS uses message zero of
the repository-owned protocol-reference topic and its `Header.stamp`, and the
timing source must share that signal's stream. Standalone MuJoCo uses sample
zero of the real `mjData.time`; the scheduled value remains
`protocol_player.q_rad`, because desired references are not MuJoCo state
fields. ROS source topics must declare the matching clock and be required for
conversion. This prevents an unrelated field or an internally consistent
group of shifted source windows from being presented as a different interval
of the protocol.

Collection may use a temporary staging directory. Finalization atomically
creates a sealed bundle. Project APIs do not modify a sealed raw bundle.
Corrections, redaction, normalization, and annotations create new derived
artifacts that reference it.

Acquisition outcome, data-quality assessment, scientific split, and
publication status are deliberately separate concepts. A controller fault is
an observed run outcome; it is not a train/test label.

## 3. Normalized trajectory

A normalized trajectory is a derived, portable, contiguous interval:

```text
trajectories/<trajectory_id>/
  trajectory.json
  signals.npz
  checksums.sha256
  seal.json
```

The raw recording remains the source of truth. The normalized manifest records
the source bundle digest, exact source bounds, converter revision and
configuration, clock alignment, resampling method, exclusions, and selected
source message/timestamp bounds. It also binds the trajectory to one named
protocol segment and an exact half-open protocol-knot interval. Dataset closure
requires that interval to lie inside the segment and have the same duration as
the normalized state grid.

The normalized artifact names `protocol_reference_signals` for compiled
desired position, velocity, and acceleration, plus effort feedforward whenever
the protocol contains it. These are native, unshifted, identity-scaled traces
from the run recording. Their raw message indices must equal the selected
protocol-knot indices, the full raw reference streams must have the protocol's
exact cardinality, and their numerical values must equal the compiled arrays.
The desired-position binding must name the exact acquisition signal selected by
`protocol_timing`; a separate controller-desired diagnostic stream cannot
substitute for it.
State references include every selected knot; effort feedforward contains the
executed intervals only and excludes the stored terminal knot. This prevents a
converter from relabeling an equal-duration source window even when position
repeats but another reference channel differs.

ROS backends must therefore record a repository-owned scheduled-reference
topic at compiled-knot cardinality in addition to any controller-state desired
signal used for diagnostics. Standalone playback records the same scheduled
references alongside the plant signals. A future converter review will decide
whether a per-output source-index/bracket map is required in addition to these
reference traces and the existing source bounds.

Every measured-state and effort binding must also occupy the same canonical
source-time window as the scheduled references. State endpoints align with the
position-reference endpoints; control endpoints align with the corresponding
half-open control interval. The permitted discrepancy is one half normalized
sample period plus the declared source-clock alignment residual. Identity and
affine clock mappings are supported for this gate; piecewise mappings fail
closed until their numeric mapping format is implemented.

The position-reference endpoints are additionally anchored to
`protocol_timing`: for selected knot interval `[i, j)`, they must map to
`protocol_start + i * period` and `protocol_start + (j - 1) * period`. This
binds both the absolute interval and its duration to the compiled protocol
grid, rather than only checking that all normalized signals agree with one
another.

The catalog validates sealed normalized evidence, declared raw indices, and
cross-signal times; it does not re-decode an opaque MCAP recording. Thus this
gate detects inconsistent or relabelled artifact metadata, while backend
converter tests remain responsible for proving that each source binding was
decoded correctly from the immutable raw bytes.

### Boundary/interval convention

For `M` physical transitions, the normalized core contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `state_time_s` | `(M + 1,)` | State-boundary times `t_0 ... t_M` |
| `control_time_s` | `(M,)` | Interval starts `t_0 ... t_(M-1)` |
| `q_rad` | `(M + 1, 7)` | Measured or ground-truth joint position |
| `dq_rad_s` | `(M + 1, 7)` | Measured or ground-truth joint velocity |

Both time arrays start at zero, `control_time_s` equals
`state_time_s[:-1]`, and the grid is fixed to the selected compiled protocol
knots. A control row applies over the half-open interval
`[t_i, t_(i+1))`.

Every normalized signal explicitly declares `time_base: state` or
`time_base: control`. Its first dimension must therefore be exactly `M + 1` or
`M`. The core `q_rad` and `dq_rad_s` signals use the state time base, and at
least one joint-effort candidate must use the control time base.

The canonical floating-point time arrays are generated as
`arange(sample_count, dtype="<f8") * (period_ns * 1e-9)`. Validators require
that exact representation; writers must not accumulate the period iteratively.

Control effort is aligned causally using zero-order hold. State interpolation
or filtering is selected independently and recorded. Conversion trims to the
common valid interval and rejects unsupported gaps; it never relies on
endpoint clamping.

For `resampling.method: none`, the selected raw interval and normalized array
must have equal cardinality and the applied time shift must be zero. Other
resampling methods remain explicit metadata rather than being inferred from
sample counts.

Invalid samples do not enter MuJoCo `TimeSeries` objects as NaNs. The converter
splits the recording into complete contiguous intervals and records why any
source region was excluded.

### Torque channels

There is no generic `effort` or authoritative `torque` field. Each retained
channel has an identifier, source topic/message/field, stage, sign convention,
unit, joint order, and structured composition:

- gravity included, excluded, or unknown;
- Coriolis included, excluded, or unknown;
- before, after, or unknown relative to rate limiting; and
- controller, hardware, sensor, simulator, estimator, or model origin.

Expected FER channels include:

- controller effort request;
- Agimus/libfranka desired link-side torque `tau_J_d`;
- Agimus/libfranka measured total link-side torque `tau_J`;
- MuJoCo actuator contribution `qfrc_actuator`; and
- estimated external joint torque.

They remain separate even when two recordings happen to be numerically close.
An unknown composition is valid diagnostic data but is not fit-eligible.

The normalized artifact preserves candidate inputs but does not choose the
forward-model input by name. The fit result selects one explicit channel plus
any transformation after its semantics have been validated.

The version-1 fit-result schema records that selection structurally.
`validate_fit_against_dataset` resolves the selection across every scientific
trajectory and rejects unknown composition, incoherent stage/location
semantics, controller commands before rate limiting, and disagreement with the
raw acquisition descriptor. Version 1 additionally requires the selected
effort to have an identity numeric transform, causal `none` or
`zero_order_hold` resampling, and zero applied time shift. A later structured
fit contract may relax those restrictions by representing the transformation
or delay as an explicit fitted input; unstructured preprocessing is rejected.
The fit's nominal source model must equal the one canonical nominal source
shared by every scientific protocol. These checks are intentionally stronger
than the generic schema validator.

Numerically differentiated signals receive distinct names and provenance.
They never replace measured `dq`, and differentiated `qdd` is not the primary
fit target.

### Clock alignment

Raw domains remain distinct:

- robot monotonic time;
- ROS message-header time;
- MCAP/bag receive or log time; and
- simulation `/clock`.

The normalized artifact defines one relative experiment clock and an explicit
identity, affine, or piecewise mapping from every source clock it uses. The
mapping records its evidence, valid interval, RMS error, maximum error, and
whether extrapolation was forbidden.

Every normalized signal is timestamped on the canonical clock while its source
binding retains the raw source-clock ID and selected message/timestamp bounds.
Each used noncanonical source must have exactly one mapping to the canonical
clock, its declared valid interval must cover all selected source samples, and
extrapolation must be forbidden.

Absolute ROS timestamps are retained as integer nanoseconds where needed for
traceability. They are not passed to MuJoCo, whose rollout time starts at zero.

## 4. Fit and export result

An identification result is immutable and self-contained:

```text
results/<result_id>/
  result.json
  parameters.json
  numerical.npz
  metrics.json
  models/
    hydrax.xml
    sbmpc_ros.xml
  compatibility.json
  report.html
  plots/
  checksums.sha256
  seal.json
```

`result.json` pins the source model, normalized trajectories, dataset split,
dependency lock, project revision, command, random seeds, parameter stages,
optimizer configuration, residual definition, weighting, window recipe and
termination status.

`parameters.json` is the canonical reusable physical result. It stores
realized SI-valued parameters rather than an optimizer's transformed
coordinates. Each parameter records its semantic target, nominal, initial and
fitted values, bounds, uncertainty, identifiability status, and whether it hit
a bound. Delay, torque scaling and sensor bias remain explicit nuisance
parameters and are never silently exported into MJCF.

Covariance, correlation, Jacobian summaries and optimization traces use
numeric NPZ. Pickle files are not portable artifacts. Metrics are split-labelled
and preserve nominal-versus-identified comparisons. HTML and plots are
presentation derivatives, not the numerical source of truth.

There is one canonical physical parameter result. Hydrax and `sbmpc_ros`
MJCFs are deterministic exports from it and must pass target-specific
structural and runtime compatibility checks.

## Dataset catalog and splits

A published or locally curated dataset is a versioned catalog over protocols,
runs and normalized trajectories:

```text
datasets/<dataset_id>/<version>/
  README.md
  LICENSE
  CITATION.cff
  dataset.json
  splits.json
  checksums.sha256
```

`splits.json` uses the roles:

- `fit`;
- `development`;
- `held_out_test`;
- `diagnostic_only`; and
- `excluded`, with a reason.

Splits are assigned to immutable trajectory identities, selected executable
protocol intervals, and their lineage. For each trajectory, the validator
fingerprints the executable joint order, command interface, exact sample
period, adjusted array contracts, and canonical command values independently
of protocol artifact IDs, titles, and enclosing unused knots. State references
include both interval boundaries; effort feedforward includes only the
physical control intervals, so a non-executed terminal effort knot cannot
create a false distinction. Floating signed zero is normalized before hashing,
so `-0.0` cannot create a false distinction from `+0.0`. Two trajectories with
the same interval execution fingerprint cannot occur in different `fit`,
`development`, or
`held_out_test` partitions, including when one interval is embedded inside a
longer protocol. Diagnostic-only use does not consume a scientific partition.
Segments, cached windows, copied protocols, or repeated derivatives from one
source lineage cannot accidentally cross scientific roles. Changing a split
creates a new dataset version and digest.

The code license does not license a dataset automatically. A published dataset
must provide its own license and citation. Machine paths, credentials, robot
network addresses, and unrelated ROS traffic are not public metadata.

## Integrity and identity

Artifact identifiers are readable names, not self-referential hashes.

The sealed-bundle contract uses `checksums.sha256` to list the SHA-256 digest
of each bundle file in sorted POSIX-relative path order, excluding itself and
`seal.json`. `seal.json` records the digest of the checksum manifest.

`finalize_bundle` writes each control file atomically, refuses to overwrite an
existing seal, and is intended to run inside a staging directory before that
directory is published. `verify_sealed_bundle` validates the packaged seal
schema, canonical checksum-manifest bytes, every payload digest, the exact
regular-file set, and the absence of symlinks or special files. The older
`verify_checksum_manifest` helper intentionally verifies listed entries only;
it does not certify sealing.

Where an artifact reference carries `bundle_sha256`, that digest is the
SHA-256 of the canonical `checksums.sha256` bytes pinned by `seal.json`.

A scientific content fingerprint is separate from byte-level bundle
integrity. Version 1 hashes canonical manifest metadata and each array's key,
dtype, shape and C-order bytes. It excludes top-level creation time and the
byte-level hash, size and media type of numeric NPZ references, so different
ZIP encodings of the same arrays produce the same fingerprint. It deliberately
retains artifact IDs, provenance, relative logical paths, and locators; those
fields are part of version-1 content identity. It therefore does not claim
that arbitrary renaming or relocation preserves the fingerprint.

Paths inside manifests are relative, contained within the artifact bundle, and
use POSIX separators. Absolute paths and `..` traversal are rejected.

## Migration policy

Every manifest carries an exact schema discriminator such as
`fer-mujoco-sysid/normalized-trajectory@1`. Unknown versions fail closed.

A schema migration reads an immutable version-N artifact and writes a new
version-(N+1) artifact with explicit `derived_from` provenance. It never
rewrites or silently upgrades the original. Frozen valid and invalid fixtures
protect each supported version.

## MuJoCo adapter boundary

The portable contracts are not MuJoCo Python objects. The project-owned adapter
will:

1. validate and load a normalized trajectory;
2. create an in-memory identification-only model copy with named joint-position
   and joint-velocity sensors;
3. construct the initial `mjSTATE_FULLPHYSICS` state from the first `q/dq`;
4. append one terminal duplicate control at `t_M`;
5. construct public `mujoco.sysid.TimeSeries` and `ModelSequences` objects with
   explicit signal names; and
6. fit with MuJoCo's implicit control resampling disabled.

MuJoCo drops the terminal control row and predicts the `M` post-step states
corresponding to measurements at `t_1 ... t_M`. The duplicate exists only in
the adapter and is never represented as a physical interval in the portable
artifact.

The source and exported controller MJCFs do not retain the temporary sensors.
MuJoCo's public containers are used in memory; their NPZ helpers may be used as
trusted caches, but they are not the public data contract.
