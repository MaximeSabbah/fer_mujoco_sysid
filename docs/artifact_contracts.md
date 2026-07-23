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

The first M1 implementation slice provides strict JSON/NPZ I/O, all version-1
structural schemas, scientific-content hashing, file-reference/checksum
verification, and full semantic validators for motion protocols and normalized
trajectories. The acquisition, split, parameter, fit-result, and dataset
schemas are structural contracts in this slice.

Complete sealed-bundle finalization, semantic validation across those remaining
artifact types, split-overlap enforcement, the example dataset, and the
dataset-validation command are the next M1 slice. Passing the current
checksum-manifest helper therefore proves the listed files, but does not by
itself certify a complete sealed bundle.

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

A protocol-validation certificate is a separate future artifact keyed by the
protocol, model, scene, payload, and constraint-profile digests. A motion that
was checked in one environment is not implicitly certified in every
environment.

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

- the exact compiled protocol and validation-certificate digests;
- backend: standalone MuJoCo, ROS MuJoCo, or Agimus FER;
- source model, robot, hand, payload, controller, software and container
  revisions;
- start/end times and observed outcome;
- raw files, topics or streams, message counts and expected rates;
- all available clock domains; and
- complete signal-source and torque semantics.

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
source message/timestamp bounds. M1b will decide and validate whether a
per-output source-index/bracket map is required in addition to those bounds.

### Boundary/interval convention

For `M` physical transitions, the normalized core contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `state_time_s` | `(M + 1,)` | State-boundary times `t_0 ... t_M` |
| `control_time_s` | `(M,)` | Interval starts `t_0 ... t_(M-1)` |
| `q_rad` | `(M + 1, 7)` | Measured or ground-truth joint position |
| `dq_rad_s` | `(M + 1, 7)` | Measured or ground-truth joint velocity |

Both time arrays start at zero, `control_time_s` equals
`state_time_s[:-1]`, and the grid is fixed. A control row applies over the
half-open interval `[t_i, t_(i+1))`.

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
Cross-artifact resolution of the selected signal and rejection of a channel
with ambiguous composition are part of the next M1 semantic-validation slice;
they are not inferred by the current generic schema validator.

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

Splits are assigned to immutable trajectory identities and their lineage.
Segments, cached windows, or repeated derivatives from one source lineage
cannot accidentally cross fit, development, and held-out roles. Changing a
split creates a new dataset version and digest.

The code license does not license a dataset automatically. A published dataset
must provide its own license and citation. Machine paths, credentials, robot
network addresses, and unrelated ROS traffic are not public metadata.

## Integrity and identity

Artifact identifiers are readable names, not self-referential hashes.

The target sealed-bundle contract uses `checksums.sha256` to list the SHA-256
digest of each bundle file in sorted POSIX-relative path order, excluding
itself and `seal.json`. `seal.json` records the digest of the checksum
manifest. The M1a verifier checks every listed file and rejects malformed,
unsorted, duplicate, escaping, or mismatched entries. Exact file-set closure
and `seal.json` verification are deliberately reserved for the M1b
finalizer/verifier; until then, an artifact is not considered sealed.

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
