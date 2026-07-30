# FER hardware campaign — 2026-07-30

The first identification data recorded from the physical robot. Every other
artifact in this repository can be regenerated; this cannot, without the arm
moving again. That is why it is committed here rather than left under
`output/`, which is disposable by convention.

Fit it with:

```bash
./scripts/identify campaigns/real-2026-07-30 --output <new-directory>
```

## What was recorded

Six protocols, played at full speed through the effort-mode
`JointTrajectoryController` on the Agimus FCI stack (robot `172.17.1.2`,
`ROS_DOMAIN_ID=29`). Each bundle carries its own analysis windows, so the fit
needs nothing from `protocols/` to interpret it.

| protocol | family | role | protocol content sha256 | loop rate | worst gap |
| --- | --- | --- | --- | --- | --- |
| `fer-friction-a` | friction | train | `513414de362775fc…` | 992 Hz | 5.5 ms |
| `fer-friction-b` | friction | train | `11ce877687e43c0b…` | 992 Hz | 5.7 ms |
| `fer-friction-holdout` | friction | holdout | `ec3bd11e2447ec21…` | 991 Hz | 6.2 ms |
| `fer-inertial-a` | inertial | train | `754dbee1dd3c2e9b…` | 990 Hz | 5.5 ms |
| `fer-inertial-b` | inertial | train | `1fd7e298b6529038…` | 991 Hz | 4.9 ms |
| `fer-inertial-holdout` | inertial | holdout | `394bdab92ec4af71…` | 991 Hz | 5.6 ms |

Nominal source model: `panda.xml`, sha256 `7e8dfc2155ff5b87…`. All six bundles
bind that same hash, so they are one campaign against one model.

## What is not here

The raw MCAP bags (248 MB) stayed in `output/real/*/raw/`. The committed
`recording/` bundles are everything the fit reads, and each one verifies
against its own `checksums.sha256`. Reconverting from the bags would only be
necessary if the converter's semantics changed — in which case the honest
answer is to record again.

## Conditions worth knowing when interpreting a fit

* **Loop rate.** The control loop held ~991 Hz against its 1 kHz target on a
  machine without an RT kernel; 1.4-1.7% of cycles arrived late, none dropped
  by more than 6.2 ms. The fit resamples onto the model's 2 ms grid, so the
  worst gap is three grid steps.
* **Torque channel.** The fit input is the controller's commanded effort
  (`tau_cmd_Nm`). It was measured to agree with the robot's own `tau_J_d`
  telemetry to **1.7 mNm** on every joint, which is the evidence that nothing
  is being clamped or gravity-compensated behind our back. `tau_J_d` and
  `tau_J` are recorded alongside as the cross-check, not as inputs.
* **Standing friction is visible in the approach.** Joint 7 parked 9-13 mrad
  off the commanded start pose on every single run — friction over gain, with
  `i: 0` by design. Peak commanded torque during the friction protocols was
  2.7 Nm.
* **The wrist friction curve falls with speed.** Measured medians on joint 7:
  0.52 Nm at 0.04 rad/s, 0.47 at 0.05, 0.30 at 0.15. Joints 5 and 6 do the
  same. That is a Stribeck characteristic, which MuJoCo's
  `frictionloss + damping` class can only express with a negative viscous
  coefficient it is not allowed to have — so a fit that leaves those joints'
  damping at the nominal value is reporting the data, not failing.
* **Joint 1 is the least excited joint.** In the inertial family it reaches
  27% of its acceleration limit where the wrist reaches 80%, because its
  velocity limit binds first. Expect `link1` inertial coordinates to be
  reported unidentifiable.
