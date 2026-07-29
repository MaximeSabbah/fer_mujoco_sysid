# Protocol review — what these motions are for

Read this first, then watch the videos (`*.mp4`), then look at
`../identification_demo/result.md` for what the identification actually
achieves in simulation.

Regenerate everything with `./scripts/identification-demo`
(videos are not committed; they are rebuilt on demand).

## The problem these motions exist to solve

The real robot parks ~62 mm short of its pregrasp target and stays there,
pushing constant torque against joints that do not move (run
`sbmpc_runs/pregrasp_real_20260722T132356Z`). The planner's MuJoCo model has
**no friction at all**, so it believes that push should be accelerating the
arm. To fix the controller we must first measure the friction the real robot
actually has, and put it in the model.

## What friction looks like, and why several speeds

Joint friction has two parts that behave differently:

```
friction torque = frictionloss * sign(velocity)  +  damping * velocity
                  \_______________________/         \_______________/
                   dry / Coulomb / breakaway         viscous
                   constant, flips with direction    grows with speed
```

Measured at **one** speed, these two are indistinguishable: one number, two
unknowns. Any pair `(frictionloss, damping)` that sums to the same torque
fits equally well.

So each protocol sweeps every joint back and forth at **three different
constant speeds** (default 0.05, 0.15 and 0.4 rad/s), in **both
directions**, with **stops in between**:

| Part of the motion | What it measures |
| --- | --- |
| constant-speed cruise (flat velocity plateau) | one point on the friction curve: inertia contributes nothing at constant speed, so measured torque = gravity + friction |
| three different cruise speeds | the slope (viscous damping) vs the offset (dry friction) |
| both directions | the sign flip of dry friction — and any asymmetry between directions |
| the slowest cruise (0.05 rad/s) | the near-standstill regime that caused the 62 mm error |
| holds at the stops | pure standstill: gravity only, plus whatever the joint holds against |

That is the entire reason for the "regimes": each one is a sample point on
the friction curve, and together they pin down its shape. The velocity
profile is a **jerk-limited trapezoid (S-curve)** — smooth ramps into a flat
cruise — chosen precisely so those flat plateaus exist and stay clean.

## The second family, and why one family cannot do both

The friction motions move every joint on **one shared schedule**, which is
what makes the cruise plateaus readable. That same property makes them
useless for inertia: the joint velocities come out perfectly correlated
(measured: 1.0000), so the data cannot tell which link's mass produced which
torque. Inertia needs the joints moving **independently** and accelerating
hard — the opposite of a clean cruise.

So there is a second family: a **Fourier series per joint, each at its own
base frequency**, amplitude-scaled until whichever of position, velocity or
acceleration binds first, and selected out of 24 random candidates by
regressor conditioning. That drops the inter-joint velocity correlation to
0.07 mean / 0.28 max and raises the acceleration from 13 % of the limit to
16 rad/s².

## The six protocols

| Protocol | Family | Purpose | Duration |
| --- | --- | --- | --- |
| `fer-friction-a` | friction | fitting data | ~42 s |
| `fer-friction-b` | friction | fitting data (different seed → different per-joint amplitudes) | ~42 s |
| `fer-friction-holdout` | friction | **never used for fitting** — different amplitudes and speeds, used only to judge the result | ~32 s |
| `fer-inertial-a` | inertial | fitting data for armature and link inertias | ~21 s |
| `fer-inertial-b` | inertial | fitting data (different seed) | ~21 s |
| `fer-inertial-holdout` | inertial | **never used for fitting** — different frequency mix | ~21 s |

In the friction family all joints move together on a shared schedule; each
joint has its own amplitude (~0.3 rad around the home pose) with a small
seeded variation, so the joints do not all trace identical paths.

Every protocol is checked before it is written: FER position, velocity,
acceleration, jerk and torque limits with a 20 % margin; clearance against
the table the robot is bolted to; and consistency between the position,
velocity and acceleration arrays. Each is byte-reproducible from its seed,
and every one of them starts and ends at rest at the home pose.

That last property is not decoration. In revision r1 the inertial protocols
did **not** have it — their per-joint frequencies did not divide the protocol
duration, so each joint stopped mid-cycle and the trajectory ended with a
step back to the home pose, up to 0.59 rad in a single 10 ms sample. Played
through the trajectory controller in ROS simulation, that saturated four
joints and aborted the run. r2 fixes it and the consistency check now
catches the whole class.

## What to check in the videos

- Does the motion look safe and sane in your cell (the arm stays near the
  home pose and returns to it)?
- Are the amplitudes big enough to be useful, small enough to be safe?
- Are the slow sweeps slow enough for the regime you care about?
- The inertial motions are much livelier than the friction ones — peak
  2.0 rad/s and 16 rad/s². Is that acceptable in your cell?
- Are ~42 s (friction) and ~21 s (inertial) acceptable run lengths?

Changing any of this is still cheap: edit the specs in
`src/fer_mujoco_sysid/campaign.py`, bump `CAMPAIGN_REVISION`, and regenerate.

## Status: what has actually been achieved so far

Everything below is **simulation only**; the robot has not moved.

| Stage | State |
| --- | --- |
| Fitting engine (friction, armature, inertia; staged fits, conditioning and uncertainty reporting) | done, proven on synthetic data with known hidden values |
| Excitation protocols (these motions) | done, committed, limit-checked — revision r2 |
| Identification demonstrated end-to-end in simulation | see `../identification_demo/result.md` |
| Playback in ROS simulation | done — all six protocols play to completion through the trajectory controller |
| Recording the run into a dataset | not started |
| Playback on the real robot | not started |
| New model delivered to hydrax / sbmpc_ros | not started |

The demonstration run answers the question that matters before touching
hardware: *if the robot really had friction of this magnitude, would this
machinery find it, and would the identified model predict the robot's motion
better than the current frictionless one?*
