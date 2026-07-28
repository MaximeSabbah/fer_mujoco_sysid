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

## The three protocols

| Protocol | Purpose | Duration |
| --- | --- | --- |
| `fer-friction-a` | fitting data | ~42 s |
| `fer-friction-b` | fitting data (different seed → different per-joint amplitudes) | ~42 s |
| `fer-friction-holdout` | **never used for fitting** — different amplitudes and speeds, used only to judge the result | ~32 s |

All joints move together on a shared schedule; each joint has its own
amplitude (~0.3 rad around the home pose) with a small seeded variation, so
the joints do not all trace identical paths. Every protocol is checked
against FER position, velocity, acceleration, jerk and torque limits (with a
20 % margin) before it is written, and is byte-reproducible from its seed.

## What to check in the videos

- Does the motion look safe and sane in your cell (the arm stays near the
  home pose and returns to it)?
- Are the amplitudes big enough to be useful, small enough to be safe?
- Are the slow sweeps slow enough for the regime you care about?
- Is ~42 s per protocol an acceptable run length?

Changing any of this is cheap right now: edit the specs in
`src/fer_mujoco_sysid/sysid/campaign.py`, regenerate, and the bundles become
revision `r2`.

## Status: what has actually been achieved so far

Everything below is **simulation only**; the robot has not moved.

| Stage | State |
| --- | --- |
| Fitting engine (friction, armature, inertia; staged fits, conditioning and uncertainty reporting) | done, proven on synthetic data with known hidden values |
| Excitation protocols (these motions) | done, committed, limit-checked |
| Identification demonstrated end-to-end in simulation | see `../identification_demo/result.md` |
| Playback in ROS simulation | not started |
| Playback on the real robot | not started |
| New model delivered to hydrax / sbmpc_ros | not started |

The demonstration run answers the question that matters before touching
hardware: *if the robot really had friction of this magnitude, would this
machinery find it, and would the identified model predict the robot's motion
better than the current frictionless one?*
