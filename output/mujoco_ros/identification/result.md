# Identification from recorded runs

Fitted on fer-friction-a, fer-friction-b, fer-inertial-a, fer-inertial-b; held out fer-friction-holdout, fer-inertial-holdout.

## Excitation quality

Worst regressor condition number: **2.30** (1 is ideal; above ~100 the motion cannot separate the parameters).

## Stage 1 — classical regressor solve

Direct least squares on `tau - tau_rigid = frictionloss*sign(dq) + damping*dq`, over the samples where each joint is clearly sliding. The relative standard deviations are the classical acceptance rule: above ~20% a parameter is not determined by this data.

| joint | frictionloss [Nm] | sigma% | damping [Nm s/rad] | sigma% | cond(Y) | samples |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.4019 | 0.01% | 1.9890 | 0.03% | 2.37 | 53015 |
| 2 | 1.2013 | 0.03% | 1.7739 | 0.08% | 2.41 | 53634 |
| 3 | 1.0963 | 0.04% | 1.4864 | 0.07% | 2.10 | 53783 |
| 4 | 1.4833 | 0.05% | 1.8823 | 0.09% | 1.99 | 54112 |
| 5 | 0.3438 | 0.06% | 0.8002 | 0.05% | 1.96 | 54340 |
| 6 | 1.0714 | 0.07% | 0.7177 | 0.19% | 1.95 | 53782 |
| 7 | 0.5299 | 0.09% | 0.5141 | 0.22% | 2.08 | 52633 |

## Stage 2 — rollout refinement

Seeded from stage 1 and refined by matching simulated to measured trajectories. The two methods minimize different things — stage 1 balances the torque equation sample by sample, stage 2 reproduces the motion — so they agree only while the model class holds.

**Worst disagreement between the two: 3.76%.** A large gap here is not noise; it means the friction model class is wrong (Stribeck, temperature, transmission) and inertial fitting should pause for a model-class review rather than absorb the error.

## Identified parameters

| joint | frictionloss [Nm] | damping [Nm s/rad] | sigma% fl | sigma% d |
| --- | --- | --- | --- | --- |
| 1 | 1.3990 | 2.0062 | 0.00% | 0.02% |
| 2 | 1.1983 | 1.8195 | 0.01% | 0.05% |
| 3 | 1.0995 | 1.5065 | 0.00% | 0.02% |
| 4 | 1.4998 | 1.9005 | 0.00% | 0.03% |
| 5 | 0.3497 | 0.8026 | 0.00% | 0.02% |
| 6 | 1.0999 | 0.6995 | 0.00% | 0.03% |
| 7 | 0.5498 | 0.5018 | 0.00% | 0.04% |

## Held-out validation — the acceptance metric

Protocol `fer-friction-holdout`, never used for fitting. The simulator starts from a measured state, is driven with the recorded torque, and runs open loop; the error is how far it has drifted from the measurement by the end of each window.

**This is what decides whether a model is good enough.** The parameter values above are worth reading — they are how a fit is checked against the measured breakaway bounds, and how a physically absurd result is spotted — but on the real robot there is no true value to compare them against. What can always be measured is whether the rollout still resembles what the arm did, and for how long. A model can look excellent over 0.1 s and drift badly over 2 s; that is the difference between one a planner can use and one it cannot.

| horizon | model | worst joint q RMSE [rad] | gripper RMSE [mm] | gripper max [mm] |
| --- | --- | --- | --- | --- |
| 0.1s | nominal | 0.00975 | 3.901 | 10.231 |
| 0.1s | classical | 0.00030 | 0.074 | 0.764 |
| 0.1s | identified | 0.00010 | 0.030 | 0.799 |
| 0.5s | nominal | 0.14023 | 71.327 | 186.787 |
| 0.5s | classical | 0.00448 | 1.143 | 2.941 |
| 0.5s | identified | 0.00034 | 0.122 | 1.105 |
| 2s | nominal | 0.78354 | 350.268 | 753.537 |
| 2s | classical | 0.02600 | 7.363 | 19.658 |
| 2s | identified | 0.00081 | 0.394 | 1.071 |

`classical` is the stage-1 estimate taken as a model on its own; `identified` is after rollout refinement. If refinement does not improve this table it is not earning its runtime — and that judgement can be made on hardware, where parameter truth cannot.

Torque reconstruction on the same protocol (inverse dynamics — a diagnostic, not the acceptance metric):

| model | worst joint torque RMSE [Nm] |
| --- | --- |
| nominal | 1.424 |
| classical | 0.200 |
| identified | 0.200 |

## Plots

* `rollout_vs_measurement.png` — **look at this first.** The simulator run open loop against what the arm actually did, over several 2 s windows. Where the coloured line leaves the black one, the model is wrong.
* `error_vs_horizon.png` — how fast that agreement decays with rollout length, per model.
* `end_effector.png` — the same rollout judged at the gripper. Joint error is what the fit minimizes; this is what the task cares about, and small joint errors compound along the chain.
* `friction_curves.png` — friction torque against velocity: the step at zero is Coulomb friction, the slope is viscous. The nominal model has no step at all, which is the problem this project exists to fix.
* `torque_tracking.png` — recorded commanded torque against what each model reconstructs. The spikes at the velocity corners are an artifact of estimating acceleration by differentiating filtered velocity through a sharp transition, not a model defect: the error is 0.002 Nm where jerk is low and 0.61 Nm in the top jerk percentile, and the rollout — which never uses acceleration — is exact throughout.
* `../<protocol>/replay.mp4` — the motion each run actually performed, with a translucent ghost of what the controller was asking for. Where the ghost pulls ahead, the arm did not follow.
* `../<protocol>/tracking.png` — the same thing as numbers: commanded minus measured, per joint. A property of the *run* rather than of the model.

## The identified model

`fer_identified.xml`, written only after passing the export whitelist and reload checks.

| joint | field | nominal | identified | change |
| --- | --- | --- | --- | --- |
| joint1 | frictionloss | 0.0000 | 1.3990 | +1.3990 |
| joint1 | damping | 1.0000 | 2.0063 | +1.0063 |
| joint2 | frictionloss | 0.0000 | 1.1983 | +1.1983 |
| joint2 | damping | 1.0000 | 1.8195 | +0.8195 |
| joint3 | frictionloss | 0.0000 | 1.0995 | +1.0995 |
| joint3 | damping | 1.0000 | 1.5065 | +0.5065 |
| joint4 | frictionloss | 0.0000 | 1.4998 | +1.4998 |
| joint4 | damping | 1.0000 | 1.9005 | +0.9005 |
| joint5 | frictionloss | 0.0000 | 0.3497 | +0.3497 |
| joint5 | damping | 1.0000 | 0.8026 | -0.1974 |
| joint6 | frictionloss | 0.0000 | 1.0999 | +1.0999 |
| joint6 | damping | 1.0000 | 0.6995 | -0.3005 |
| joint7 | frictionloss | 0.0000 | 0.5498 | +0.5498 |
| joint7 | damping | 1.0000 | 0.5018 | -0.4982 |

