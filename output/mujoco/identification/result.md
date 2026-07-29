# Identification from recorded runs

Fitted on fer-friction-a, fer-friction-b, fer-inertial-a, fer-inertial-b; held out fer-friction-holdout, fer-inertial-holdout.

## Excitation quality

Worst regressor condition number: **2.33** (1 is ideal; above ~100 the motion cannot separate the parameters).

## Stage 1 — classical regressor solve

Direct least squares on `tau - tau_rigid = frictionloss*sign(dq) + damping*dq`, over the samples where each joint is clearly sliding. The relative standard deviations are the classical acceptance rule: above ~20% a parameter is not determined by this data.

| joint | frictionloss [Nm] | sigma% | damping [Nm s/rad] | sigma% | cond(Y) | samples |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.4021 | 0.02% | 1.9884 | 0.05% | 2.39 | 50441 |
| 2 | 1.2013 | 0.04% | 1.7739 | 0.10% | 2.43 | 51017 |
| 3 | 1.0958 | 0.05% | 1.4869 | 0.09% | 2.13 | 51205 |
| 4 | 1.4824 | 0.07% | 1.8831 | 0.11% | 2.02 | 51522 |
| 5 | 0.3433 | 0.11% | 0.8015 | 0.09% | 1.99 | 51742 |
| 6 | 1.0716 | 0.09% | 0.7200 | 0.23% | 1.98 | 51287 |
| 7 | 0.5291 | 0.18% | 0.5213 | 0.43% | 2.11 | 50249 |

## Stage 2 — rollout refinement

Seeded from stage 1 and refined by matching simulated to measured trajectories. The two methods minimize different things — stage 1 balances the torque equation sample by sample, stage 2 reproduces the motion — so they agree only while the model class holds.

**Worst disagreement between the two: 4.37%.** A large gap here is not noise; it means the friction model class is wrong (Stribeck, temperature, transmission) and inertial fitting should pause for a model-class review rather than absorb the error.

## Identified parameters

| joint | frictionloss [Nm] | damping [Nm s/rad] | sigma% fl | sigma% d |
| --- | --- | --- | --- | --- |
| 1 | 1.3947 | 2.0344 | 0.01% | 0.05% |
| 2 | 1.1990 | 1.8215 | 0.01% | 0.11% |
| 3 | 1.0976 | 1.5016 | 0.01% | 0.05% |
| 4 | 1.4992 | 1.9024 | 0.01% | 0.06% |
| 5 | 0.3488 | 0.8162 | 0.01% | 0.04% |
| 6 | 1.1001 | 0.6928 | 0.00% | 0.08% |
| 7 | 0.5498 | 0.4985 | 0.01% | 0.09% |

## Held-out validation — the acceptance metric

Protocol `fer-friction-holdout`, never used for fitting. The simulator starts from a measured state, is driven with the recorded torque, and runs open loop; the error is how far it has drifted from the measurement by the end of each window.

**This is what decides whether a model is good enough.** The parameter values above are worth reading — they are how a fit is checked against the measured breakaway bounds, and how a physically absurd result is spotted — but on the real robot there is no true value to compare them against. What can always be measured is whether the rollout still resembles what the arm did, and for how long. A model can look excellent over 0.1 s and drift badly over 2 s; that is the difference between one a planner can use and one it cannot.

| horizon | model | worst joint q RMSE [rad] | gripper RMSE [mm] | gripper max [mm] |
| --- | --- | --- | --- | --- |
| 0.1s | nominal | 0.00975 | 3.933 | 10.241 |
| 0.1s | classical | 0.00035 | 0.074 | 0.369 |
| 0.1s | identified | 0.00021 | 0.027 | 0.224 |
| 0.5s | nominal | 0.14102 | 72.371 | 186.867 |
| 0.5s | classical | 0.00503 | 1.162 | 3.522 |
| 0.5s | identified | 0.00250 | 0.232 | 1.337 |
| 2s | nominal | 0.84219 | 368.879 | 786.157 |
| 2s | classical | 0.02677 | 7.793 | 19.550 |
| 2s | identified | 0.00752 | 0.684 | 1.648 |

`classical` is the stage-1 estimate taken as a model on its own; `identified` is after rollout refinement. If refinement does not improve this table it is not earning its runtime — and that judgement can be made on hardware, where parameter truth cannot.

Torque reconstruction on the same protocol (inverse dynamics — a diagnostic, not the acceptance metric):

| model | worst joint torque RMSE [Nm] |
| --- | --- |
| nominal | 1.444 |
| classical | 0.242 |
| identified | 0.242 |

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
| joint1 | frictionloss | 0.0000 | 1.3947 | +1.3947 |
| joint1 | damping | 1.0000 | 2.0344 | +1.0344 |
| joint2 | frictionloss | 0.0000 | 1.1990 | +1.1990 |
| joint2 | damping | 1.0000 | 1.8215 | +0.8215 |
| joint3 | frictionloss | 0.0000 | 1.0976 | +1.0976 |
| joint3 | damping | 1.0000 | 1.5016 | +0.5016 |
| joint4 | frictionloss | 0.0000 | 1.4992 | +1.4992 |
| joint4 | damping | 1.0000 | 1.9024 | +0.9024 |
| joint5 | frictionloss | 0.0000 | 0.3488 | +0.3488 |
| joint5 | damping | 1.0000 | 0.8162 | -0.1838 |
| joint6 | frictionloss | 0.0000 | 1.1000 | +1.1000 |
| joint6 | damping | 1.0000 | 0.6928 | -0.3072 |
| joint7 | frictionloss | 0.0000 | 0.5498 | +0.5498 |
| joint7 | damping | 1.0000 | 0.4985 | -0.5015 |

