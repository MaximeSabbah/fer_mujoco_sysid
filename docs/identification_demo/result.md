# Simulated identification — result

Hidden truth: friction, damping and armature of realistic magnitude (informed by the real 2026-07-22 breakaway bounds; NOT the real robot's values). Recordings carry encoder and torque-sensor noise and are preprocessed before fitting.

## 1. Excitation quality (measured before fitting)

Condition number of the regressor per joint — the classical excitation measure. 1 is ideal; above ~100 the motion cannot separate the parameters.

| family | parameters | worst cond(Y) | verdict |
| --- | --- | --- | --- |
| friction | frictionloss, damping | 2.43 | well excited |
| inertial | frictionloss, damping, armature | 2.78 | well excited |

## 2. Preprocessing

Zero-phase Butterworth low-pass, order 4, cutoff 8.0 Hz, applied identically to positions, velocities and torques. Zero phase matters: a causal filter would shift the joint states against the recorded torque and bias friction exactly at the velocity reversals.

## 3. Identified parameters

| joint | frictionloss truth | identified | damping truth | identified |
| --- | --- | --- | --- | --- |
| 1 | 1.400 | 1.396 | 2.000 | 2.071 |
| 2 | 1.200 | 1.201 | 1.800 | 1.777 |
| 3 | 1.100 | 1.099 | 1.500 | 1.530 |
| 4 | 1.500 | 1.496 | 1.900 | 2.025 |
| 5 | 0.350 | 0.349 | 0.800 | 0.821 |
| 6 | 1.100 | 1.097 | 0.700 | 0.772 |
| 7 | 0.550 | 0.549 | 0.500 | 0.527 |

## 4. Held-out validation

A protocol never used for fitting. Open-loop prediction over 0.5 s windows re-initialized from the measured state.

| metric | nominal model | identified model |
| --- | --- | --- |
| worst joint q RMSE [rad] | 0.1273 | 0.0026 |
| gripper position RMSE [mm] | 68.81 | 0.52 |
| gripper position max [mm] | 198.76 | 3.29 |
| worst joint torque RMSE [Nm] | 1.494 | 0.203 |

Per-joint torque reconstruction error [Nm]:

| joint | nominal | identified |
| --- | --- | --- |
| 1 | 1.365 | 0.085 |
| 2 | 1.257 | 0.203 |
| 3 | 1.049 | 0.055 |
| 4 | 1.494 | 0.114 |
| 5 | 0.303 | 0.033 |
| 6 | 0.974 | 0.098 |
| 7 | 0.465 | 0.053 |

Same check on the **inertial** protocol, where only friction has been fitted so far (armature and link inertias are still nominal):

| metric | nominal model | friction-identified model |
| --- | --- | --- |
| worst joint torque RMSE [Nm] | 1.981 | 0.339 |

## 5. Plots

* `friction_curves.png` — friction torque vs velocity at the cruises: truth (thick orange), identified (dashed blue), nominal (grey — no step at zero, because it has no dry friction).
* `torque_tracking_friction.png` — measured joint torque against what each model reconstructs, on the held-out friction protocol.
* `torque_tracking_inertial.png` — the same on an inertial protocol: the residual there is what armature and inertial identification still has to remove.
* `torque_residuals.png` — per-joint torque reconstruction error.

## What this does and does not prove

**Does prove**: the protocols are well conditioned for what they target; the pipeline recovers realistic friction from noisy data without seeing the truth; the identified model reconstructs joint torques and predicts unseen motion far better than the nominal one.

**Does not prove anything about the real robot.** The truth here differs from the model only in parameters MuJoCo can express, the noise is white, and the plant is the same simulator the fit uses. The real FER has position- and temperature-dependent friction, transmission effects, and dynamics outside this model class.
