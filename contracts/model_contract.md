# FER model contract

## Purpose

System identification starts from the MuJoCo model that already transfers
successfully enough to run the FER controller. It estimates corrections to
that model; it does not reconstruct a new robot description from URDF.

## Model roles

| Model | Role |
| --- | --- |
| `hydrax/hydrax/models/panda/panda.xml` | Nominal system-identification template and controller model |
| `sbmpc_ros/sbmpc_bringup/mujoco/fer_ros2_control.xml` | ROS simulation and deployment wrapper |
| Agimus FER description | Reference for fixed hardware conventions, kinematics, limits, and nominal parameter provenance |

The deprecated standalone `sbmpc` repository is not an input, output, or
runtime dependency of this project.

The ROS MJCF currently has a legacy absolute `meshdir` pointing into a local
`sbmpc` checkout. That is an existing packaging defect in `sbmpc_ros`; it is
not part of the physical model. This project resolves the ROS MJCF against the
Hydrax asset directory in memory for compatibility tests. The overlay's asset
path will be addressed separately before the deprecated checkout can be
removed.

The exact nominal revisions and model-file hashes used to establish this
contract are recorded in `contracts/nominal_sources.toml`. Updating a baseline
model is therefore an explicit review rather than an accidental consequence of
a sibling checkout changing.

For local compatibility tests, the two read-only source files can be selected
explicitly:

```bash
export FER_SYSID_HYDRAX_MODEL=/path/to/hydrax/hydrax/models/panda/panda.xml
export FER_SYSID_SBMPC_ROS_MODEL=/path/to/sbmpc_ros/sbmpc_bringup/mujoco/fer_ros2_control.xml
./scripts/test
```

When these variables are absent, the test suite looks for sibling repository
checkouts.

## One physical model, two wrappers

The Hydrax and ROS MJCFs have the same body tree, fixed transforms, inertials,
joint axes and ranges, gripper frame, collision geometry, armature, damping,
and arm torque limits. The wrapper-specific names and actuators must remain
unchanged.

| Semantic element | Hydrax | `sbmpc_ros` |
| --- | --- | --- |
| Arm joint `i` | `joint{i}` | `fer_joint{i}` |
| Finger joint 1 | `finger_joint1` | `fer_finger_joint1` |
| Finger joint 2 | `finger_joint2` | `fer_finger_joint2` |
| Arm actuator `i` | `motor{i}` direct-torque `general`, control-limited | `fer_joint{i}` effort `motor`, control- and force-limited |
| Gripper actuator | Width position servo `actuator8` | Effort motor `fer_finger_joint1` |

Identification produces one canonical parameter manifest keyed by semantic
body and joint identity. Deterministic exporters will apply that manifest to
both wrappers. The two MJCFs must never be fitted independently.

## Parameters

The first dynamic parameter groups intended for identification are:

- joint friction loss and damping, as the primary real-robot hypothesis;
- joint armature; and
- body mass, center of mass, and physically consistent inertia.

The following remain fixed unless a future experiment introduces suitable
external measurements:

- body hierarchy and joint transforms;
- joint axes and position limits;
- visual and collision geometry;
- the hand-to-gripper site transform;
- joint and actuator ordering; and
- wrapper-specific actuator and gripper interfaces.

Measurement bias, timing delay, and torque scale may be estimated as nuisance
parameters. They are not automatically exported to MJCF: the Hydrax planning
model assumes direct torque actuation, so any such integration needs an
explicit controller-level review.

## Hydrax identification projection

Hydrax derives its seven-axis planning model from `panda.xml` at runtime. It
removes the finger joints, gripper actuator, and equality constraint while
retaining the hand and finger bodies and their inertia. Contacts are disabled.

This project uses the same projection for arm identification. The integration
timestep comes from the identification dataset; it is not forced to Hydrax's
40 ms planning period.

The current `run_open_loop_effort_protocol` helper is deliberately a
contact-free plant primitive. It recompiles the projection from the exact
declared source, applies each nonterminal effort-feedforward knot for one
integration interval, rejects actuator saturation and process-global MuJoCo
callbacks, and returns the canonical `M + 1` state / `M` control time grid. It
does not consume desired `q/dq/ddq` after initialization and therefore does not
claim equivalence with ROS effort trajectory control. That equivalence requires
the shared interpolation and feedback-controller layer defined in M4/M5.

## Compatibility gates

Before an identified model can be proposed for integration:

1. Both wrapper MJCFs compile under the pinned MuJoCo version.
2. The Hydrax output compiles and steps under the controller's MuJoCo/MJX
   version.
3. Fixed kinematics, limits, names, dimensions, and actuator ordering are
   unchanged.
4. Canonical identified parameters are equal across the Hydrax and ROS
   wrappers after applying the name mapping.
5. With contacts disabled and equivalent arm inputs, both wrappers produce
   matching arm bias forces and accelerations.
6. Held-out simulated and real trajectories show a measurable improvement over
   the nominal model.
