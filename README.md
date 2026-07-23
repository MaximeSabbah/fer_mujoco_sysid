# FER MuJoCo system identification

Tools and datasets for identifying a Franka Research 3 (FER) robot directly in
MuJoCo.

The objective is to turn simulated or recorded robot motion into a physically
consistent MuJoCo model that predicts the FER dynamics more accurately.

## Intended workflow

1. Generate persistently exciting, limit- and collision-checked motions.
2. Execute the same motion protocol in simulation or on the robot.
3. Record joint states, applied torques, timing, and relevant robot state.
4. Convert recordings into MuJoCo system-identification sequences.
5. Estimate friction and damping first, then physically valid armature and
   inertial parameters.
6. Validate the identified model on motions that were not used for fitting.
7. Export a reproducible MJCF model, parameter manifest, uncertainty report,
   and diagnostic plots.

The fitting backend is MuJoCo's native `mujoco.sysid` toolbox. The project pins
the exact MuJoCo version so that datasets and results remain reproducible.

The versioned [implementation plan](docs/implementation_plan.md) defines the
milestones, validation gates, and review checkpoints used to develop the
project.

## Project status

The repository currently provides:

- a Python 3.12 environment pinned to `mujoco[sysid]==3.10.0`;
- a reviewed nominal FER model contract;
- a seven-axis identification-model projection;
- versioned, pickle-free artifact contracts; and
- physical-parity, dynamics, provenance, artifact, and sysid API tests.

Synthetic parameter recovery, excitation generation, recording adapters, and
real-data fitting will be added incrementally.

## Protocols and datasets

Reusable excitation trajectories are first-class project assets under
[`protocols/`](protocols/). The compiled motion arrays are stored with their
generator, model, payload, segment, and validation provenance so another FER
setup can check and execute the exact same motion.

Robot recordings remain outside the source repository by default. A simulated
or real recording enters [`datasets/`](datasets/) only when it is deliberately
curated, licensed, and useful to share. Compact curated data can live directly
in Git; unusually large public recordings can be referenced by immutable
release URL and SHA-256.

The [artifact contracts](docs/artifact_contracts.md) define the exact protocol,
acquisition, normalized-trajectory, fit-result, and dataset formats.

## Setup

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked --all-groups
./scripts/test
```

The nominal model provenance and compatibility guarantees are documented in
[`contracts/nominal_sources.toml`](contracts/nominal_sources.toml) and the
[model contract](docs/model_contract.md).
