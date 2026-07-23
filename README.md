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
- a seven-axis identification-model projection; and
- physical-parity, dynamics, provenance, and sysid API tests.

Synthetic parameter recovery, excitation generation, recording adapters, and
real-data fitting will be added incrementally.

## Datasets

Reusable excitation trajectories and curated simulated or real recordings are
first-class project assets and may be committed under [`datasets/`](datasets/).
Each dataset should include its motion protocol, signal definitions, model and
software versions, acquisition metadata, and train/validation role.

Small, reusable datasets should live directly in Git. Large binary recordings
can use Git LFS or a versioned external release, while their metadata, hashes,
conversion recipe, and derived compact data remain in this repository.

## Setup

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked --all-groups
./scripts/test
```

The nominal model provenance and compatibility guarantees are documented in
[`contracts/nominal_sources.toml`](contracts/nominal_sources.toml) and the
[model contract](docs/model_contract.md).
