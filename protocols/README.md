# Motion protocols

This directory is the versioned library of reusable FER motions.

Each protocol contains the exact compiled time, position, velocity and
acceleration arrays that can be validated and played in standalone MuJoCo, ROS
MuJoCo, or through the reviewed Agimus hardware path. Generator settings,
source-model hashes, payload assumptions and segment meanings accompany the
compiled arrays.

Motion protocols are intended to remain compact and live directly in Git.
Robot recordings produced by executing them are local artifacts by default;
they belong under `datasets/` only when deliberately curated and licensed for
publication.

The normative format is documented in
[`docs/artifact_contracts.md`](../docs/artifact_contracts.md) and validated by
`src/fer_mujoco_sysid/schemas/motion-protocol-v1.schema.json`.
