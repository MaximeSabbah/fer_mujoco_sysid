# Datasets

This directory is for deliberately curated FER system-identification data.
Reusable motions live separately under [`protocols/`](../protocols/).

Every contributed dataset should contain or reference:

- a stable dataset identifier and short description;
- the commanded motion protocol;
- joint names, units, sampling clock, and signal semantics;
- raw or losslessly converted measured states and torques;
- robot, end-effector, payload, and environment configuration;
- software and model revisions;
- acquisition and conversion commands;
- integrity hashes; and
- whether each trajectory belongs to the fitting or held-out validation set.

Robot recordings are local by default and do not have to be published. When a
recording is intentionally shared, prefer the portable normalized JSON/NPZ
contract. MCAP or another large binary source may live in a versioned release,
provided that this directory retains its immutable URL, size, SHA-256, and
exact retrieval/conversion recipe.

Machine-specific paths, credentials, robot network information, and unrelated
ROS traffic must not be included in published datasets.
