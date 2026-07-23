# Datasets

This directory is for reusable FER motion protocols and system-identification
data.

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

Prefer portable, inspectable formats for motion protocols and compact derived
data. MCAP or other large binary recordings may be stored with Git LFS or in a
versioned release, provided that this directory retains a manifest, hashes, and
an exact retrieval/conversion recipe.

Machine-specific paths, credentials, robot network information, and unrelated
ROS traffic must not be included in published datasets.
