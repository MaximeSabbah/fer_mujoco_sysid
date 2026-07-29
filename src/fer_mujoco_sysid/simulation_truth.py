"""One deterministic hidden-plant definition shared by simulation backends.

The fitter never reads this module.  The standalone and ROS simulation
acquisition paths use it to construct the plant, while a separate
simulation-only evaluator uses it after fitting to report parameter recovery.
Real recordings carry no simulation truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from mujoco import sysid

from fer_mujoco_sysid.export import (
    IdentifiedParameters,
    apply_parameters,
)
from fer_mujoco_sysid.fitting import (
    BodyInertialCorrection,
    cad_prior_inertial_parameters,
)
from fer_mujoco_sysid.io import sha256_file, write_json
from fer_mujoco_sysid.model import build_hydrax_arm_spec, resolve_model_paths

SIMULATION_TRUTH_FORMAT = "fer-mujoco-sysid/simulation-truth@1"

# Deliberately different from the CAD nominal, but comfortably inside the
# estimator bounds.  These are synthetic values, not estimates of the FER.
TRUTH_FRICTIONLOSS = (1.4, 1.2, 1.1, 1.5, 0.35, 1.1, 0.55)
TRUTH_DAMPING = (2.0, 1.8, 1.5, 1.9, 0.8, 0.7, 0.5)
TRUTH_ARMATURE = (0.14, 0.13, 0.12, 0.12, 0.08, 0.07, 0.06)

# The oracle exposes only the coordinates it actually perturbs.  A separate
# Jacobian gate must certify which of these can be claimed as individually
# recoverable; behavior remains the primary full-model closure criterion.
TRUTH_BODY_PARAMETERIZATION = (
    BodyInertialCorrection(
        "link4",
        com_axes=(0, 2),
        estimate_inertia_scale=False,
    ),
    BodyInertialCorrection(
        "link5",
        estimate_mass=False,
        estimate_inertia_scale=True,
    ),
    BodyInertialCorrection(
        "link6",
        com_axes=(1,),
        estimate_inertia_scale=False,
    ),
    BodyInertialCorrection(
        "link7",
        estimate_mass=False,
        com_axes=(0,),
        estimate_inertia_scale=True,
    ),
)
TRUTH_BODY_COORDINATES = {
    "link4_mass_scale": 1.10,
    "link4_com_x_offset_m": 0.004,
    "link4_com_z_offset_m": -0.003,
    "link5_inertia_scale": 1.08,
    "link6_mass_scale": 0.90,
    "link6_com_y_offset_m": 0.005,
    "link7_com_x_offset_m": -0.004,
    "link7_inertia_scale": 0.92,
}
TRUTH_BODIES = tuple(correction.body for correction in TRUTH_BODY_PARAMETERIZATION)


def _source_path(source_path: str | Path | None) -> Path:
    return Path(
        source_path
        if source_path is not None
        else resolve_model_paths().require().hydrax
    ).resolve()


def truth_parameters(
    source_path: str | Path | None = None,
) -> IdentifiedParameters:
    """Return exact absolute parameters for the deterministic hidden plant."""
    source = _source_path(source_path)
    spec = build_hydrax_arm_spec(source)
    nominal = spec.compile()
    inertials = cad_prior_inertial_parameters(
        spec,
        nominal,
        TRUTH_BODY_PARAMETERIZATION,
    )
    for name, value in TRUTH_BODY_COORDINATES.items():
        inertials[name].update_from_vector(np.array([value], dtype=np.float64))
    sysid.apply_param_modifiers_spec(inertials, spec)

    for index in range(1, 8):
        joint = spec.joint(f"joint{index}")
        joint.frictionloss = TRUTH_FRICTIONLOSS[index - 1]
        damping = np.asarray(joint.damping, dtype=np.float64).copy()
        damping[0] = TRUTH_DAMPING[index - 1]
        joint.damping = damping
        joint.armature = TRUTH_ARMATURE[index - 1]

    return IdentifiedParameters.from_model(
        spec.compile(),
        bodies=TRUTH_BODIES,
    )


def truth_model(
    source_path: str | Path | None = None,
    *,
    gravity: bool = False,
    joint_state_sensors: bool = True,
) -> mujoco.MjModel:
    """Compile the exact seven-axis hidden plant used for identification."""
    source = _source_path(source_path)
    spec = build_hydrax_arm_spec(
        source,
        joint_state_sensors=joint_state_sensors,
    )
    apply_parameters(spec, truth_parameters(source))
    if not gravity:
        spec.option.gravity = [0.0, 0.0, 0.0]
    return spec.compile()


def truth_manifest(
    source_path: str | Path | None = None,
) -> dict[str, object]:
    """Return the deterministic, portable simulation-truth manifest."""
    source = _source_path(source_path)
    payload: dict[str, object] = {
        "format": SIMULATION_TRUTH_FORMAT,
        "source_model": {
            "filename": source.name,
            "sha256": sha256_file(source),
        },
        "parameters": truth_parameters(source).as_dict(),
        "perturbed_coordinates": dict(TRUTH_BODY_COORDINATES),
        "conventions": {
            "acquisition_gravity_enabled": False,
            "consumer_model_gravity_enabled": True,
            "contacts_enabled": False,
            "integrator": "mjINT_IMPLICITFAST",
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(
        b"fer-mujoco-sysid/simulation-truth@1\0" + canonical
    ).hexdigest()
    return payload


def write_truth_manifest(
    destination: str | Path,
    source_path: str | Path | None = None,
) -> Path:
    destination = Path(destination)
    write_json(destination, truth_manifest(source_path))
    return destination


def load_truth_manifest(
    path: str | Path,
    *,
    source_path: str | Path | None = None,
) -> dict[str, object]:
    """Load a truth manifest and verify both its content and source model."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != SIMULATION_TRUTH_FORMAT:
        raise ValueError(f"{path}: not a {SIMULATION_TRUTH_FORMAT} manifest")
    claimed_hash = raw.get("content_sha256")
    unsigned = dict(raw)
    unsigned.pop("content_sha256", None)
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    expected_hash = hashlib.sha256(
        b"fer-mujoco-sysid/simulation-truth@1\0" + canonical
    ).hexdigest()
    if claimed_hash != expected_hash:
        raise ValueError(f"{path}: simulation truth content hash does not match")

    source = _source_path(source_path)
    source_metadata = raw.get("source_model")
    if not isinstance(source_metadata, dict):
        raise ValueError(f"{path}: simulation truth has no source_model mapping")
    if source_metadata.get("sha256") != sha256_file(source):
        raise ValueError(
            f"{path}: simulation truth source does not match {source}"
        )
    return raw


def parameters_from_truth_manifest(
    path: str | Path,
    *,
    source_path: str | Path | None = None,
) -> IdentifiedParameters:
    raw = load_truth_manifest(path, source_path=source_path)
    parameters = raw.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{path}: simulation truth has no parameter mapping")
    return IdentifiedParameters.from_mapping(parameters)


def truth_model_from_manifest(
    path: str | Path,
    source_path: str | Path | None = None,
    *,
    gravity: bool = False,
    joint_state_sensors: bool = True,
) -> mujoco.MjModel:
    """Compile the hidden plant described by a verified truth manifest."""
    source = _source_path(source_path)
    spec = build_hydrax_arm_spec(
        source,
        joint_state_sensors=joint_state_sensors,
    )
    apply_parameters(
        spec,
        parameters_from_truth_manifest(path, source_path=source),
    )
    if not gravity:
        spec.option.gravity = [0.0, 0.0, 0.0]
    return spec.compile()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("--model", default=None)
    arguments = parser.parse_args(argv)
    print(write_truth_manifest(arguments.destination, arguments.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
