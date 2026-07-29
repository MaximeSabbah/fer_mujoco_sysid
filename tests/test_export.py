"""Gates for writing identified parameters back into a model.

An identified model is the nominal model plus what was identified, and
nothing else. These check both halves of that claim: the values land, and
nothing outside the whitelist moves.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid.export import (
    ExportCheck,
    ExportError,
    IdentifiedParameters,
    export_identified_model,
    verify_export,
)
from fer_mujoco_sysid.model import HYDRAX_ARM_JOINT_NAMES, ModelPaths

_FRICTION = (1.40, 1.20, 1.10, 1.50, 0.35, 1.10, 0.55)
_DAMPING = (2.00, 1.80, 1.50, 1.90, 0.80, 0.70, 0.50)
_ARMATURE = (0.12, 0.13, 0.11, 0.14, 0.09, 0.08, 0.07)


@pytest.fixture
def parameters() -> IdentifiedParameters:
    return IdentifiedParameters(
        frictionloss=_FRICTION, damping=_DAMPING, armature=_ARMATURE
    )


@pytest.fixture
def exported(
    tmp_path: Path, model_paths: ModelPaths, parameters: IdentifiedParameters
) -> tuple[Path, ExportCheck]:
    destination = tmp_path / "identified.xml"
    check = export_identified_model(
        destination,
        parameters,
        source_path=model_paths.hydrax,
        manifest_path=tmp_path / "identified.json",
    )
    return destination, check


def test_identified_values_reach_the_compiled_model(
    exported: tuple[Path, ExportCheck],
) -> None:
    destination, _ = exported
    model = mujoco.MjModel.from_xml_path(str(destination))
    for index, name in enumerate(HYDRAX_ARM_JOINT_NAMES):
        dof = int(model.joint(name).dofadr[0])
        assert model.dof_frictionloss[dof] == pytest.approx(_FRICTION[index], rel=1e-5)
        assert model.dof_damping[dof] == pytest.approx(_DAMPING[index], rel=1e-5)
        assert model.dof_armature[dof] == pytest.approx(_ARMATURE[index], rel=1e-5)


def test_export_reports_what_it_changed(exported: tuple[Path, ExportCheck]) -> None:
    _, check = exported
    assert set(check.changed) == set(HYDRAX_ARM_JOINT_NAMES)
    assert set(check.changed["joint1"]) == {"frictionloss", "damping", "armature"}
    nominal_friction, identified_friction = check.changed["joint1"]["frictionloss"]
    assert nominal_friction == pytest.approx(0.0)
    assert identified_friction == pytest.approx(_FRICTION[0], rel=1e-5)
    assert "joint1" in check.table()


def test_manifest_pins_both_models_by_hash(
    exported: tuple[Path, ExportCheck], tmp_path: Path
) -> None:
    """A model without provenance cannot be traced back to what produced it."""
    manifest = json.loads((tmp_path / "identified.json").read_text())
    assert manifest["format"] == "fer-mujoco-sysid/identified-model@1"
    assert len(manifest["source_model"]["sha256"]) == 64
    assert len(manifest["exported_model"]["sha256"]) == 64
    assert manifest["exported_fields"] == ["armature", "damping", "frictionloss"]
    assert manifest["parameters"]["frictionloss"] == pytest.approx(list(_FRICTION))


def test_export_leaves_the_robot_alone(
    exported: tuple[Path, ExportCheck], model_paths: ModelPaths
) -> None:
    """Kinematics, masses and actuator limits are not identification outputs."""
    destination, _ = exported
    nominal = mujoco.MjModel.from_xml_path(str(model_paths.hydrax))
    identified = mujoco.MjModel.from_xml_path(str(destination))
    for name in ("body_mass", "body_ipos", "jnt_range", "actuator_forcerange"):
        np.testing.assert_allclose(
            getattr(identified, name), getattr(nominal, name), rtol=1e-5, atol=1e-9
        )


def test_a_model_that_changed_something_else_is_rejected(
    tmp_path: Path, model_paths: ModelPaths, parameters: IdentifiedParameters
) -> None:
    """The verification must actually bite, not just pass on good input."""
    spec = mujoco.MjSpec.from_file(str(model_paths.hydrax))
    spec.meshdir = str((model_paths.hydrax.parent / spec.meshdir).resolve())
    spec.texturedir = str((model_paths.hydrax.parent / spec.texturedir).resolve())
    from fer_mujoco_sysid.export import apply_parameters

    apply_parameters(spec, parameters)
    # A plausible-looking edit that is not an identification result.
    spec.body("link3").mass = float(spec.body("link3").mass) * 1.05
    tampered = tmp_path / "tampered.xml"
    spec.compile()
    tampered.write_text(spec.to_xml(), encoding="utf-8")

    with pytest.raises(ExportError, match="body_mass"):
        verify_export(model_paths.hydrax, tampered, parameters)


def test_values_that_are_not_physical_are_refused() -> None:
    with pytest.raises(ExportError, match="negative"):
        IdentifiedParameters(frictionloss=(-0.1,) + _FRICTION[1:]).validate()
    with pytest.raises(ExportError, match="7 joints"):
        IdentifiedParameters(damping=(1.0, 2.0)).validate()
    with pytest.raises(ExportError, match="non-finite"):
        IdentifiedParameters(armature=(float("nan"),) + _ARMATURE[1:]).validate()


def test_partial_exports_touch_only_what_was_fitted(
    tmp_path: Path, model_paths: ModelPaths
) -> None:
    """The friction stage releases friction, not armature."""
    friction_only = IdentifiedParameters(frictionloss=_FRICTION, damping=_DAMPING)
    destination = tmp_path / "friction_only.xml"
    check = export_identified_model(
        destination, friction_only, source_path=model_paths.hydrax
    )
    assert set(check.changed["joint1"]) == {"frictionloss", "damping"}

    nominal = mujoco.MjModel.from_xml_path(str(model_paths.hydrax))
    identified = mujoco.MjModel.from_xml_path(str(destination))
    np.testing.assert_allclose(
        identified.dof_armature, nominal.dof_armature, rtol=1e-5
    )


def test_exported_model_is_not_pinned_to_this_machine(
    exported: tuple[Path, ExportCheck],
) -> None:
    """Asset paths must be relative, or the model loads on one machine only."""
    destination, _ = exported
    text = destination.read_text()
    for attribute in ("meshdir", "texturedir"):
        value = text.split(f'{attribute}="', 1)[1].split('"', 1)[0]
        assert not Path(value).is_absolute(), f"{attribute} is absolute: {value}"


def test_exported_model_still_loads_after_the_workspace_moves(
    tmp_path: Path, model_paths: ModelPaths, parameters: IdentifiedParameters
) -> None:
    """Relative asset paths are only worth having if they actually resolve.

    Export into a copy of the workspace layout, move the whole thing, and
    load the model from its new home.
    """
    import shutil

    original = tmp_path / "before"
    assets = original / "nominal"
    assets.mkdir(parents=True)
    shutil.copytree(model_paths.hydrax.parent, assets, dirs_exist_ok=True)

    destination = original / "out" / "identified.xml"
    export_identified_model(
        destination, parameters, source_path=assets / model_paths.hydrax.name
    )

    moved = tmp_path / "after"
    original.rename(moved)
    model = mujoco.MjModel.from_xml_path(str(moved / "out" / "identified.xml"))
    assert model.njnt > 0
