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
    BodyInertial,
    ConsumerModelExportCheck,
    ExportCheck,
    ExportError,
    IdentifiedParameters,
    apply_parameters,
    export_consumer_model,
    export_full_source_patched_model,
    export_identified_model,
    spec_to_xml_exact_dynamics,
    verify_consumer_model_export,
    verify_export,
)
from fer_mujoco_sysid.model import (
    HYDRAX_ARM_JOINT_NAMES,
    ModelPaths,
    build_hydrax_arm_model,
    build_hydrax_arm_spec,
)

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
    assert manifest["artifact_kind"] == "full-source-parameter-patch"
    assert manifest["consumer_ready"] is False
    assert "consumer planning derivation not validated" in manifest["validation_scope"]
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
    tampered.write_text(spec_to_xml_exact_dynamics(spec), encoding="utf-8")

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
    np.testing.assert_allclose(identified.dof_armature, nominal.dof_armature, rtol=1e-5)


def _armature_and_body_parameters(model_paths: ModelPaths) -> IdentifiedParameters:
    nominal = mujoco.MjModel.from_xml_path(str(model_paths.hydrax))
    body = BodyInertial.from_model(nominal, "link3")
    return IdentifiedParameters(
        armature=_ARMATURE,
        body_inertials=(
            BodyInertial(
                body="link3",
                mass=body.mass * 1.08,
                ipos=tuple(np.asarray(body.ipos) + np.array((0.003, -0.002, 0.004))),
                inertia=tuple(np.asarray(body.inertia) * 1.05),
            ),
        ),
    )


def test_armature_and_body_inertials_export_exactly_together(
    tmp_path: Path, model_paths: ModelPaths
) -> None:
    parameters = _armature_and_body_parameters(model_paths)
    destination = tmp_path / "armature_and_body.xml"
    manifest_path = tmp_path / "armature_and_body.json"

    check = export_identified_model(
        destination,
        parameters,
        source_path=model_paths.hydrax,
        manifest_path=manifest_path,
    )
    written = mujoco.MjModel.from_xml_path(str(destination))
    written_body = BodyInertial.from_model(written, "link3")
    wanted_body = parameters.body_inertials[0]

    assert written_body.mass == pytest.approx(wanted_body.mass, rel=1e-5)
    np.testing.assert_allclose(
        written_body.ipos, wanted_body.ipos, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        written_body.inertia, wanted_body.inertia, rtol=1e-5, atol=1e-8
    )
    for index, name in enumerate(HYDRAX_ARM_JOINT_NAMES):
        dof = int(written.joint(name).dofadr[0])
        assert written.dof_armature[dof] == pytest.approx(_ARMATURE[index], rel=1e-5)
    assert set(check.body_changes["link3"]) == {"mass", "ipos", "inertia"}
    assert check.roundtrip_error < 1e-5
    assert "link3" in check.table()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["exported_fields"] == ["armature", "body_inertials"]
    assert manifest["parameters"]["body_inertials"]["link3"]["mass"] == pytest.approx(
        wanted_body.mass
    )
    restored = IdentifiedParameters.from_mapping(
        manifest["parameters"] | {"joints": manifest["joints"]}
    )
    assert restored == parameters


def test_from_model_captures_exact_armature_and_body_values(
    model_paths: ModelPaths,
) -> None:
    wanted = _armature_and_body_parameters(model_paths)
    spec = mujoco.MjSpec.from_file(str(model_paths.hydrax))
    fitted = apply_parameters(spec, wanted).compile()

    captured = IdentifiedParameters.from_model(
        fitted,
        bodies=("link3",),
        include_frictionloss=False,
        include_damping=False,
    )

    assert captured.armature == pytest.approx(_ARMATURE)
    assert captured.body_inertials[0].mass == pytest.approx(
        wanted.body_inertials[0].mass
    )
    np.testing.assert_allclose(
        captured.body_inertials[0].ipos,
        wanted.body_inertials[0].ipos,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        captured.body_inertials[0].inertia,
        wanted.body_inertials[0].inertia,
        rtol=1e-9,
        atol=5e-12,
    )


def test_body_export_still_rejects_a_structural_edit(
    tmp_path: Path, model_paths: ModelPaths
) -> None:
    parameters = _armature_and_body_parameters(model_paths)
    spec = mujoco.MjSpec.from_file(str(model_paths.hydrax))
    spec.meshdir = str((model_paths.hydrax.parent / spec.meshdir).resolve())
    spec.texturedir = str((model_paths.hydrax.parent / spec.texturedir).resolve())
    apply_parameters(spec, parameters)
    geom = spec.geoms[0]
    position = np.asarray(geom.pos, dtype=np.float64).copy()
    position[0] += 0.01
    geom.pos = position
    tampered = tmp_path / "body_with_tampered_geom.xml"
    spec.compile()
    tampered.write_text(spec_to_xml_exact_dynamics(spec), encoding="utf-8")

    with pytest.raises(ExportError, match="geom_pos"):
        verify_export(model_paths.hydrax, tampered, parameters)


def _accepted_fitted_model(
    model_paths: ModelPaths, parameters: IdentifiedParameters
) -> mujoco.MjModel:
    spec = build_hydrax_arm_spec(
        model_paths.hydrax,
        joint_state_sensors=True,
    )
    spec.option.gravity = [0.0, 0.0, 0.0]
    apply_parameters(spec, parameters)
    return spec.compile()


def test_consumer_export_validates_full_and_hydrax_planning_loads(
    tmp_path: Path, model_paths: ModelPaths
) -> None:
    parameters = _armature_and_body_parameters(model_paths)
    accepted = _accepted_fitted_model(model_paths, parameters)
    destination = tmp_path / "fer_consumer.xml"

    check = export_consumer_model(
        destination,
        parameters,
        accepted_fitted_model=accepted,
        source_path=model_paths.hydrax,
    )

    assert isinstance(check, ConsumerModelExportCheck)
    full = mujoco.MjModel.from_xml_path(str(destination))
    assert (full.nq, full.nv, full.nu, full.njnt, full.neq) == (9, 9, 8, 9, 1)
    assert full.joint("finger_joint1").name == "finger_joint1"
    np.testing.assert_allclose(full.opt.gravity, (0.0, 0.0, -9.81))
    assert not (int(full.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))
    assert int(full.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_EULER)

    planning = build_hydrax_arm_model(destination)
    assert (planning.nq, planning.nv, planning.nu, planning.njnt, planning.neq) == (
        7,
        7,
        7,
        7,
        0,
    )
    np.testing.assert_allclose(planning.opt.gravity, (0.0, 0.0, -9.81))
    assert int(planning.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    assert int(planning.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    assert check.planning_compiled_error < 2e-5
    assert check.planning_behavior_error < 1e-4
    assert check.fit_convention_behavior_error < 1e-4

    manifest = json.loads(destination.with_suffix(".json").read_text())
    assert manifest["format"] == "fer-mujoco-sysid/consumer-model@1"
    assert manifest["artifact_kind"] == "gravity-enabled-full-panda-consumer-model"
    assert manifest["consumer_ready"] is True
    assert len(manifest["source_model"]["sha256"]) == 64
    assert manifest["projection"]["removed_source_joint_names"] == [
        "finger_joint1",
        "finger_joint2",
    ]
    assert manifest["projection"]["removed_source_actuator_names"] == ["actuator8"]
    assert manifest["projection"]["source_equality_count"] == 1
    assert manifest["projection"]["planning_equality_count"] == 0
    conventions = manifest["simulation_conventions"]
    assert conventions["accepted_fit"]["gravity_enabled"] is False
    assert conventions["full_consumer_model"]["gravity_enabled"] is True
    assert conventions["full_consumer_model"]["contacts_enabled"] is True
    assert conventions["hydrax_planning_model"]["gravity_enabled"] is True
    assert conventions["hydrax_planning_model"]["contacts_enabled"] is False
    assert (
        conventions["hydrax_planning_model"]["integrator"]["name"]
        == "mjINT_IMPLICITFAST"
    )
    torque = conventions["torque_semantics"]
    assert torque["mppi_inverse_dynamics_includes_gravity"] is True
    assert torque["real_lfc_remove_gravity_compensation_effort"] is True
    assert torque["simulation_remove_gravity_compensation_effort"] is False
    assert manifest["verification"]["ordinary_full_mjmodel_reload"] is True
    assert manifest["verification"]["hydrax_planning_derivation_reload"] is True


def test_consumer_verifier_rejects_a_changed_gravity_convention(
    tmp_path: Path, model_paths: ModelPaths
) -> None:
    parameters = _armature_and_body_parameters(model_paths)
    accepted = _accepted_fitted_model(model_paths, parameters)
    destination = tmp_path / "consumer.xml"
    export_full_source_patched_model(
        destination,
        parameters,
        source_path=model_paths.hydrax,
    )

    text = destination.read_text()
    destination.write_text(
        text.replace(
            '<mujoco model="panda">',
            '<mujoco model="panda">\n  <option gravity="0 0 0"/>',
            1,
        )
    )

    with pytest.raises(ExportError, match="opt.gravity|lost gravity"):
        verify_consumer_model_export(
            accepted,
            destination,
            parameters,
            source_path=model_paths.hydrax,
        )


def test_consumer_export_refuses_a_non_gravity_free_fit_before_writing(
    tmp_path: Path, model_paths: ModelPaths
) -> None:
    parameters = IdentifiedParameters(armature=_ARMATURE)
    spec = build_hydrax_arm_spec(model_paths.hydrax)
    apply_parameters(spec, parameters)
    gravity_enabled_fit = spec.compile()
    destination = tmp_path / "must_not_exist.xml"

    with pytest.raises(ExportError, match="gravity-free Franka telemetry"):
        export_consumer_model(
            destination,
            parameters,
            accepted_fitted_model=gravity_enabled_fit,
            source_path=model_paths.hydrax,
        )
    assert not destination.exists()


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
