"""Protocol identity, recording alignment, and analysis-window gates."""

from __future__ import annotations

import copy
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid.excitation import (
    FrictionProtocolSpec,
    InertialProtocolSpec,
    generate_friction_protocol,
    generate_inertial_protocol,
    write_protocol_bundle,
)
from fer_mujoco_sysid.io import ArtifactError
from fer_mujoco_sysid.model import build_hydrax_arm_spec
from fer_mujoco_sysid.protocol import (
    DEFAULT_ANALYSIS_EDGE_GUARD_S,
    FRICTION_CRUISE,
    FRICTION_FAMILY,
    INERTIAL_EXCITATION,
    INERTIAL_FAMILY,
    TRAIN_ROLE,
    align_protocol_reference,
    alignment_from_start_time,
    friction_cruise_mask,
    inertial_excitation_mask,
    load_protocol_bundle,
    map_analysis_windows,
    map_protocol_segments,
    protocol_analysis_windows,
    protocol_family,
    protocol_role,
    validate_protocol_manifest,
)


@pytest.fixture(scope="module")
def nominal_model(model_paths) -> mujoco.MjModel:
    return build_hydrax_arm_spec(
        model_paths.hydrax, joint_state_sensors=True
    ).compile()


@pytest.fixture(scope="module")
def bundles(nominal_model: mujoco.MjModel, tmp_path_factory):
    root = tmp_path_factory.mktemp("protocol-masks")
    friction_spec = FrictionProtocolSpec(
        protocol_id="name-does-not-declare-role",
        seed=31,
        role=TRAIN_ROLE,
        sample_period_s=0.01,
    )
    inertial_spec = InertialProtocolSpec(
        protocol_id="also-not-name-derived",
        seed=32,
        role=TRAIN_ROLE,
        candidates=2,
    )
    friction = generate_friction_protocol(friction_spec, nominal_model)
    inertial = generate_inertial_protocol(inertial_spec, nominal_model)
    friction_root = write_protocol_bundle(
        friction,
        root,
        model=nominal_model,
        created_at="2026-07-29T00:00:00Z",
    )
    inertial_root = write_protocol_bundle(
        inertial,
        root,
        model=nominal_model,
        created_at="2026-07-29T00:00:00Z",
    )
    return {
        "friction": load_protocol_bundle(friction_root),
        "inertial": load_protocol_bundle(inertial_root),
    }


def test_family_and_role_are_explicit_not_name_derived(bundles) -> None:
    manifest, arrays = bundles["friction"]
    assert protocol_family(manifest) == FRICTION_FAMILY
    assert protocol_role(manifest) == TRAIN_ROLE

    renamed = copy.deepcopy(manifest)
    renamed["protocol_id"] = "fer-inertial-holdout"
    validate_protocol_manifest(renamed, arrays)
    assert protocol_family(renamed) == FRICTION_FAMILY
    assert protocol_role(renamed) == TRAIN_ROLE

    missing_role = copy.deepcopy(manifest)
    del missing_role["role"]
    with pytest.raises(ArtifactError, match="protocol role"):
        validate_protocol_manifest(missing_role, arrays)


def test_typed_generators_reject_wrong_identity(
    nominal_model: mujoco.MjModel,
) -> None:
    friction = FrictionProtocolSpec(protocol_id="bad-family", seed=1)
    with pytest.raises(ValueError, match="family must be"):
        generate_friction_protocol(
            replace(friction, family_id=INERTIAL_FAMILY), nominal_model
        )
    with pytest.raises(ValueError, match="protocol role"):
        generate_friction_protocol(replace(friction, role="validation"), nominal_model)


def test_friction_mask_selects_only_guarded_true_cruises(bundles) -> None:
    manifest, arrays = bundles["friction"]
    start_time_s = 2.73
    recording_time_s = np.arange(
        0.0, start_time_s + float(arrays["time_s"][-1]) + 0.5, 0.01
    )
    alignment = alignment_from_start_time(manifest, start_time_s=start_time_s)
    mask = friction_cruise_mask(
        manifest, arrays, recording_time_s, alignment
    )

    assert mask.any()
    assert not mask[recording_time_s < start_time_s].any()
    windows = protocol_analysis_windows(manifest)
    assert len(windows) == 6
    assert {window.kind for window in windows} == {FRICTION_CRUISE}

    protocol_indices = np.rint(
        (recording_time_s[mask] - start_time_s) / 0.01
    ).astype(np.intp)
    np.testing.assert_allclose(arrays["ddq_rad_s2"][protocol_indices], 0.0)
    allowed_velocities = np.asarray(
        [window.nominal_velocity_rad_s for window in windows]
    )
    for velocity in arrays["dq_rad_s"][protocol_indices]:
        assert np.any(np.all(np.isclose(allowed_velocities, velocity), axis=1))

    selected_times = recording_time_s[mask]
    mapped_segments = map_protocol_segments(manifest, arrays, alignment)
    segment_index = np.full(len(recording_time_s), -1, dtype=np.intp)
    protocol_rows = (
        (recording_time_s >= start_time_s)
        & (recording_time_s <= start_time_s + float(arrays["time_s"][-1]))
    )
    for index, segment in enumerate(mapped_segments):
        segment_index[segment.mask(recording_time_s)] = index
    assert np.all(segment_index[protocol_rows] >= 0)

    for segment in mapped_segments:
        if segment.kind != "excitation":
            assert not segment.mask(selected_times).any(), segment.interval_id

    guarded = map_analysis_windows(
        manifest,
        arrays,
        alignment,
        kind=FRICTION_CRUISE,
        edge_guard_s=DEFAULT_ANALYSIS_EDGE_GUARD_S,
    )
    raw = map_analysis_windows(
        manifest,
        arrays,
        alignment,
        kind=FRICTION_CRUISE,
        edge_guard_s=0.0,
    )
    for guarded_window, raw_window in zip(guarded, raw, strict=True):
        assert guarded_window.start_time_s - raw_window.start_time_s == pytest.approx(
            DEFAULT_ANALYSIS_EDGE_GUARD_S
        )
        assert raw_window.end_time_s - guarded_window.end_time_s == pytest.approx(
            DEFAULT_ANALYSIS_EDGE_GUARD_S
        )


def test_reference_alignment_ignores_approach_and_respects_speed_scale(
    bundles,
) -> None:
    manifest, arrays = bundles["friction"]
    start_time_s = 2.73
    speed_scale = 0.5
    played_duration = float(arrays["time_s"][-1]) / speed_scale
    recording_time_s = np.arange(
        0.0, start_time_s + played_duration + 0.5, 0.01
    )
    home = arrays["q_rad"][0]
    approach_origin = home + np.linspace(0.04, 0.16, 7)
    progress = np.minimum(recording_time_s / 2.0, 1.0)
    q_reference = (
        approach_origin[None, :]
        + progress[:, None] * (home - approach_origin)[None, :]
    )
    protocol_clock = (recording_time_s - start_time_s) * speed_scale
    playing = recording_time_s >= start_time_s
    for joint in range(7):
        q_reference[playing, joint] = np.interp(
            protocol_clock[playing],
            arrays["time_s"],
            arrays["q_rad"][:, joint],
        )

    alignment = align_protocol_reference(
        manifest,
        arrays,
        recording_time_s,
        q_reference,
        speed_scale=speed_scale,
        probe_samples=128,
    )
    assert alignment.start_time_s == pytest.approx(start_time_s, abs=0.01)
    mapped = map_analysis_windows(
        manifest,
        arrays,
        alignment,
        kind=FRICTION_CRUISE,
        edge_guard_s=0.0,
    )
    first_compiled_velocity = np.asarray(
        protocol_analysis_windows(manifest)[0].nominal_velocity_rad_s
    )
    np.testing.assert_allclose(
        mapped[0].nominal_velocity_rad_s,
        speed_scale * first_compiled_velocity,
    )

    with pytest.raises(ArtifactError, match="does not match"):
        align_protocol_reference(
            manifest,
            arrays,
            recording_time_s,
            np.zeros_like(q_reference),
            speed_scale=speed_scale,
            probe_samples=128,
        )


def test_inertial_mask_excludes_both_settle_segments(bundles) -> None:
    manifest, arrays = bundles["inertial"]
    assert protocol_family(manifest) == INERTIAL_FAMILY
    start_time_s = 1.2
    recording_time_s = np.arange(
        0.0, start_time_s + float(arrays["time_s"][-1]) + 0.5, 0.01
    )
    alignment = alignment_from_start_time(manifest, start_time_s=start_time_s)
    mask = inertial_excitation_mask(
        manifest, arrays, recording_time_s, alignment
    )
    assert mask.any()

    windows = map_analysis_windows(
        manifest,
        arrays,
        alignment,
        kind=INERTIAL_EXCITATION,
        edge_guard_s=DEFAULT_ANALYSIS_EDGE_GUARD_S,
    )
    assert len(windows) == 1
    selected = recording_time_s[mask]
    assert selected.min() >= windows[0].start_time_s
    assert selected.max() <= windows[0].end_time_s
    for segment in map_protocol_segments(manifest, arrays, alignment):
        if segment.kind == "settle":
            assert not segment.mask(selected).any()

    with pytest.raises(ArtifactError, match="requires family"):
        friction_cruise_mask(manifest, arrays, recording_time_s, alignment)
    with pytest.raises(ArtifactError, match="finite and non-negative"):
        map_analysis_windows(
            manifest,
            arrays,
            alignment,
            edge_guard_s=np.nan,
        )
