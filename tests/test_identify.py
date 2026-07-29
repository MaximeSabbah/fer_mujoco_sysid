"""Gates for turning a recording into something the fit can roll out.

The recording arrives on the controller's clock; the fit rolls the model out
at the model's timestep. Getting that hand-off wrong is silent — the fit just
explains the misalignment with wrong parameters — so it is gated here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid import identify
from fer_mujoco_sysid.dataset import (
    BOUND_RECORDING_FORMAT,
    LEGACY_RECORDING_FORMAT,
    write_recording,
)
from fer_mujoco_sysid.io import sha256_file
from fer_mujoco_sysid.model import ModelPaths, build_hydrax_arm_spec
from fer_mujoco_sysid.preparation import prepare_recording
from fer_mujoco_sysid.protocol import (
    FRICTION_CRUISE,
    FRICTION_FAMILY,
    HOLDOUT_ROLE,
    ROS_ARM_JOINT_NAMES,
    TRAIN_ROLE,
)

_RATE_HZ = 500.0
_DURATION_S = 3.0


@pytest.fixture(scope="module")
def model(model_paths: ModelPaths) -> mujoco.MjModel:
    return build_hydrax_arm_spec(model_paths.hydrax, joint_state_sensors=True).compile()


def _write(
    root: Path,
    *,
    source_model: Path,
    protocol_id: str = "fixture-without-name-semantics",
    role: str = TRAIN_ROLE,
    rate_hz: float = _RATE_HZ,
    artifact_format: str = BOUND_RECORDING_FORMAT,
) -> Path:
    time_s = np.arange(int(_DURATION_S * rate_hz)) / rate_hz
    velocity = np.array([0.05, -0.07, 0.09, -0.11, 0.13, -0.15, 0.17])
    start_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    dq = np.repeat(velocity[None, :], len(time_s), axis=0)
    q = start_q[None, :] + time_s[:, None] * velocity[None, :]
    duration_s = float(time_s[-1])
    guard_s = 0.1
    analysis_eligible = (time_s >= guard_s) & (time_s <= duration_s - guard_s)
    arrays = {
        "time_s": time_s,
        "q_rad": q,
        "dq_rad_s": dq,
        "tau_cmd_Nm": 1.1 * np.sign(dq) + 0.7 * dq,
        "protocol_segment_index": np.zeros(len(time_s), dtype=np.int16),
        "analysis_eligible": analysis_eligible,
    }
    protocol_sha256 = hashlib.sha256(
        f"{protocol_id}:{FRICTION_FAMILY}:{role}".encode()
    ).hexdigest()
    segment = {
        "segment_id": "constant_velocity",
        "kind": "excitation",
        "start_index": 0,
        "end_index_exclusive": len(time_s),
        "analysis_eligible": True,
    }
    window = {
        "window_id": "constant_velocity_cruise",
        "kind": FRICTION_CRUISE,
        "parent_segment_id": "constant_velocity",
        "start_index": 0,
        "end_index_exclusive": len(time_s),
        "nominal_velocity_rad_s": velocity.tolist(),
    }
    start_stamp_ns = 1_000_000_000
    end_stamp_ns = start_stamp_ns + int(round(duration_s * 1e9))
    manifest = {
        "format": artifact_format,
        "joint_order": list(ROS_ARM_JOINT_NAMES),
        "backend": "mujoco",
        "speed_scale": 1.0,
        "torque_source": (
            "synthetic controller request; applied/post-limiter status is not asserted"
        ),
        "torque_channels": {
            "tau_cmd_Nm": {
                "source_topic": "test fixture",
                "message_field": "synthetic torque law",
                "meaning": "known friction torque used by the fixture",
                "clock": "fixture sample clock",
                "rate_limiting": "none",
                "application_status": "not_asserted",
            }
        },
        "torque_limit_Nm": [87.0] * 4 + [12.0] * 3,
        "protocol": {
            "protocol_id": protocol_id,
            "family": FRICTION_FAMILY,
            "role": role,
            "content_sha256": protocol_sha256,
            "duration_s": duration_s,
            "segments": [segment],
            "analysis_windows": [window],
        },
        "protocol_timing": {
            "marker_format": "fer-mujoco-sysid/test-protocol-event@1",
            "marker_topic": "test_protocol_boundary",
            "start_stamp_ns": start_stamp_ns,
            "end_stamp_ns": end_stamp_ns,
            "selected_first_stamp_ns": start_stamp_ns,
            "selected_last_stamp_ns": end_stamp_ns,
            "marked_duration_s": duration_s,
            "speed_scale": 1.0,
            "recording_time_origin": "synthetic protocol start marker",
        },
        "mapped_segments": [
            {
                "interval_id": "constant_velocity",
                "kind": "excitation",
                "start_time_s": 0.0,
                "end_time_s": duration_s,
                "analysis_eligible": True,
            }
        ],
        "mapped_analysis_windows": [
            {
                "interval_id": "constant_velocity_cruise",
                "kind": FRICTION_CRUISE,
                "parent_segment_id": "constant_velocity",
                "start_time_s": guard_s,
                "end_time_s": duration_s - guard_s,
                "analysis_eligible": True,
                "nominal_velocity_rad_s": velocity.tolist(),
            }
        ],
        "analysis_policy": {
            "edge_guard_s": guard_s,
            "protocol_rows_array": "analysis_eligible",
            "telemetry_rows_array": None,
        },
        "source_model": {
            "repository": "test fixture",
            "revision": "fixture-r1",
            "path": str(source_model),
            "sha256": sha256_file(source_model),
        },
    }
    return write_recording(root, manifest, arrays)


def test_recording_is_resampled_onto_the_model_timestep(
    tmp_path: Path, model: mujoco.MjModel, model_paths: ModelPaths
) -> None:
    """The fit rolls out at the model's step; the recording arrives on another."""
    prepared = prepare_recording(
        _write(tmp_path / "run", source_model=model_paths.hydrax),
        model,
        expected_source_model=model_paths.hydrax,
    )
    times = prepared.run.control_times
    step = float(model.opt.timestep)

    np.testing.assert_allclose(np.diff(times), step, rtol=1e-9)
    assert prepared.run.control.shape[1] == model.nu
    assert prepared.run.measured.shape[1] == 14


def test_row_convention_matches_the_rollout_skew(
    tmp_path: Path, model: mujoco.MjModel, model_paths: ModelPaths
) -> None:
    """``measured`` carries the post-step stamp ``mujoco.rollout`` emits.

    Getting this wrong shifts the torque against the state by one step, which
    biases friction exactly at the velocity reversals it is read from.
    """
    prepared = prepare_recording(
        _write(tmp_path / "run", source_model=model_paths.hydrax),
        model,
        expected_source_model=model_paths.hydrax,
    )
    step = float(model.opt.timestep)

    np.testing.assert_allclose(prepared.run.control_times[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(prepared.run.measured_times[0], step, rtol=1e-9)
    np.testing.assert_allclose(
        prepared.run.measured_times - prepared.run.control_times, step, rtol=1e-9
    )
    np.testing.assert_allclose(
        prepared.run.qpos0, prepared.run.measured[0, :7], atol=1e-12
    )
    np.testing.assert_allclose(
        prepared.run.qvel0, prepared.run.measured[0, 7:14], atol=1e-12
    )


def test_a_recording_at_a_different_rate_still_lands_on_the_grid(
    tmp_path: Path, model: mujoco.MjModel, model_paths: ModelPaths
) -> None:
    """Hardware will not publish at exactly the model's rate."""
    prepared = prepare_recording(
        _write(
            tmp_path / "run",
            source_model=model_paths.hydrax,
            rate_hz=333.0,
        ),
        model,
        expected_source_model=model_paths.hydrax,
    )
    step = float(model.opt.timestep)
    np.testing.assert_allclose(np.diff(prepared.run.control_times), step, rtol=1e-9)
    assert prepared.run.control.shape[0] > 0


def test_holdout_role_comes_only_from_explicit_protocol_metadata(
    tmp_path: Path, model: mujoco.MjModel, model_paths: ModelPaths
) -> None:
    """A misleading identifier cannot promote or demote a held-out run."""
    train = prepare_recording(
        _write(
            tmp_path / "a",
            source_model=model_paths.hydrax,
            protocol_id="this-name-says-holdout",
            role=TRAIN_ROLE,
        ),
        model,
        expected_source_model=model_paths.hydrax,
    )
    held = prepare_recording(
        _write(
            tmp_path / "b",
            source_model=model_paths.hydrax,
            protocol_id="ordinary-training-looking-name",
            role=HOLDOUT_ROLE,
        ),
        model,
        expected_source_model=model_paths.hydrax,
    )
    assert not train.is_holdout
    assert held.is_holdout
    assert train.family == FRICTION_FAMILY
    assert held.family == FRICTION_FAMILY


def test_legacy_recording_remains_rejected_by_preparation(
    tmp_path: Path, model: mujoco.MjModel, model_paths: ModelPaths
) -> None:
    legacy = _write(
        tmp_path / "legacy",
        source_model=model_paths.hydrax,
        artifact_format=LEGACY_RECORDING_FORMAT,
    )
    with pytest.raises(ValueError, match="predates protocol binding"):
        prepare_recording(legacy, model)


def _mock_identification_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    accepted: bool,
    media_error: Exception | None = None,
) -> tuple[SimpleNamespace, list[dict[str, object]]]:
    """Replace fitting with sentinels while exercising publication itself."""
    fit_model = object()
    stage_mask = np.ones(1, dtype=bool)

    def recording(protocol_id: str, role: str) -> SimpleNamespace:
        return SimpleNamespace(
            root=tmp_path / "recordings" / protocol_id / "recording",
            label=protocol_id,
            content_sha256="a" * 64,
            protocol_id=protocol_id,
            protocol_content_sha256="b" * 64,
            family=FRICTION_FAMILY,
            role=role,
            backend="mujoco_ros",
            control_channel="tau_cmd_Nm",
            source_model_sha256="c" * 64,
            preprocessing={},
            health={"usable": True, "problems": []},
            stage=SimpleNamespace(
                classical_mask=stage_mask,
                analysis_mask=stage_mask,
                protocol_mask=stage_mask,
            ),
        )

    prepared = [
        recording("friction-train", TRAIN_ROLE),
        recording("friction-holdout", HOLDOUT_ROLE),
    ]
    queue = iter(prepared)
    model_path = tmp_path / "nominal.xml"
    stages = SimpleNamespace(
        nominal=object(),
        classical=object(),
        identified=fit_model,
        parameters=object(),
        summary=lambda: {"mock": True},
    )
    acceptance = identify.ReproductionAcceptance(
        accepted=accepted,
        problems=() if accepted else ("fixture rejection",),
        relative_scores={},
    )
    export_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        identify,
        "resolve_model_paths",
        lambda: SimpleNamespace(require=lambda: SimpleNamespace(hydrax=model_path)),
    )
    monkeypatch.setattr(
        identify,
        "fitting_spec",
        lambda _path: SimpleNamespace(compile=lambda: object()),
    )
    monkeypatch.setattr(identify, "find_recordings", lambda _root: [1, 2])
    monkeypatch.setattr(identify, "prepare_recording", lambda *_a, **_k: next(queue))
    monkeypatch.setattr(identify, "validate_campaign_lineage", lambda *_a, **_k: None)
    monkeypatch.setattr(identify, "fit_stages", lambda *_a, **_k: stages)
    monkeypatch.setattr(
        identify,
        "_holdout_evaluation",
        lambda record, _models: (
            {},
            {},
            {
                "protocol_id": record.protocol_id,
                "family": record.family,
                "control_channel": record.control_channel,
                "horizons": {},
                "worst_torque_rmse_Nm": {},
            },
        ),
    )
    monkeypatch.setattr(
        identify,
        "accept_reproduction",
        lambda *_a, **_k: acceptance,
    )

    def fake_export(destination, parameters, **kwargs):
        export_calls.append(
            {
                "destination": destination,
                "parameters": parameters,
                **kwargs,
            }
        )
        Path(destination).write_text("<mujoco/>", encoding="utf-8")
        identify.write_json(
            kwargs["manifest_path"],
            {"consumer_model": {"path": str(destination)}},
        )
        return SimpleNamespace(
            roundtrip_error=0.0,
            planning_compiled_error=0.0,
            planning_behavior_error=0.0,
            fit_convention_behavior_error=0.0,
            table=lambda: "| mock |",
        )

    monkeypatch.setattr(identify, "export_consumer_model", fake_export)
    monkeypatch.setattr(
        identify,
        "_export_behavior_parity",
        lambda *_a, **_k: {"worst_q_rmse_difference_rad": 0.0},
    )

    def fake_media(output_dir, *_args):
        if media_error is not None:
            raise media_error
        if accepted:
            assert not (
                output_dir.parent / "identification" / "fer_identified.xml"
            ).exists()
        plot = output_dir / "diagnostic.png"
        plot.write_bytes(b"plot")
        return ["diagnostic.png"]

    monkeypatch.setattr(identify, "_render_outputs", fake_media)

    def fake_report(path, summary, _stages, export, acceptance):
        path.write_text(
            f"{summary['status']} export={export is not None} "
            f"accepted={acceptance.accepted}\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(identify, "_write_report", fake_report)
    return SimpleNamespace(fit_model=fit_model), export_calls


def test_accepted_consumer_is_published_only_after_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinels, export_calls = _mock_identification_attempt(
        monkeypatch,
        tmp_path,
        accepted=True,
    )
    output = tmp_path / "identification"

    summary = identify.run(tmp_path / "recordings", output)

    assert summary["status"] == "accepted"
    assert (output / "diagnostic.png").is_file()
    assert (output / "result.md").is_file()
    assert (output / "fer_identified.xml").is_file()
    call = export_calls[0]
    assert call["accepted_fitted_model"] is sentinels.fit_model
    assert Path(call["destination"]).parent != output
    manifest = json.loads((output / "fer_identified.json").read_text())
    assert manifest["consumer_model"]["path"] == str(output / "fer_identified.xml")
    status = json.loads((output / identify.STATUS_FILENAME).read_text())
    assert status["state"] == "accepted"
    assert status["consumer_model_current"] is True


def test_accepted_fit_with_failed_media_does_not_publish_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_identification_attempt(
        monkeypatch,
        tmp_path,
        accepted=True,
        media_error=RuntimeError("renderer unavailable"),
    )
    output = tmp_path / "identification"

    with pytest.raises(RuntimeError, match="renderer unavailable"):
        identify.run(tmp_path / "recordings", output)

    assert not (output / "fer_identified.xml").exists()
    assert not (output / "fer_identified.json").exists()
    assert not (output / "identification.json").exists()
    status = json.loads((output / identify.STATUS_FILENAME).read_text())
    assert status["state"] == "failed"
    assert status["consumer_model_current"] is False


def test_rejected_fit_keeps_diagnostics_and_retires_stale_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, export_calls = _mock_identification_attempt(
        monkeypatch,
        tmp_path,
        accepted=False,
    )
    output = tmp_path / "identification"
    output.mkdir()
    (output / "fer_identified.xml").write_text("stale xml", encoding="utf-8")
    (output / "fer_identified.json").write_text("{}", encoding="utf-8")
    (output / "user-notes.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fixture rejection"):
        identify.run(tmp_path / "recordings", output)

    assert export_calls == []
    assert (output / "identification.json").is_file()
    assert (output / "result.md").read_text().startswith("rejected export=False")
    assert (output / "diagnostic.png").is_file()
    assert not (output / "fer_identified.xml").exists()
    assert not (output / "fer_identified.json").exists()
    assert (output / "user-notes.txt").read_text() == "keep me"
    archived = list((output / "previous_consumer_models").rglob("fer_identified.xml"))
    assert len(archived) == 1
    assert archived[0].read_text() == "stale xml"
    result = json.loads((output / "identification.json").read_text())
    assert result["exported_model"] is None
    assert result["consumer_model_manifest"] is None
    status = json.loads((output / identify.STATUS_FILENAME).read_text())
    assert status["state"] == "rejected"
    assert status["consumer_model_current"] is False


def test_report_distinguishes_fit_and_consumer_gravity_conventions(
    tmp_path: Path,
) -> None:
    stages = SimpleNamespace(
        linear=SimpleNamespace(
            covariance_method="Newey-West",
            covariance_lags=10,
            table=lambda: "| friction |",
        ),
        first_friction=SimpleNamespace(objective_reduction=0.5),
        dynamic=None,
        friction_refit=SimpleNamespace(objective_reduction=0.6),
        method_disagreement_percent=1.0,
        coupling_refinement_rounds=2,
        coupling_max_change_fraction=1e-6,
        parameters=SimpleNamespace(
            frictionloss=(0.1,) * 7,
            damping=(0.2,) * 7,
        ),
    )
    acceptance = identify.ReproductionAcceptance(True, (), {})
    summary = {
        "recordings": [],
        "train": ["friction-train"],
        "holdout": ["friction-holdout"],
        "held_out": {},
        "exported_model": str(tmp_path / "fer_identified.xml"),
    }
    export = SimpleNamespace(table=lambda: "| exported parameters |")
    report = tmp_path / "result.md"

    identify._write_report(report, summary, stages, export, acceptance)

    text = report.read_text()
    assert "gravity-free effective-effort convention" in text
    assert "gravity-enabled full Panda consumer model" in text
    assert "Hydrax/MPPI" in text
    assert "final LFC adapter subtracts the gravity contribution" in text
