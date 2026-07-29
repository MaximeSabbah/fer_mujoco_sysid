"""Fast contract tests for the complete simulation orchestration layer."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from fer_mujoco_sysid import simulation_pipeline
from fer_mujoco_sysid.io import read_json, write_json


def _validation(
    *,
    estimate: float = 1.00000001,
    truth_sha256: str = "shared-truth",
) -> dict[str, object]:
    return {
        "accepted": True,
        "problems": [],
        "simulation_truth": {"content_sha256": truth_sha256},
        "parameters": [
            {
                "parameter": "joint1.damping",
                "kind": "damping",
                "truth": 1.0,
                "estimated": estimate,
                "bound_span": 8.0,
            }
        ],
        "mppi_horizon": {
            "fer-friction-holdout": {
                "worst_q_rmse_rad": 2.0e-8,
                "worst_dq_rmse_rad_s": 3.0e-8,
                "gripper_rmse_mm": 4.0e-7,
            }
        },
        "torque_equation_closure": {
            "fer-friction-holdout": {
                "identified": {"worst_rmse_Nm": 3.0e-8}
            }
        },
    }


def _recordings(
    *,
    protocol_sha256: str = "protocol-sha",
    source_model_sha256: str = "source-sha",
) -> list[dict[str, object]]:
    return [
        {
            "protocol_id": "fer-friction-holdout",
            "family": "friction",
            "role": "holdout",
            "path": "fer-friction-holdout",
            "recording_sha256": "recording-sha",
            "protocol_sha256": protocol_sha256,
            "source_model_sha256": source_model_sha256,
            "samples": 160,
            "usable": True,
        }
    ]


def _backend_result(
    backend: str,
    *,
    estimate: float = 1.00000001,
    truth_sha256: str = "shared-truth",
    protocol_sha256: str = "protocol-sha",
) -> dict[str, object]:
    return {
        "backend": backend,
        "recordings": _recordings(protocol_sha256=protocol_sha256),
        "validation": _validation(
            estimate=estimate,
            truth_sha256=truth_sha256,
        ),
    }


def _patch_fast_single_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    acquisition_error: BaseException | None = None,
) -> None:
    truth = {
        "content_sha256": "shared-truth",
        "source_model": {"sha256": "source-sha"},
    }

    def fake_write_truth(path: str | Path) -> Path:
        destination = Path(path)
        write_json(destination, truth)
        return destination

    def fake_acquire(backend_root: Path, _truth_path: Path) -> None:
        if acquisition_error is not None:
            raise acquisition_error
        backend_root.mkdir(parents=True)

    monkeypatch.setattr(
        simulation_pipeline,
        "write_truth_manifest",
        fake_write_truth,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "load_truth_manifest",
        lambda _path: truth,
    )
    monkeypatch.setattr(simulation_pipeline, "_acquire_mujoco", fake_acquire)
    monkeypatch.setattr(
        simulation_pipeline,
        "_recording_inventory",
        lambda *_args, **_kwargs: _recordings(),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "identify_campaign",
        lambda *_args, **_kwargs: {"status": "accepted"},
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "validate_simulation_result",
        lambda *_args, **_kwargs: _validation(),
    )


def test_compare_backends_accepts_matching_near_exact_results() -> None:
    direct = _backend_result("mujoco", estimate=1.00000001)
    ros = _backend_result("mujoco_ros", estimate=1.00000002)

    comparison = simulation_pipeline.compare_backends(direct, ros)

    assert comparison["accepted"] is True
    assert comparison["problems"] == []
    assert comparison["truth_match"] is True
    assert comparison["protocol_lineage_match"] is True
    assert comparison["parameter_names_match"] is True
    assert (
        comparison["maximum_parameter_difference_fraction_of_bound_span"]
        < comparison["parameter_difference_limit_fraction_of_bound_span"]
    )


def test_compare_backends_rejects_lineage_and_parameter_disagreement() -> None:
    direct = _backend_result("mujoco")
    ros = copy.deepcopy(_backend_result("mujoco_ros"))
    ros["validation"]["simulation_truth"]["content_sha256"] = "different-truth"
    ros["recordings"][0]["protocol_sha256"] = "different-protocol"
    ros["validation"]["parameters"][0]["estimated"] = 1.1

    comparison = simulation_pipeline.compare_backends(direct, ros)

    assert comparison["accepted"] is False
    assert comparison["truth_match"] is False
    assert comparison["protocol_lineage_match"] is False
    assert any("same simulation truth" in item for item in comparison["problems"])
    assert any("lineages differ" in item for item in comparison["problems"])
    assert any(
        "parameter disagreement" in item for item in comparison["problems"]
    )


def test_single_backend_run_is_diagnostic_and_root_is_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_single_backend(monkeypatch)
    root = tmp_path / "simulation-run"

    result = simulation_pipeline.run(
        root,
        backends=("mujoco",),
        max_iters=2,
        dynamic_starts=1,
        render_media=False,
    )

    assert result["state"] == "diagnostic_accepted"
    assert result["release_ready"] is False
    assert result["primary_model"] is None
    assert result["release_model_manifest"] is None
    assert result["cross_backend"] is None
    assert not (root / "release_model.json").exists()

    persisted = read_json(root / "simulation_pipeline.json")
    assert persisted["state"] == "diagnostic_accepted"
    assert persisted["release_ready"] is False
    assert persisted["primary_model"] is None
    assert persisted["release_model_manifest"] is None

    status = read_json(root / "simulation_pipeline_status.json")
    assert status["state"] == "diagnostic_accepted"
    assert status["release_ready"] is False
    assert status["primary_model"] is None
    assert status["result_manifest"] == "simulation_pipeline.json"

    with pytest.raises(FileExistsError):
        simulation_pipeline.run(
            root,
            backends=("mujoco",),
            render_media=False,
        )


def test_failure_writes_failed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_single_backend(
        monkeypatch,
        acquisition_error=RuntimeError("synthetic acquisition failure"),
    )
    root = tmp_path / "failed-run"

    with pytest.raises(RuntimeError, match="synthetic acquisition failure"):
        simulation_pipeline.run(
            root,
            backends=("mujoco",),
            render_media=False,
        )

    status = read_json(root / "simulation_pipeline_status.json")
    assert status["state"] == "failed"
    assert status["release_ready"] is False
    assert status["primary_model"] is None
    assert status["result_manifest"] is None
    assert status["error"] == {
        "type": "RuntimeError",
        "message": "synthetic acquisition failure",
    }
    assert not (root / "simulation_pipeline.json").exists()
    assert not (root / "release_model.json").exists()
