from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from fer_mujoco_sysid import cli
from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError


def _validated_dataset() -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(
            manifest=MappingProxyType(
                {
                    "artifact_id": "dataset-artifact",
                    "dataset_id": "fer-example",
                    "version": "2.1.0",
                }
            )
        ),
        splits=SimpleNamespace(
            manifest=MappingProxyType(
                {
                    "artifact_id": "splits-example",
                    "partitions": MappingProxyType(
                        {
                            "fit": (),
                            "development": (),
                            "held_out_test": (),
                            "diagnostic_only": (),
                            "excluded": (),
                        }
                    ),
                }
            )
        ),
        protocols=MappingProxyType(
            {
                "protocol-z": object(),
                "protocol-a": object(),
            }
        ),
        runs=MappingProxyType(
            {
                "run-b": object(),
                "run-a": object(),
            }
        ),
        trajectories=MappingProxyType(
            {
                "trajectory-heldout": object(),
                "trajectory-fit-b": object(),
                "trajectory-fit-a": object(),
            }
        ),
        partition_by_trajectory=MappingProxyType(
            {
                "trajectory-heldout": "held_out_test",
                "trajectory-fit-b": "fit",
                "trajectory-fit-a": "fit",
            }
        ),
    )


def test_validate_dataset_is_sealed_and_summary_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, Path | None, bool]] = []

    def validate(
        dataset_root: Path,
        *,
        catalog_root: Path | None,
        require_sealed: bool,
    ) -> SimpleNamespace:
        calls.append((dataset_root, catalog_root, require_sealed))
        return _validated_dataset()

    monkeypatch.setattr(cli, "validate_dataset_bundle", validate)
    dataset_root = tmp_path / "dataset"
    catalog_root = tmp_path / "catalog"

    status = cli.main(
        [
            "validate-dataset",
            str(dataset_root),
            "--catalog-root",
            str(catalog_root),
        ]
    )

    assert status == 0
    assert calls == [(dataset_root, catalog_root, True)]
    assert capsys.readouterr() == (
        "valid sealed dataset\n"
        "dataset id: fer-example\n"
        "dataset artifact id: dataset-artifact\n"
        "dataset version: 2.1.0\n"
        "splits artifact id: splits-example\n"
        "protocols (2): protocol-a, protocol-z\n"
        "runs (2): run-a, run-b\n"
        "trajectories (3): trajectory-fit-a, trajectory-fit-b, "
        "trajectory-heldout\n"
        "partitions:\n"
        "  development: 0\n"
        "  diagnostic_only: 0\n"
        "  excluded: 0\n"
        "  fit: 2\n"
        "  held_out_test: 1\n",
        "",
    )


@pytest.mark.parametrize(
    "error",
    [
        ArtifactValidationError("seal.json is missing"),
        FileNotFoundError("dataset.json is missing"),
    ],
)
def test_validate_dataset_reports_expected_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(cli, "validate_dataset_bundle", fail)

    assert cli.main(["validate-dataset", "missing"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error: {error}\n"
    assert "Traceback" not in captured.err
