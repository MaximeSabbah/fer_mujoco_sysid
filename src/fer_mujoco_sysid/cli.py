"""Command-line entry points for FER MuJoCo system identification."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from fer_mujoco_sysid.artifacts.catalog import (
    ValidatedDataset,
    validate_dataset_bundle,
)
from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fer-mujoco-sysid")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-dataset",
        help="validate a sealed dataset and its local artifact closure",
    )
    validate.add_argument("dataset_root", type=Path)
    validate.add_argument(
        "--catalog-root",
        type=Path,
        help="root against which relative artifact locators are resolved",
    )
    return parser


def _sorted_ids(records: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(sorted(records))


def _print_collection(label: str, identifiers: tuple[str, ...]) -> None:
    rendered = ", ".join(identifiers) if identifiers else "-"
    print(f"{label} ({len(identifiers)}): {rendered}")


def _print_dataset_summary(dataset: ValidatedDataset) -> None:
    manifest = dataset.dataset.manifest
    splits_manifest = dataset.splits.manifest
    partition_counts = Counter(dataset.partition_by_trajectory.values())
    partition_names = sorted(set(splits_manifest["partitions"]) | set(partition_counts))

    print("valid sealed dataset")
    print(f"dataset id: {manifest['dataset_id']}")
    print(f"dataset artifact id: {manifest['artifact_id']}")
    print(f"dataset version: {manifest['version']}")
    print(f"splits artifact id: {splits_manifest['artifact_id']}")
    _print_collection("protocols", _sorted_ids(dataset.protocols))
    _print_collection("runs", _sorted_ids(dataset.runs))
    _print_collection("trajectories", _sorted_ids(dataset.trajectories))
    print("partitions:")
    for partition in partition_names:
        print(f"  {partition}: {partition_counts[partition]}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit status."""

    args = _parser().parse_args(argv)
    if args.command != "validate-dataset":  # pragma: no cover - argparse owns this
        raise AssertionError(f"unhandled command: {args.command}")

    try:
        dataset = validate_dataset_bundle(
            args.dataset_root,
            catalog_root=args.catalog_root,
            require_sealed=True,
        )
    except (ArtifactValidationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_dataset_summary(dataset)
    return 0
