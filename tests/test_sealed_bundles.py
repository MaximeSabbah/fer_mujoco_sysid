from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

import fer_mujoco_sysid.artifacts.bundle as bundle_module
from fer_mujoco_sysid.artifacts import (
    CHECKSUM_MANIFEST_NAME,
    SEAL_NAME,
    SEAL_SCHEMA,
    ArtifactValidationError,
    finalize_bundle,
    load_json,
    sha256_file,
    verify_sealed_bundle,
    write_json,
)


def _payload(root: Path) -> None:
    (root / "nested").mkdir(parents=True)
    (root / "alpha.txt").write_bytes(b"alpha\n")
    (root / "nested" / ".hidden").write_bytes(b"\x00\x01\x02")


def _seal_for_manifest(root: Path, manifest_content: bytes) -> None:
    (root / CHECKSUM_MANIFEST_NAME).write_bytes(manifest_content)
    write_json(
        root / SEAL_NAME,
        {
            "schema": SEAL_SCHEMA,
            "checksum_manifest": CHECKSUM_MANIFEST_NAME,
            "checksum_manifest_sha256": hashlib.sha256(manifest_content).hexdigest(),
        },
    )


def _valid_bundle(root: Path) -> dict[str, str]:
    root.mkdir()
    _payload(root)
    return finalize_bundle(root)


def test_finalize_is_deterministic_and_covers_exact_payload(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _payload(first)
    _payload(second)

    first_entries = finalize_bundle(first)
    second_entries = finalize_bundle(second)

    assert list(first_entries) == ["alpha.txt", "nested/.hidden"]
    assert first_entries == second_entries
    expected_manifest = (
        f"{sha256_file(first / 'alpha.txt')}  alpha.txt\n"
        f"{sha256_file(first / 'nested' / '.hidden')}  nested/.hidden\n"
    ).encode()
    assert (first / CHECKSUM_MANIFEST_NAME).read_bytes() == expected_manifest
    assert (second / CHECKSUM_MANIFEST_NAME).read_bytes() == expected_manifest
    assert (first / SEAL_NAME).read_bytes() == (second / SEAL_NAME).read_bytes()
    assert load_json(first / SEAL_NAME) == {
        "schema": SEAL_SCHEMA,
        "checksum_manifest": CHECKSUM_MANIFEST_NAME,
        "checksum_manifest_sha256": hashlib.sha256(expected_manifest).hexdigest(),
    }


def test_verify_returns_sorted_payload_entries(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    expected = _valid_bundle(root)

    assert verify_sealed_bundle(root) == expected
    assert list(verify_sealed_bundle(root)) == sorted(expected)


@pytest.mark.parametrize("control_file", [CHECKSUM_MANIFEST_NAME, SEAL_NAME])
def test_control_files_are_not_listed(
    tmp_path: Path,
    control_file: str,
) -> None:
    root = tmp_path / "bundle"
    entries = _valid_bundle(root)
    manifest = (root / CHECKSUM_MANIFEST_NAME).read_bytes()

    assert control_file not in entries
    assert f"  {control_file}\n".encode() not in manifest


def test_finalize_rejects_already_sealed_bundle_without_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    _valid_bundle(root)
    before_manifest = (root / CHECKSUM_MANIFEST_NAME).read_bytes()
    before_seal = (root / SEAL_NAME).read_bytes()

    with pytest.raises(ArtifactValidationError, match="already sealed"):
        finalize_bundle(root)

    assert (root / CHECKSUM_MANIFEST_NAME).read_bytes() == before_manifest
    assert (root / SEAL_NAME).read_bytes() == before_seal


def test_finalize_safely_replaces_an_unsealed_checksum_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    _payload(root)
    (root / CHECKSUM_MANIFEST_NAME).write_text("incomplete\n", encoding="utf-8")

    entries = finalize_bundle(root)

    assert verify_sealed_bundle(root) == entries


@pytest.mark.parametrize("failure_on_replace", [1, 2])
def test_finalize_writes_each_control_file_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_on_replace: int,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    _payload(root)
    old_manifest = b"preexisting incomplete manifest\n"
    (root / CHECKSUM_MANIFEST_NAME).write_bytes(old_manifest)
    original_replace = os.replace
    replace_count = 0

    def failing_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == failure_on_replace:
            raise OSError("injected replacement failure")
        original_replace(source, destination)

    monkeypatch.setattr(bundle_module.os, "replace", failing_replace)

    with pytest.raises(ArtifactValidationError, match="atomically write"):
        finalize_bundle(root)

    if failure_on_replace == 1:
        assert (root / CHECKSUM_MANIFEST_NAME).read_bytes() == old_manifest
    else:
        assert (root / CHECKSUM_MANIFEST_NAME).read_bytes() != old_manifest
    assert not (root / SEAL_NAME).exists()
    assert not list(root.glob(".*.tmp"))


def test_finalize_rolls_back_seal_when_final_verification_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    _payload(root)

    def fail_verification(_: Path) -> dict[str, str]:
        raise ArtifactValidationError("injected final verification failure")

    monkeypatch.setattr(
        bundle_module,
        "verify_sealed_bundle",
        fail_verification,
    )

    with pytest.raises(ArtifactValidationError, match="injected final"):
        finalize_bundle(root)

    assert not (root / SEAL_NAME).exists()
    assert (root / CHECKSUM_MANIFEST_NAME).is_file()


@pytest.mark.parametrize("mutation", ["change", "delete", "add"])
def test_verify_rejects_payload_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "bundle"
    _valid_bundle(root)
    if mutation == "change":
        (root / "alpha.txt").write_bytes(b"changed\n")
        match = "SHA-256 mismatch"
    elif mutation == "delete":
        (root / "alpha.txt").unlink()
        match = "missing listed files"
    else:
        (root / "extra.txt").write_bytes(b"not listed")
        match = "unlisted regular files"

    with pytest.raises(ArtifactValidationError, match=match):
        verify_sealed_bundle(root)


def test_verify_rejects_manifest_that_omits_existing_file(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    entries = _valid_bundle(root)
    reduced = f"{entries['alpha.txt']}  alpha.txt\n".encode()
    _seal_for_manifest(root, reduced)

    with pytest.raises(ArtifactValidationError, match="unlisted regular files"):
        verify_sealed_bundle(root)


def test_verify_rejects_manifest_that_lists_missing_file(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _valid_bundle(root)
    content = f"{'0' * 64}  absent.bin\n".encode()
    _seal_for_manifest(root, content)

    with pytest.raises(ArtifactValidationError, match="missing listed files"):
        verify_sealed_bundle(root)


@pytest.mark.parametrize(
    "manifest_content",
    [
        b"not a checksum line\n",
        (f"{'0' * 64}  ../outside\n").encode(),
        (f"{'0' * 64}  alpha.txt\n{'1' * 64}  alpha.txt\n").encode(),
        (f"{'0' * 64}  nested/.hidden\n{'1' * 64}  alpha.txt\n").encode(),
        (f"{'A' * 64}  alpha.txt\n").encode(),
        (f"{'0' * 64}  alpha.txt\n\n").encode(),
        (f"{'0' * 64} *alpha.txt\n").encode(),
        (f"{'0' * 64}  alpha.txt").encode(),
    ],
    ids=[
        "malformed",
        "traversal",
        "duplicate",
        "unsorted",
        "uppercase-digest",
        "blank-line",
        "noncanonical-marker",
        "missing-final-newline",
    ],
)
def test_verify_rejects_noncanonical_checksum_manifests(
    tmp_path: Path,
    manifest_content: bytes,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    _payload(root)
    _seal_for_manifest(root, manifest_content)

    with pytest.raises(ArtifactValidationError):
        verify_sealed_bundle(root)


@pytest.mark.parametrize("reserved", [CHECKSUM_MANIFEST_NAME, SEAL_NAME])
def test_verify_rejects_manifest_listing_control_files(
    tmp_path: Path,
    reserved: str,
) -> None:
    root = tmp_path / "bundle"
    _valid_bundle(root)
    content = f"{'0' * 64}  {reserved}\n".encode()
    _seal_for_manifest(root, content)

    with pytest.raises(ArtifactValidationError, match="reserved control files"):
        verify_sealed_bundle(root)


def _replace_seal(root: Path, changes: dict[str, Any], *, remove: str = "") -> None:
    seal = load_json(root / SEAL_NAME)
    assert isinstance(seal, dict)
    seal.update(changes)
    if remove:
        del seal[remove]
    write_json(root / SEAL_NAME, seal)


@pytest.mark.parametrize(
    ("changes", "remove"),
    [
        ({"schema": "fer-mujoco-sysid/seal@2"}, ""),
        ({"checksum_manifest": "other.sha256"}, ""),
        ({"checksum_manifest_sha256": "A" * 64}, ""),
        ({"checksum_manifest_sha256": 7}, ""),
        ({"unknown": "forbidden"}, ""),
        ({}, "checksum_manifest"),
    ],
    ids=[
        "unknown-version",
        "wrong-manifest",
        "uppercase-digest",
        "wrong-digest-type",
        "unknown-field",
        "missing-field",
    ],
)
def test_verify_rejects_invalid_seal_schema(
    tmp_path: Path,
    changes: dict[str, Any],
    remove: str,
) -> None:
    root = tmp_path / "bundle"
    _valid_bundle(root)
    _replace_seal(root, changes, remove=remove)

    with pytest.raises(ArtifactValidationError):
        verify_sealed_bundle(root)


def test_verify_rejects_duplicate_seal_fields(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _valid_bundle(root)
    digest = sha256_file(root / CHECKSUM_MANIFEST_NAME)
    (root / SEAL_NAME).write_text(
        "{"
        f'"schema":"{SEAL_SCHEMA}",'
        f'"schema":"{SEAL_SCHEMA}",'
        f'"checksum_manifest":"{CHECKSUM_MANIFEST_NAME}",'
        f'"checksum_manifest_sha256":"{digest}"'
        "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="duplicate JSON object key"):
        verify_sealed_bundle(root)


def test_verify_rejects_checksum_manifest_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _valid_bundle(root)
    _replace_seal(root, {"checksum_manifest_sha256": "0" * 64})

    with pytest.raises(
        ArtifactValidationError,
        match="SHA-256 mismatch for checksums.sha256",
    ):
        verify_sealed_bundle(root)


@pytest.mark.parametrize("mutation", ["file_set", "manifest", "seal"])
def test_verify_rejects_bundle_changes_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = tmp_path / "bundle"
    _valid_bundle(root)
    original_verify_payload = bundle_module._verify_payload

    def mutate_after_payload_check(*args: Any, **kwargs: Any) -> None:
        original_verify_payload(*args, **kwargs)
        if mutation == "file_set":
            (root / "late.txt").write_text("late\n", encoding="utf-8")
        elif mutation == "manifest":
            with (root / CHECKSUM_MANIFEST_NAME).open("ab") as handle:
                handle.write(b"\n")
        else:
            with (root / SEAL_NAME).open("ab") as handle:
                handle.write(b" ")

    monkeypatch.setattr(
        bundle_module,
        "_verify_payload",
        mutate_after_payload_check,
    )

    with pytest.raises(ArtifactValidationError, match="changed during"):
        verify_sealed_bundle(root)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_finalize_rejects_symlinks(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    target = tmp_path / "target"
    if kind == "file":
        target.write_bytes(b"target")
    else:
        target.mkdir()
    (root / "link").symlink_to(target, target_is_directory=kind == "directory")

    with pytest.raises(ArtifactValidationError, match="symlink"):
        finalize_bundle(root)


def test_finalize_rejects_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "payload").write_bytes(b"payload")
    root = tmp_path / "bundle"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(ArtifactValidationError, match="root must not be a symlink"):
        finalize_bundle(root)


def test_finalize_rejects_nonregular_files(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    fifo = root / "events.fifo"
    os.mkfifo(fifo)

    with pytest.raises(ArtifactValidationError, match="not a regular file"):
        finalize_bundle(root)


@pytest.mark.parametrize("operation", [finalize_bundle, verify_sealed_bundle])
def test_public_operations_reject_invalid_roots(
    tmp_path: Path,
    operation: Any,
) -> None:
    regular_file = tmp_path / "not-a-directory"
    regular_file.write_bytes(b"file")

    with pytest.raises(ArtifactValidationError):
        operation(tmp_path / "missing")
    with pytest.raises(ArtifactValidationError):
        operation(regular_file)


def test_finalize_rejects_empty_bundle(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()

    with pytest.raises(ArtifactValidationError, match="no payload files"):
        finalize_bundle(root)
