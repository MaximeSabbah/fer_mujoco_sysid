"""Run the complete simulation-first identification release gate.

One invocation creates a fresh, immutable run root, binds both simulation
backends to the same hidden-truth manifest, records the six canonical
protocols, identifies a model independently from each backend, and writes the
plots, videos, numerical validation, and consumer-model handoff metadata.

The ROS-backed result is the primary handoff because it exercises the path
that will later be used on the real robot.  It is advertised as release-ready
only when the independent in-process MuJoCo result passes as well.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fer_mujoco_sysid.campaign import CAMPAIGN, repository_root  # noqa: E402
from fer_mujoco_sysid.dataset import find_recordings, load_recording  # noqa: E402
from fer_mujoco_sysid.identify import run as identify_campaign  # noqa: E402
from fer_mujoco_sysid.io import read_json, sha256_file, write_json  # noqa: E402
from fer_mujoco_sysid.simulate import run as simulate_campaign  # noqa: E402
from fer_mujoco_sysid.simulation_truth import (  # noqa: E402
    load_truth_manifest,
    write_truth_manifest,
)
from fer_mujoco_sysid.simulation_validation import (  # noqa: E402
    SimulationThresholds,
    validate_simulation_result,
)

PIPELINE_FORMAT = "fer-mujoco-sysid/simulation-pipeline@1"
PIPELINE_STATUS_FORMAT = "fer-mujoco-sysid/simulation-pipeline-status@1"
BACKEND_RESULT_FORMAT = "fer-mujoco-sysid/simulation-backend-result@1"
RELEASE_MODEL_FORMAT = "fer-mujoco-sysid/release-model@1"
SUPPORTED_BACKENDS = ("mujoco", "mujoco_ros")
DEFAULT_MAX_ITERS = 10
DEFAULT_DYNAMIC_STARTS = 1
CROSS_BACKEND_THRESHOLD_MULTIPLIER = 2.0

_METRIC_THRESHOLDS = {
    "worst_q_rmse_rad": "max_q_rmse_rad_at_mppi_horizon",
    "worst_dq_rmse_rad_s": "max_dq_rmse_rad_s_at_mppi_horizon",
    "gripper_rmse_mm": "max_gripper_rmse_mm_at_mppi_horizon",
}


def _default_output_root() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return repository_root() / "output" / "simulation" / f"{timestamp}-{suffix}"


def _write_status(
    root: Path,
    *,
    run_id: str,
    state: str,
    requested_backends: tuple[str, ...],
    release_ready: bool = False,
    error: BaseException | None = None,
) -> None:
    status: dict[str, object] = {
        "format": PIPELINE_STATUS_FORMAT,
        "run_id": run_id,
        "state": state,
        "requested_backends": list(requested_backends),
        "release_ready": release_ready,
        "result_manifest": (
            "simulation_pipeline.json"
            if state in {"accepted", "diagnostic_accepted", "rejected"}
            else None
        ),
        "primary_model": (
            "mujoco_ros/identification/fer_identified.xml" if release_ready else None
        ),
    }
    if error is not None:
        status["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    write_json(root / "simulation_pipeline_status.json", status)


def _expected_protocols() -> dict[str, dict[str, str]]:
    protocols: dict[str, dict[str, str]] = {}
    root = repository_root() / "protocols"
    for spec in CAMPAIGN:
        manifest = read_json(root / spec.protocol_id / "protocol.json")
        protocols[spec.protocol_id] = {
            "family": str(manifest["family"]),
            "role": str(manifest["role"]),
            "protocol_sha256": str(manifest["content_sha256"]),
            "source_model_sha256": str(manifest["source_model"]["sha256"]),
        }
    return protocols


def _recording_inventory(
    backend_root: Path,
    *,
    backend: str,
    truth_sha256: str,
) -> list[dict[str, object]]:
    expected = _expected_protocols()
    recordings = find_recordings(backend_root)
    if len(recordings) != len(expected):
        raise RuntimeError(
            f"{backend}: expected {len(expected)} recordings, found {len(recordings)}"
        )

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in recordings:
        manifest, arrays = load_recording(path)
        protocol = manifest.get("protocol")
        if not isinstance(protocol, dict):
            raise RuntimeError(f"{path}: missing protocol lineage")
        protocol_id = str(protocol.get("protocol_id", ""))
        if protocol_id not in expected:
            raise RuntimeError(f"{path}: unexpected protocol {protocol_id!r}")
        if protocol_id in seen:
            raise RuntimeError(f"{backend}: duplicate recording for {protocol_id}")
        seen.add(protocol_id)

        declared = expected[protocol_id]
        for field, actual in (
            ("family", protocol.get("family")),
            ("role", protocol.get("role")),
            ("protocol_sha256", protocol.get("content_sha256")),
        ):
            if actual != declared[field]:
                raise RuntimeError(
                    f"{path}: {field} {actual!r} does not match the committed "
                    f"value {declared[field]!r}"
                )
        if manifest.get("backend") != backend:
            raise RuntimeError(
                f"{path}: backend {manifest.get('backend')!r} != {backend!r}"
            )
        source = manifest.get("source_model")
        if not isinstance(source, dict):
            raise RuntimeError(f"{path}: missing source-model lineage")
        if source.get("sha256") != declared["source_model_sha256"]:
            raise RuntimeError(f"{path}: source-model SHA does not match protocol")
        simulation = manifest.get("simulation")
        truth = simulation.get("truth") if isinstance(simulation, dict) else None
        if not isinstance(truth, dict):
            raise RuntimeError(f"{path}: missing simulation-truth lineage")
        if truth.get("content_sha256") != truth_sha256:
            raise RuntimeError(f"{path}: simulation-truth SHA does not match run")

        rows.append(
            {
                "protocol_id": protocol_id,
                "family": declared["family"],
                "role": declared["role"],
                "path": path.relative_to(backend_root).as_posix(),
                "recording_sha256": manifest["content_sha256"],
                "protocol_sha256": declared["protocol_sha256"],
                "source_model_sha256": declared["source_model_sha256"],
                "samples": int(len(arrays["time_s"])),
                "usable": bool(manifest["health"]["usable"]),
            }
        )

    missing = set(expected) - seen
    if missing:
        raise RuntimeError(f"{backend}: missing protocols {sorted(missing)}")
    return sorted(rows, key=lambda row: str(row["protocol_id"]))


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _acquire_mujoco(backend_root: Path, truth_path: Path) -> None:
    print("\n=== Acquisition: in-process MuJoCo ===", flush=True)
    simulate_campaign(
        backend_root,
        simulation_truth_path=truth_path,
    )


def _acquire_mujoco_ros(backend_root: Path, truth_path: Path) -> None:
    print("\n=== Acquisition: ROS-controlled MuJoCo ===", flush=True)
    runner = repository_root() / "scripts" / "run-protocol"
    for index, spec in enumerate(CAMPAIGN, start=1):
        print(
            f"\n--- mujoco_ros protocol {index}/{len(CAMPAIGN)}: "
            f"{spec.protocol_id} ---",
            flush=True,
        )
        subprocess.run(
            [
                str(runner),
                spec.protocol_id,
                "--backend",
                "mujoco_ros",
                "--record",
                str(backend_root),
                "--simulation-truth",
                str(truth_path),
            ],
            cwd=repository_root(),
            check=True,
        )


def _parameter_map(
    validation: dict[str, object],
) -> dict[str, dict[str, object]]:
    parameters = validation.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError("simulation validation has no parameter rows")
    return {str(row["parameter"]): row for row in parameters if isinstance(row, dict)}


def compare_backends(
    mujoco_result: dict[str, object],
    mujoco_ros_result: dict[str, object],
    *,
    thresholds: SimulationThresholds | None = None,
) -> dict[str, object]:
    """Compare two independently fitted simulation backends."""
    thresholds = thresholds or SimulationThresholds()
    direct = mujoco_result["validation"]
    ros = mujoco_ros_result["validation"]
    if not isinstance(direct, dict) or not isinstance(ros, dict):
        raise ValueError("backend result is missing simulation validation")
    problems: list[str] = []

    direct_truth = direct.get("simulation_truth")
    ros_truth = ros.get("simulation_truth")
    truth_match = direct_truth == ros_truth
    if not truth_match:
        problems.append("backends do not share the same simulation truth")

    direct_protocols = {
        str(row["protocol_id"]): (
            str(row["protocol_sha256"]),
            str(row["source_model_sha256"]),
        )
        for row in mujoco_result["recordings"]
    }
    ros_protocols = {
        str(row["protocol_id"]): (
            str(row["protocol_sha256"]),
            str(row["source_model_sha256"]),
        )
        for row in mujoco_ros_result["recordings"]
    }
    protocol_lineage_match = direct_protocols == ros_protocols
    if not protocol_lineage_match:
        problems.append("backend protocol/source lineages differ")

    direct_parameters = _parameter_map(direct)
    ros_parameters = _parameter_map(ros)
    parameter_names_match = set(direct_parameters) == set(ros_parameters)
    parameter_rows: list[dict[str, object]] = []
    max_parameter_delta = 0.0
    if not parameter_names_match:
        problems.append("backend validation parameter sets differ")
    for name in sorted(set(direct_parameters) & set(ros_parameters)):
        first = direct_parameters[name]
        second = ros_parameters[name]
        span = float(first["bound_span"])
        delta = abs(float(first["estimated"]) - float(second["estimated"])) / span
        max_parameter_delta = max(max_parameter_delta, delta)
        parameter_rows.append(
            {
                "parameter": name,
                "kind": first["kind"],
                "truth": float(first["truth"]),
                "mujoco": float(first["estimated"]),
                "mujoco_ros": float(second["estimated"]),
                "bound_span": span,
                "fraction_of_bound_span": delta,
            }
        )
    parameter_limit = (
        CROSS_BACKEND_THRESHOLD_MULTIPLIER
        * thresholds.max_parameter_fraction_of_bound_span
    )
    if max_parameter_delta > parameter_limit:
        problems.append(
            "maximum parameter disagreement "
            f"{max_parameter_delta:.3g} > {parameter_limit:.3g} of bound span"
        )

    metric_differences: dict[str, dict[str, float]] = {}
    direct_mppi = direct.get("mppi_horizon")
    ros_mppi = ros.get("mppi_horizon")
    if not isinstance(direct_mppi, dict) or not isinstance(ros_mppi, dict):
        raise ValueError("backend validation has no MPPI-horizon metrics")
    if set(direct_mppi) != set(ros_mppi):
        problems.append("backend MPPI holdout sets differ")
    for protocol_id in sorted(set(direct_mppi) & set(ros_mppi)):
        first = direct_mppi[protocol_id]
        second = ros_mppi[protocol_id]
        if not isinstance(first, dict) or not isinstance(second, dict):
            raise ValueError(f"{protocol_id}: malformed MPPI metrics")
        differences: dict[str, float] = {}
        for metric, threshold_name in _METRIC_THRESHOLDS.items():
            delta = abs(float(first[metric]) - float(second[metric]))
            differences[metric] = delta
            limit = CROSS_BACKEND_THRESHOLD_MULTIPLIER * float(
                getattr(thresholds, threshold_name)
            )
            if delta > limit:
                problems.append(
                    f"{protocol_id} {metric} disagreement {delta:.3g} > {limit:.3g}"
                )
        metric_differences[protocol_id] = differences

    direct_torque = direct.get("torque_equation_closure")
    ros_torque = ros.get("torque_equation_closure")
    if not isinstance(direct_torque, dict) or not isinstance(ros_torque, dict):
        raise ValueError("backend validation has no torque-equation closure")
    torque_differences: dict[str, float] = {}
    if set(direct_torque) != set(ros_torque):
        problems.append("backend torque holdout sets differ")
    torque_limit = (
        CROSS_BACKEND_THRESHOLD_MULTIPLIER * thresholds.max_torque_equation_rmse_Nm
    )
    for protocol_id in sorted(set(direct_torque) & set(ros_torque)):
        delta = abs(
            float(direct_torque[protocol_id]["identified"]["worst_rmse_Nm"])
            - float(ros_torque[protocol_id]["identified"]["worst_rmse_Nm"])
        )
        torque_differences[protocol_id] = delta
        if delta > torque_limit:
            problems.append(
                f"{protocol_id} torque-equation disagreement "
                f"{delta:.3g} Nm > {torque_limit:.3g} Nm"
            )

    return {
        "accepted": not problems,
        "problems": problems,
        "truth_match": truth_match,
        "protocol_lineage_match": protocol_lineage_match,
        "parameter_names_match": parameter_names_match,
        "parameter_difference_limit_fraction_of_bound_span": parameter_limit,
        "maximum_parameter_difference_fraction_of_bound_span": (max_parameter_delta),
        "parameters": parameter_rows,
        "mppi_horizon_metric_differences": metric_differences,
        "torque_equation_rmse_differences_Nm": torque_differences,
    }


def _plot_cross_backend(
    destination: Path,
    comparison: dict[str, object],
) -> None:
    rows = comparison["parameters"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("cross-backend comparison has no parameter rows")
    names = [str(row["parameter"]) for row in rows]
    truth = np.asarray([float(row["truth"]) for row in rows])
    span = np.asarray([float(row["bound_span"]) for row in rows])
    mujoco_error = np.maximum(
        np.abs(np.asarray([float(row["mujoco"]) for row in rows]) - truth) / span,
        1e-16,
    )
    ros_error = np.maximum(
        np.abs(np.asarray([float(row["mujoco_ros"]) for row in rows]) - truth) / span,
        1e-16,
    )
    positions = np.arange(len(rows))
    height = 0.38
    figure, axis = plt.subplots(
        figsize=(11, max(5.0, 0.27 * len(rows))),
        constrained_layout=True,
    )
    axis.barh(
        positions - height / 2,
        mujoco_error,
        height,
        label="mujoco",
        color="#2563eb",
    )
    axis.barh(
        positions + height / 2,
        ros_error,
        height,
        label="mujoco_ros",
        color="#b45309",
    )
    axis.set_yticks(positions, names, fontsize=7)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("absolute truth error / parameter bound span")
    axis.set_title("Independent parameter recovery by simulation backend")
    axis.grid(True, axis="x", alpha=0.25)
    axis.legend()
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def _write_backend_result(
    backend_root: Path,
    *,
    backend: str,
    recordings: list[dict[str, object]],
    identification: dict[str, object],
    validation: dict[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {
        "format": BACKEND_RESULT_FORMAT,
        "backend": backend,
        "accepted": bool(validation["accepted"]),
        "recordings": recordings,
        "identification": {
            "status": identification["status"],
            "manifest": "identification/identification.json",
            "report": "identification/result.md",
            "consumer_model": "identification/fer_identified.xml",
            "consumer_model_manifest": "identification/fer_identified.json",
        },
        "simulation_validation": {
            "accepted": validation["accepted"],
            "problems": validation["problems"],
            "manifest": "identification/simulation_validation.json",
            "report": "identification/simulation_validation.md",
            "parameter_plot": "identification/parameter_recovery.png",
            "torque_equation_plot": ("identification/torque_equation_closure.png"),
        },
    }
    write_json(backend_root / "backend_result.json", result)
    result["artifacts"] = _artifact_inventory(backend_root)
    result["validation"] = validation
    return result


def _write_release_pointer(root: Path, backend_result: dict[str, object]) -> None:
    model = root / "mujoco_ros" / "identification" / "fer_identified.xml"
    manifest = root / "mujoco_ros" / "identification" / "fer_identified.json"
    write_json(
        root / "release_model.json",
        {
            "format": RELEASE_MODEL_FORMAT,
            "backend": "mujoco_ros",
            "model": {
                "path": model.relative_to(root).as_posix(),
                "sha256": sha256_file(model),
            },
            "model_manifest": {
                "path": manifest.relative_to(root).as_posix(),
                "sha256": sha256_file(manifest),
            },
            "simulation_validation": {
                "path": ("mujoco_ros/identification/simulation_validation.json"),
                "accepted": backend_result["validation"]["accepted"],
            },
            "gravity": {
                "model": "enabled",
                "mujoco_command": "send full modeled torque",
                "real_franka_command": (
                    "subtract model gravity exactly once in the final adapter"
                ),
            },
        },
    )


def _write_report(path: Path, result: dict[str, object]) -> None:
    lines = [
        "# Complete simulation identification pipeline",
        "",
        f"Verdict: **{str(result['state']).upper()}**.",
        "",
        f"- Run ID: `{result['run_id']}`",
        f"- Shared truth SHA-256: `{result['simulation_truth']['content_sha256']}`",
        f"- Release ready: **{result['release_ready']}**",
        "",
        "## Backend gates",
        "",
        "| backend | recordings | identification | truth validation | model |",
        "| --- | ---: | --- | --- | --- |",
    ]
    backends = result["backends"]
    for backend, backend_result in backends.items():
        lines.append(
            f"| `{backend}` | {len(backend_result['recordings'])} | "
            f"{backend_result['identification']['status']} | "
            f"{'accepted' if backend_result['validation']['accepted'] else 'rejected'} "
            f"| `{backend}/identification/fer_identified.xml` |"
        )
    comparison = result.get("cross_backend")
    if isinstance(comparison, dict):
        lines += [
            "",
            "## Cross-backend agreement",
            "",
            f"Verdict: **{'ACCEPTED' if comparison['accepted'] else 'REJECTED'}**.",
            "",
            "Maximum parameter disagreement: "
            "`"
            f"{comparison['maximum_parameter_difference_fraction_of_bound_span']:.3g}"
            "` "
            "of its allowed fit span.",
            "",
            "See `cross_backend_parameter_recovery.png` for the independent "
            "truth-recovery comparison.",
        ]
        problems = comparison["problems"]
        if problems:
            lines += ["", "Problems:", ""]
            lines.extend(f"- {problem}" for problem in problems)
    lines += [
        "",
        "## Deliverables",
        "",
        "Each backend directory contains all six immutable recordings, replay "
        "videos and tracking plots for every protocol, family-tagged held-out "
        "rollout/torque/end-effector plots, numerical truth-recovery plots, "
        "reports, and a gravity-enabled consumer MuJoCo model.",
        "",
    ]
    if result["release_ready"]:
        lines += [
            "The authoritative handoff is `release_model.json`, pointing to "
            "`mujoco_ros/identification/fer_identified.xml`.",
            "",
        ]
    else:
        lines += [
            "No primary model is advertised because the complete two-backend "
            "release gate did not pass.",
            "",
        ]
    lines += [
        "For MuJoCo rollouts, send the full modeled torque. On the real Franka, "
        "retain gravity in this model and subtract the model gravity term "
        "exactly once in the final torque adapter because Franka compensates "
        "gravity internally.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    output_root: str | Path | None = None,
    *,
    backends: tuple[str, ...] = SUPPORTED_BACKENDS,
    max_iters: int = DEFAULT_MAX_ITERS,
    dynamic_starts: int = DEFAULT_DYNAMIC_STARTS,
    render_media: bool = True,
) -> dict[str, object]:
    """Execute a fresh simulation campaign and return its top-level manifest."""
    requested = tuple(dict.fromkeys(backends))
    if not requested or any(name not in SUPPORTED_BACKENDS for name in requested):
        raise ValueError(
            f"backends must be a non-empty subset of {SUPPORTED_BACKENDS}, "
            f"got {requested}"
        )
    root = Path(output_root) if output_root is not None else _default_output_root()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    _write_status(
        root,
        run_id=run_id,
        state="running",
        requested_backends=requested,
    )
    print(f"fresh simulation run: {root}", flush=True)

    try:
        truth_path = write_truth_manifest(root / "simulation_truth.json").resolve()
        truth = load_truth_manifest(truth_path)
        backend_results: dict[str, dict[str, object]] = {}
        for backend in requested:
            backend_root = root / backend
            if backend == "mujoco":
                _acquire_mujoco(backend_root, truth_path)
            else:
                _acquire_mujoco_ros(backend_root, truth_path)

            recordings = _recording_inventory(
                backend_root,
                backend=backend,
                truth_sha256=str(truth["content_sha256"]),
            )
            print(
                f"\n=== Identification and media: {backend} ===",
                flush=True,
            )
            identification_dir = backend_root / "identification"
            identification = identify_campaign(
                backend_root,
                identification_dir,
                max_iters=max_iters,
                dynamic_starts=dynamic_starts,
                render_media=render_media,
            )
            print(
                f"\n=== Strict simulation truth validation: {backend} ===",
                flush=True,
            )
            validation = validate_simulation_result(
                backend_root,
                identification_dir,
            )
            backend_results[backend] = _write_backend_result(
                backend_root,
                backend=backend,
                recordings=recordings,
                identification=identification,
                validation=validation,
            )
            print(
                f"{backend}: {'ACCEPTED' if validation['accepted'] else 'REJECTED'}",
                flush=True,
            )

        all_selected_accepted = all(
            bool(result["validation"]["accepted"])
            for result in backend_results.values()
        )
        complete = set(requested) == set(SUPPORTED_BACKENDS)
        comparison: dict[str, object] | None = None
        if complete:
            comparison = compare_backends(
                backend_results["mujoco"],
                backend_results["mujoco_ros"],
            )
            _plot_cross_backend(
                root / "cross_backend_parameter_recovery.png",
                comparison,
            )
        release_ready = bool(
            complete
            and all_selected_accepted
            and comparison is not None
            and comparison["accepted"]
        )
        if release_ready:
            _write_release_pointer(root, backend_results["mujoco_ros"])
        state = (
            "accepted"
            if release_ready
            else "diagnostic_accepted"
            if not complete and all_selected_accepted
            else "rejected"
        )
        result = {
            "format": PIPELINE_FORMAT,
            "run_id": run_id,
            "state": state,
            "release_ready": release_ready,
            "requested_backends": list(requested),
            "simulation_truth": {
                "path": "simulation_truth.json",
                "content_sha256": truth["content_sha256"],
                "source_model_sha256": truth["source_model"]["sha256"],
            },
            "optimizer": {
                "max_iters": max_iters,
                "dynamic_starts": dynamic_starts,
            },
            "media_rendered": render_media,
            "backends": backend_results,
            "cross_backend": comparison,
            "primary_model": (
                "mujoco_ros/identification/fer_identified.xml"
                if release_ready
                else None
            ),
            "release_model_manifest": ("release_model.json" if release_ready else None),
        }
        # The full validation dictionaries are intentionally retained here:
        # this top-level manifest is the machine-readable, one-file audit.
        write_json(root / "simulation_pipeline.json", result)
        _write_report(root / "simulation_pipeline.md", result)
        _write_status(
            root,
            run_id=run_id,
            state=state,
            requested_backends=requested,
            release_ready=release_ready,
        )
        print(
            f"\nPipeline verdict: {state.upper()}\n"
            f"Artifacts: {root}\n"
            f"Report: {root / 'simulation_pipeline.md'}",
            flush=True,
        )
        return result
    except BaseException as error:
        _write_status(
            root,
            run_id=run_id,
            state="failed",
            requested_backends=requested,
            error=error,
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="fresh destination; defaults to output/simulation/<UTC-run-id>",
    )
    parser.add_argument(
        "--backend",
        choices=("all", *SUPPORTED_BACKENDS),
        default="all",
        help="all is the release gate; one backend is diagnostic only",
    )
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument(
        "--dynamic-starts",
        type=int,
        default=DEFAULT_DYNAMIC_STARTS,
    )
    parser.add_argument(
        "--no-media",
        action="store_true",
        help="skip videos/ordinary plots for a faster diagnostic run",
    )
    arguments = parser.parse_args(argv)
    selected = (
        SUPPORTED_BACKENDS if arguments.backend == "all" else (arguments.backend,)
    )
    try:
        result = run(
            arguments.output,
            backends=selected,
            max_iters=arguments.max_iters,
            dynamic_starts=arguments.dynamic_starts,
            render_media=not arguments.no_media,
        )
    except FileExistsError as error:
        parser.error(str(error))
    except Exception as error:
        print(
            json.dumps(
                {
                    "state": "failed",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
                indent=2,
            )
        )
        return 1
    return 0 if result["state"] in {"accepted", "diagnostic_accepted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
