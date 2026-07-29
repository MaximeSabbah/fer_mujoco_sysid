"""Simulation-only numerical closure and hidden-truth recovery evidence.

Production identification never reads the hidden truth.  This module runs
after a fit has been published and answers the extra question available only
in simulation: did the pipeline recover the certified coordinates and the
behavior of the plant that generated the recordings?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import sysid  # noqa: E402

from fer_mujoco_sysid.dataset import find_recordings, load_recording  # noqa: E402
from fer_mujoco_sysid.diagnostics import (  # noqa: E402
    predicted_torque,
    torque_residuals,
)
from fer_mujoco_sysid.fitting import (  # noqa: E402
    armature_parameters,
    body_full_inertia,
    cad_prior_inertial_parameters,
    combine_parameters,
    measurement_sequences,
    set_hinge_damping,
)
from fer_mujoco_sysid.io import read_json, write_json  # noqa: E402
from fer_mujoco_sysid.model import (  # noqa: E402
    HYDRAX_ARM_JOINT_NAMES,
    build_hydrax_arm_model,
    resolve_model_paths,
)
from fer_mujoco_sysid.preparation import prepare_recording  # noqa: E402
from fer_mujoco_sysid.protocol import INERTIAL_FAMILY  # noqa: E402
from fer_mujoco_sysid.simulation_truth import (  # noqa: E402
    TRUTH_ARMATURE,
    TRUTH_BODY_COORDINATES,
    TRUTH_BODY_PARAMETERIZATION,
    TRUTH_DAMPING,
    TRUTH_FRICTIONLOSS,
)
from fer_mujoco_sysid.simulation_truth import (
    truth_model as build_truth_model,
)
from fer_mujoco_sysid.stages import (  # noqa: E402
    FIT_MEASUREMENT_STRIDE,
    FIT_WINDOW_S,
    INERTIAL_WINDOWS_PER_RECORDING,
    fitting_spec,
)

SIMULATION_VALIDATION_FORMAT = "fer-mujoco-sysid/simulation-validation@1"


@dataclass(frozen=True)
class SimulationThresholds:
    """Strict oracle tolerances, separate from hardware acceptance limits."""

    max_parameter_fraction_of_bound_span: float = 1e-4
    max_q_rmse_rad_at_mppi_horizon: float = 1e-6
    max_dq_rmse_rad_s_at_mppi_horizon: float = 1e-5
    max_gripper_rmse_mm_at_mppi_horizon: float = 1e-3
    max_torque_equation_rmse_Nm: float = 1e-5


_PARAMETER_SPANS = {
    "frictionloss": 3.0,
    "damping": 8.0,
    "armature": 2.0,
    "mass_scale": 0.6,
    "com_offset_m": 0.04,
    "inertia_scale": 0.6,
}


def _objective(residuals: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(value).ravel() for value in residuals])
    return float(values @ values), float(np.max(np.abs(values), initial=0.0))


def oracle_residual_floor(recordings_root: str | Path) -> dict[str, object]:
    """Evaluate the dynamic training residual at the known exact parameters."""
    source = resolve_model_paths().require().hydrax
    nominal = fitting_spec(source).compile()
    prepared = [
        prepare_recording(path, nominal, expected_source_model=source)
        for path in find_recordings(recordings_root)
    ]
    inertial = [
        recording
        for recording in prepared
        if recording.family == INERTIAL_FAMILY and recording.role == "train"
    ]
    if not inertial:
        raise ValueError("simulation campaign has no inertial training recordings")

    spec = fitting_spec(source)
    for index, name in enumerate(HYDRAX_ARM_JOINT_NAMES):
        joint = spec.joint(name)
        joint.frictionloss = TRUTH_FRICTIONLOSS[index]
        set_hinge_damping(joint, TRUTH_DAMPING[index])
    model = spec.compile()
    parameters = combine_parameters(
        armature_parameters(model),
        cad_prior_inertial_parameters(
            spec,
            model,
            TRUTH_BODY_PARAMETERIZATION,
        ),
    )
    for index, name in enumerate(HYDRAX_ARM_JOINT_NAMES):
        parameters[f"{name}_armature"].update_from_vector(
            np.array([TRUTH_ARMATURE[index]], dtype=np.float64)
        )
    for name, value in TRUTH_BODY_COORDINATES.items():
        parameters[name].update_from_vector(np.array([value], dtype=np.float64))

    runs = [run for recording in inertial for run in recording.stage.analysis_runs()]
    sequences = measurement_sequences(
        spec,
        runs,
        window_s=FIT_WINDOW_S,
        max_windows_per_run=INERTIAL_WINDOWS_PER_RECORDING,
        measurement_stride=FIT_MEASUREMENT_STRIDE,
    )
    residual_fn = sysid.build_residual_fn(models_sequences=[sequences])
    residuals, _, _ = residual_fn(parameters.as_vector(), parameters.copy())
    squared, maximum = _objective(residuals)
    return {
        "squared_residual": squared,
        "maximum_absolute_residual": maximum,
        "windows": len(sequences.sequence_name),
        "active_parameters": list(parameters.get_non_frozen_parameter_names()),
    }


def _body_coordinate_values(
    source: mujoco.MjModel,
    identified: mujoco.MjModel,
) -> dict[str, float]:
    values: dict[str, float] = {}
    axes = ("x", "y", "z")
    for correction in TRUTH_BODY_PARAMETERIZATION:
        nominal = source.body(correction.body)
        fitted = identified.body(correction.body)
        if correction.estimate_mass:
            values[f"{correction.body}_mass_scale"] = float(
                fitted.mass[0] / nominal.mass[0]
            )
        for axis in correction.com_axes:
            values[f"{correction.body}_com_{axes[axis]}_offset_m"] = float(
                fitted.ipos[axis] - nominal.ipos[axis]
            )
        if correction.estimate_inertia_scale:
            nominal_tensor = body_full_inertia(source, correction.body)
            fitted_tensor = body_full_inertia(identified, correction.body)
            values[f"{correction.body}_inertia_scale"] = float(
                fitted_tensor @ nominal_tensor / (nominal_tensor @ nominal_tensor)
            )
    return values


def _parameter_rows(
    source: mujoco.MjModel,
    identified: mujoco.MjModel,
    accepted_dynamic: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    joint_truth = (
        (
            "frictionloss",
            TRUTH_FRICTIONLOSS,
            source.dof_frictionloss,
            identified.dof_frictionloss,
        ),
        ("damping", TRUTH_DAMPING, source.dof_damping, identified.dof_damping),
        ("armature", TRUTH_ARMATURE, source.dof_armature, identified.dof_armature),
    )
    for kind, truth, _, estimated_values in joint_truth:
        for index, name in enumerate(HYDRAX_ARM_JOINT_NAMES):
            dof = int(identified.joint(name).dofadr[0])
            parameter_name = f"{name}_{kind}"
            error = abs(float(estimated_values[dof]) - truth[index])
            rows.append(
                {
                    "parameter": parameter_name,
                    "kind": kind,
                    "bound_span": _PARAMETER_SPANS[kind],
                    "truth": truth[index],
                    "estimated": float(estimated_values[dof]),
                    "absolute_error": error,
                    "fraction_of_bound_span": error / _PARAMETER_SPANS[kind],
                    "status": (
                        "identified"
                        if kind != "armature" or parameter_name in accepted_dynamic
                        else "not-certified"
                    ),
                }
            )

    estimated_body = _body_coordinate_values(source, identified)
    for name, truth in TRUTH_BODY_COORDINATES.items():
        if "_mass_scale" in name:
            kind = "mass_scale"
        elif "_com_" in name:
            kind = "com_offset_m"
        else:
            kind = "inertia_scale"
        estimated = estimated_body[name]
        error = abs(estimated - truth)
        rows.append(
            {
                "parameter": name,
                "kind": kind,
                "bound_span": _PARAMETER_SPANS[kind],
                "truth": truth,
                "estimated": estimated,
                "absolute_error": error,
                "fraction_of_bound_span": error / _PARAMETER_SPANS[kind],
                "status": (
                    "identified" if name in accepted_dynamic else "not-certified"
                ),
            }
        )
    return rows


def _truth_lineage(recordings_root: str | Path) -> dict[str, object]:
    hashes: set[str] = set()
    for path in find_recordings(recordings_root):
        manifest, _ = load_recording(path)
        simulation = manifest.get("simulation")
        if not isinstance(simulation, dict):
            raise ValueError(f"{path}: recording has no simulation metadata")
        truth = simulation.get("truth")
        if not isinstance(truth, dict):
            raise ValueError(f"{path}: recording is not bound to simulation truth")
        hashes.add(str(truth.get("content_sha256", "")))
    if len(hashes) != 1 or "" in hashes:
        raise ValueError(f"campaign uses inconsistent simulation truths: {hashes}")
    return {"content_sha256": next(iter(hashes))}


def _truth_consistent_acceleration(
    truth: mujoco.MjModel,
    q_rad: np.ndarray,
    dq_rad_s: np.ndarray,
    torque_Nm: np.ndarray,
) -> np.ndarray:
    """Evaluate continuous acceleration of the declared hidden plant.

    Acceleration is not measured directly on the robot. A centered numerical
    derivative is therefore retained in the ordinary hardware-style torque
    plot, but it has an integration/discretization floor even for perfect
    simulation data. This simulation-only oracle uses the truth plant to ask
    the stricter equation-level question without feeding truth into fitting.
    """
    data = mujoco.MjData(truth)
    acceleration = np.empty_like(dq_rad_s, dtype=np.float64)
    for index in range(len(q_rad)):
        data.qpos[:] = q_rad[index]
        data.qvel[:] = dq_rad_s[index]
        data.ctrl[:] = torque_Nm[index]
        mujoco.mj_forward(truth, data)
        acceleration[index] = data.qacc
    return acceleration


def _torque_equation_closure(
    recordings_root: str | Path,
    *,
    source_path: Path,
    identified: mujoco.MjModel,
) -> dict[str, dict[str, object]]:
    """Compare torque equations using truth-consistent acceleration."""
    nominal = fitting_spec(source_path).compile()
    truth = build_truth_model(
        source_path,
        gravity=False,
        joint_state_sensors=True,
    )
    prepared = [
        prepare_recording(path, nominal, expected_source_model=source_path)
        for path in find_recordings(recordings_root)
    ]
    closure: dict[str, dict[str, object]] = {}
    for recording in prepared:
        if recording.role != "holdout":
            continue
        run = recording.protocol_run()
        q = run.measured[:, :7]
        dq = run.measured[:, 7:14]
        torque = recording.protocol_classical_torque()
        ddq = _truth_consistent_acceleration(truth, q, dq, torque)
        candidates: dict[str, object] = {}
        for name, model in (("nominal", nominal), ("identified", identified)):
            report = torque_residuals(
                torque,
                predicted_torque(model, q, dq, ddq),
            )
            candidates[name] = {
                "rmse_Nm": report.rmse_Nm.tolist(),
                "worst_rmse_Nm": report.worst_rmse,
                "maximum_absolute_error_Nm": float(np.max(report.max_abs_Nm)),
            }
        closure[recording.protocol_id] = candidates
    return closure


def validate_simulation_result(
    recordings_root: str | Path,
    identification_dir: str | Path,
    *,
    thresholds: SimulationThresholds | None = None,
) -> dict[str, object]:
    """Write and return strict truth-recovery evidence for one backend."""
    thresholds = thresholds or SimulationThresholds()
    identification_dir = Path(identification_dir)
    result = read_json(identification_dir / "identification.json")
    if result.get("status") != "accepted":
        raise ValueError("simulation validation requires an accepted identification")
    model_path = Path(str(result["exported_model"]))
    identified = build_hydrax_arm_model(model_path)
    source_path = resolve_model_paths().require().hydrax
    source = build_hydrax_arm_model(source_path)
    identified_fit_projection = fitting_spec(model_path).compile()
    dynamic = result["stages"].get("dynamic")
    accepted_dynamic = set(
        dynamic.get("accepted_parameters", []) if isinstance(dynamic, dict) else []
    )
    rows = _parameter_rows(source, identified, accepted_dynamic)

    problems = [
        f"{row['parameter']} was not certified as observable"
        for row in rows
        if row["status"] != "identified"
    ]
    problems.extend(
        f"{row['parameter']} error is {row['fraction_of_bound_span']:.3g} "
        "of its bound span"
        for row in rows
        if float(row["fraction_of_bound_span"])
        > thresholds.max_parameter_fraction_of_bound_span
    )

    mppi: dict[str, dict[str, float]] = {}
    held_out = result["held_out"]
    for protocol_id, evaluation in held_out.items():
        metrics = evaluation["horizons"]["0.32s"]["identified"]
        values = {
            "worst_q_rmse_rad": float(metrics["worst_q_rmse_rad"]),
            "worst_dq_rmse_rad_s": float(metrics["worst_dq_rmse_rad_s"]),
            "gripper_rmse_mm": float(metrics["gripper_rmse_mm"]),
        }
        mppi[protocol_id] = values
        for name, value, limit in (
            (
                "q",
                values["worst_q_rmse_rad"],
                thresholds.max_q_rmse_rad_at_mppi_horizon,
            ),
            (
                "dq",
                values["worst_dq_rmse_rad_s"],
                thresholds.max_dq_rmse_rad_s_at_mppi_horizon,
            ),
            (
                "gripper",
                values["gripper_rmse_mm"],
                thresholds.max_gripper_rmse_mm_at_mppi_horizon,
            ),
        ):
            if value > limit:
                problems.append(
                    f"{protocol_id} 0.32 s {name} error {value:.3g} > {limit:.3g}"
                )

    torque_equation = _torque_equation_closure(
        recordings_root,
        source_path=source_path,
        identified=identified_fit_projection,
    )
    for protocol_id, models in torque_equation.items():
        identified_torque = models["identified"]
        value = float(identified_torque["worst_rmse_Nm"])
        if value > thresholds.max_torque_equation_rmse_Nm:
            problems.append(
                f"{protocol_id} truth-consistent torque RMSE "
                f"{value:.3g} Nm > {thresholds.max_torque_equation_rmse_Nm:.3g} Nm"
            )

    validation: dict[str, object] = {
        "format": SIMULATION_VALIDATION_FORMAT,
        "accepted": not problems,
        "problems": problems,
        "backend": result["backend"],
        "simulation_truth": _truth_lineage(recordings_root),
        "oracle_residual_floor": oracle_residual_floor(recordings_root),
        "thresholds": thresholds.__dict__,
        "parameters": rows,
        "mppi_horizon": mppi,
        "torque_equation_closure": torque_equation,
    }
    write_json(identification_dir / "simulation_validation.json", validation)
    _write_report(identification_dir / "simulation_validation.md", validation)
    _plot_recovery(identification_dir / "parameter_recovery.png", rows)
    _plot_torque_closure(
        identification_dir / "torque_equation_closure.png",
        torque_equation,
    )
    return validation


def _write_report(path: Path, validation: dict[str, object]) -> None:
    lines = [
        "# Simulation numerical-closure validation",
        "",
        f"Verdict: **{'ACCEPTED' if validation['accepted'] else 'REJECTED'}**.",
        "",
        "| parameter | status | truth | estimated | absolute error | span fraction |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in validation["parameters"]:
        lines.append(
            f"| {row['parameter']} | {row['status']} | {row['truth']:.9g} | "
            f"{row['estimated']:.9g} | {row['absolute_error']:.3g} | "
            f"{row['fraction_of_bound_span']:.3g} |"
        )
    lines += [
        "",
        "## Truth-consistent torque equation closure",
        "",
        "The ordinary torque plot differentiates measured velocity, as the real "
        "pipeline must, and therefore has a numerical differentiation floor. "
        "This simulation-only check evaluates acceleration with the declared "
        "hidden plant and tests the identified inverse-dynamics equation "
        "without using truth during fitting.",
        "",
        "| protocol | model | worst RMSE [Nm] | maximum error [Nm] |",
        "| --- | --- | ---: | ---: |",
    ]
    for protocol_id, models in validation["torque_equation_closure"].items():
        for model_name, metrics in models.items():
            lines.append(
                f"| {protocol_id} | {model_name} | "
                f"{metrics['worst_rmse_Nm']:.6g} | "
                f"{metrics['maximum_absolute_error_Nm']:.6g} |"
            )
    lines += ["", "## Problems", ""]
    problems = validation["problems"]
    lines.extend(
        [f"- {problem}" for problem in problems]
        if problems
        else ["No simulation-closure problems."]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_recovery(path: Path, rows: list[dict[str, object]]) -> None:
    names = [str(row["parameter"]) for row in rows]
    errors = [float(row["fraction_of_bound_span"]) for row in rows]
    figure, axis = plt.subplots(figsize=(11, max(5.0, 0.25 * len(rows))))
    positions = np.arange(len(rows))
    axis.barh(positions, errors, color="#2563eb")
    axis.set_yticks(positions, labels=names, fontsize=7)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("absolute error / parameter bound span")
    axis.set_title("Simulation truth recovery")
    axis.grid(True, axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_torque_closure(
    path: Path,
    closure: dict[str, dict[str, object]],
) -> None:
    protocols = list(closure)
    positions = np.arange(len(protocols))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for offset, name, color in (
        (-width / 2, "nominal", "#9ca3af"),
        (width / 2, "identified", "#2563eb"),
    ):
        axis.bar(
            positions + offset,
            [float(closure[protocol][name]["worst_rmse_Nm"]) for protocol in protocols],
            width,
            label=name,
            color=color,
        )
    axis.set_xticks(positions, protocols, rotation=15, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("worst joint torque-equation RMSE [Nm]")
    axis.set_title("Simulation-only truth-consistent torque closure")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings")
    parser.add_argument("identification")
    arguments = parser.parse_args(argv)
    result = validate_simulation_result(
        arguments.recordings,
        arguments.identification,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
