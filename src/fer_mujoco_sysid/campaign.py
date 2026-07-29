"""The reviewable friction-identification campaign (P3 slice 2).

Defines the canonical friction and inertial protocols plus held-out variants
as fixed, seeded specs; generates their content-addressed bundles under
``protocols/`` and the human-review material under ``protocols/review/``
(per-joint motion plots and a conditioning summary). ``--check`` regenerates
every spec into a temporary directory and compares content hashes against the
committed bundles, so generator drift cannot go unnoticed.

Family and train/holdout role are explicit protocol metadata. Identification
must consume those fields rather than infer semantics from protocol names.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from fer_mujoco_sysid.excitation import (  # noqa: E402
    CompiledProtocol,
    FrictionProtocolSpec,
    InertialProtocolSpec,
    generate_friction_protocol,
    generate_inertial_protocol,
    load_protocol_bundle,
    write_protocol_bundle,
)
from fer_mujoco_sysid.fitting import (  # noqa: E402
    CONDITIONING_RATIO_MINIMUM,
    CORRELATION_FREEZE_LIMIT,
    MeasuredRun,
    conditioning_report,
    friction_parameters,
    measurement_sequences,
)
from fer_mujoco_sysid.model import (  # noqa: E402
    build_hydrax_arm_spec,
    resolve_model_paths,
)
from fer_mujoco_sysid.protocol import HOLDOUT_ROLE, TRAIN_ROLE  # noqa: E402

# Committed campaign timestamp: fixed so regeneration is byte-reproducible.
CAMPAIGN_CREATED_AT = "2026-07-27T00:00:00Z"
# Family/role and fit-eligible subwindows are immutable manifest metadata.
# Friction windows are only the exact constant-velocity cruise plateaus;
# inertial windows are only the Fourier excitation. The complete trajectories
# and all settle/hold/return samples remain recorded. Inertial frequencies are
# commensurate with the protocol duration, so every trajectory returns to rest
# at home without a terminal step; see InertialProtocolSpec.

# The campaign: two canonical protocols (different seeds, so different
# per-joint amplitude jitter) and one held-out variant with a different
# configuration. Committed protocols use a 100 Hz knot grid: dense enough
# for the trajectory controller to interpolate the S-curves, compact enough
# to live in git.
ProtocolSpec = FrictionProtocolSpec | InertialProtocolSpec

# The friction family: slow constant-velocity cruises that separate the
# Coulomb offset from the viscous slope. All joints share one schedule,
# which is fine here (each joint's friction acts only on its own torque).
FRICTION_CAMPAIGN: tuple[FrictionProtocolSpec, ...] = (
    FrictionProtocolSpec(
        protocol_id="fer-friction-a",
        seed=101,
        role=TRAIN_ROLE,
        sample_period_s=0.01,
    ),
    FrictionProtocolSpec(
        protocol_id="fer-friction-b",
        seed=102,
        role=TRAIN_ROLE,
        sample_period_s=0.01,
    ),
    FrictionProtocolSpec(
        protocol_id="fer-friction-holdout",
        seed=901,
        role=HOLDOUT_ROLE,
        sample_period_s=0.01,
        amplitudes_rad=(0.27,) * 7,
        cruise_speeds_rad_s=(0.07, 0.2, 0.35),
    ),
)

# The inertial family: per-joint Fourier series at different base
# frequencies, so the joints move independently and accelerate hard. Both
# properties are required to identify link inertias and armature, and
# neither holds in the friction family.
INERTIAL_CAMPAIGN: tuple[InertialProtocolSpec, ...] = (
    InertialProtocolSpec(protocol_id="fer-inertial-a", seed=201, role=TRAIN_ROLE),
    InertialProtocolSpec(protocol_id="fer-inertial-b", seed=202, role=TRAIN_ROLE),
    InertialProtocolSpec(
        protocol_id="fer-inertial-holdout",
        seed=902,
        role=HOLDOUT_ROLE,
        # Different multiples of the same fundamental as the canonical
        # protocols, so the holdout excites a different frequency mix while
        # still returning to rest at home (see InertialProtocolSpec).
        base_frequency_hz=(0.10, 0.20, 0.30, 0.45, 0.55, 0.65, 0.75),
    ),
)

CAMPAIGN: tuple[ProtocolSpec, ...] = FRICTION_CAMPAIGN + INERTIAL_CAMPAIGN


def compile_protocol(
    spec: ProtocolSpec,
    model: mujoco.MjModel,
    model_path: str | Path | None = None,
) -> CompiledProtocol:
    """Compile any campaign protocol with its family's generator.

    Passing ``model_path`` additionally checks the motion against the table
    the robot is bolted to.
    """
    if model_path is None:
        model_path = resolve_model_paths().require().hydrax
    if isinstance(spec, FrictionProtocolSpec):
        return generate_friction_protocol(spec, model, model_path)
    return generate_inertial_protocol(spec, model, model_path)


# Simulation-check tracking gains (NOT robot gains): only used to play a
# protocol on the nominal model for conditioning evidence.
_TRACKING_KP = np.array([100.0, 100.0, 100.0, 100.0, 40.0, 25.0, 15.0])
_TRACKING_KD = np.array([10.0, 10.0, 10.0, 10.0, 4.0, 3.0, 2.0])

_INK = "#374151"
_Q_COLOR = "#2563eb"
_DQ_COLOR = "#b45309"
_HOLD_SHADE = "#e5e7eb"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tracking_run(model: mujoco.MjModel, compiled: CompiledProtocol) -> MeasuredRun:
    """Play a compiled protocol on *model* with a PD tracker; record data."""
    dt = model.opt.timestep
    protocol_times = compiled.time_s
    steps = int(round(protocol_times[-1] / dt))
    times = np.arange(steps) * dt
    q_des = np.column_stack(
        [np.interp(times, protocol_times, compiled.q_rad[:, j]) for j in range(7)]
    )
    dq_des = np.column_stack(
        [np.interp(times, protocol_times, compiled.dq_rad_s[:, j]) for j in range(7)]
    )

    data = mujoco.MjData(model)
    data.qpos[:] = compiled.q_rad[0]
    mujoco.mj_forward(model, data)
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    control = np.empty((steps, 7))
    measured_times = np.empty(steps)
    measured = np.empty((steps, model.nsensordata))
    for k in range(steps):
        tau = np.clip(
            _TRACKING_KP * (q_des[k] - data.qpos)
            + _TRACKING_KD * (dq_des[k] - data.qvel),
            low,
            high,
        )
        control[k] = tau
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        measured_times[k] = data.time
        measured[k] = data.sensordata
    return MeasuredRun(
        label=f"{compiled.spec.protocol_id}_playback",
        qpos0=compiled.q_rad[0].copy(),
        qvel0=np.zeros(7),
        control_times=times,
        control=control,
        measured_times=measured_times,
        measured=measured,
    )


def plot_protocol(compiled: CompiledProtocol, path: Path) -> None:
    """Per-joint small multiples: position and velocity vs time."""
    time_s = compiled.time_s
    spec = compiled.spec
    figure, axes = plt.subplots(
        7, 2, figsize=(11, 12), sharex=True, constrained_layout=True
    )
    is_friction = isinstance(spec, FrictionProtocolSpec)
    subtitle = (
        f"cruises {list(spec.cruise_speeds_rad_s)} rad/s"
        if is_friction
        else f"base frequencies {list(spec.base_frequency_hz)} Hz"
    )
    figure.suptitle(
        f"{spec.protocol_id} — {spec.family_id}, seed {spec.seed}, {subtitle}",
        color=_INK,
    )
    holds = [
        segment for segment in compiled.segments if segment.kind in ("hold", "settle")
    ]
    for joint in range(7):
        for column, (values, color, unit) in enumerate(
            (
                (compiled.q_rad[:, joint], _Q_COLOR, "rad"),
                (compiled.dq_rad_s[:, joint], _DQ_COLOR, "rad/s"),
            )
        ):
            axis = axes[joint, column]
            for segment in holds:
                axis.axvspan(
                    time_s[segment.start_index],
                    time_s[segment.end_index_exclusive - 1],
                    color=_HOLD_SHADE,
                    lw=0,
                    zorder=0,
                )
            axis.plot(time_s, values, color=color, lw=1.4)
            axis.set_ylabel(f"j{joint + 1} [{unit}]", color=_INK, fontsize=8)
            axis.tick_params(labelsize=7, colors=_INK)
            axis.grid(True, alpha=0.25, lw=0.5)
            for spine in axis.spines.values():
                spine.set_alpha(0.3)
            if column == 1 and is_friction:
                for speed in spec.cruise_speeds_rad_s:
                    for sign in (1.0, -1.0):
                        axis.axhline(
                            sign * speed,
                            color=_DQ_COLOR,
                            lw=0.6,
                            ls="--",
                            alpha=0.35,
                        )
    axes[0, 0].set_title("desired position (holds shaded)", color=_INK, fontsize=9)
    axes[0, 1].set_title(
        "desired velocity (cruise speeds dashed)"
        if is_friction
        else "desired velocity",
        color=_INK,
        fontsize=9,
    )
    axes[-1, 0].set_xlabel("time [s]", color=_INK, fontsize=8)
    axes[-1, 1].set_xlabel("time [s]", color=_INK, fontsize=8)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def generate_campaign(
    protocols_root: Path,
    review_dir: Path,
    *,
    workspace_root: str | Path | None = None,
    conditioning: bool = True,
) -> list[dict[str, object]]:
    """Generate all campaign bundles plus the review material."""
    model_paths = resolve_model_paths(workspace_root).require()
    model = build_hydrax_arm_spec(
        model_paths.hydrax, joint_state_sensors=True
    ).compile()
    review_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for spec in CAMPAIGN:
        compiled = compile_protocol(spec, model)
        root = write_protocol_bundle(
            compiled,
            protocols_root,
            model=model,
            workspace_root=workspace_root,
            created_at=CAMPAIGN_CREATED_AT,
        )
        plot_protocol(compiled, review_dir / f"{spec.protocol_id}.png")
        manifest = json.loads((root / "protocol.json").read_text())
        row: dict[str, object] = {
            "protocol_id": spec.protocol_id,
            "duration_s": float(compiled.time_s[-1]),
            "samples": int(len(compiled.q_rad)),
            "content_sha256": str(manifest["content_sha256"]),
        }
        if conditioning:
            run = tracking_run(model, compiled)
            sequences = measurement_sequences(
                build_hydrax_arm_spec(model_paths.hydrax, joint_state_sensors=True),
                [run],
                window_s=0.5,
            )
            parameters = friction_parameters(model).move_off_bounds()
            report = conditioning_report(parameters, sequences)
            correlations = report.parameter_correlations
            worst = max(abs(float(correlations[2 * j, 2 * j + 1])) for j in range(7))
            row["conditioning_ratio"] = report.conditioning_ratio
            row["worst_frictionloss_damping_correlation"] = worst
        rows.append(row)

    _write_summary(review_dir / "summary.md", rows)
    return rows


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Campaign review summary",
        "",
        f"Generated deterministically ({CAMPAIGN_CREATED_AT}). "
        "Each protocol is identified by its content SHA-256. Conditioning "
        "numbers come from simulated "
        "position-tracked playback on the nominal model: the 14-parameter "
        "friction block (frictionloss + damping, all joints) must be "
        f"identifiable (ratio >= {CONDITIONING_RATIO_MINIMUM:.0e}) with every "
        "per-joint frictionloss/damping correlation below "
        f"{CORRELATION_FREEZE_LIMIT}.",
        "",
        "| protocol | duration [s] | samples | conditioning ratio "
        "| worst fl/damping corr | content sha256 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        ratio = row.get("conditioning_ratio")
        worst = row.get("worst_frictionloss_damping_correlation")
        lines.append(
            f"| {row['protocol_id']} | {row['duration_s']:.1f} "
            f"| {row['samples']} "
            f"| {'' if ratio is None else format(ratio, '.2e')} "
            f"| {'' if worst is None else format(worst, '.3f')} "
            f"| `{str(row['content_sha256'])[:16]}...` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def verify_campaign(
    protocols_root: Path, *, workspace_root: str | Path | None = None
) -> list[str]:
    """Regenerate every spec and compare content hashes with the committed
    bundles; returns human-readable mismatch descriptions (empty = clean)."""
    model_paths = resolve_model_paths(workspace_root).require()
    model = build_hydrax_arm_spec(
        model_paths.hydrax, joint_state_sensors=True
    ).compile()
    problems: list[str] = []
    for spec in CAMPAIGN:
        committed_root = protocols_root / spec.protocol_id
        if not committed_root.is_dir():
            problems.append(f"{spec.protocol_id}: missing {committed_root}")
            continue
        committed, _ = load_protocol_bundle(committed_root)
        compiled = compile_protocol(spec, model)
        with tempfile.TemporaryDirectory() as scratch:
            regenerated_root = write_protocol_bundle(
                compiled,
                Path(scratch),
                model=model,
                workspace_root=workspace_root,
                created_at=CAMPAIGN_CREATED_AT,
            )
            regenerated = json.loads((regenerated_root / "protocol.json").read_text())
        if regenerated["content_sha256"] != committed["content_sha256"]:
            problems.append(
                f"{spec.protocol_id}: regenerated content "
                f"{regenerated['content_sha256'][:16]}... != committed "
                f"{str(committed['content_sha256'])[:16]}..."
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate and compare against committed bundles",
    )
    parser.add_argument(
        "--no-conditioning",
        action="store_true",
        help="skip the simulated-playback conditioning summary",
    )
    arguments = parser.parse_args(argv)
    root = repository_root()
    if arguments.check:
        problems = verify_campaign(root / "protocols")
        for problem in problems:
            print(f"MISMATCH: {problem}")
        print("campaign check:", "FAILED" if problems else "clean")
        return 1 if problems else 0
    rows = generate_campaign(
        root / "protocols",
        root / "protocols" / "review",
        conditioning=not arguments.no_conditioning,
    )
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
