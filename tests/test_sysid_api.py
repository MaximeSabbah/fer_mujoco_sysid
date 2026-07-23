import inspect
from importlib.metadata import version

import mujoco.sysid as sysid
import numpy as np


def test_pinned_mujoco_version_is_loaded() -> None:
    assert version("mujoco") == "3.10.0"


def test_required_public_sysid_api_is_available() -> None:
    required_symbols = (
        "ModelSequences",
        "Parameter",
        "ParameterDict",
        "SystemTrajectory",
        "TimeSeries",
        "apply_body_inertia",
        "body_inertia_param",
        "build_residual_fn",
        "default_report",
        "optimize",
        "save_results",
    )
    missing = [name for name in required_symbols if not hasattr(sysid, name)]
    assert not missing, f"missing public mujoco.sysid symbols: {missing}"


def test_optimizer_exposes_conditioning_diagnostics() -> None:
    assert "check_conditioning" in inspect.signature(sysid.optimize).parameters


def test_parameter_and_timeseries_containers_are_usable() -> None:
    parameter = sysid.Parameter(
        name="joint1_damping",
        nominal=np.array([1.0]),
        min_value=np.array([0.0]),
        max_value=np.array([5.0]),
    )
    parameters = sysid.ParameterDict({parameter.name: parameter})
    series = sysid.TimeSeries(
        times=np.array([0.0, 0.001]),
        data=np.array([[0.0], [0.1]]),
    )

    np.testing.assert_array_equal(
        parameters[parameter.name].value,
        np.array([1.0]),
    )
    np.testing.assert_array_equal(series.times, np.array([0.0, 0.001]))
