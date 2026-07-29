"""FER MuJoCo system-identification tools.

Deliberately empty of imports. Importing the package must not import MuJoCo,
because :mod:`fer_mujoco_sysid.protocol` and :mod:`fer_mujoco_sysid.playback`
run inside ROS processes on the robot, where the ``mujoco`` name may resolve
to whatever a ROS package happened to put on the path. Import the module you
need — ``from fer_mujoco_sysid.model import ...`` — not the package.
"""
