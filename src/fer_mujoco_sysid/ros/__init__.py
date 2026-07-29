"""ROS 2 entry points for playing and recording identification protocols.

Everything in this subpackage imports ``rclpy`` and therefore only runs
inside a sourced ROS environment. The rest of ``fer_mujoco_sysid`` never
imports it, so the fitting pipeline and its test suite stay ROS-free.
"""
