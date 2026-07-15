#!/usr/bin/env python3
"""
camera_vfh.launch.py
====================
Runs the depth camera + VFH obstacle avoidance (classic) pipeline.
Run after sitl.launch.py is already running (~18 seconds).

Nodes launched:
    gz_depth_image_bridge  — bridge /depth_camera Gazebo -> ROS2 (32FC1, meters)
    gz_depth_bridge_node   — convert -> /realsense/depth/* pipeline
    gz_depth_viewer_node   — colorized depth to /depth_viz/image & /depth_viz/debug
    vfh_obstacle_avoidance — compute safe direction, publish /vfh/movement_direction

Usage:
    ros2 launch drone camera_vfh.launch.py
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    # ── rqt_image_view ───────────────────────────────────────────────────────
    image_viewer = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "run", "rqt_image_view", "rqt_image_view",
                     "/vfh/visual"],
                name="rqt_image_view",
                output="screen",
            ),
        ],
    )

    # ── Bridge: Gazebo /depth_camera -> ROS2 (32FC1, meters) ─────────────────
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_depth_image_bridge",
        arguments=[
            "/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
        output="screen",
    )

    # ── Convert -> /realsense/depth/* (D455-compatible) ──────────────────────
    depth_bridge = Node(
        package="fm_planner",
        executable="gz_depth_bridge_node",
        name="gz_depth_bridge_node",
        output="screen",
    )

    # ── Colorized depth viewer ────────────────────────────────────────────────
    depth_viewer = Node(
        package="fm_planner",
        executable="gz_depth_viewer_node",
        name="gz_depth_viewer_node",
        output="screen",
    )

    # ── VFH obstacle avoidance (classic) ─────────────────────────────────────
    vfh_node = Node(
        package="fm_planner",
        executable="vfh_obstacle_avoidance_node",
        name="vfh_obstacle_avoidance",
        output="screen",
        parameters=[{
            # 2026-07-14: 88 -> 91 (Orbbec Gemini 2; the OakD-Lite sim camera
            # was matched to 91 deg too — the old value of 88 was already
            # wrong, the old sim camera was actually 1.274 rad = 73 deg).
            "hfov_deg":          91.0,
            "d_min":            300.0,   # 0.3m minimum valid depth (mm)
            "d_max":           4000.0,   # 4m effective
            "threshold":         0.45,   # single threshold, classic VFH
            "smooth_window":        2,
            "min_valley_width":     3,   # 3 sectors = 10 degrees minimum valley width
            "robot_radius":      346.0,  # mm — half the diagonal of the 49x49cm drone
            "safety_dist":       500.0,  # mm — safety buffer
            "show_debug":        False,
            "publish_visual":     True,
        }],
    )

    return LaunchDescription([
        gz_bridge,
        depth_bridge,
        depth_viewer,
        vfh_node,
        image_viewer,
    ])
