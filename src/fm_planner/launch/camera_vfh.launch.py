#!/usr/bin/env python3
"""
camera_vfh.launch.py
====================
Runs the depth camera + VFH obstacle avoidance (classic) pipeline.
Run after sitl.launch.py is already running (~18 seconds).

Nodes launched:
    gz_depth_image_bridge  — bridge /depth_camera Gazebo -> ROS2 (32FC1, meters)
    gz_depth_bridge_node   — convert -> /realsense/depth/* pipeline
    gz_depth_viewer_node   — colorized depth (only if show_camera:=true)
    vfh_obstacle_avoidance — compute safe direction, publish /vfh/movement_direction
    rqt_image_view         — camera window (only if show_camera:=true)

Usage:
    ros2 launch fm_planner camera_vfh.launch.py                 # no camera window
    ros2 launch fm_planner camera_vfh.launch.py show_camera:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # show_camera: default FALSE -> no rqt window / no colorized depth viewer.
    # The depth BRIDGE (needed by the expert to record depth images) always runs.
    show_camera_arg = DeclareLaunchArgument(
        "show_camera", default_value="false",
        description="true = open the rqt_image_view camera window + depth "
                    "viewer; false (default) = headless camera (bridge only).")
    show_camera = LaunchConfiguration("show_camera")

    # ── rqt_image_view (only when show_camera:=true) ─────────────────────────
    image_viewer = TimerAction(
        period=6.0,
        condition=IfCondition(show_camera),
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

    # ── Colorized depth viewer (only when show_camera:=true) ─────────────────
    depth_viewer = Node(
        package="fm_planner",
        executable="gz_depth_viewer_node",
        name="gz_depth_viewer_node",
        output="screen",
        condition=IfCondition(show_camera),
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
        show_camera_arg,
        gz_bridge,
        depth_bridge,
        depth_viewer,
        vfh_node,
        image_viewer,
    ])
