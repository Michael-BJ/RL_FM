#!/usr/bin/env python3
"""
nn_inference.launch.py
======================
LIGHTWEIGHT inference deployment node (nn_inference_node) — separate from the
data collection pipeline. Same infrastructure as nn_planning_unknown
(TF, depth->cloud, dynamic octomap band), only the planner node differs.

Usage:
  Terminal 1: ros2 launch neo_planner sitl.launch.py world:=bricks
  Terminal 2: ros2 launch neo_planner mavros_only.launch.py
  Terminal 3: ros2 launch neo_planner camera_vfh.launch.py
  Terminal 4: ros2 launch neo_planner nn_inference.launch.py \
                model_path:=/home/michael/training_nn_planner/saved_net_01/planner_net.pth \
                goal_x:=20.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path", description="Path to the model file (.pth or .onnx)")
    goal_x_arg = DeclareLaunchArgument(
        "goal_x", default_value="15.0", description="Goal X coordinate (m)")
    target_alt_arg = DeclareLaunchArgument(
        "target_alt", default_value="2.0", description="Flight altitude (m)")
    replan_period_arg = DeclareLaunchArgument(
        "replan_period", default_value="0.5", description="Replan period (s)")
    cmd_hz_arg = DeclareLaunchArgument(
        "cmd_hz", default_value="100", description="Setpoint rate (Hz)")
    v_max_arg = DeclareLaunchArgument(
        "v_max", default_value="1.0", description="Max velocity (m/s)")
    safe_dis_arg = DeclareLaunchArgument(
        "safe_dis", default_value="0.7", description="Soft safe distance (m)")
    hard_clearance_arg = DeclareLaunchArgument(
        "hard_clearance", default_value="0.35",
        description="Hard minimum clearance (m) for trajectory post-check")
    guard_clearance_arg = DeclareLaunchArgument(
        "guard_clearance", default_value="0.4",
        description="Real-time guard & look-ahead threshold (m)")
    collision_cost_tol_arg = DeclareLaunchArgument(
        "collision_cost_tol", default_value="1.0",
        description="Optimizer weighted collision-cost tolerance")
    depth_max_lag_arg = DeclareLaunchArgument(
        "depth_max_lag", default_value="0.5",
        description="Maximum depth lag before a replan is skipped (s)")
    auto_reverse_arg = DeclareLaunchArgument(
        "auto_reverse", default_value="true", description="Shuttle back and forth")

    resolution_arg = DeclareLaunchArgument(
        "resolution", default_value="0.15", description="Octomap resolution (m)")
    max_range_arg = DeclareLaunchArgument(
        "max_range", default_value="4.0", description="Max depth range (m)")
    # Initial band values — will be overridden dynamically by nn_inference_node
    # (_configure_octomap_band) once the node is ready.
    occ_min_z_arg = DeclareLaunchArgument("occ_min_z", default_value="1.3")
    occ_max_z_arg = DeclareLaunchArgument("occ_max_z", default_value="3.0")

    cam_x_arg     = DeclareLaunchArgument("cam_x",     default_value="0.10")
    cam_y_arg     = DeclareLaunchArgument("cam_y",     default_value="0.00")
    cam_z_arg     = DeclareLaunchArgument("cam_z",     default_value="-0.05")
    cam_roll_arg  = DeclareLaunchArgument("cam_roll",  default_value="-1.5708")
    cam_pitch_arg = DeclareLaunchArgument("cam_pitch", default_value="0.0")
    cam_yaw_arg   = DeclareLaunchArgument("cam_yaw",   default_value="-1.5708")

    mavros_tf_node = Node(
        package="neo_planner",
        executable="mavros_tf_broadcaster_node",
        name="mavros_tf_broadcaster_unknown",
        output="screen",
        parameters=[{
            "pose_topic":   "/mavros/local_position/pose",
            "parent_frame": "odom",
            "child_frame":  "base_link",
        }],
    )

    static_tf_camera = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_to_camera_depth_frame",
        arguments=[
            "--x",     LaunchConfiguration("cam_x"),
            "--y",     LaunchConfiguration("cam_y"),
            "--z",     LaunchConfiguration("cam_z"),
            "--roll",  LaunchConfiguration("cam_roll"),
            "--pitch", LaunchConfiguration("cam_pitch"),
            "--yaw",   LaunchConfiguration("cam_yaw"),
            "--frame-id",       "base_link",
            "--child-frame-id", "camera_depth_frame",
        ],
        output="screen",
    )

    depth_to_cloud = Node(
        package="neo_planner",
        executable="depth_to_pointcloud_node",
        name="depth_to_pointcloud",
        output="screen",
        parameters=[{
            "min_range":   0.15,
            "max_range":   LaunchConfiguration("max_range"),
            "skip_pixels": 2,
        }],
    )

    octomap_server_node = Node(
        package="octomap_server",
        executable="octomap_server_node",
        name="octomap_server_unknown",
        output="screen",
        parameters=[{
            "frame_id":               "odom",
            "resolution":             ParameterValue(
                LaunchConfiguration("resolution"), value_type=float),
            "sensor_model.max_range": ParameterValue(
                LaunchConfiguration("max_range"), value_type=float),
            "publish_2d_map":         True,
            "publish_3d_map":         False,
            "queue_size":             50,
            "occupancy_min_z":        ParameterValue(
                LaunchConfiguration("occ_min_z"), value_type=float),
            "occupancy_max_z":        ParameterValue(
                LaunchConfiguration("occ_max_z"), value_type=float),
            "filter_ground_plane":    False,
            "use_height_map":         False,
            "latch":                  False,
        }],
        remappings=[
            ("cloud_in", "/realsense/depth/points"),
        ],
    )

    nn_inference_node = Node(
        package="neo_planner",
        executable="nn_inference_node",
        name="nn_inference_node",
        output="screen",
        parameters=[{
            "model_path":          LaunchConfiguration("model_path"),
            "goal_x":              LaunchConfiguration("goal_x"),
            "target_alt":          LaunchConfiguration("target_alt"),
            "replan_period":       LaunchConfiguration("replan_period"),
            "cmd_hz":              LaunchConfiguration("cmd_hz"),
            "v_max":               LaunchConfiguration("v_max"),
            "safe_dis":            LaunchConfiguration("safe_dis"),
            "hard_clearance":      LaunchConfiguration("hard_clearance"),
            "guard_clearance":     LaunchConfiguration("guard_clearance"),
            "collision_cost_tol":  LaunchConfiguration("collision_cost_tol"),
            "depth_max_lag":       LaunchConfiguration("depth_max_lag"),
            "auto_reverse":        LaunchConfiguration("auto_reverse"),
        }],
    )

    return LaunchDescription([
        model_path_arg, goal_x_arg, target_alt_arg, replan_period_arg,
        cmd_hz_arg, v_max_arg, safe_dis_arg, hard_clearance_arg,
        guard_clearance_arg, collision_cost_tol_arg, depth_max_lag_arg,
        auto_reverse_arg, resolution_arg, max_range_arg,
        occ_min_z_arg, occ_max_z_arg,
        cam_x_arg, cam_y_arg, cam_z_arg, cam_roll_arg, cam_pitch_arg,
        cam_yaw_arg,
        mavros_tf_node, static_tf_camera, depth_to_cloud,
        octomap_server_node, nn_inference_node,
    ])
