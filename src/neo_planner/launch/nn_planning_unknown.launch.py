#!/usr/bin/env python3
"""
nn_planning_unknown.launch.py  (v2 — FIX ground capture)
=========================================================
NEO-Planner deployment in an UNKNOWN ENVIRONMENT (octomap from LIVE depth).

v2 FIX (FIX "drone is always considered in collision"):
  Problem: occupancy_min_z/max_z = [1.0, 3.0] captured the GROUND PLANE (which
  appears at z ~ ground_z in the odom frame). The ground became a false
  obstacle -> ESDF full -> the drone cannot plan -> escape -> drift -> stuck.

  FIX: the occupancy window is built as a BAND around the drone's flight
  altitude, EXCLUDING the ground. Only obstacles at drone height enter the 2D map.

  Relationship:  cruise_z = ground_z + target_alt
  Correct band:  [cruise_z - band_below, cruise_z + band_above]

  Example: ground_z measured 2.26m, target_alt 2.0 -> cruise 4.26m
           occ_min_z = 3.5  (above ground 2.26, below the drone)
           occ_max_z = 5.0  (above the drone, captures obstacle tops)

  ADJUST occ_min_z/occ_max_z to the ground_z of your world! Check the log
  "[NEO] Ground Z: X.XXm" then set occ_min_z slightly above ground_z.

TF chain: odom -> base_link -> camera_depth_frame (same as v1).

Nodes:
  1. mavros_tf_broadcaster  - TF odom->base_link
  2. static_tf_camera       - TF base_link->camera_depth_frame
  3. depth_to_pointcloud    - LIVE depth -> PointCloud2 (skip empty clouds)
  4. octomap_server         - LIVE cloud -> /projected_map (band excludes ground)
  5. neo_planner_node       - NN + optimizer + ESDF

Prerequisites:
  Terminal 1: ros2 launch nn_planner sitl.launch.py world:=forest1
  Terminal 2: ros2 launch nn_planner mavros_only.launch.py
  Terminal 3: ros2 launch nn_planner camera_vfh.launch.py
  Terminal 4: ros2 launch nn_planner nn_planning_unknown.launch.py \
                model_path:=/home/michael/training_nn_planner/saved_net/planner_net.pth \
                goal_x:=20.0 occ_min_z:=3.5 occ_max_z:=5.0
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
        "target_alt", default_value="2.0", description="Altitude above ground (m)")
    replan_period_arg = DeclareLaunchArgument(
        "replan_period", default_value="1.0", description="Replan period (s)")
    cmd_hz_arg = DeclareLaunchArgument(
        "cmd_hz", default_value="100", description="Setpoint rate (Hz)")
    v_max_arg = DeclareLaunchArgument(
        "v_max", default_value="1.0", description="Max velocity (m/s)")
    safe_dis_arg = DeclareLaunchArgument(
        "safe_dis", default_value="0.7", description="Safe distance (m)")
    safe_dis_emergency_arg = DeclareLaunchArgument(
        "safe_dis_emergency", default_value="0.4", description="Emergency safe distance")
    collision_cost_tol_arg = DeclareLaunchArgument(
        "collision_cost_tol", default_value="1.0",
        description="Optimizer weighted collision-cost tolerance")
    hard_clearance_arg = DeclareLaunchArgument(
        "hard_clearance", default_value="0.35",
        description="Hard minimum clearance (m) — a trajectory is rejected if "
                    "any point is closer than this to the inflated ESDF")
    planning_time_ahead_arg = DeclareLaunchArgument(
        "planning_time_ahead", default_value="1.0", description="dTf (s)")
    auto_reverse_arg = DeclareLaunchArgument(
        "auto_reverse", default_value="true", description="Shuttle back and forth")

    resolution_arg = DeclareLaunchArgument(
        "resolution", default_value="0.15", description="Octomap resolution (m)")
    max_range_arg = DeclareLaunchArgument(
        "max_range", default_value="4.0", description="Max depth range (m)")

    # ── IMPORTANT: occupancy band, ADJUST to your world's ground_z ────────────
    # This planner is 2D — collision is only computed in XY. No Z planning.
    # The drone flies at a FIXED cruise_z. Everything within the band [occ_min_z,
    # occ_max_z] is projected to the 2D map and avoided horizontally.
    #
    # Correct occ_min_z FORMULA (accounting for the drone legs):
    #   occ_min_z = ground_z + target_alt - drone_leg_length - margin
    #
    # Example: ground_z=2.26, target_alt=2.0, legs=0.2m, margin=0.15
    #   cruise_z = 2.26 + 2.0 = 4.26m
    #   drone_lowest_point = 4.26 - 0.2 = 4.06m
    #   occ_min_z = 4.06 - 0.15 = 3.91m  <- obstacles this tall are detected
    #
    # DANGER if occ_min_z is too high: obstacles whose height is between occ_min_z
    # and drone_lowest_point will NOT be detected -> the legs hit them!
    #
    # Default 2.8m: tolerant of ground_z 1.0-2.3 with drone legs ~0.2m.
    occ_min_z_arg = DeclareLaunchArgument(
        "occ_min_z", default_value="2.8",
        description="Min Z of 2D projection — MUST be below (cruise_z - drone_leg_length)")
    occ_max_z_arg = DeclareLaunchArgument(
        "occ_max_z", default_value="5.0",
        description="Max Z of 2D projection — above the drone's flight altitude")

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
            # FIX: explicit cast to float — LaunchConfiguration = string,
            # octomap_server needs double (see note in the expert launch).
            "resolution":             ParameterValue(
                LaunchConfiguration("resolution"), value_type=float),
            "sensor_model.max_range": ParameterValue(
                LaunchConfiguration("max_range"), value_type=float),
            "publish_2d_map":         True,
            "publish_3d_map":         False,
            "queue_size":             50,
            # Band: only capture obstacles at drone height, EXCLUDE the ground.
            # NOTE: these values are OVERRIDDEN dynamically by neo_planner_node
            # once ground_z is measured (see _configure_octomap_band) so the band
            # always follows the actual cruise_z, whatever the world & EKF drift.
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

    neo_planner_node = Node(
        package="neo_planner",
        executable="neo_planner_node",
        name="neo_planner_node",
        output="screen",
        parameters=[{
            "model_path":          LaunchConfiguration("model_path"),
            "goal_x":              LaunchConfiguration("goal_x"),
            "target_alt":          LaunchConfiguration("target_alt"),
            "replan_period":       LaunchConfiguration("replan_period"),
            "cmd_hz":              LaunchConfiguration("cmd_hz"),
            "v_max":               LaunchConfiguration("v_max"),
            "safe_dis":            LaunchConfiguration("safe_dis"),
            "safe_dis_emergency":  LaunchConfiguration("safe_dis_emergency"),
            "collision_cost_tol":  LaunchConfiguration("collision_cost_tol"),
            "hard_clearance":      LaunchConfiguration("hard_clearance"),
            "planning_time_ahead": LaunchConfiguration("planning_time_ahead"),
            "auto_reverse":        LaunchConfiguration("auto_reverse"),
        }],
    )

    return LaunchDescription([
        model_path_arg, goal_x_arg, target_alt_arg, replan_period_arg,
        cmd_hz_arg, v_max_arg, safe_dis_arg, safe_dis_emergency_arg,
        collision_cost_tol_arg, hard_clearance_arg,
        planning_time_ahead_arg, auto_reverse_arg,
        resolution_arg, max_range_arg, occ_min_z_arg, occ_max_z_arg,
        cam_x_arg, cam_y_arg, cam_z_arg, cam_roll_arg, cam_pitch_arg, cam_yaw_arg,
        mavros_tf_node, static_tf_camera, depth_to_cloud,
        octomap_server_node, neo_planner_node,
    ])
