#!/usr/bin/env python3
"""
expert_planning.launch.py

Launcher for the expert data collection pipeline.
4 nodes: publish_pcd, octomap_server, mavros_tf_broadcaster, expert_planner.

FIX: data_save_dir is auto-timestamped when not set, so every run is saved
in a different folder.
"""

import os
from datetime import datetime
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_data_save_dir(context):
    """
    If data_save_dir = '' or 'auto', generate one with a timestamp.
    Format: ~/training_data/run_YYYYMMDD_HHMMSS/
    If an explicit path is given, use it as-is.
    """
    raw = LaunchConfiguration("data_save_dir").perform(context)
    if raw in ("", "auto"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.expanduser(f"~/training_data/run_{timestamp}")
    else:
        path = os.path.expanduser(raw)

    # Grab the other parameters to pass to expert_planner_node
    pcd_path        = LaunchConfiguration("pcd_path").perform(context)
    target_samples  = LaunchConfiguration("target_samples").perform(context)
    goal_x          = LaunchConfiguration("goal_x").perform(context)
    target_alt      = LaunchConfiguration("target_alt").perform(context)
    replan_period   = LaunchConfiguration("replan_period").perform(context)
    cmd_hz          = LaunchConfiguration("cmd_hz").perform(context)
    v_max           = LaunchConfiguration("v_max").perform(context)
    safe_dis        = LaunchConfiguration("safe_dis").perform(context)
    use_ekf_stable  = LaunchConfiguration("use_ekf_stable").perform(context)
    arena_y_min     = LaunchConfiguration("arena_y_min").perform(context)
    arena_y_max     = LaunchConfiguration("arena_y_max").perform(context)
    arena_x_min     = LaunchConfiguration("arena_x_min").perform(context)
    arena_x_max     = LaunchConfiguration("arena_x_max").perform(context)

    print("=" * 62)
    print(f"  Data collection folder: {path}")
    print(f"  Arena X: [{arena_x_min}, {arena_x_max}]  Y: [{arena_y_min}, {arena_y_max}]")
    print("=" * 62)

    # expert_planner node
    expert_node = Node(
        package="neo_planner",
        executable="expert_planner_node",
        name="expert_planner_node",
        output="screen",
        parameters=[{
            "data_save_dir":  path,
            "target_samples": int(target_samples),
            "goal_x":         float(goal_x),
            "target_alt":     float(target_alt),
            "replan_period":  float(replan_period),
            "cmd_hz":         int(cmd_hz),
            "v_max":          float(v_max),
            "safe_dis":       float(safe_dis),
            "use_ekf_stable": use_ekf_stable.lower() in ("true", "1", "yes"),
            "arena_x_min":    float(arena_x_min),
            "arena_x_max":    float(arena_x_max),
            "arena_y_min":    float(arena_y_min),
            "arena_y_max":    float(arena_y_max),
        }],
    )

    return [expert_node]


def generate_launch_description():
    # ── Launch arguments ──────────────────────────────────────────────────────
    pcd_path_arg = DeclareLaunchArgument(
        "pcd_path",
        description="Path to the .pcd file used to publish the point cloud")

    target_samples_arg = DeclareLaunchArgument(
        "target_samples", default_value="5000",
        description="Target number of samples to collect")

    goal_x_arg = DeclareLaunchArgument(
        "goal_x", default_value="20.0",
        description="Goal X coordinate (m)")

    target_alt_arg = DeclareLaunchArgument(
        "target_alt", default_value="2.0",
        description="Target drone altitude (m)")

    replan_period_arg = DeclareLaunchArgument(
        "replan_period", default_value="1.0",
        description="Replanning period (seconds)")

    cmd_hz_arg = DeclareLaunchArgument(
        "cmd_hz", default_value="100",
        description="Setpoint publish rate to MAVROS (Hz)")

    v_max_arg = DeclareLaunchArgument(
        "v_max", default_value="1.0",
        description="Maximum drone speed (m/s)")

    safe_dis_arg = DeclareLaunchArgument(
        "safe_dis", default_value="0.5",
        description=(
            "EXTRA safety margin outside the drone body (m). The drone body "
            "(r~0.25m) is already counted via the ESDF inflation_radius, so this "
            "is the extra body->pole distance. 0.5 = the drone passes >=0.5m from "
            "a pole (smooth in poles); raise to 0.8 for an extra-wide margin, "
            "lower to 0.35 for aggressive slalom / straighter paths."))

    use_ekf_stable_arg = DeclareLaunchArgument(
        "use_ekf_stable", default_value="true",
        description="Wait for EKF Z to converge before takeoff (true/false)")

    # Arena boundary — limit the drone navigation area so it does not stray too far.
    # The local target is always clamped within these bounds + lateral_step.
    # poles world: X [0,23], Y [-4,+4] obstacles. Default Y ±6 gives room to move.
    arena_x_min_arg = DeclareLaunchArgument(
        "arena_x_min", default_value="-2.0",
        description="Minimum X bound of the navigation arena (m)")
    arena_x_max_arg = DeclareLaunchArgument(
        "arena_x_max", default_value="23.0",
        description="Maximum X bound of the navigation arena (m)")
    arena_y_min_arg = DeclareLaunchArgument(
        "arena_y_min", default_value="-6.0",
        description="Minimum Y bound of the navigation arena (m) — poles default -6")
    arena_y_max_arg = DeclareLaunchArgument(
        "arena_y_max", default_value="6.0",
        description="Maximum Y bound of the navigation arena (m) — poles default +6")

    # Occupancy band — ADJUST to the world:
    #   poles  : occ_min_z=1.0  occ_max_z=3.0  (short obstacle=3.36m, drone=1.5m)
    #   bricks : occ_min_z=0.5  occ_max_z=3.0  (short obstacle=1.5m, drone=1.5m)
    occ_min_z_arg = DeclareLaunchArgument(
        "occ_min_z", default_value="1.0",
        description="Min Z of 2D projection (m) — poles=1.0, bricks=0.5")
    occ_max_z_arg = DeclareLaunchArgument(
        "occ_max_z", default_value="3.0",
        description="Max Z of 2D projection (m)")

    # FIX: default 'auto' = generate a timestamped folder
    data_save_dir_arg = DeclareLaunchArgument(
        "data_save_dir",
        default_value="auto",
        description="Folder to save data. 'auto' = ~/training_data/run_TIMESTAMP/")

    # ── Other nodes (do not need data_save_dir) ───────────────────────────────
    publish_pcd_node = Node(
        package="neo_planner",
        executable="publish_pcd_node",
        name="publish_pcd_node",
        output="screen",
        parameters=[{
            "pcd_path":  LaunchConfiguration("pcd_path"),
            "frame_id":  "map",
            "rate_hz":   2.0,
        }],
    )

    octomap_server_node = Node(
        package="octomap_server",
        executable="octomap_server_node",
        name="octomap_server_expert",
        output="screen",
        parameters=[{
            "frame_id":          "map",
            # Resolution 0.2 -> 0.1: the ESDF takes this resolution from
            # /projected_map then computes n_cells = ceil(inflation_radius /
            # resolution) to fatten obstacles. At 0.2m, an inflation of 0.25m
            # rounds to 2 cells = 0.4m (0.15m excess hidden). At 0.1m it rounds
            # to 3 cells = 0.3m (only 0.05m excess left), so safe_dis is measured
            # almost exactly from the drone body edge. Bonus: more accurate
            # obstacle boundaries.
            "resolution":        0.1,
            "publish_2d_map":    True,
            "publish_3d_map":    False,
            # FIX: LaunchConfiguration produces a STRING-typed parameter, whereas
            # octomap_server declares a double. Without ParameterValue(
            # value_type=float), the occupancy band could be REJECTED / not
            # applied -> ground (z=0.05 in poles.pcd) leaks into the projected_map
            # OR the band is wrong. Must be cast explicitly.
            "occupancy_min_z":   ParameterValue(
                LaunchConfiguration("occ_min_z"), value_type=float),
            "occupancy_max_z":   ParameterValue(
                LaunchConfiguration("occ_max_z"), value_type=float),
            "filter_ground_plane": False,
            "use_height_map":    False,
        }],
        remappings=[
            ("cloud_in", "/pointcloud/map"),
        ],
    )

    mavros_tf_node = Node(
        package="neo_planner",
        executable="mavros_tf_broadcaster_node",
        name="mavros_tf_broadcaster_expert",
        output="screen",
    )

    return LaunchDescription([
        pcd_path_arg,
        target_samples_arg,
        goal_x_arg,
        target_alt_arg,
        replan_period_arg,
        cmd_hz_arg,
        v_max_arg,
        safe_dis_arg,
        use_ekf_stable_arg,
        arena_x_min_arg,
        arena_x_max_arg,
        arena_y_min_arg,
        arena_y_max_arg,
        occ_min_z_arg,
        occ_max_z_arg,
        data_save_dir_arg,
        publish_pcd_node,
        octomap_server_node,
        mavros_tf_node,
        OpaqueFunction(function=generate_data_save_dir),
    ])
