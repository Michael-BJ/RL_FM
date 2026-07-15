#!/usr/bin/env python3
"""
fm_planning_unknown.launch.py
=============================
FM-Planner deployment in an UNKNOWN ENVIRONMENT (octomap from LIVE depth).
Same infrastructure as nn_inference.launch.py (TF, depth->cloud, dynamic
octomap band) — only the planner node differs: fm_inference_node samples
K trajectory candidates per replan and warm-starts MINCO with the best one.

Usage:
  Terminal 1: ros2 launch fm_planner sitl.launch.py world:=forest1
  Terminal 2: ros2 launch fm_planner mavros_only.launch.py
  Terminal 3: ros2 launch fm_planner camera_vfh.launch.py
  Terminal 4: ros2 launch fm_planner fm_planning_unknown.launch.py \
                model_path:=/home/michael/saved_net/fm/run_XXXX/fm_planner_XXXX.pth \
                goal_x:=20.0 K:=8 n_steps:=2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path", description="Path to the FM model file (.pth or .onnx)")
    k_arg = DeclareLaunchArgument(
        "K", default_value="8", description="FM candidates sampled per replan")
    n_steps_arg = DeclareLaunchArgument(
        "n_steps", default_value="2",
        description="Euler integration steps (.pth backend only)")
    use_persistence_arg = DeclareLaunchArgument(
        "use_mode_persistence", default_value="false",
        description="FM+ ablation: hysteresis against mode-switching between "
                    "replans (false = FM-minimal, official comparison vs NN)")
    use_steering_arg = DeclareLaunchArgument(
        "use_bimodal_steering", default_value="false",
        description="FM+ ablation: bimodal candidate spread shifts the local "
                    "target earlier than the octomap does (false = FM-minimal)")
    use_anchor_arg = DeclareLaunchArgument(
        "use_anchor_sampling", default_value="false",
        description="EXPERIMENTAL stage 2: 1 of the K candidates is anchored "
                    "from the previous replan's solution. The 2026-07-06 test "
                    "actually made oscillation worse (escape 5->10) — default "
                    "OFF until the staleness bug is fixed & revalidated.")
    candidate_log_arg = DeclareLaunchArgument(
        "candidate_log_path", default_value="",
        description="Optional path to dump ALL K raw candidates per replan "
                    "as JSONL, for visualizing what FM actually considered "
                    "(not just the one flown). '' = disabled (default).")
    use_eff_rank_arg = DeclareLaunchArgument(
        "use_efficiency_ranking", default_value="false",
        description="EXPERIMENTAL stage 2 (optional): ranking = clearance - "
                    "lambda*polyline_length. Default OFF — see the note on "
                    "use_anchor_sampling (interacts badly with a stale anchor)")
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
        "safe_dis", default_value="0.5",
        description="Soft safe distance (m) — aligned with the expert planner "
                    "(the training data was collected at 0.5)")
    hard_clearance_arg = DeclareLaunchArgument(
        "hard_clearance", default_value="0.35",
        description="Hard minimum clearance (m) for trajectory post-check")
    guard_clearance_arg = DeclareLaunchArgument(
        "guard_clearance", default_value="0.4",
        description="Real-time guard & look-ahead threshold (m)")
    collision_cost_tol_arg = DeclareLaunchArgument(
        "collision_cost_tol", default_value="20.0",
        description="Optimizer weighted collision-cost tolerance — same as the "
                    "expert planner (1.0 rejected nearly every plan in tight gaps)")
    depth_max_lag_arg = DeclareLaunchArgument(
        "depth_max_lag", default_value="0.5",
        description="Maximum depth lag before a replan is skipped (s)")
    auto_reverse_arg = DeclareLaunchArgument(
        "auto_reverse", default_value="true", description="Shuttle back and forth")
    use_safety_guards_arg = DeclareLaunchArgument(
        "use_safety_guards", default_value="true",
        description="false = 'pure' NEO-Planner ablation: no proximity/"
                    "look-ahead guard, no no-progress or replan-fail escape "
                    "— only the learned initializer + MINCO optimizer decide "
                    "(matches the original paper, no reactive scaffolding). "
                    "Hard post-check + tracking-gap invalidation stay on.")
    use_lookahead_guard_arg = DeclareLaunchArgument(
        "use_lookahead_guard", default_value="false",
        description="v2.3 middle ground for the fair comparison: when "
                    "use_safety_guards:=false, keep ONLY the map-based "
                    "look-ahead guard, acting as invalidate+hover (stop short "
                    "of an unmapped wall) instead of escape. Prevents the "
                    "octomap-lag collisions seen on wall_gauntlet without "
                    "adding any active rescue. Ignored if guards are on.")
    arena_x_min_arg = DeclareLaunchArgument(
        "arena_x_min", default_value="-2.0",
        description="Arena X min (m) — escape/ESDF treat outside this as "
                    "occupied instead of 'unmapped=safe' (v2.2 fix)")
    arena_x_max_arg = DeclareLaunchArgument(
        "arena_x_max", default_value="25.0", description="Arena X max (m)")
    arena_y_min_arg = DeclareLaunchArgument(
        "arena_y_min", default_value="-6.0", description="Arena Y min (m)")
    arena_y_max_arg = DeclareLaunchArgument(
        "arena_y_max", default_value="6.0", description="Arena Y max (m)")

    resolution_arg = DeclareLaunchArgument(
        "resolution", default_value="0.15", description="Octomap resolution (m)")
    max_range_arg = DeclareLaunchArgument(
        "max_range", default_value="4.0", description="Max depth range (m)")
    # Initial band values — will be overridden dynamically by fm_inference_node
    # (_configure_octomap_band, inherited) once the node is ready.
    occ_min_z_arg = DeclareLaunchArgument("occ_min_z", default_value="1.3")
    occ_max_z_arg = DeclareLaunchArgument("occ_max_z", default_value="3.0")

    cam_x_arg     = DeclareLaunchArgument("cam_x",     default_value="0.10")
    cam_y_arg     = DeclareLaunchArgument("cam_y",     default_value="0.00")
    cam_z_arg     = DeclareLaunchArgument("cam_z",     default_value="-0.05")
    cam_roll_arg  = DeclareLaunchArgument("cam_roll",  default_value="-1.5708")
    cam_pitch_arg = DeclareLaunchArgument("cam_pitch", default_value="0.0")
    cam_yaw_arg   = DeclareLaunchArgument("cam_yaw",   default_value="-1.5708")

    mavros_tf_node = Node(
        package="fm_planner",
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
        package="fm_planner",
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

    fm_inference_node = Node(
        package="fm_planner",
        executable="fm_inference_node",
        name="fm_inference_node",
        output="screen",
        parameters=[{
            "model_path":          LaunchConfiguration("model_path"),
            "K":                   LaunchConfiguration("K"),
            "n_steps":             LaunchConfiguration("n_steps"),
            "use_mode_persistence": LaunchConfiguration("use_mode_persistence"),
            "use_bimodal_steering": LaunchConfiguration("use_bimodal_steering"),
            "use_anchor_sampling":  LaunchConfiguration("use_anchor_sampling"),
            "use_efficiency_ranking": LaunchConfiguration("use_efficiency_ranking"),
            "candidate_log_path":  LaunchConfiguration("candidate_log_path"),
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
            "use_safety_guards":   LaunchConfiguration("use_safety_guards"),
            "use_lookahead_guard": LaunchConfiguration("use_lookahead_guard"),
            "arena_x_min":         LaunchConfiguration("arena_x_min"),
            "arena_x_max":         LaunchConfiguration("arena_x_max"),
            "arena_y_min":         LaunchConfiguration("arena_y_min"),
            "arena_y_max":         LaunchConfiguration("arena_y_max"),
        }],
    )

    return LaunchDescription([
        model_path_arg, k_arg, n_steps_arg, use_persistence_arg,
        use_steering_arg, use_anchor_arg, use_eff_rank_arg, candidate_log_arg,
        goal_x_arg, target_alt_arg,
        replan_period_arg, cmd_hz_arg, v_max_arg, safe_dis_arg,
        hard_clearance_arg, guard_clearance_arg, collision_cost_tol_arg,
        depth_max_lag_arg, auto_reverse_arg, use_safety_guards_arg,
        use_lookahead_guard_arg,
        arena_x_min_arg, arena_x_max_arg, arena_y_min_arg, arena_y_max_arg,
        resolution_arg, max_range_arg,
        occ_min_z_arg, occ_max_z_arg,
        cam_x_arg, cam_y_arg, cam_z_arg, cam_roll_arg, cam_pitch_arg,
        cam_yaw_arg,
        mavros_tf_node, static_tf_camera, depth_to_cloud,
        octomap_server_node, fm_inference_node,
    ])
