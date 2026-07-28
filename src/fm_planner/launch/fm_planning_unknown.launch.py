#!/usr/bin/env python3
"""
fm_planning_unknown.launch.py
=============================
FM-Planner deployment in an UNKNOWN ENVIRONMENT (octomap from LIVE depth).
Brings up the full inference stack (TF, depth->cloud, dynamic octomap band)
plus fm_inference_node, which samples K trajectory candidates per replan and
warm-starts MINCO with the best one.

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
        "replan_period", default_value="1.0", description="Replan period (s)")
    cmd_hz_arg = DeclareLaunchArgument(
        "cmd_hz", default_value="100", description="Setpoint rate (Hz)")
    v_max_arg = DeclareLaunchArgument(
        "v_max", default_value="1.0", description="Max velocity (m/s)")
    # 2026-07-24: the three clearances below are now measured FROM THE CENTER OF
    # MASS, identical in frame AND value to expert_planning.launch.py — the model
    # is refined by the same MINCO cost that produced its labels, so the two must
    # agree. Previously these were body-EDGE numbers with a hidden +0.30 m body
    # radius. Add/subtract BODY_RADIUS_M (0.30, the real ESDF inflation) to
    # convert. See the constants block in nn_inference_node.py.
    safe_dis_arg = DeclareLaunchArgument(
        "safe_dis", default_value="0.65",
        description="Soft-zone clearance FROM THE CENTER OF MASS (m) — same "
                    "value as the expert (SAFE_DIS_CENTER). 0.65 at rest, "
                    "+0.25 per m/s -> ~0.90 m at cruise 1.0 m/s. The hard "
                    "barrier is 0.60 m (center), set in nn_inference_node.py.")
    hard_clearance_arg = DeclareLaunchArgument(
        "hard_clearance", default_value="0.55",
        description="Trajectory post-check reject line, FROM THE CENTER OF MASS "
                    "(m). Set just UNDER the 0.60 m hard barrier and decoupled "
                    "from safe_dis (expert parity), so the post-check no longer "
                    "rejects the grazing plans the soft cost is built to accept.")
    guard_clearance_arg = DeclareLaunchArgument(
        # 2026-07-27: default lowered 0.70 -> 0.55 at user request, to match
        # hard_clearance (the planner's own post-check floor) — previously the
        # guard was STRICTER than what the planner already accepted, so plans
        # through tight gauntlet_train2 passages (e.g. the G6 slalom, ~0.64 m
        # center clearance) got accepted then immediately vetoed in a loop,
        # with no escape to recover -> permanent hover -> stuck_abort_s land.
        "guard_clearance", default_value="0.55",
        description="Real-time guard & look-ahead threshold FROM THE CENTER OF "
                    "MASS (m). Same effective trip line as before (was 0.40 "
                    "body-edge) — only the frame changed; inference-only, the "
                    "expert has no counterpart.")
    collision_cost_tol_arg = DeclareLaunchArgument(
        "collision_cost_tol", default_value="20.0",
        description="Optimizer weighted collision-cost tolerance — same as the "
                    "expert planner (1.0 rejected nearly every plan in tight gaps)")
    # ── Flight-dynamics levers (2026-07-27) — see "NEO-PARITY" in command.txt.
    # Defaults = current behaviour. Together with replan_period/safe_dis these
    # reproduce the older legacy_inference_node (neo_planner_ws) flight feel.
    planning_time_ahead_arg = DeclareLaunchArgument(
        "planning_time_ahead", default_value="0.3",
        description="dTf (s): the replan starts from where the drone will be "
                    "this far ahead, but playback starts NOW — so the setpoint "
                    "leads the drone by ~dTf x speed and PX4 converts that "
                    "standing error into extra speed. Legacy node used 1.0.")
    speed_margin_k_arg = DeclareLaunchArgument(
        "speed_margin_k", default_value="0.25",
        description="Speed-aware clearance margin (m per m/s) added to "
                    "safe_dis. 0.0 = fixed margin like the legacy node.")
    px4_vel_cap_arg = DeclareLaunchArgument(
        "px4_vel_cap", default_value="0.0",
        description="PX4 horizontal speed limit written at startup: 0 (DEFAULT, "
                    "2026-07-27) = PX4 stock 12/5 m/s, i.e. PX4 decides the "
                    "physical ceiling/cruise reference instead of v_max*1.5 -- "
                    "MPC_XY_CRUISE=5.0/MPC_XY_VEL_MAX=12.0 are NOT derived from "
                    "v_max in this mode. The commanded SETPOINT velocity is "
                    "still governed by v_max (MINCO cost soft penalty + the "
                    "explicit clamps in fm_inference_base.py) -- those are the "
                    "safety net kept from the 2026-07-27 PHYSICS EXPLOSION fix "
                    "(13-29 m/s on contact with no cap AND no v_max shaping); "
                    "only PX4's own ceiling/cruise stopped being tied to v_max. "
                    "<0 = auto (v_max x1.5, old default), >0 = explicit cap.")
    depth_max_lag_arg = DeclareLaunchArgument(
        "depth_max_lag", default_value="0.5",
        description="Maximum depth lag before a replan is skipped (s)")
    auto_reverse_arg = DeclareLaunchArgument(
        "auto_reverse", default_value="true", description="Shuttle back and forth")
    use_safety_guards_arg = DeclareLaunchArgument(
        # 2026-07-27: false (was true) at user request — escape OFF by default.
        "use_safety_guards", default_value="false",
        description="false = 'pure' NEO-Planner ablation: no proximity/"
                    "look-ahead guard, no no-progress or replan-fail escape "
                    "— only the learned initializer + MINCO optimizer decide "
                    "(matches the original paper, no reactive scaffolding). "
                    "Hard post-check + tracking-gap invalidation stay on.")
    use_lookahead_guard_arg = DeclareLaunchArgument(
        # 2026-07-27 (later same day): default flipped back to false (had been
        # true for one session) — user watched the pure/no-guard run in
        # gauntlet_train2 and judged the model's raw behaviour good enough.
        # Guard 1+2 and escape are now ALL off by default; only the
        # always-on nets remain (post-check, cost-gate, tracking-gap,
        # speed-limiter, blind/stuck abort timers — see fm_inference_base.py).
        # Set true here (with use_safety_guards still false) to restore the
        # "GUARDS 1+2, invalidate+hover, NO escape" middle mode.
        "use_lookahead_guard", default_value="false",
        description="v2.3 middle ground for the fair comparison: when "
                    "use_safety_guards:=false, keep ONLY the map-based "
                    "look-ahead guard, acting as invalidate+hover (stop short "
                    "of an unmapped wall) instead of escape. Prevents the "
                    "octomap-lag collisions seen on wall_gauntlet without "
                    "adding any active rescue. Ignored if guards are on.")
    # ── Anti-collision hardening (2026-07-27) ────────────────────────────────
    # Defaults ARE the new behaviour; the values below exist so a run can be
    # reverted to the pre-fix pipeline for comparison. See the constants block
    # in fm_inference_base.py for the four crashes each one closes.
    max_plan_cost_arg = DeclareLaunchArgument(
        "max_plan_cost", default_value="50.0",
        description="Reject a plan whose TOTAL MINCO cost exceeds this — "
                    "catches diverged solves that plan_once() passes because "
                    "only its collision term is checked (run gb accepted "
                    "cost=2007.6 and flew into a wall; p90 of healthy plans "
                    "is 24). 0 = disable the gate.")
    use_speed_limit_arg = DeclareLaunchArgument(
        "use_speed_limit", default_value="true",
        description="Cap speed by the clearance available "
                    "(v = (clearance - guard_clearance)/speed_margin_k), so "
                    "the drone reaches the guard trip line slowly enough to "
                    "stop. Scales speed only, never direction — it cannot "
                    "rescue a bad candidate the way escape does. "
                    "false = old behaviour (v_max everywhere).")
    blind_abort_arg = DeclareLaunchArgument(
        "blind_abort_s", default_value="20.0",
        description="Land and end the mission after this many seconds "
                    "continuously off-map or outside the arena — with escape "
                    "off such a run is unrecoverable, so it should terminate "
                    "instead of hovering (run gb hovered ~10 min). 0 = never.")
    stuck_abort_arg = DeclareLaunchArgument(
        "stuck_abort_s", default_value="90.0",
        description="Land and end the mission after this many seconds without "
                    "closing on the goal — a drone parked at the guard line is "
                    "the CORRECT fair-mode failure, but the run is over, so it "
                    "should not need a manual cutoff. Landing is not a rescue. "
                    "0 = never.")
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
    # 2026-07-26: publish the 3D octomap voxel markers (/occupied_cells_vis_array,
    # rainbow by height) so RViz can show the perceived map like the expert does.
    # Default false = unchanged (only the 2D /projected_map, cheaper).
    publish_3d_map_arg = DeclareLaunchArgument(
        "publish_3d_map", default_value="false",
        description="true = also publish /occupied_cells_vis_array (3D octomap "
                    "voxels) for RViz, same as the expert planner.")

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
            # 2026-07-26: 0.15 -> 0.5. Depth returns closer than 0.5 m are almost
            # always near-field noise / self-view (the drone body is ~0.35 m and
            # the camera near-plane is 0.15 m); at 0.15 m these burned SPURIOUS
            # occupied cells into the octomap right at the drone's start (x~0.1-0.5)
            # -> esdf=0.00 at its own cell -> every plan born in collision -> the
            # "stuck at origin" / "chaotic escape" failures. Already-mapped
            # obstacles (seen at up to max_range) stay in the octomap as the drone
            # closes in, so raising the FLOOR loses nothing real. NOTE: this only
            # feeds the octomap; the MODEL input is the raw depth IMAGE, unaffected.
            "min_range":   0.5,
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
            "publish_3d_map":         ParameterValue(
                LaunchConfiguration("publish_3d_map"), value_type=bool),
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
            "planning_time_ahead": LaunchConfiguration("planning_time_ahead"),
            "speed_margin_k":      LaunchConfiguration("speed_margin_k"),
            "px4_vel_cap":         LaunchConfiguration("px4_vel_cap"),
            "depth_max_lag":       LaunchConfiguration("depth_max_lag"),
            "auto_reverse":        LaunchConfiguration("auto_reverse"),
            "use_safety_guards":   LaunchConfiguration("use_safety_guards"),
            "use_lookahead_guard": LaunchConfiguration("use_lookahead_guard"),
            "max_plan_cost":       LaunchConfiguration("max_plan_cost"),
            "use_speed_limit":     LaunchConfiguration("use_speed_limit"),
            "blind_abort_s":       LaunchConfiguration("blind_abort_s"),
            "stuck_abort_s":       LaunchConfiguration("stuck_abort_s"),
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
        planning_time_ahead_arg, speed_margin_k_arg, px4_vel_cap_arg,
        depth_max_lag_arg, auto_reverse_arg, use_safety_guards_arg,
        use_lookahead_guard_arg,
        max_plan_cost_arg, use_speed_limit_arg, blind_abort_arg,
        stuck_abort_arg,
        arena_x_min_arg, arena_x_max_arg, arena_y_min_arg, arena_y_max_arg,
        resolution_arg, max_range_arg,
        occ_min_z_arg, occ_max_z_arg, publish_3d_map_arg,
        cam_x_arg, cam_y_arg, cam_z_arg, cam_roll_arg, cam_pitch_arg,
        cam_yaw_arg,
        mavros_tf_node, static_tf_camera, depth_to_cloud,
        octomap_server_node, fm_inference_node,
    ])
