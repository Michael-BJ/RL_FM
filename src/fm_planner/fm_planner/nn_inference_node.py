#!/usr/bin/env python3
"""
nn_inference_node.py  (LIGHTWEIGHT inference — separate from the dataset pipeline)
==========================================================================
A STANDALONE NEO-Planner inference node, no longer a mirror of
expert_planner_node (data collection). The rigidity only needed while collecting
the dataset is DROPPED; the safety guards are KEPT.

Dropped compared to legacy_inference_node.py (expert v25 mirror):
  - Replan loop of 11 seeds x 2 (normal/shorter) -> just 1 NN guess per variant
    (warm_start_plan already retries 5x internally with perturbation)
  - Random local-target perturbation (np.random) -> deterministic target
  - Two stuck mechanisms (speed + progress) -> no-progress only is enough
  - Emergency-stop with a persistence timer -> large gap = invalidate immediately
  - EKF stable wait + GPS alt-trim -> cruise_z stays = target_alt
  - Depth lag gating 0.3s -> relaxed to 0.5s (replan skipped less often)

Kept (safety, NOT dataset rigidity):
  - Hard post-check on the trajectory  (hard_clearance, reject grazing trajectory)
  - Real-time collision guard + look-ahead guard 0.3-1.5s ahead
  - Time-scaling + clamp v_max
  - Dynamic octomap band following cruise_z + map reset
  - Simple escape mode (8-direction probe) on replan failure / no progress

Reuse: PlannerNet/DroneState/TrajectorySegment from legacy_inference_node.py,
ESDF from esdf_ros2.py, MinJerkPlanner from min_jerk_planner.py.
"""
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, ParamSetV2
from rcl_interfaces.msg import Parameter as RclParameter
from rcl_interfaces.msg import ParameterValue as RclParameterValue
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Empty as EmptySrv
from nav_msgs.msg import OccupancyGrid
from pyquaternion import Quaternion
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from esdf_ros2 import ESDF
from min_jerk_planner import MinJerkPlanner, PlannerConfig
# Reuse the architecture & utils from the old inference node — DO NOT duplicate.
from legacy_inference_node import (
    PlannerNet, DroneState, TrajectorySegment,
    IMG_WIDTH, IMG_HEIGHT, MOTION_INPUT_SIZE, _SENSOR_QOS,
)

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# Real-time guard (safety, NOT dataset rigidity — do not relax carelessly)
# 2026-07-14 INPUT CONTRACT MIGRATION (Gemini 2): fixed-range depth
# normalization to 10.0 m — MUST be identical to expert_planner_node.py
# (see the note there). A model trained on the per-frame-max dataset
# (before 2026-07-14) is NOT compatible with this input convention.
DEPTH_NORM_MAX_M = 10.0

LOOKAHEAD_HORIZON_S    = 1.5
LOOKAHEAD_STEP_S       = 0.3
LOOKAHEAD_CHECK_PERIOD = 10    # check every N publish ticks (10 Hz @ cmd_hz=100)
GAP_INVALIDATE         = 2.5   # m, tracking gap -> trajectory discarded immediately

# Simple escape
ESCAPE_VELOCITY   = 0.4
ESCAPE_PROBE_DIST = 1.5
ESCAPE_EXIT_DIST  = 1.0
ESCAPE_MAX_TIME   = 10.0
NO_PROGRESS_TIME  = 8.0
NO_PROGRESS_DELTA = 0.8
# v2.2 FIX (2026-07-09): get_edt_dis() returns 10000.0 ("safe") for any
# point outside the CURRENTLY-MAPPED octomap region -- correct for MINCO's
# collision cost (can't penalize what hasn't been observed yet) but
# pathological for escape's own max-clearance scoring: a bounded bias term
# (center_w <= 1.5) can never outweigh a ~10000 vs ~3 gap, so ANY probe
# direction that happens to poke into unexplored space (common near an
# occluding obstacle cluster, or the arena edge where camera FOV has less
# history) wins regardless of center/goal bias -- confirmed via isolated
# replay of the real wall_gauntlet failure. Cap it here, LOCAL to escape
# scoring only (not the shared ESDF query -- other consumers, e.g. MINCO's
# own collision cost, are correct to treat unobserved space as unpenalized).
# 2.5 m ~= what a genuinely open, already-explored corridor reads as in
# practice (see [INF] esdf= log values) -- "as good as clearly open",
# not an unbeatable outlier that swamps every other direction.
ESCAPE_UNMAPPED_CAP = 2.5


class NNInferenceNode(Node):
    STATE_IDLE    = "IDLE"
    STATE_TAKEOFF = "TAKEOFF"
    STATE_FLYING  = "FLYING"
    STATE_LANDING = "LANDING"
    STATE_DONE    = "DONE"

    def __init__(self):
        super().__init__("nn_inference_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("goal_x",            20.0)
        self.declare_parameter("target_alt",         2.0)
        self.declare_parameter("replan_period",      0.5)
        self.declare_parameter("cmd_hz",           100)
        # 2026-07-14: STAYS 5.0 — follows the expert (the target_pos
        # distribution in the model input must match between training and
        # inference). See the 7.5 -> 5.0 revert note in
        # expert_planner_node.py (it destabilized dense worlds: cost blew up
        # -> escape ping-pong -> physics explosion).
        self.declare_parameter("longitu_step",       5.0)
        self.declare_parameter("lateral_step",       1.0)
        self.declare_parameter("v_max",              1.0)
        self.declare_parameter("arm_timeout",       30.0)
        self.declare_parameter("connect_timeout",   30.0)
        self.declare_parameter("safe_dis",           0.7)
        self.declare_parameter("hard_clearance",     0.35)
        self.declare_parameter("guard_clearance",    0.4)
        self.declare_parameter("collision_cost_tol", 1.0)
        self.declare_parameter("planning_time_ahead", 0.3)
        self.declare_parameter("depth_max_lag",      0.5)
        self.declare_parameter("auto_reverse",      True)
        # "Pure" ablation (2026-07-06): the original NEO-Planner paper has NO
        # escape mode / reactive guards / stuck-detection — only the learned
        # initializer + MINCO optimization ("the subsequent optimization step
        # ensures flight safety"). Those extra layers are OUR scaffolding on
        # top, and they can mask a genuinely bad NN/FM candidate (drone gets
        # rescued before we see it fail). Set false to strip them out for a
        # head-to-head NN-vs-FM comparison of the MODEL itself: no proximity
        # guard, no look-ahead guard, no no-progress escape, no
        # replan-fail escape. The hard post-check on a NEWLY PLANNED
        # trajectory (_validate_planned_traj) and the tracking-gap
        # invalidation (Guard 3 — discards a stale/unreachable reference, not
        # an obstacle decision) stay ON regardless; they are core plumbing,
        # not safety scaffolding. WARNING: with this off, a bad candidate can
        # genuinely fly into an obstacle (Gazebo PHYSICS EXPLOSION) — that is
        # the point (measuring true failure), but expect to restart SITL.
        self.declare_parameter("use_safety_guards", True)
        # v2.3 (2026-07-10): middle ground for the fair NN-vs-FM comparison.
        # `use_safety_guards:=false` turned wall_gauntlet into a "both models
        # fail" world (measured 2026-07-10: NN stalled at the widest wall, FM
        # collided + physics-exploded) because removing the LOOK-AHEAD guard
        # also removes the only thing compensating for octomap lag -> a
        # candidate that looked clear at plan time flies into a wall the map
        # had not filled in yet. That is an OCTOMAP-latency artifact, not a
        # model-quality signal. This flag keeps ONLY the map-based look-ahead
        # guard (Guard 2) active while use_safety_guards is false, but its
        # action is to INVALIDATE + hover (stop short of the obstacle), NOT to
        # enter the escape maneuver. So the drone is prevented from crashing
        # into an unmapped wall, yet gets NO active rescue (no escape probe,
        # no no-progress recovery, no proximity guard) that could mask a
        # genuinely bad candidate. Same map-safety for both models -> a fair
        # comparison that can actually COMPLETE instead of both crashing.
        # Ignored when use_safety_guards is true (full scaffolding already on).
        self.declare_parameter("use_lookahead_guard", False)
        # v2.2 (2026-07-09): arena bounds -- MISSING entirely before this,
        # unlike expert_planner_node.py which always passes these to ESDF()
        # AND biases escape back toward the center when out of bounds. Root
        # cause of a real wall_gauntlet failure: get_edt_dis() returns
        # 10000.0 ("safe") for any point outside the CURRENTLY-MAPPED
        # octomap region (which only grows where the depth camera has
        # looked, unlike the expert's ground-truth-pcd octomap that covers
        # the whole world quickly). _find_escape_direction() then reliably
        # scores "toward unexplored space" as safest -- near an occluding
        # pole/wall cluster or the arena edge, unexplored is systematically
        # OUTWARD, and nothing pulled the drone back (no arena awareness at
        # all here before this fix) -- escapes compounded in the same
        # direction across replans until the drone got pinned at the arena
        # edge with collapsing altitude. Same defaults as expert_planner_node.
        self.declare_parameter("arena_x_min", -2.0)
        self.declare_parameter("arena_x_max", 25.0)
        self.declare_parameter("arena_y_min", -6.0)
        self.declare_parameter("arena_y_max", 6.0)

        self._model_path     = str(self.get_parameter("model_path").value)
        self._goal_x         = float(self.get_parameter("goal_x").value)
        self._alt            = float(self.get_parameter("target_alt").value)
        self._replan_period  = float(self.get_parameter("replan_period").value)
        self._cmd_hz         = int(self.get_parameter("cmd_hz").value)
        self._longitu_step   = float(self.get_parameter("longitu_step").value)
        self._lateral_step   = float(self.get_parameter("lateral_step").value)
        self._v_max          = float(self.get_parameter("v_max").value)
        self._arm_timeout    = float(self.get_parameter("arm_timeout").value)
        self._conn_timeout   = float(self.get_parameter("connect_timeout").value)
        self._safe_dis       = float(self.get_parameter("safe_dis").value)
        self._hard_clearance = float(self.get_parameter("hard_clearance").value)
        self._guard_clear    = float(self.get_parameter("guard_clearance").value)
        self._coll_tol       = float(self.get_parameter("collision_cost_tol").value)
        self._dt_ahead       = float(self.get_parameter("planning_time_ahead").value)
        self._depth_max_lag  = float(self.get_parameter("depth_max_lag").value)
        self._auto_reverse   = bool(self.get_parameter("auto_reverse").value)
        self._use_guards     = bool(self.get_parameter("use_safety_guards").value)
        # look-ahead-only mode (see declare): active only when full guards off.
        self._use_lookahead  = (bool(self.get_parameter("use_lookahead_guard").value)
                                and not self._use_guards)
        self._arena_x_min    = float(self.get_parameter("arena_x_min").value)
        self._arena_x_max    = float(self.get_parameter("arena_x_max").value)
        self._arena_y_min    = float(self.get_parameter("arena_y_min").value)
        self._arena_y_max    = float(self.get_parameter("arena_y_max").value)
        self._arena_center   = np.array([
            (self._arena_x_min + self._arena_x_max) / 2.0,
            (self._arena_y_min + self._arena_y_max) / 2.0])

        if not os.path.isfile(self._model_path):
            self.get_logger().error(f"Model not found: {self._model_path}")
            raise FileNotFoundError(self._model_path)

        self._use_pth = self._model_path.endswith('.pth')
        self._device  = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        backend_str = self._load_model()

        cfg = PlannerConfig()
        cfg.v_max              = self._v_max
        cfg.T_min              = 0.5
        cfg.T_max              = 5.0
        cfg.safe_dis           = self._safe_dis
        cfg.init_wpts_num      = 2
        cfg.init_T             = 2.5
        cfg.delta_t            = 0.1
        cfg.collision_cost_tol = self._coll_tol
        cfg.opt_tol            = 1e-2
        cfg.weights            = [1.0, 1.0, 1.0, 10000.0]
        self._planner          = MinJerkPlanner(cfg)

        # arena_bounds -> virtual wall in the ESDF (see esdf_ros2.py): cells
        # outside the arena are marked occupied, not just "not yet mapped".
        self._esdf = ESDF(arena_bounds=(
            self._arena_x_min, self._arena_x_max,
            self._arena_y_min, self._arena_y_max))

        self._connected   = False
        self._armed       = False
        self._mode        = ""
        self._drone_state = DroneState()
        self._cruise_z    = None

        self._traj        = TrajectorySegment()
        self._traj.cmd_hz = self._cmd_hz
        self._traj_lock   = threading.Lock()

        self._global_target  = np.array([self._goal_x, 0.0])
        self._target_state   = np.zeros((2, 2))
        self._reached_target = False
        self._mission_state  = self.STATE_IDLE

        self._replan_fail_count = 0
        self._escape_active     = False
        self._escape_vel_xy     = np.zeros(2)
        self._escape_start_t    = None
        self._progress_ref_dist = None
        self._progress_ref_time = None
        self._lookahead_tick    = 0

        self._latest_depth       = None
        self._latest_depth_state = None
        self._latest_depth_time  = None
        self._bridge             = CvBridge()
        self._depth_lock         = threading.Lock()

        self._inference_times = []
        self._replan_count    = 0

        self._pub_sp = self.create_publisher(
            PositionTarget, "/mavros/setpoint_raw/local", 10)

        # ── Visualization: model candidate paths (bimodal fan) + flown path ──
        # Publishes the RAW model guesses (before MINCO) as coarse polylines so
        # RViz shows what the MODEL proposed: NN -> 1 line (unimodal, aimed at
        # obstacle centre); FM -> K lines fanning left/right (bimodal). The
        # MINCO-refined path that is actually flown is drawn separately (yellow).
        # Purely a diagnostic publisher — never touches control.
        self.declare_parameter("publish_markers", True)
        self._publish_markers = bool(
            self.get_parameter("publish_markers").value)
        # /projected_map (the octomap RViz shows) is published in the `odom`
        # frame; the ESDF world coordinates + waypoints share that frame, so
        # markers must live in `odom` to overlay the map (NOT the `map` frame
        # used to label MAVROS setpoints).
        self.declare_parameter("marker_frame", "odom")
        self._marker_frame = str(self.get_parameter("marker_frame").value)
        self._pub_markers = self.create_publisher(
            MarkerArray, "/planner/candidates", 1) if self._publish_markers \
            else None
        self.create_subscription(String, "/px4/state",   self._cb_state,   10)
        self.create_subscription(String, "/px4/sensors", self._cb_sensors, 10)
        self.create_subscription(
            Image, "/realsense/depth/float32", self._cb_depth, _SENSOR_QOS)
        self.create_subscription(
            OccupancyGrid, "/projected_map", self._esdf.occupancy_map_cb, 10)

        self._arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self._mode_client   = self.create_client(SetMode,     "/mavros/set_mode")
        self._param_client  = self.create_client(ParamSetV2,  "/mavros/param/set")
        self._octo_param_client = self.create_client(
            SetParameters, "/octomap_server_unknown/set_parameters")
        self._octo_reset_client = self.create_client(
            EmptySrv, "/octomap_server_unknown/reset")

        self._cmd_timer    = self.create_timer(1.0 / self._cmd_hz, self._publish_cmd)
        self._status_timer = self.create_timer(5.0, self._print_status)

        self.get_logger().info("=" * 62)
        self.get_logger().info("  NN Inference (standalone, relaxed motion)")
        self.get_logger().info("=" * 62)
        self.get_logger().info(f"  Model         : {os.path.basename(self._model_path)}")
        self.get_logger().info(f"  Backend       : {backend_str}")
        self.get_logger().info(f"  v_max         : {self._v_max} m/s")
        self.get_logger().info(f"  Goal X        : {self._goal_x} m  alt={self._alt} m")
        self.get_logger().info(f"  safe_dis      : {self._safe_dis} m")
        self.get_logger().info(f"  hard_clearance: {self._hard_clearance} m")
        self.get_logger().info(f"  guard_clear   : {self._guard_clear} m")
        self.get_logger().info(f"  Replan        : {self._replan_period} s")
        guard_mode = ("FULL (guards+escape)" if self._use_guards
                      else "LOOK-AHEAD only (invalidate+hover, no escape)"
                      if self._use_lookahead else "OFF (no guards/escape)")
        self.get_logger().info(f"  Guard mode    : {guard_mode}")
        self.get_logger().info("=" * 62)

    # ── Model backend (overridden by fm_inference_node) ──────────────────────

    def _load_model(self):
        """Load the network from self._model_path. Returns a backend string."""
        if self._use_pth:
            self._net = PlannerNet().to(self._device)
            self._net.load_state_dict(
                torch.load(self._model_path, map_location=self._device))
            self._net.eval()
            return f"PyTorch ({self._device})"
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime not installed")
        self._session = ort.InferenceSession(
            self._model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        return f"ONNX ({self._session.get_providers()})"

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _cb_state(self, msg):
        try:
            d = json.loads(msg.data)
            self._connected = d.get("connected", False)
            self._armed     = d.get("armed",     False)
            self._mode      = d.get("mode",      "")
        except Exception:
            pass

    def _cb_sensors(self, msg):
        try:
            d = json.loads(msg.data)
            x  = float(d.get("local_x", 0)); y  = float(d.get("local_y", 0))
            z  = float(d.get("local_z", 0))
            vx = float(d.get("vel_x",  0)); vy = float(d.get("vel_y",  0))
            vz = float(d.get("vel_z",  0)); yaw = float(d.get("yaw", 0))
            self._drone_state.global_pos = np.array([x, y, z])
            self._drone_state.global_vel = np.array([vx, vy, vz])
            self._drone_state.yaw        = yaw
            qw = math.cos(yaw / 2); qz = math.sin(yaw / 2)
            self._drone_state.attitude = Quaternion(qw, 0.0, 0.0, qz)
            self._drone_state.local_vel = (
                self._drone_state.attitude.inverse.rotate(
                    self._drone_state.global_vel))
        except Exception:
            pass

    def _cb_depth(self, msg):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            if depth.shape != (IMG_HEIGHT, IMG_WIDTH):
                depth = cv2.resize(depth, (IMG_WIDTH, IMG_HEIGHT))
            snap = DroneState()
            snap.copy_from(self._drone_state)
            with self._depth_lock:
                self._latest_depth       = depth.copy()
                self._latest_depth_state = snap
                self._latest_depth_time  = time.time()
        except Exception:
            pass

    # ── Setpoint publisher + real-time guard ────────────────────────────────

    def _publish_cmd(self):
        msg = PositionTarget()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = "map"
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = (
            PositionTarget.IGNORE_VZ  | PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE)

        if (self._mission_state == self.STATE_FLYING
                and self._escape_active and self._cruise_z is not None):
            drone_pos = self._drone_state.global_pos[:2]
            tgt = drone_pos + self._escape_vel_xy * 0.5
            msg.position.x = float(tgt[0]); msg.position.y = float(tgt[1])
            msg.position.z = float(self._cruise_z)
            msg.velocity.x = float(self._escape_vel_xy[0])
            msg.velocity.y = float(self._escape_vel_xy[1])
            msg.velocity.z = 0.0
            if np.linalg.norm(self._escape_vel_xy) > 0.05:
                msg.yaw = math.atan2(self._escape_vel_xy[1],
                                     self._escape_vel_xy[0])
            else:
                msg.yaw = self._drone_state.yaw
            self._pub_sp.publish(msg)
            return

        if self._mission_state == self.STATE_FLYING and self._cruise_z is not None:
            t_now = time.time()
            with self._traj_lock:
                pos, vel, _ = self._traj.sample_at(t_now)
            if pos is None:
                pos = self._drone_state.global_pos[:2]
                vel = np.zeros(2)

            drone_pos = self._drone_state.global_pos[:2]

            # Guard 1: the drone's own position is too close to an obstacle -> escape.
            if (self._use_guards and self._esdf.is_ready() and not self._escape_active
                    and self._esdf.get_edt_dis(drone_pos) < self._guard_clear):
                self.get_logger().warn(
                    "[INF] Collision guard! ESDF too close — escape",
                    throttle_duration_sec=0.5)
                with self._traj_lock:
                    self._traj.invalidate()
                self._enter_escape()
                pos, vel = drone_pos, np.zeros(2)

            # Guard 2: look-ahead — trajectory 0.3..1.5s ahead checked against the
            # latest ESDF (the map changes after planning). v2.3: also runs in
            # look-ahead-only mode (self._use_lookahead), where its action is
            # invalidate + hover instead of escape (see use_lookahead_guard).
            self._lookahead_tick += 1
            if ((self._use_guards or self._use_lookahead) and self._esdf.is_ready()
                    and not self._escape_active
                    and self._lookahead_tick % LOOKAHEAD_CHECK_PERIOD == 0):
                hit = None
                with self._traj_lock:
                    if self._traj.is_valid():
                        for dt in np.arange(LOOKAHEAD_STEP_S,
                                            LOOKAHEAD_HORIZON_S + 1e-6,
                                            LOOKAHEAD_STEP_S):
                            p, _, _ = self._traj.sample_at(t_now + dt)
                            if p is None:
                                break
                            d = self._esdf.get_edt_dis(p)
                            if d < self._guard_clear:
                                hit = d
                                break
                if hit is not None:
                    with self._traj_lock:
                        self._traj.invalidate()
                    pos, vel = drone_pos, np.zeros(2)
                    if self._use_guards:
                        self.get_logger().warn(
                            f"[INF] Look-ahead guard! clearance ahead {hit:.2f}m"
                            " — escape", throttle_duration_sec=0.5)
                        self._enter_escape()
                    else:
                        # look-ahead-only: stop short, no escape maneuver.
                        self.get_logger().warn(
                            f"[INF] Look-ahead guard! clearance ahead {hit:.2f}m"
                            " — hover (no escape)", throttle_duration_sec=0.5)

            # Guard 3: large tracking gap -> stale trajectory, discard immediately
            # (without a persistence timer like the dataset pipeline).
            if np.linalg.norm(pos - drone_pos) > GAP_INVALIDATE:
                self.get_logger().warn(
                    "[INF] Large tracking gap — trajectory discarded",
                    throttle_duration_sec=1.0)
                with self._traj_lock:
                    self._traj.invalidate()
                pos, vel = drone_pos, np.zeros(2)

            v_norm = float(np.linalg.norm(vel))
            if v_norm > self._v_max:
                vel = vel * (self._v_max / v_norm)

            msg.position.x = float(pos[0]); msg.position.y = float(pos[1])
            msg.position.z = float(self._cruise_z)
            msg.velocity.x = float(vel[0]); msg.velocity.y = float(vel[1])
            msg.velocity.z = 0.0
            if abs(vel[0]) > 0.05 or abs(vel[1]) > 0.05:
                msg.yaw = math.atan2(vel[1], vel[0])
            else:
                msg.yaw = self._drone_state.yaw
        else:
            pos      = self._drone_state.global_pos
            target_z = self._cruise_z if self._cruise_z is not None else self._alt
            msg.position.x = float(pos[0]); msg.position.y = float(pos[1])
            msg.position.z = float(target_z)
            msg.velocity.x = 0.0; msg.velocity.y = 0.0; msg.velocity.z = 0.0
            msg.yaw        = self._drone_state.yaw

        self._pub_sp.publish(msg)

    # ── Simple escape ─────────────────────────────────────────────────────────

    def _is_out_of_bounds(self, pos) -> bool:
        return not (self._arena_x_min <= pos[0] <= self._arena_x_max
                    and self._arena_y_min <= pos[1] <= self._arena_y_max)

    def _find_escape_direction(self):
        """v2.2 FIX (2026-07-09): add the SAME arena-center bias
        expert_planner_node.py already uses. Without it, clearance alone
        reliably steers escape toward UNEXPLORED octomap space (which
        get_edt_dis() treats as 10000.0 = "safe", see arena_bounds comment
        at __init__) rather than genuinely open space -- near an occluding
        obstacle cluster or the arena edge, unexplored is systematically
        OUTWARD, so escapes compounded outward across replans with nothing
        pulling back (real wall_gauntlet failure: drone pinned at the arena
        edge, altitude collapsed). center_bias is weak normally (matches
        goal_bias's old 0.2 order of magnitude) but dominates once actually
        out of bounds -- same weighting expert_planner_node.py uses."""
        drone_pos = self._drone_state.global_pos[:2]
        to_goal = self._global_target - drone_pos
        n = np.linalg.norm(to_goal)
        to_goal_dir = to_goal / n if n > 1e-3 else np.array([1.0, 0.0])
        to_center = self._arena_center - drone_pos
        cn = np.linalg.norm(to_center)
        to_center_dir = to_center / cn if cn > 1e-3 else np.array([0.0, 0.0])
        out = self._is_out_of_bounds(drone_pos)
        center_w = 1.5 if out else 0.15
        goal_w   = 0.1 if out else 0.2
        best_score, best_dir = -np.inf, np.array([1.0, 0.0])
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            d = np.array([np.cos(ang), np.sin(ang)])
            far = min(self._esdf.get_edt_dis(drone_pos + d * ESCAPE_PROBE_DIST),
                      ESCAPE_UNMAPPED_CAP)
            mid = min(self._esdf.get_edt_dis(drone_pos + d * ESCAPE_PROBE_DIST * 0.5),
                      ESCAPE_UNMAPPED_CAP)
            score = (min(far, mid) + goal_w * float(d @ to_goal_dir)
                     + center_w * float(d @ to_center_dir))
            if score > best_score:
                best_score, best_dir = score, d
        return best_dir

    def _enter_escape(self):
        if self._escape_active or not self._esdf.is_ready():
            return
        self._escape_vel_xy  = self._find_escape_direction() * ESCAPE_VELOCITY
        self._escape_active  = True
        self._escape_start_t = time.time()
        self._replan_fail_count = 0
        self._reset_progress()
        with self._traj_lock:
            self._traj.invalidate()
        self.get_logger().warn(
            f"[INF] >>> ESCAPE vel=({self._escape_vel_xy[0]:.2f},"
            f"{self._escape_vel_xy[1]:.2f})")

    def _maybe_exit_escape(self):
        if not self._escape_active:
            return
        timeout = time.time() - self._escape_start_t > ESCAPE_MAX_TIME
        clear   = (self._esdf.get_edt_dis(
            self._drone_state.global_pos[:2]) > ESCAPE_EXIT_DIST)
        if timeout or clear:
            self.get_logger().info("[INF] <<< Exit escape")
            self._escape_active  = False
            self._escape_start_t = None

    def _reset_progress(self):
        self._progress_ref_dist = None
        self._progress_ref_time = None

    def _check_no_progress(self):
        now = time.time()
        d = float(np.linalg.norm(
            self._drone_state.global_pos[:2] - self._global_target))
        if (self._progress_ref_dist is None
                or d < self._progress_ref_dist - NO_PROGRESS_DELTA):
            self._progress_ref_dist = d
            self._progress_ref_time = now
            return False
        return (now - self._progress_ref_time) > NO_PROGRESS_TIME

    # ── Local target (deterministic — no random seed like the dataset) ──────

    def _set_local_target(self, shorter=False, ref_pos=None):
        curr = self._drone_state.global_pos[:2] if ref_pos is None else ref_pos
        step = 2.0 if shorter else self._longitu_step
        to_goal = self._global_target - curr
        dist = np.linalg.norm(to_goal)
        if dist < step:
            self._target_state[0] = self._global_target
            self._target_state[1] = np.zeros(2)
            return
        fwd = to_goal / dist
        lat = np.array([[fwd[1], -fwd[0]], [-fwd[1], fwd[0]]])
        local_tgt = curr + step * fwd
        lat_flag, lat_move = 0, self._lateral_step
        for _ in range(8):
            if not (self._esdf.is_ready()
                    and self._esdf.has_collision(local_tgt)):
                break
            local_tgt = local_tgt + lat_move * lat[lat_flag]
            lat_flag  = 1 - lat_flag
            lat_move += self._lateral_step
        goal_dir = self._global_target - local_tgt
        n = np.linalg.norm(goal_dir)
        self._target_state[0] = local_tgt
        self._target_state[1] = (goal_dir / n * self._v_max * 0.8
                                 if n > 1e-3 else np.zeros(2))

    # ── RViz markers: model candidates (bimodal fan) + flown path ───────────

    def _publish_candidate_markers(self, guesses, head, tail_xy):
        """Draw each raw model guess as a coarse polyline head->w1->w2->tail.
        Best (rank 0) = green & thick; others coloured by side (left=blue,
        right=red). For NN there is a single green line; for FM the K lines
        fan out to both sides at an ambiguous obstacle = bimodality made
        visible. No effect on control."""
        if not self._publish_markers or self._pub_markers is None:
            return
        z = float(self._cruise_z) if self._cruise_z is not None else self._alt
        hx, hy = float(head[0][0]), float(head[0][1])
        tx, ty = float(tail_xy[0]), float(tail_xy[1])
        ax, ay = tx - hx, ty - hy
        an = math.hypot(ax, ay) or 1.0
        ax, ay = ax / an, ay / an
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self._marker_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, (int_wpts, ts) in enumerate(guesses):
            m = Marker()
            m.header.frame_id = self._marker_frame
            m.header.stamp = now
            m.ns = "candidates"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.10 if i == 0 else 0.05
            w1 = (float(int_wpts[0, 0]), float(int_wpts[1, 0]))
            w2 = (float(int_wpts[0, 1]), float(int_wpts[1, 1]))
            for px, py in [(hx, hy), w1, w2, (tx, ty)]:
                p = Point()
                p.x, p.y, p.z = px, py, z
                m.points.append(p)
            lat = ax * (w2[1] - hy) - ay * (w2[0] - hx)   # +left / -right
            if i == 0:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 1.0
            elif lat >= 0.0:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.5, 1.0, 0.7
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.4, 0.2, 0.7
            arr.markers.append(m)
        self._pub_markers.publish(arr)

    def _publish_flown_marker(self):
        """Draw the MINCO-refined trajectory that is actually flown (yellow)."""
        if not self._publish_markers or self._pub_markers is None:
            return
        try:
            pos = self._planner.get_pos_array()
        except Exception:
            return
        z = float(self._cruise_z) if self._cruise_z is not None else self._alt
        m = Marker()
        m.header.frame_id = self._marker_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "flown"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.13
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.95, 0.1, 1.0
        for p in pos:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = float(p[2]) if len(p) > 2 else z
            m.points.append(pt)
        arr = MarkerArray()
        arr.markers.append(m)
        self._pub_markers.publish(arr)

    def _get_head_state(self):
        """Planning head state: from the active trajectory if it is still healthy,
        otherwise from the drone's actual position."""
        t_now = time.time()
        with self._traj_lock:
            if self._traj.is_valid():
                remaining = self._traj.time_remaining(t_now)
                pos, vel, _ = self._traj.sample_at(t_now + self._dt_ahead)
            else:
                remaining, pos, vel = 0.0, None, None
        head = np.zeros((3, 2))
        drone_pos = self._drone_state.global_pos[:2]
        if (pos is not None and remaining > self._dt_ahead + 0.3
                and np.linalg.norm(pos - drone_pos) < GAP_INVALIDATE):
            head[0], head[1] = pos, vel
        else:
            head[0] = drone_pos
            v = self._drone_state.global_vel[:2].copy()
            s = float(np.linalg.norm(v))
            if s > self._v_max:
                v *= self._v_max / s
            if s < 0.1:
                v = np.zeros(2)
            head[1] = v
        return head

    # ── NN initial guess ─────────────────────────────────────────────────────

    def _form_nn_input(self, depth_img, drone_state, head_pos, head_vel):
        # Normalisasi RENTANG-TETAP (DEPTH_NORM_MAX_M) — identik dgn expert.
        d = np.nan_to_num(depth_img, nan=0.0,
                          posinf=DEPTH_NORM_MAX_M, neginf=0.0)
        depth_u8 = np.clip(
            d / DEPTH_NORM_MAX_M * 255.0, 0, 255).astype(np.uint8)

        quat = drone_state.attitude
        z = self._cruise_z or self._alt
        init_pos = np.array([head_pos[0], head_pos[1], z])
        init_vel = np.array([head_vel[0], head_vel[1], 0.0])
        tgt_pos  = np.array([self._target_state[0, 0],
                             self._target_state[0, 1], z])
        tgt_vel  = np.array([self._target_state[1, 0],
                             self._target_state[1, 1], 0.0])
        motion_info = np.concatenate([
            drone_state.local_vel,
            quat.rotation_matrix.reshape(-1),
            quat.inverse.rotate(init_pos - drone_state.global_pos),
            quat.inverse.rotate(init_vel - drone_state.global_vel),
            quat.inverse.rotate(tgt_pos  - drone_state.global_pos),
            quat.inverse.rotate(tgt_vel  - drone_state.global_vel),
        ]).astype(np.float32)
        return depth_u8, motion_info

    def _nn_initial_guess(self, depth_img, drone_state, head_state):
        depth_u8, motion_info = self._form_nn_input(
            depth_img, drone_state, head_state[0], head_state[1])
        input_flat = np.concatenate(
            [depth_u8.reshape(-1).astype(np.float32), motion_info])
        t0 = time.time()
        if self._use_pth:
            with torch.no_grad():
                x = torch.tensor(input_flat).unsqueeze(0).to(self._device)
                output = self._net(x).cpu().numpy().flatten()
        else:
            output = self._session.run(
                None, {'input': input_flat.reshape(1, -1)})[0].flatten()
        self._inference_times.append((time.time() - t0) * 1000)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)

        quat, dpos = drone_state.attitude, drone_state.global_pos
        w1 = quat.rotate(np.array(output[0:3])) + dpos
        w2 = quat.rotate(np.array(output[3:6])) + dpos
        int_wpts = np.array([[w1[0], w2[0]], [w1[1], w2[1]]])
        ts = np.clip(output[6:9], self._planner.T_min + 1e-3,
                     self._planner.T_max - 1e-3)
        return int_wpts, ts

    def _initial_guesses(self, depth_img, drone_state, head_state, tail_state):
        """Warm-start candidates, best first. The NN regressor yields exactly
        one; fm_inference_node overrides this with K ranked samples."""
        return [self._nn_initial_guess(depth_img, drone_state, head_state)]

    # ── Replan (LIGHTWEIGHT: max 2 attempts, deterministic) ──────────────────

    def _validate_planned_traj(self):
        """Hard post-check: reject a trajectory with clearance < hard_clearance
        (ESDF inflated by the drone radius). Closes the soft collision-cost gap.
        Adaptive threshold: if the drone is already closer than hard_clearance,
        do not reject a trajectory that takes it OUT."""
        cur_clear = float(self._esdf.get_edt_dis(
            self._drone_state.global_pos[:2]))
        thr = min(self._hard_clearance, max(cur_clear - 0.05, 0.1))
        for p in self._planner.get_pos_array():
            d = self._esdf.get_edt_dis(p[:2])
            if d < thr:
                raise ValueError(
                    f"Trajectory grazing: {d:.2f}m < hard {thr:.2f}m")

    def _replan(self):
        if not self._esdf.is_ready():
            self.get_logger().warn("[INF] ESDF not ready yet",
                                   throttle_duration_sec=2.0)
            return
        if self._escape_active:
            self._maybe_exit_escape()
            if self._escape_active:
                return
        if self._use_guards and self._check_no_progress():
            self.get_logger().warn("[INF] No progress — escape")
            self._enter_escape()
            return

        with self._depth_lock:
            if self._latest_depth is None:
                return
            lag = time.time() - self._latest_depth_time
            if lag > self._depth_max_lag:
                self.get_logger().warn(
                    f"[INF] Skip replan, depth lag {lag*1000:.0f}ms",
                    throttle_duration_sec=2.0)
                return
            depth_snapshot = self._latest_depth.copy()
            sync_state = DroneState()
            sync_state.copy_from(self._latest_depth_state)

        if np.linalg.norm(
                sync_state.global_pos[:2] - self._global_target) < 0.8:
            self._reached_target = True
            return

        # Max 2 attempts: normal target then short target. For each attempt,
        # warm_start_plan already retries 5x internally with perturbation.
        #
        # ADAPTIVE safe_dis (anti-deadlock): if the drone is already closer than
        # safe_dis to an obstacle, the (fixed) trajectory start point must violate
        # it -> the collision cost at the start alone already > tol -> all plans
        # fail and the drone hovers in place. Lower the reference to follow the
        # current clearance (-margin), floor hard_clearance.
        cur_clear = float(self._esdf.get_edt_dis(sync_state.global_pos[:2]))
        self._planner.safe_dis = float(np.clip(
            cur_clear - 0.05, self._hard_clearance + 0.05, self._safe_dis))
        ok = False
        for shorter in (False, True):
            self._set_local_target(shorter=shorter,
                                   ref_pos=sync_state.global_pos[:2])
            head = self._get_head_state()
            tail = np.zeros((3, 2))
            tail[0], tail[1] = self._target_state[0], self._target_state[1]
            t0 = time.time()
            try:
                guesses = self._initial_guesses(
                    depth_snapshot, sync_state, head, tail)
            except Exception:
                continue
            self._publish_candidate_markers(guesses, head, self._target_state[0])
            for cand_i, (int_wpts, ts) in enumerate(guesses):
                try:
                    self._planner.warm_start_plan(
                        self._esdf, head, tail, int_wpts, ts)
                    self._validate_planned_traj()
                except Exception:
                    continue
                self._replan_count += 1
                avg_inf = (np.mean(self._inference_times)
                           if self._inference_times else 0)
                cand_str = (f" cand#{cand_i + 1}/{len(guesses)}"
                            if len(guesses) > 1 else "")
                self.get_logger().info(
                    f"[INF] Replan ok #{self._replan_count} "
                    f"cost={self._planner.final_cost:.1f} "
                    f"t={(time.time()-t0)*1000:.0f}ms (nn={avg_inf:.1f}ms)"
                    f"{cand_str}{' short' if shorter else ''}",
                    throttle_duration_sec=1.0)
                self._publish_flown_marker()
                ok = True
                break
            if ok:
                break

        if not ok:
            self._replan_fail_count += 1
            self.get_logger().warn(
                f"[INF] Replan failed ({self._replan_fail_count}x)")
            if self._use_guards and self._replan_fail_count >= 2:
                self._enter_escape()
            return
        self._replan_fail_count = 0

        state_cmd = self._planner.get_full_state_cmd(hz=self._cmd_hz)
        self._install_trajectory(state_cmd, float(np.sum(self._planner.ts)))

    def _install_trajectory(self, state_cmd, base_total_time):
        peak = 0.0
        try:
            speeds = np.linalg.norm(state_cmd[:, 1, :], axis=1)
            peak = float(np.max(speeds)) if len(speeds) else 0.0
        except Exception:
            pass
        scale = peak / self._v_max if (self._v_max > 1e-3
                                       and peak > self._v_max) else 1.0
        with self._traj_lock:
            self._traj.t_start         = time.time()
            self._traj.state_cmd       = state_cmd
            self._traj.base_total_time = base_total_time
            self._traj.time_scale      = scale
            self._traj.total_time      = base_total_time * scale
            self._traj.cmd_hz          = self._cmd_hz

    # ── Status / util ────────────────────────────────────────────────────────

    def _print_status(self):
        pos = self._drone_state.global_pos
        speed = float(np.linalg.norm(self._drone_state.global_vel[:2]))
        esc = " [ESCAPE]" if self._escape_active else ""
        esdf_info = (f" | esdf={float(self._esdf.get_edt_dis(pos[:2])):.2f}"
                     if self._esdf.is_ready() else "")
        avg_inf = np.mean(self._inference_times) if self._inference_times else 0
        self.get_logger().info(
            f"[INF] {self._mission_state}{esc} | "
            f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}){esdf_info} | "
            f"speed={speed:.2f} | "
            f"goal=({self._global_target[0]:.1f},{self._global_target[1]:.1f})"
            f" | nn={avg_inf:.1f}ms")

    def _call_srv(self, client, request, timeout=8.0):
        fut = client.call_async(request)
        t0  = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                return None
            time.sleep(0.05)
        return fut.result()

    def _set_px4_param_int(self, name, value):
        if not self._param_client.wait_for_service(timeout_sec=5.0):
            return False
        req = ParamSetV2.Request()
        req.param_id = name
        pv = RclParameterValue()
        pv.type, pv.integer_value = ParameterType.PARAMETER_INTEGER, int(value)
        req.value = pv
        res = self._call_srv(self._param_client, req)
        return bool(res and res.success)

    def _set_px4_param_float(self, name, value):
        if not self._param_client.wait_for_service(timeout_sec=5.0):
            return False
        req = ParamSetV2.Request()
        req.param_id = name
        pv = RclParameterValue()
        pv.type, pv.double_value = ParameterType.PARAMETER_DOUBLE, float(value)
        req.value = pv
        res = self._call_srv(self._param_client, req)
        ok = bool(res and res.success)
        self.get_logger().info(
            f"[INF] {'ok' if ok else 'FAILED'} PX4 param {name} = {value}")
        return ok

    def _configure_octomap_band(self, cruise_z):
        """Occupancy band follows the actual cruise_z (important collision fix)."""
        occ_min = max(0.35, cruise_z - 0.7)
        occ_max = cruise_z + 1.0
        if not self._octo_param_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                f"[INF] octomap set_parameters absent — set manually: "
                f"occ_min_z~{occ_min:.2f} occ_max_z~{occ_max:.2f}")
            return
        req = SetParameters.Request()
        for name, val in (("occupancy_min_z", occ_min),
                          ("occupancy_max_z", occ_max)):
            p = RclParameter()
            p.name = name
            pv = RclParameterValue()
            pv.type, pv.double_value = ParameterType.PARAMETER_DOUBLE, float(val)
            p.value = pv
            req.parameters.append(p)
        res = self._call_srv(self._octo_param_client, req)
        ok  = bool(res and all(r.successful for r in res.results))
        self.get_logger().info(
            f"[INF] {'ok' if ok else 'FAILED'} Octomap band "
            f"[{occ_min:.2f}, {occ_max:.2f}]")
        if ok and self._octo_reset_client.wait_for_service(timeout_sec=3.0):
            self._call_srv(self._octo_reset_client, EmptySrv.Request(),
                           timeout=5.0)

    def _set_mode(self, mode):
        if not self._mode_client.wait_for_service(timeout_sec=3.0):
            return False
        req = SetMode.Request(); req.custom_mode = mode
        res = self._call_srv(self._mode_client, req)
        return bool(res and res.mode_sent)

    def _arm(self, value):
        if not self._arming_client.wait_for_service(timeout_sec=3.0):
            return False
        req = CommandBool.Request(); req.value = value
        res = self._call_srv(self._arming_client, req)
        return bool(res and res.success)

    def _wait_altitude(self, target_z, tol=0.2, timeout=30.0):
        t0, stable = time.time(), None
        while time.time() - t0 < timeout:
            z  = self._drone_state.global_pos[2]
            vz = abs(self._drone_state.global_vel[2])
            if abs(z - target_z) < tol and vz < 0.2:
                stable = stable or time.time()
                if time.time() - stable >= 2.0:
                    return True
            else:
                stable = None
            time.sleep(0.1)
        return False

    def _wait_ekf_stable(self, tol: float = 0.08, stable_dur: float = 3.0,
                         timeout: float = 45.0) -> float:
        """Wait for EKF Z to converge before takeoff (std < tol for stable_dur).
        Identical to expert_planner — required so ground_z is accurate."""
        self.get_logger().info(
            f"[INF] Waiting for EKF Z to stabilize (tol={tol}m, dur={stable_dur}s)...")
        t0 = time.time()
        z_history: list = []
        stable_start = None
        while time.time() - t0 < timeout:
            z_now = self._drone_state.global_pos[2]
            z_history.append(z_now)
            if len(z_history) > 50:
                z_history.pop(0)
            if len(z_history) >= 10:
                z_std = float(np.std(z_history[-10:]))
                if z_std < tol:
                    if stable_start is None:
                        stable_start = time.time()
                    elif time.time() - stable_start >= stable_dur:
                        z_ground = float(np.mean(z_history[-10:]))
                        self.get_logger().info(
                            f"[INF] ✓ EKF stable. ground_z={z_ground:.3f}m "
                            f"(std={z_std:.4f})")
                        return z_ground
                else:
                    stable_start = None
            elapsed = time.time() - t0
            if int(elapsed) % 5 == 0 and elapsed % 1.0 < 0.15:
                z_std_log = float(np.std(z_history[-10:])) if len(z_history) >= 10 else 999.0
                self.get_logger().info(
                    f"[INF] EKF Z={z_now:.3f}m std={z_std_log:.4f} "
                    f"({elapsed:.0f}s/{timeout:.0f}s)...")
            time.sleep(0.1)
        z_ground = float(np.mean(z_history[-10:])) if z_history else 0.0
        self.get_logger().warn(
            f"[INF] EKF stable timeout — using ground_z={z_ground:.3f}m")
        return z_ground

    def _warm_up(self):
        dummy = np.zeros(IMG_WIDTH * IMG_HEIGHT + MOTION_INPUT_SIZE,
                         dtype=np.float32)
        if self._use_pth:
            with torch.no_grad():
                x = torch.tensor(dummy).unsqueeze(0).to(self._device)
                for _ in range(3):
                    self._net(x)
        else:
            for _ in range(3):
                self._session.run(None, {'input': dummy.reshape(1, -1)})
        time.sleep(50 / self._cmd_hz + 0.5)

    # ── Mission sequence ─────────────────────────────────────────────────────

    def run_sequence(self):
        self.get_logger().info("[INF] Waiting for MAVROS...")
        t0 = time.time()
        while not self._connected:
            if time.time() - t0 > self._conn_timeout:
                rclpy.shutdown(); return
            time.sleep(0.5)

        self.get_logger().info("[INF] Waiting for depth + ESDF...")
        t0 = time.time()
        while self._latest_depth is None or not self._esdf.is_ready():
            if time.time() - t0 > 60.0:
                self.get_logger().error("[INF] Depth/ESDF timeout!")
                rclpy.shutdown(); return
            time.sleep(0.5)

        # Force the EKF to use GPS as the height reference (not the drifting SITL
        # barometer). Wait 2s so PX4 processes the parameter before calibration.
        self._set_px4_param_int("EKF2_HGT_REF", 1)

        # Cap the PX4 horizontal speed (same as expert _configure_px4_height_ref):
        # v_max only lives in the planner — without the MPC caps PX4 chases a
        # position error at up to ~12 m/s (escape/resume near obstacles) and a
        # Gazebo contact impulse then triggers a PHYSICS EXPLOSION.
        vmax_phys = self._v_max * 1.5
        self._set_px4_param_float("MPC_XY_VEL_MAX", vmax_phys)
        self._set_px4_param_float("MPC_XY_CRUISE",  self._v_max)
        self._set_px4_param_float("MPC_XY_VEL_ALL", vmax_phys)
        time.sleep(2.0)

        # Wait for EKF Z to converge then record ground_z as the height reference.
        # Without this, cruise_z = target_alt assumes ground=0.0m, but the EKF can
        # have an offset of a few decimeters → the physical drone flies lower.
        ground_z = self._wait_ekf_stable()
        if abs(ground_z) > 2.0:
            self.get_logger().warn(
                f"[INF] ground_z unreasonable ({ground_z:.2f}m) — using 0.0m")
            ground_z = 0.0

        self.get_logger().info(
            f"[INF] ground_z={ground_z:.3f}m → cruise target="
            f"{ground_z + self._alt:.3f}m (physical ≈{self._alt:.1f}m)")

        # Set cruise_z BEFORE warm-up so the pre-ARM setpoint is already correct.
        # The ground_z offset ensures physical = target_alt even if the EKF frame is non-zero.
        self._cruise_z = ground_z + self._alt

        self._configure_octomap_band(self._alt)
        self._warm_up()

        self.get_logger().info("[INF] ARM...")
        t0, armed = time.time(), False
        while time.time() - t0 < self._arm_timeout:
            if self._arm(True):
                time.sleep(0.8)
                if self._armed:
                    armed = True; break
            time.sleep(2.0)
        if not armed:
            rclpy.shutdown(); return

        if not self._set_mode("OFFBOARD"):
            self._arm(False); rclpy.shutdown(); return
        t0 = time.time()
        while self._mode != "OFFBOARD" and time.time() - t0 < 5.0:
            time.sleep(0.1)

        self._mission_state = self.STATE_TAKEOFF
        takeoff_target      = self._cruise_z
        self.get_logger().info(f"[INF] Takeoff z={takeoff_target:.2f}m (physical ≈{self._alt:.1f}m)...")
        if not self._wait_altitude(takeoff_target):
            self.get_logger().warn("[INF] Takeoff not stable — continuing")
        time.sleep(1.0)

        self._mission_state = self.STATE_FLYING
        self._reset_progress()
        self.get_logger().info(
            f"[INF] Start -> goal=({self._global_target[0]:.1f},"
            f"{self._global_target[1]:.1f})")

        last_replan, direction = 0.0, 1
        while rclpy.ok():
            now = time.time()
            if now - last_replan >= self._replan_period:
                last_replan = now
                self._replan()
            dist = np.linalg.norm(
                self._drone_state.global_pos[:2] - self._global_target)
            if not self._escape_active and (dist < 1.5 or self._reached_target):
                if not self._auto_reverse:
                    self.get_logger().info("[INF] ok Goal reached!")
                    break
                self._reached_target = False
                direction = -direction
                self._global_target = (np.array([0.0, 0.0]) if direction < 0
                                       else np.array([self._goal_x, 0.0]))
                self._reset_progress()
                with self._traj_lock:
                    self._traj.invalidate()
                self.get_logger().info(
                    f"[INF] New goal: ({self._global_target[0]:.1f},"
                    f"{self._global_target[1]:.1f})")
            time.sleep(0.05)

        if not rclpy.ok():
            return

        self._mission_state = self.STATE_LANDING
        self._cruise_z      = None
        self.get_logger().info("[INF] Landing...")
        land_floor = ground_z + 0.25
        curr_z = self._drone_state.global_pos[2]
        while curr_z > land_floor and rclpy.ok():
            curr_z = max(ground_z + 0.05, curr_z - 0.08)
            sp = PositionTarget()
            sp.header.stamp     = self.get_clock().now().to_msg()
            sp.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
            sp.type_mask = (
                PositionTarget.IGNORE_VX  | PositionTarget.IGNORE_VY |
                PositionTarget.IGNORE_VZ  | PositionTarget.IGNORE_AFX |
                PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                PositionTarget.IGNORE_YAW_RATE)
            pos = self._drone_state.global_pos
            sp.position.x = float(pos[0]); sp.position.y = float(pos[1])
            sp.position.z = float(curr_z)
            self._pub_sp.publish(sp)
            time.sleep(0.1)

        time.sleep(1.5)
        try:
            self._arm(False)
            self._set_mode("MANUAL")
        except Exception:
            pass
        self._mission_state = self.STATE_DONE
        self.get_logger().info("[INF] DONE ok")
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = NNInferenceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    seq = threading.Thread(target=node.run_sequence, daemon=True)
    seq.start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        try:
            node._arm(False)
        except Exception:
            pass
    finally:
        executor.shutdown()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
