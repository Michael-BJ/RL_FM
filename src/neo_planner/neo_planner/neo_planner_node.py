#!/usr/bin/env python3
"""
neo_planner_node.py  (v2 — NN initial guess + MinJerk optimizer)
=================================================================
INFERENCE node for NEO-Planner deployment.

This is a MIRROR of expert_planner_node.py v25 (which has been tested), with
ONE core change in the replan step:

  Expert (data collection):
      self._planner.batch_plan(esdf, head_state, tail_state)
      # 3 initial trajectories from batch sampling -> pick the lowest cost

  NEO-Planner (inference):
      int_wpts, ts = NN(depth, state, init, target)   # initial guess
      self._planner.warm_start_plan(esdf, head_state, tail_state, int_wpts, ts)
      # 1 initial trajectory from the NN -> optimizer refines using the ESDF

Everything else (ESDF, MINCO optimizer, escape mode, EKF stable wait, IGNORE_VZ,
discontinuity detection, online replanning dTf) is EXACTLY the same as v25.

IMPORTANT: reuse the existing modules -- esdf_ros2.py & min_jerk_planner.py.
           There is NO duplicated ESDF/MinJerkPlanner class here anymore.

Auto-detect model format:
  .pth  -> PyTorch inference
  .onnx -> ONNX Runtime inference
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
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
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

# Reuse the existing modules (DO NOT duplicate)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from esdf_ros2 import ESDF
from min_jerk_planner import MinJerkPlanner, PlannerConfig

# Model loading
import torch
import torch.nn as nn
import torchvision.models as models

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# Constants (same as v25)
IMG_WIDTH         = 640
IMG_HEIGHT        = 480
MOTION_INPUT_SIZE = 24
OUTPUT_SIZE       = 9
IMG_FEATURE_SIZE  = 24
MOTION_FEATURE_SIZE = 24

DISCONTINUITY_THRESHOLD  = 3.0
EMERGENCY_STOP_THRESHOLD = 3.0
STUCK_SPEED_THRESHOLD    = 0.15
STUCK_TIME_THRESHOLD     = 5.0
EMERGENCY_STOP_PERSIST_TIME = 2.0
# Detect "no progress to the goal": if the distance to the goal does not decrease
# by at least NO_PROGRESS_DELTA meters within NO_PROGRESS_TIME seconds -> escape.
# Robust against small jitter that resets the speed-based stuck detection.
NO_PROGRESS_TIME  = 8.0
NO_PROGRESS_DELTA = 0.8

ESCAPE_VELOCITY   = 0.4
ESCAPE_PROBE_DIST = 1.5
ESCAPE_EXIT_DIST  = 1.2
ESCAPE_MAX_TIME   = 15.0

DEPTH_MAX_LAG_S = 0.3

# Look-ahead collision guard: check trajectory points 0.3..1.5 s ahead against
# the ESDF. The map can change AFTER the trajectory is created (octomap keeps
# updating), so an old trajectory must be cancelled as soon as the new map shows
# an obstacle in its path — without waiting for the next replan.
LOOKAHEAD_HORIZON_S    = 1.5
LOOKAHEAD_STEP_S       = 0.3
LOOKAHEAD_CHECK_PERIOD = 10   # check every N publish ticks (10 Hz at cmd_hz=100)

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


class PlannerNet(nn.Module):
    """Architecture IDENTICAL to nn_trainer.py."""
    def __init__(self):
        super().__init__()
        self.img_backbone = models.resnet18(weights='DEFAULT')
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        self.img_backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.img_backbone.fc = nn.Linear(
            self.img_backbone.fc.in_features, IMG_FEATURE_SIZE)
        self.motion_backbone = nn.Sequential(
            nn.Linear(MOTION_INPUT_SIZE, 48), nn.LeakyReLU(),
            nn.Linear(48, 24), nn.LeakyReLU(),
            nn.Linear(24, 24), nn.LeakyReLU(),
            nn.Linear(24, MOTION_FEATURE_SIZE),
        )
        self.mlp = nn.Sequential(
            nn.Linear(IMG_FEATURE_SIZE + MOTION_FEATURE_SIZE, 48), nn.LeakyReLU(),
            nn.Linear(48, 96), nn.LeakyReLU(),
            nn.Linear(96, 96), nn.LeakyReLU(),
            nn.Linear(96, OUTPUT_SIZE),
        )

    def forward(self, x):
        img = x[:, :IMG_WIDTH * IMG_HEIGHT].reshape(-1, 1, IMG_HEIGHT, IMG_WIDTH)
        vector = x[:, IMG_WIDTH * IMG_HEIGHT:]
        img_feature = self.img_backbone(img)
        motion_feature = self.motion_backbone(vector)
        fused = torch.cat([img_feature, motion_feature], dim=1)
        return self.mlp(fused)


class DroneState:
    def __init__(self):
        self.global_pos = np.zeros(3)
        self.global_vel = np.zeros(3)
        self.local_vel  = np.zeros(3)
        self.attitude   = Quaternion()
        self.yaw        = 0.0

    def copy_from(self, other):
        self.global_pos = other.global_pos.copy()
        self.global_vel = other.global_vel.copy()
        self.local_vel  = other.local_vel.copy()
        self.attitude   = other.attitude
        self.yaw        = other.yaw


class TrajectorySegment:
    def __init__(self):
        self.t_start         = 0.0
        self.state_cmd       = None
        self.cmd_hz          = 100
        self.base_total_time = 0.0   # ORIGINAL trajectory duration (before time-scale)
        self.time_scale      = 1.0   # >1 = trajectory slowed down so v <= v_max
        self.total_time      = 0.0   # = base_total_time * time_scale

    def is_valid(self):
        return self.state_cmd is not None and len(self.state_cmd) > 0

    def time_remaining(self, t_now):
        if not self.is_valid():
            return 0.0
        return max(0.0, self.total_time - (t_now - self.t_start))

    def invalidate(self):
        self.state_cmd = None

    def sample_at(self, t_now):
        """
        Sampling with TIME-SCALING. If the base trajectory has a peak speed >
        v_max, time_scale > 1 slows it down: the spatial path is exactly the
        same, just traversed over a longer time. Velocity is divided by
        time_scale, acceleration by time_scale^2.
        """
        if not self.is_valid():
            return None, None, None
        elapsed = t_now - self.t_start
        if elapsed < 0:
            elapsed = 0.0
        if elapsed >= self.total_time:
            last = self.state_cmd[-1]
            return last[0], np.zeros(2), np.zeros(2)
        k = self.time_scale if self.time_scale > 1e-6 else 1.0
        orig_t = elapsed / k                  # time on the original trajectory
        idx_float = orig_t * self.cmd_hz
        idx_low   = int(idx_float)
        idx_high  = min(idx_low + 1, len(self.state_cmd) - 1)
        alpha     = idx_float - idx_low
        s_low  = self.state_cmd[idx_low]
        s_high = self.state_cmd[idx_high]
        pos = s_low[0] * (1 - alpha) + s_high[0] * alpha
        vel = (s_low[1] * (1 - alpha) + s_high[1] * alpha) / k
        acc = (s_low[2] * (1 - alpha) + s_high[2] * alpha) / (k * k)
        return pos, vel, acc


class NEOPlannerNode(Node):
    STATE_IDLE    = "IDLE"
    STATE_TAKEOFF = "TAKEOFF"
    STATE_FLYING  = "FLYING"
    STATE_LANDING = "LANDING"
    STATE_DONE    = "DONE"

    def __init__(self):
        super().__init__("neo_planner_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("goal_x",          20.0)
        self.declare_parameter("target_alt",       2.0)
        self.declare_parameter("replan_period",    1.0)
        self.declare_parameter("cmd_hz",         100)
        self.declare_parameter("longitu_step",     5.0)
        self.declare_parameter("lateral_step",     1.0)
        self.declare_parameter("v_max",            1.0)
        self.declare_parameter("arm_timeout",     30.0)
        self.declare_parameter("connect_timeout", 30.0)
        self.declare_parameter("safe_dis",         0.7)
        self.declare_parameter("safe_dis_emergency", 0.4)
        self.declare_parameter("collision_cost_tol", 1.0)
        # Hard post-check: the optimizer's trajectory is REJECTED if any point has
        # an ESDF distance < hard_clearance. Closes the gap of the soft collision
        # cost that can still let a grazing trajectory pass below the tolerance.
        # The ESDF is already inflated by the drone radius, so this is BODY clearance.
        self.declare_parameter("hard_clearance",   0.35)
        self.declare_parameter("planning_time_ahead", 1.0)
        self.declare_parameter("auto_reverse",    True)
        self.declare_parameter("use_ekf_stable",  False)

        self._model_path    = str(self.get_parameter("model_path").value)
        self._goal_x        = float(self.get_parameter("goal_x").value)
        self._alt           = float(self.get_parameter("target_alt").value)
        self._replan_period = float(self.get_parameter("replan_period").value)
        self._cmd_hz        = int(self.get_parameter("cmd_hz").value)
        self._longitu_step  = float(self.get_parameter("longitu_step").value)
        self._lateral_step  = float(self.get_parameter("lateral_step").value)
        self._v_max         = float(self.get_parameter("v_max").value)
        self._arm_timeout   = float(self.get_parameter("arm_timeout").value)
        self._conn_timeout  = float(self.get_parameter("connect_timeout").value)
        self._safe_dis_norm = float(self.get_parameter("safe_dis").value)
        self._safe_dis_emer = float(self.get_parameter("safe_dis_emergency").value)
        self._coll_tol      = float(self.get_parameter("collision_cost_tol").value)
        self._hard_clearance = float(self.get_parameter("hard_clearance").value)
        self._dt_ahead      = float(self.get_parameter("planning_time_ahead").value)
        self._auto_reverse   = bool(self.get_parameter("auto_reverse").value)
        self._use_ekf_stable = bool(self.get_parameter("use_ekf_stable").value)

        if not os.path.isfile(self._model_path):
            self.get_logger().error(f"Model not found: {self._model_path}")
            raise FileNotFoundError(self._model_path)

        self._use_pth = self._model_path.endswith('.pth')
        self._device  = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if self._use_pth:
            self._net = PlannerNet().to(self._device)
            self._net.load_state_dict(
                torch.load(self._model_path, map_location=self._device))
            self._net.eval()
            backend_str = f"PyTorch ({self._device})"
        else:
            if not HAS_ONNX:
                raise RuntimeError("onnxruntime not installed")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self._session = ort.InferenceSession(
                self._model_path, providers=providers)
            backend_str = f"ONNX ({self._session.get_providers()})"

        # Planner config (SAME as expert v25)
        cfg = PlannerConfig()
        cfg.v_max              = self._v_max
        cfg.T_min              = 0.5
        cfg.T_max              = 5.0
        cfg.safe_dis           = self._safe_dis_norm
        cfg.init_wpts_num      = 2
        cfg.init_T             = 2.5
        cfg.delta_t            = 0.1
        cfg.collision_cost_tol = self._coll_tol
        cfg.opt_tol            = 1e-2
        cfg.weights            = [1.0, 1.0, 1.0, 10000.0]
        self._planner          = MinJerkPlanner(cfg)

        self._esdf = ESDF()

        self._connected   = False
        self._armed       = False
        self._mode        = ""
        self._drone_state = DroneState()
        self._ground_z    = 0.0
        self._cruise_z    = None

        self._traj        = TrajectorySegment()
        self._traj.cmd_hz = self._cmd_hz
        self._traj_lock   = threading.Lock()

        self._global_target      = np.array([self._goal_x, 0.0])
        self._target_state       = np.zeros((2, 2))
        self._near_global_target = False
        self._reached_target     = False
        self._mission_state      = self.STATE_IDLE

        self._replan_fail_count  = 0
        self._stuck_since        = None
        self._last_pos_for_stuck = None
        self._escape_active      = False
        self._escape_velocity_xy = np.zeros(2)
        self._escape_start_time  = None
        self._progress_ref_dist  = None   # best distance to the goal so far
        self._progress_ref_time  = None
        self._emergency_stop_first_t = None
        self._emergency_stop_lock    = threading.Lock()
        self._lookahead_tick         = 0

        self._latest_depth             = None
        self._latest_depth_drone_state = None
        self._latest_depth_timestamp   = None
        self._bridge                   = CvBridge()
        self._depth_lock               = threading.Lock()
        self._lag_skip_count = 0

        self._inference_times = []
        self._planning_times  = []
        self._replan_count    = 0

        self._pub_sp = self.create_publisher(
            PositionTarget, "/mavros/setpoint_raw/local", 10)
        self.create_subscription(String, "/px4/state",   self._cb_state,   10)
        self.create_subscription(String, "/px4/sensors", self._cb_sensors, 10)
        self.create_subscription(
            Image, "/realsense/depth/float32", self._cb_depth, _SENSOR_QOS)
        self.create_subscription(
            OccupancyGrid, "/projected_map", self._esdf.occupancy_map_cb, 10)

        self._arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self._mode_client   = self.create_client(SetMode,     "/mavros/set_mode")
        self._param_client  = self.create_client(ParamSetV2,  "/mavros/param/set")
        # FIX COLLISION (inference): client to set the octomap band dynamically
        self._octo_param_client = self.create_client(
            SetParameters, "/octomap_server_unknown/set_parameters")
        self._octo_reset_client = self.create_client(
            EmptySrv, "/octomap_server_unknown/reset")

        # FIX ALTITUDE: GPS-AGL trim state
        self._gps_alt_buf     = []
        self._gps_alt_home    = None
        self._cruise_z_base   = None

        self._cmd_timer    = self.create_timer(1.0 / self._cmd_hz, self._publish_cmd)
        self._status_timer = self.create_timer(5.0, self._print_status)
        self._alt_trim_timer = self.create_timer(2.0, self._altitude_trim_cb)

        self._print_header(backend_str)

    def _print_header(self, backend_str):
        self.get_logger().info("=" * 62)
        self.get_logger().info("  NEO-Planner Inference v2 (NN + MinJerk + ESDF)")
        self.get_logger().info("=" * 62)
        self.get_logger().info(f"  Model         : {os.path.basename(self._model_path)}")
        self.get_logger().info(f"  Backend       : {backend_str}")
        self.get_logger().info(f"  v_max         : {self._v_max} m/s")
        self.get_logger().info(f"  Alt           : {self._alt} m")
        self.get_logger().info(f"  Goal X        : {self._goal_x} m")
        self.get_logger().info(f"  safe_dis      : {self._safe_dis_norm} m")
        self.get_logger().info(f"  coll_tol      : {self._coll_tol}")
        self.get_logger().info(f"  hard_clearance: {self._hard_clearance} m")
        self.get_logger().info(f"  Replan        : {self._replan_period} s")
        self.get_logger().info(f"  dTf (ahead)   : {self._dt_ahead} s")
        self.get_logger().info("=" * 62)

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
            x   = float(d.get("local_x", 0)); y = float(d.get("local_y", 0))
            z   = float(d.get("local_z", 0))
            vx  = float(d.get("vel_x",  0)); vy = float(d.get("vel_y",  0))
            vz  = float(d.get("vel_z",  0)); yaw = float(d.get("yaw", 0))
            self._drone_state.global_pos = np.array([x, y, z])
            self._drone_state.global_vel = np.array([vx, vy, vz])
            self._drone_state.yaw        = yaw
            qw = math.cos(yaw / 2); qz = math.sin(yaw / 2)
            self._drone_state.attitude = Quaternion(qw, 0.0, 0.0, qz)
            self._drone_state.local_vel = (
                self._drone_state.attitude.inverse.rotate(
                    self._drone_state.global_vel))
            # FIX ALTITUDE: raw GPS altitude (not EKF) for physical AGL
            gps = d.get("gps")
            if gps and gps.get("altitude_m") is not None:
                alt = float(gps["altitude_m"])
                if math.isfinite(alt):
                    self._gps_alt_buf.append(alt)
                    if len(self._gps_alt_buf) > 20:
                        self._gps_alt_buf.pop(0)
        except Exception:
            pass

    def _cb_depth(self, msg):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            if depth.shape != (IMG_HEIGHT, IMG_WIDTH):
                depth = cv2.resize(depth, (IMG_WIDTH, IMG_HEIGHT))
            ds_snapshot = DroneState()
            ds_snapshot.copy_from(self._drone_state)
            now = time.time()
            with self._depth_lock:
                self._latest_depth             = depth.copy()
                self._latest_depth_drone_state = ds_snapshot
                self._latest_depth_timestamp   = now
        except Exception:
            pass

    def _publish_cmd(self):
        msg = PositionTarget()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = "map"
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = (
            PositionTarget.IGNORE_VZ   |
            PositionTarget.IGNORE_AFX  |
            PositionTarget.IGNORE_AFY  |
            PositionTarget.IGNORE_AFZ  |
            PositionTarget.IGNORE_YAW_RATE
        )

        if (self._mission_state == self.STATE_FLYING
                and self._escape_active
                and self._cruise_z is not None):
            drone_pos  = self._drone_state.global_pos[:2]
            target_pos = drone_pos + self._escape_velocity_xy * 0.5
            msg.position.x = float(target_pos[0])
            msg.position.y = float(target_pos[1])
            msg.position.z = float(self._cruise_z)
            msg.velocity.x = float(self._escape_velocity_xy[0])
            msg.velocity.y = float(self._escape_velocity_xy[1])
            msg.velocity.z = 0.0
            vel_xy = self._escape_velocity_xy
            if abs(vel_xy[0]) > 0.05 or abs(vel_xy[1]) > 0.05:
                msg.yaw = math.atan2(vel_xy[1], vel_xy[0])
            else:
                msg.yaw = self._drone_state.yaw
            self._pub_sp.publish(msg)
            return

        if (self._mission_state == self.STATE_FLYING
                and self._cruise_z is not None):
            t_now = time.time()
            with self._traj_lock:
                pos, vel, acc = self._traj.sample_at(t_now)
            if pos is None:
                pos = self._drone_state.global_pos[:2]
                vel = np.zeros(2)

            drone_pos = self._drone_state.global_pos[:2]

            # Real-time collision guard: if the ESDF is ready and the drone is too
            # close to an obstacle, cancel the trajectory and enter escape now —
            # no need to wait for the next replan (which could be 1 second away).
            if (self._esdf.is_ready()
                    and not self._escape_active
                    and self._esdf.get_edt_dis(drone_pos) < self._safe_dis_emer):
                self.get_logger().warn(
                    "[NEO] Collision guard! ESDF too close — cancel traj",
                    throttle_duration_sec=0.5)
                with self._traj_lock:
                    self._traj.invalidate()
                self._enter_escape_safe()
                pos = drone_pos
                vel = np.zeros(2)

            # Look-ahead guard: check trajectory points 0.3..1.5s AHEAD against
            # the latest ESDF. The trajectory is validated at planning time, but
            # the map keeps updating — a new obstacle on the path must cancel the
            # trajectory now, not wait for a replan/until the drone itself enters
            # the danger zone.
            self._lookahead_tick += 1
            if (self._esdf.is_ready()
                    and not self._escape_active
                    and self._lookahead_tick % LOOKAHEAD_CHECK_PERIOD == 0):
                hit_dist = None
                with self._traj_lock:
                    if self._traj.is_valid():
                        for dt in np.arange(LOOKAHEAD_STEP_S,
                                            LOOKAHEAD_HORIZON_S + 1e-6,
                                            LOOKAHEAD_STEP_S):
                            p_ahead, _, _ = self._traj.sample_at(t_now + dt)
                            if p_ahead is None:
                                break
                            d_ahead = self._esdf.get_edt_dis(p_ahead)
                            if d_ahead < self._safe_dis_emer:
                                hit_dist = d_ahead
                                break
                if hit_dist is not None:
                    self.get_logger().warn(
                        f"[NEO] Look-ahead guard! Trajectory ahead "
                        f"clearance {hit_dist:.2f}m — cancel traj",
                        throttle_duration_sec=0.5)
                    with self._traj_lock:
                        self._traj.invalidate()
                    self._enter_escape_safe()
                    pos = drone_pos
                    vel = np.zeros(2)

            gap = np.linalg.norm(pos - drone_pos)
            if gap > EMERGENCY_STOP_THRESHOLD:
                with self._emergency_stop_lock:
                    if self._emergency_stop_first_t is None:
                        self._emergency_stop_first_t = t_now
                duration = t_now - self._emergency_stop_first_t
                self.get_logger().warn(
                    f"[NEO] Emergency stop! gap {gap:.1f}m (persist={duration:.1f}s)",
                    throttle_duration_sec=1.0)
                if duration > EMERGENCY_STOP_PERSIST_TIME:
                    with self._emergency_stop_lock:
                        self._emergency_stop_first_t = None
                    with self._traj_lock:
                        self._traj.invalidate()
                    self._enter_escape_safe()
                    pos = drone_pos
                    vel = np.zeros(2)
            else:
                with self._emergency_stop_lock:
                    self._emergency_stop_first_t = None

            # Safety net: clamp velocity to v_max (time-scaling already handles
            # most of it; this is a safety net against any leftover spikes).
            v_norm = float(np.linalg.norm(vel))
            if v_norm > self._v_max:
                vel = vel * (self._v_max / v_norm)

            msg.position.x = float(pos[0])
            msg.position.y = float(pos[1])
            msg.position.z = float(self._cruise_z)
            msg.velocity.x = float(vel[0])
            msg.velocity.y = float(vel[1])
            msg.velocity.z = 0.0
            if abs(vel[0]) > 0.05 or abs(vel[1]) > 0.05:
                msg.yaw = math.atan2(vel[1], vel[0])
            else:
                msg.yaw = self._drone_state.yaw
        else:
            pos      = self._drone_state.global_pos
            target_z = self._cruise_z if self._cruise_z is not None else self._alt
            msg.position.x = float(pos[0])
            msg.position.y = float(pos[1])
            msg.position.z = float(target_z)
            msg.velocity.x = 0.0
            msg.velocity.y = 0.0
            msg.velocity.z = 0.0
            msg.yaw        = self._drone_state.yaw

        self._pub_sp.publish(msg)

    def _enter_escape_safe(self):
        if self._escape_active:
            return
        if not self._esdf.is_ready():
            return
        self._enter_escape()

    def _find_escape_direction(self):
        drone_pos  = self._drone_state.global_pos[:2]
        global_tgt = self._global_target
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        best_score = -np.inf
        best_dir   = np.array([1.0, 0.0])
        to_goal = global_tgt - drone_pos
        to_goal_norm = np.linalg.norm(to_goal)
        to_goal_dir = (to_goal / to_goal_norm if to_goal_norm > 1e-3
                       else np.array([1.0, 0.0]))
        for angle in angles:
            dir_vec  = np.array([np.cos(angle), np.sin(angle)])
            probe_pt = drone_pos + dir_vec * ESCAPE_PROBE_DIST
            esdf_dist = self._esdf.get_edt_dis(probe_pt)
            mid_pt   = drone_pos + dir_vec * (ESCAPE_PROBE_DIST * 0.5)
            mid_dist = self._esdf.get_edt_dis(mid_pt)
            goal_bias = float(dir_vec @ to_goal_dir)
            score = min(esdf_dist, mid_dist) + 0.2 * goal_bias
            if score > best_score:
                best_score = score
                best_dir   = dir_vec
        return best_dir

    def _check_stuck(self):
        speed = float(np.linalg.norm(self._drone_state.global_vel[:2]))
        pos = self._drone_state.global_pos[:2].copy()
        if speed > STUCK_SPEED_THRESHOLD:
            self._stuck_since = None
            self._last_pos_for_stuck = None
            return False
        if self._stuck_since is None:
            self._stuck_since = time.time()
            self._last_pos_for_stuck = pos
            return False
        if self._last_pos_for_stuck is not None:
            drift = np.linalg.norm(pos - self._last_pos_for_stuck)
            if drift > 0.5:
                self._stuck_since = time.time()
                self._last_pos_for_stuck = pos
                return False
        return (time.time() - self._stuck_since) > STUCK_TIME_THRESHOLD

    def _reset_progress(self):
        self._progress_ref_dist = None
        self._progress_ref_time = None

    def _check_no_progress(self):
        """
        Stuck based on PROGRESS TO THE GOAL (not speed). Robust against small
        jitter: as long as the distance to the goal does not decrease by >=
        NO_PROGRESS_DELTA within NO_PROGRESS_TIME seconds, consider it trapped -> escape.
        """
        now = time.time()
        pos = self._drone_state.global_pos[:2]
        d = float(np.linalg.norm(pos - self._global_target))
        if self._progress_ref_dist is None:
            self._progress_ref_dist = d
            self._progress_ref_time = now
            return False
        if d < self._progress_ref_dist - NO_PROGRESS_DELTA:
            # Real progress toward the goal -> reset the reference
            self._progress_ref_dist = d
            self._progress_ref_time = now
            return False
        return (now - self._progress_ref_time) > NO_PROGRESS_TIME

    def _enter_escape(self):
        escape_dir = self._find_escape_direction()
        self._escape_velocity_xy = escape_dir * ESCAPE_VELOCITY
        self._escape_active      = True
        self._escape_start_time  = time.time()
        self._replan_fail_count  = 0
        self._reset_progress()
        with self._traj_lock:
            self._traj.invalidate()
        with self._emergency_stop_lock:
            self._emergency_stop_first_t = None
        self.get_logger().warn(
            f"[NEO] >>> ESCAPE: vel=({escape_dir[0]:.2f},{escape_dir[1]:.2f})")

    def _exit_escape(self):
        if self._escape_active:
            self.get_logger().info("[NEO] <<< Exit escape mode")
            self._escape_active = False
            self._escape_start_time = None
            self._stuck_since = None

    def _should_exit_escape(self):
        if not self._escape_active:
            return False
        if (self._escape_start_time is not None
                and time.time() - self._escape_start_time > ESCAPE_MAX_TIME):
            self.get_logger().warn("[NEO] Escape timeout — exit")
            return True
        drone_pos = self._drone_state.global_pos[:2]
        esdf_dist = self._esdf.get_edt_dis(drone_pos)
        if esdf_dist > ESCAPE_EXIT_DIST:
            return True
        future_pt = drone_pos + self._escape_velocity_xy * 1.0
        if self._esdf.get_edt_dis(future_pt) < esdf_dist * 0.8:
            new_dir = self._find_escape_direction()
            self._escape_velocity_xy = new_dir * ESCAPE_VELOCITY
        return False

    def _set_local_target(self, seed=0, shorter=False, ref_pos=None):
        if ref_pos is None:
            curr_pos = self._drone_state.global_pos[:2]
        else:
            curr_pos = ref_pos
        global_tgt = self._global_target
        step = 2.0 if shorter else self._longitu_step
        dist = np.linalg.norm(global_tgt - curr_pos)
        if dist < step:
            self._target_state[0] = global_tgt
            self._target_state[1] = np.zeros(2)
            self._near_global_target = True
            return
        self._near_global_target = False
        fwd = global_tgt - curr_pos
        fwd = fwd / np.linalg.norm(fwd)
        lat = np.array([[fwd[1], -fwd[0]], [-fwd[1], fwd[0]]])
        if seed > 0:
            local_tgt = curr_pos + step * fwd + np.random.normal(0, 1, 2)
        else:
            local_tgt = curr_pos + step * fwd
        lat_flag = 0
        lat_move = self._lateral_step
        n_iter = 0
        while (self._esdf.is_ready()
               and self._esdf.has_collision(local_tgt)
               and n_iter < 8):
            local_tgt += lat_move * lat[lat_flag]
            lat_flag = 1 - lat_flag
            lat_move += self._lateral_step
            n_iter += 1
        goal_dir = global_tgt - local_tgt
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-3:
            goal_dir = goal_dir / goal_norm
        self._target_state[0] = local_tgt
        self._target_state[1] = goal_dir * self._v_max * 0.8

    def _get_drone_state_at(self, t_future):
        t_now = time.time()
        with self._traj_lock:
            if self._traj.is_valid():
                remaining = self._traj.time_remaining(t_now)
                traj_pos, traj_vel, _ = self._traj.sample_at(t_now + t_future)
            else:
                remaining = 0.0
                traj_pos, traj_vel = None, None
        state = DroneState()
        state.attitude  = self._drone_state.attitude
        state.local_vel = self._drone_state.local_vel
        state.yaw       = self._drone_state.yaw
        use_traj = False
        if traj_pos is not None and remaining > t_future + 0.3:
            gap = np.linalg.norm(traj_pos[:2] - self._drone_state.global_pos[:2])
            if gap < DISCONTINUITY_THRESHOLD:
                use_traj = True
        if use_traj:
            state.global_pos = np.append(traj_pos, self._cruise_z or self._alt)
            state.global_vel = np.append(traj_vel, 0.0)
        else:
            state.global_pos = self._drone_state.global_pos.copy()
            state.global_vel = self._drone_state.global_vel.copy()
            speed = float(np.linalg.norm(state.global_vel[:2]))
            if speed > self._v_max:
                state.global_vel[:2] *= (self._v_max / speed)
            if speed < 0.1:
                state.global_vel[:2] = 0.0
            state.global_pos[:2] += state.global_vel[:2] * t_future
        return state

    def _form_nn_input(self, depth_img, drone_state, plan_init_state, target_state):
        """IDENTICAL to expert v25 _form_nn_input."""
        valid = depth_img[np.isfinite(depth_img) & (depth_img > 0)]
        if valid.size > 0:
            max_d = float(valid.max())
            depth_norm = np.clip(depth_img / (max_d + 1e-6) * 255, 0, 255)
        else:
            depth_norm = np.zeros_like(depth_img)
        depth_u8 = depth_norm.astype(np.uint8)

        quat = drone_state.attitude
        drone_local_vel = drone_state.local_vel
        drone_R = quat.rotation_matrix.reshape(-1)

        init_pos_3d = np.array([plan_init_state.global_pos[0],
                                plan_init_state.global_pos[1],
                                self._cruise_z or self._alt])
        init_vel_3d = np.array([plan_init_state.global_vel[0],
                                plan_init_state.global_vel[1], 0.0])
        init_pos_body = quat.inverse.rotate(init_pos_3d - drone_state.global_pos)
        init_vel_body = quat.inverse.rotate(init_vel_3d - drone_state.global_vel)

        tgt_pos_3d = np.array([target_state[0, 0], target_state[0, 1],
                               self._cruise_z or self._alt])
        tgt_vel_3d = np.array([target_state[1, 0], target_state[1, 1], 0.0])
        tgt_pos_body = quat.inverse.rotate(tgt_pos_3d - drone_state.global_pos)
        tgt_vel_body = quat.inverse.rotate(tgt_vel_3d - drone_state.global_vel)

        motion_info = np.concatenate([
            drone_local_vel, drone_R,
            init_pos_body, init_vel_body,
            tgt_pos_body, tgt_vel_body,
        ]).astype(np.float32)
        return depth_u8, motion_info

    def _nn_initial_guess(self, depth_img, drone_state, plan_init_state, target_state):
        depth_u8, motion_info = self._form_nn_input(
            depth_img, drone_state, plan_init_state, target_state)
        img_flat = depth_u8.reshape(-1).astype(np.float32)
        input_flat = np.concatenate([img_flat, motion_info])

        t0 = time.time()
        if self._use_pth:
            with torch.no_grad():
                x = torch.tensor(input_flat).unsqueeze(0).to(self._device)
                output = self._net(x).cpu().numpy().flatten()
        else:
            output = self._session.run(
                None, {'input': input_flat.reshape(1, -1)})[0].flatten()
        inf_ms = (time.time() - t0) * 1000
        self._inference_times.append(inf_ms)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)

        quat = drone_state.attitude
        drone_pos = drone_state.global_pos
        wpt1_body = np.array([output[0], output[1], output[2]])
        wpt2_body = np.array([output[3], output[4], output[5]])
        wpt1_world = quat.rotate(wpt1_body) + drone_pos
        wpt2_world = quat.rotate(wpt2_body) + drone_pos
        int_wpts_world = np.array([
            [wpt1_world[0], wpt2_world[0]],
            [wpt1_world[1], wpt2_world[1]],
        ])
        ts = np.clip(output[6:9], self._planner.T_min + 1e-3,
                     self._planner.T_max - 1e-3)
        return int_wpts_world, ts

    def _validate_planned_traj(self):
        """
        Hard post-check: sample the whole optimizer trajectory (every 0.1s) and
        REJECT if any point has an ESDF distance < hard_clearance.

        The collision cost in the optimizer is a soft penalty with a tolerance —
        a trajectory that penetrates safe_dis by up to ~0.17m can still pass under
        collision_cost_tol. This hard check ensures the tracked trajectory NEVER
        gets closer than hard_clearance to an obstacle (the ESDF is already
        inflated by the drone radius).

        Raises ValueError if the trajectory grazes -> caught by the seed loop as a
        normal replan failure (retry the next seed).
        """
        # Adaptive threshold: if the drone CURRENTLY is already closer than
        # hard_clearance (e.g. squeezed in a tight gap), do not reject a
        # trajectory that actually takes it out — use the current clearance minus
        # a margin as the threshold, floor 0.1m.
        cur_clear = float(self._esdf.get_edt_dis(
            self._drone_state.global_pos[:2]))
        thr = min(self._hard_clearance, max(cur_clear - 0.05, 0.1))
        pts = self._planner.get_pos_array()   # (N, D), sampled every 0.1s
        for p in pts:
            d = self._esdf.get_edt_dis(p[:2])
            if d < thr:
                raise ValueError(
                    f"Trajectory grazing: clearance {d:.2f}m < hard {thr:.2f}m")

    def _replan(self):
        if not self._esdf.is_ready():
            self.get_logger().warn(
                "[NEO] ESDF not ready yet — /projected_map not received",
                throttle_duration_sec=2.0)
            return
        if self._escape_active:
            if self._should_exit_escape():
                self._exit_escape()
            else:
                return
        if not self._escape_active and self._check_stuck():
            self._enter_escape()
            return
        if not self._escape_active and self._check_no_progress():
            self.get_logger().warn(
                "[NEO] No progress to the goal — escape (local minimum)")
            self._enter_escape()
            return

        with self._depth_lock:
            if (self._latest_depth is None
                    or self._latest_depth_drone_state is None
                    or self._latest_depth_timestamp is None):
                return
            lag = time.time() - self._latest_depth_timestamp
            if lag > DEPTH_MAX_LAG_S:
                self._lag_skip_count += 1
                self.get_logger().warn(
                    f"[NEO] Skip replan, depth lag {lag*1000:.0f}ms",
                    throttle_duration_sec=2.0)
                return
            depth_snapshot   = self._latest_depth.copy()
            sync_drone_state = DroneState()
            sync_drone_state.copy_from(self._latest_depth_drone_state)

        dist = np.linalg.norm(sync_drone_state.global_pos[:2] - self._global_target)
        if dist < 0.8:
            self._reached_target = True
            return

        # ADAPTIVE safe_dis (anti-deadlock): if the drone is already closer than
        # safe_dis to an obstacle, the (fixed) trajectory start point must
        # violate it -> the collision cost at the start alone already > tol ->
        # ALL plans fail forever and the drone hovers in place. Solution: lower
        # the safe_dis reference to follow the current clearance (-margin), with a
        # hard_clearance floor. The collision gradient still pushes the trajectory
        # away; the hard post-check still rejects a grazing one.
        cur_clear = float(self._esdf.get_edt_dis(
            sync_drone_state.global_pos[:2]))
        self._planner.safe_dis = float(np.clip(
            cur_clear - 0.05,
            self._hard_clearance + 0.05, self._safe_dis_norm))

        head_state = None
        plan_init_state = None
        ok = False

        # Time budget: a full sweep of 22 seeds x 5 internal retries can take
        # 30-50s if everything fails — the drone "freezes" hovering for that long.
        # Limit the total sweep time; better to declare failure quickly then escape.
        replan_t0 = time.time()
        REPLAN_BUDGET_S = 2.5

        for use_shorter in [False, True]:
            for seed in range(11):
                if time.time() - replan_t0 > REPLAN_BUDGET_S:
                    break
                self._set_local_target(
                    seed, shorter=use_shorter,
                    ref_pos=sync_drone_state.global_pos[:2])
                plan_init_state = self._get_drone_state_at(self._dt_ahead)
                head_state    = np.zeros((3, 2))
                head_state[0] = plan_init_state.global_pos[:2]
                head_state[1] = plan_init_state.global_vel[:2]
                tail_state    = np.zeros((3, 2))
                tail_state[0] = self._target_state[0]
                tail_state[1] = self._target_state[1]
                try:
                    t0 = time.time()
                    int_wpts_init, ts_init = self._nn_initial_guess(
                        depth_snapshot, sync_drone_state,
                        plan_init_state, self._target_state)
                    self._planner.warm_start_plan(
                        self._esdf, head_state, tail_state,
                        int_wpts_init, ts_init)
                    self._validate_planned_traj()
                    dt = time.time() - t0
                    self._planning_times.append(dt * 1000)
                    if len(self._planning_times) > 50:
                        self._planning_times.pop(0)
                    self._replan_count += 1
                    avg_inf = (np.mean(self._inference_times)
                               if self._inference_times else 0)
                    coll = self._planner.weighted_cost[3]
                    self.get_logger().info(
                        f"[NEO] Replan ok #{self._replan_count} "
                        f"cost={self._planner.final_cost:.1f} "
                        f"t={dt*1000:.0f}ms (nn={avg_inf:.1f}ms) "
                        f"coll={coll:.2f} seed={seed}"
                        f"{' short' if use_shorter else ''}",
                        throttle_duration_sec=1.0)
                    ok = True
                    break
                except Exception:
                    pass
            if ok or time.time() - replan_t0 > REPLAN_BUDGET_S:
                break

        if not ok:
            self._replan_fail_count += 1
            self.get_logger().warn(
                f"[NEO] Replan failed ({self._replan_fail_count}x in a row)")
            if self._replan_fail_count >= 2:
                self._enter_escape()
            return

        self._replan_fail_count = 0

        drone_pos_now = self._drone_state.global_pos[:2]
        discontinuity = float(np.linalg.norm(head_state[0] - drone_pos_now))
        if discontinuity > DISCONTINUITY_THRESHOLD:
            self.get_logger().warn(
                f"[NEO] Discontinuity {discontinuity:.1f}m — replan from actual")
            self._set_local_target(0)
            tail_state    = np.zeros((3, 2))
            tail_state[0] = self._target_state[0]
            tail_state[1] = self._target_state[1]
            head_actual    = np.zeros((3, 2))
            head_actual[0] = drone_pos_now
            v_now = self._drone_state.global_vel[:2].copy()
            speed_now = float(np.linalg.norm(v_now))
            if speed_now > self._v_max:
                v_now *= (self._v_max / speed_now)
            head_actual[1] = v_now
            replan_ok = False
            for _ in range(5):
                try:
                    int_wpts_init, ts_init = self._nn_initial_guess(
                        depth_snapshot, sync_drone_state,
                        plan_init_state, self._target_state)
                    self._planner.warm_start_plan(
                        self._esdf, head_actual, tail_state,
                        int_wpts_init, ts_init)
                    self._validate_planned_traj()
                    replan_ok = True
                    head_state = head_actual
                    break
                except Exception:
                    pass
            if not replan_ok:
                return

        new_traj_array = self._planner.get_full_state_cmd(hz=self._cmd_hz)
        total_time     = float(np.sum(self._planner.ts))
        self._install_trajectory(new_traj_array, total_time)
        with self._emergency_stop_lock:
            self._emergency_stop_first_t = None

    def _install_trajectory(self, state_cmd, base_total_time):
        """
        Install a new trajectory + compute time_scale so the peak speed <= v_max.
        The spatial path does not change; it is only slowed down if needed.
        """
        peak = 0.0
        try:
            vels = state_cmd[:, 1, :]                 # (N, D)
            speeds = np.linalg.norm(vels, axis=1)
            peak = float(np.max(speeds)) if len(speeds) else 0.0
        except Exception:
            peak = 0.0
        if self._v_max > 1e-3 and peak > self._v_max:
            time_scale = peak / self._v_max
        else:
            time_scale = 1.0
        with self._traj_lock:
            self._traj.t_start         = time.time()
            self._traj.state_cmd       = state_cmd
            self._traj.base_total_time = base_total_time
            self._traj.time_scale      = time_scale
            self._traj.total_time      = base_total_time * time_scale
            self._traj.cmd_hz          = self._cmd_hz

    def _print_status(self):
        pos = self._drone_state.global_pos
        vel = self._drone_state.global_vel
        speed = float(np.linalg.norm(vel[:2]))
        esc = " [ESCAPE]" if self._escape_active else ""
        z_err = ""
        if self._cruise_z is not None:
            z_err = f" | z_err={self._cruise_z - pos[2]:+.2f}"
        esdf_info = ""
        if self._esdf.is_ready():
            esdf_info = f" | esdf={float(self._esdf.get_edt_dis(pos[:2])):.2f}"
        avg_inf = np.mean(self._inference_times) if self._inference_times else 0
        avg_pl  = np.mean(self._planning_times) if self._planning_times else 0
        self.get_logger().info(
            f"[NEO] {self._mission_state}{esc} | "
            f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}){z_err}{esdf_info} | "
            f"speed={speed:.2f}m/s | "
            f"goal=({self._global_target[0]:.1f},{self._global_target[1]:.1f}) | "
            f"nn={avg_inf:.1f}ms plan={avg_pl:.0f}ms")

    def _wait_ekf_stable(self, tol=0.08, stable_dur=2.0, timeout=20.0):
        self.get_logger().info(
            f"[NEO] Waiting for EKF Z to stabilize (tol={tol}m, dur={stable_dur}s)...")
        t0 = time.time()
        stable_since = None
        z_ref = self._drone_state.global_pos[2]
        while time.time() - t0 < timeout:
            z_now = self._drone_state.global_pos[2]
            if abs(z_now - z_ref) > tol:
                z_ref = z_now
                stable_since = None
            else:
                if stable_since is None:
                    stable_since = time.time()
                    z_ref = z_now
                if time.time() - stable_since >= stable_dur:
                    self.get_logger().info(f"[NEO] ok EKF Z stable at {z_now:.3f}m")
                    return z_now
            time.sleep(0.1)
        self.get_logger().warn(
            f"[NEO] Timeout EKF Z, using {self._drone_state.global_pos[2]:.3f}m")
        return self._drone_state.global_pos[2]

    def _call_srv(self, client, request, timeout=8.0):
        fut = client.call_async(request)
        t0  = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                return None
            time.sleep(0.05)
        return fut.result()

    # ── FIX ALTITUDE + COLLISION: EKF config, octomap band, alt-trim ─────────
    def _set_px4_param_int(self, name: str, value: int) -> bool:
        """Set a PX4 parameter via MAVROS ParamSetV2 (best-effort)."""
        if not self._param_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                f"[NEO] /mavros/param/set not available — {name} skipped")
            return False
        req = ParamSetV2.Request()
        req.param_id = name
        pv = RclParameterValue()
        pv.type          = ParameterType.PARAMETER_INTEGER
        pv.integer_value = int(value)
        req.value = pv
        res = self._call_srv(self._param_client, req)
        ok  = bool(res and res.success)
        self.get_logger().info(
            f"[NEO] {'ok' if ok else 'FAILED'} PX4 param {name} = {value}")
        return ok

    def _configure_px4_height_ref(self):
        """EKF2_HGT_REF=1 (GPS) — anti baro drift that makes the physical drone descend."""
        self._set_px4_param_int("EKF2_HGT_REF", 1)

    def _configure_octomap_band(self, cruise_z: float):
        """
        MOST IMPORTANT COLLISION FIX (inference): set the octomap occupancy band
        to FOLLOW the actual cruise_z, not a static value from the launch.

        Old problem: the default occ_min_z=2.8 assumes ground_z ~ 2.26m. If SITL
        starts clean (ground_z ~ 0, cruise_z = 2.0), the band [2.8, 5.0] sits
        ABOVE the drone -> short obstacles (bricks ~ 1.5m) do not enter the
        projected_map at all -> the ESDF is empty -> the drone is "blind" and
        collides even though octomap is running.

        New band: [cruise_z - 0.7, cruise_z + 1.0], clamped to at least
        ground_z + 0.35 so the floor is not projected.
        """
        occ_min = max(self._ground_z + 0.35, cruise_z - 0.7)
        occ_max = cruise_z + 1.0
        if not self._octo_param_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                "[NEO] octomap set_parameters not available — using the static band. "
                f"Make sure manually: occ_min_z~{occ_min:.2f}, occ_max_z~{occ_max:.2f}")
            return
        req = SetParameters.Request()
        for name, val in (("occupancy_min_z", occ_min),
                          ("occupancy_max_z", occ_max)):
            p = RclParameter()
            p.name = name
            pv = RclParameterValue()
            pv.type         = ParameterType.PARAMETER_DOUBLE
            pv.double_value = float(val)
            p.value = pv
            req.parameters.append(p)
        res = self._call_srv(self._octo_param_client, req)
        ok  = bool(res and all(r.successful for r in res.results))
        self.get_logger().info(
            f"[NEO] {'ok' if ok else 'FAILED'} Octomap band -> "
            f"[{occ_min:.2f}, {occ_max:.2f}] (cruise_z={cruise_z:.2f})")
        # Reset the map so old voxels (wrong band) are discarded and rebuilt
        if ok and self._octo_reset_client.wait_for_service(timeout_sec=3.0):
            self._call_srv(self._octo_reset_client, EmptySrv.Request(), timeout=5.0)
            self.get_logger().info("[NEO] ok Octomap reset (rebuild with the new band)")

    def _altitude_trim_cb(self):
        """Slowly trim cruise_z using raw GPS AGL (anti baro/EKF drift)."""
        if (self._mission_state != self.STATE_FLYING
                or self._cruise_z is None
                or self._cruise_z_base is None
                or self._gps_alt_home is None
                or len(self._gps_alt_buf) < 5):
            return
        agl = float(np.mean(self._gps_alt_buf[-5:])) - self._gps_alt_home
        err = self._alt - agl
        if abs(err) < 0.12:
            return
        step = float(np.clip(err * 0.5, -0.15, 0.15))
        new_cruise = float(np.clip(
            self._cruise_z + step,
            self._cruise_z_base - 2.0, self._cruise_z_base + 2.0))
        if abs(new_cruise - self._cruise_z) > 1e-4:
            self.get_logger().info(
                f"[NEO] Alt-trim: AGL(gps)={agl:.2f}m target={self._alt:.2f}m "
                f"-> cruise_z {self._cruise_z:.2f} -> {new_cruise:.2f}",
                throttle_duration_sec=4.0)
            self._cruise_z = new_cruise

    def _set_mode(self, mode):
        if not self._mode_client.wait_for_service(timeout_sec=3.0):
            return False
        req = SetMode.Request(); req.custom_mode = mode
        res = self._call_srv(self._mode_client, req)
        ok  = bool(res and res.mode_sent)
        if ok:
            self.get_logger().info(f"[MODE] -> {mode} ok")
        return ok

    def _arm(self, value):
        if not self._arming_client.wait_for_service(timeout_sec=3.0):
            return False
        req = CommandBool.Request(); req.value = value
        res = self._call_srv(self._arming_client, req)
        ok  = bool(res and res.success)
        if ok:
            self.get_logger().info(f"[ARM] {'ARMED' if value else 'DISARMED'} ok")
        return ok

    def _wait_altitude(self, target_z, tol=0.2, timeout=30.0):
        t0 = time.time()
        t_stable_start = None
        while time.time() - t0 < timeout:
            z = self._drone_state.global_pos[2]
            vz = abs(self._drone_state.global_vel[2])
            if abs(z - target_z) < tol and vz < 0.2:
                if t_stable_start is None:
                    t_stable_start = time.time()
                elif time.time() - t_stable_start >= 2.0:
                    return True
            else:
                t_stable_start = None
            time.sleep(0.1)
        return False

    def _warm_up(self, n=50):
        self.get_logger().info(f"[NEO] Warm-up: {n} setpoints @ {self._cmd_hz}Hz")
        dummy = np.zeros(IMG_WIDTH * IMG_HEIGHT + MOTION_INPUT_SIZE, dtype=np.float32)
        if self._use_pth:
            with torch.no_grad():
                x = torch.tensor(dummy).unsqueeze(0).to(self._device)
                for _ in range(3):
                    self._net(x)
        else:
            for _ in range(3):
                self._session.run(None, {'input': dummy.reshape(1, -1)})
        time.sleep(n / self._cmd_hz + 0.5)

    def run_sequence(self):
        self.get_logger().info("[NEO] Waiting for MAVROS...")
        t0 = time.time()
        while not self._connected:
            if time.time() - t0 > self._conn_timeout:
                rclpy.shutdown(); return
            time.sleep(0.5)

        self.get_logger().info("[NEO] Waiting for depth image...")
        t0 = time.time()
        while self._latest_depth is None:
            if time.time() - t0 > 30.0:
                rclpy.shutdown(); return
            time.sleep(0.5)

        self.get_logger().info("[NEO] Waiting for ESDF (/projected_map)...")
        t0 = time.time()
        while not self._esdf.is_ready():
            if time.time() - t0 > 60.0:
                self.get_logger().error(
                    "[NEO] /projected_map timeout! Check octomap_server.")
                rclpy.shutdown(); return
            time.sleep(1.0)
        self.get_logger().info("[NEO] ESDF ready ok")

        # FIX ALTITUDE #1: EKF height reference = GPS before calibration
        self._configure_px4_height_ref()
        time.sleep(2.0)

        if self._use_ekf_stable:
            self._ground_z = self._wait_ekf_stable()
            self.get_logger().info(f"[NEO] Ground Z (EKF): {self._ground_z:.3f}m")
        else:
            self._ground_z = 0.0
            self.get_logger().info(f"[NEO] Ground Z (absolute): 0.0m — cruise={self._alt:.2f}m")

        # FIX ALTITUDE #2: physical AGL reference from raw GPS while still on the ground
        if len(self._gps_alt_buf) >= 5:
            self._gps_alt_home = float(np.mean(self._gps_alt_buf[-10:]))
            self.get_logger().info(
                f"[NEO] GPS alt home = {self._gps_alt_home:.2f}m (AGL reference)")
        else:
            self._gps_alt_home = None
            self.get_logger().warn("[NEO] GPS altitude not available — alt-trim disabled")

        # FIX COLLISION: octomap band follows the actual cruise_z + reset the map
        self._configure_octomap_band(self._ground_z + self._alt)

        self._warm_up(n=50)

        self.get_logger().info("[NEO] ARM drone...")
        t0 = time.time(); armed = False
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
        takeoff_z = self._ground_z + self._alt
        self._cruise_z      = takeoff_z
        self._cruise_z_base = takeoff_z   # clamp reference for GPS alt-trim
        self.get_logger().info(f"[NEO] Takeoff to z={takeoff_z:.2f}m...")
        if not self._wait_altitude(takeoff_z, tol=0.2, timeout=30.0):
            self.get_logger().warn("[NEO] Takeoff not stable — continuing anyway")
        self.get_logger().info(
            f"[NEO] ok Takeoff! z={self._drone_state.global_pos[2]:.3f}m")
        time.sleep(1.0)

        self.get_logger().info("[NEO] Initial plan...")
        self._set_local_target(0)
        first_ok = False
        for first_seed in range(10):
            try:
                with self._depth_lock:
                    depth_snapshot = self._latest_depth.copy()
                    sync_ds = DroneState()
                    sync_ds.copy_from(self._latest_depth_drone_state)
                head0    = np.zeros((3, 2))
                head0[0] = self._drone_state.global_pos[:2]
                tail0    = np.zeros((3, 2))
                tail0[0] = self._target_state[0]
                tail0[1] = self._target_state[1]
                plan_init = self._get_drone_state_at(self._dt_ahead)
                int_wpts_init, ts_init = self._nn_initial_guess(
                    depth_snapshot, sync_ds, plan_init, self._target_state)
                self._planner.warm_start_plan(
                    self._esdf, head0, tail0, int_wpts_init, ts_init)
                self._validate_planned_traj()
                first_traj = self._planner.get_full_state_cmd(hz=self._cmd_hz)
                self._install_trajectory(
                    first_traj, float(np.sum(self._planner.ts)))
                self.get_logger().info(
                    f"[NEO] First plan ok ({self._traj.total_time:.2f}s)")
                first_ok = True
                break
            except Exception:
                time.sleep(0.3)
        if not first_ok:
            self.get_logger().warn("[NEO] First plan failed — will retry while flying")

        self._mission_state = self.STATE_FLYING
        self.get_logger().info(
            f"[NEO] ok Start navigation -> "
            f"goal=({self._global_target[0]:.1f},{self._global_target[1]:.1f})")

        last_replan_t  = time.time()
        traj_direction = 1
        while rclpy.ok():
            now = time.time()
            if now - last_replan_t >= self._replan_period:
                last_replan_t = now
                self._replan()
            pos2d = self._drone_state.global_pos[:2]
            dist  = np.linalg.norm(pos2d - self._global_target)
            if not self._escape_active and (dist < 1.5 or self._reached_target):
                if not self._auto_reverse:
                    self.get_logger().info("[NEO] ok Goal reached!")
                    break
                self._reached_target = False
                if traj_direction == 1:
                    self._global_target = np.array([0.0, 0.0])
                    traj_direction      = -1
                else:
                    self._global_target = np.array([self._goal_x, 0.0])
                    traj_direction      = 1
                self._near_global_target = False
                self._reset_progress()
                with self._traj_lock:
                    self._traj.invalidate()
                self.get_logger().info(
                    f"[NEO] New goal: "
                    f"({self._global_target[0]:.1f},{self._global_target[1]:.1f})")
            time.sleep(0.05)

        if not rclpy.ok():
            return

        self._mission_state = self.STATE_LANDING
        self._cruise_z      = None
        self.get_logger().info("[NEO] Landing...")
        curr_z = self._drone_state.global_pos[2]
        while curr_z > self._ground_z + 0.25 and rclpy.ok():
            curr_z = max(self._ground_z + 0.05, curr_z - 0.08)
            try:
                sp = PositionTarget()
                sp.header.stamp     = self.get_clock().now().to_msg()
                sp.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
                sp.type_mask        = (PositionTarget.IGNORE_VX |
                                       PositionTarget.IGNORE_VY |
                                       PositionTarget.IGNORE_VZ |
                                       PositionTarget.IGNORE_AFX |
                                       PositionTarget.IGNORE_AFY |
                                       PositionTarget.IGNORE_AFZ |
                                       PositionTarget.IGNORE_YAW_RATE)
                pos = self._drone_state.global_pos
                sp.position.x = float(pos[0])
                sp.position.y = float(pos[1])
                sp.position.z = float(curr_z)
                self._pub_sp.publish(sp)
            except Exception:
                break
            time.sleep(0.1)

        time.sleep(1.5)
        try:
            self._arm(False)
            self._set_mode("MANUAL")
        except Exception:
            pass

        self._mission_state = self.STATE_DONE
        avg_inf = np.mean(self._inference_times) if self._inference_times else 0
        avg_pl  = np.mean(self._planning_times) if self._planning_times else 0
        self.get_logger().info("=" * 62)
        self.get_logger().info("  NEO-PLANNER INFERENCE DONE ok")
        self.get_logger().info(f"  Replan count   : {self._replan_count}")
        self.get_logger().info(f"  Avg NN inf     : {avg_inf:.1f} ms")
        self.get_logger().info(f"  Avg planning   : {avg_pl:.1f} ms")
        self.get_logger().info("=" * 62)
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = NEOPlannerNode()
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
