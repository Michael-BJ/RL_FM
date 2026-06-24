#!/usr/bin/env python3
"""
nn_inference_node.py  (LIGHTWEIGHT inference — separate from the dataset pipeline)
==========================================================================
A STANDALONE NEO-Planner inference node, no longer a mirror of
expert_planner_node (data collection). The rigidity only needed while collecting
the dataset is DROPPED; the safety guards are KEPT.

Dropped compared to neo_planner_node.py (expert v25 mirror):
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

Reuse: PlannerNet/DroneState/TrajectorySegment from neo_planner_node.py,
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from esdf_ros2 import ESDF
from min_jerk_planner import MinJerkPlanner, PlannerConfig
# Reuse the architecture & utils from the old inference node — DO NOT duplicate.
from neo_planner_node import (
    PlannerNet, DroneState, TrajectorySegment,
    IMG_WIDTH, IMG_HEIGHT, MOTION_INPUT_SIZE, _SENSOR_QOS,
)

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# Real-time guard (safety, NOT dataset rigidity — do not relax carelessly)
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
            self._session = ort.InferenceSession(
                self._model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            backend_str = f"ONNX ({self._session.get_providers()})"

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

        self._esdf = ESDF()

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
        self.get_logger().info("=" * 62)

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
            if (self._esdf.is_ready() and not self._escape_active
                    and self._esdf.get_edt_dis(drone_pos) < self._guard_clear):
                self.get_logger().warn(
                    "[INF] Collision guard! ESDF too close — escape",
                    throttle_duration_sec=0.5)
                with self._traj_lock:
                    self._traj.invalidate()
                self._enter_escape()
                pos, vel = drone_pos, np.zeros(2)

            # Guard 2: look-ahead — trajectory 0.3..1.5s ahead checked against the
            # latest ESDF (the map changes after planning).
            self._lookahead_tick += 1
            if (self._esdf.is_ready() and not self._escape_active
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
                    self.get_logger().warn(
                        f"[INF] Look-ahead guard! clearance ahead {hit:.2f}m",
                        throttle_duration_sec=0.5)
                    with self._traj_lock:
                        self._traj.invalidate()
                    self._enter_escape()
                    pos, vel = drone_pos, np.zeros(2)

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

    def _find_escape_direction(self):
        drone_pos = self._drone_state.global_pos[:2]
        to_goal = self._global_target - drone_pos
        n = np.linalg.norm(to_goal)
        to_goal_dir = to_goal / n if n > 1e-3 else np.array([1.0, 0.0])
        best_score, best_dir = -np.inf, np.array([1.0, 0.0])
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            d = np.array([np.cos(ang), np.sin(ang)])
            far = self._esdf.get_edt_dis(drone_pos + d * ESCAPE_PROBE_DIST)
            mid = self._esdf.get_edt_dis(drone_pos + d * ESCAPE_PROBE_DIST * 0.5)
            score = min(far, mid) + 0.2 * float(d @ to_goal_dir)
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
        valid = depth_img[np.isfinite(depth_img) & (depth_img > 0)]
        if valid.size > 0:
            depth_norm = np.clip(
                depth_img / (float(valid.max()) + 1e-6) * 255, 0, 255)
        else:
            depth_norm = np.zeros_like(depth_img)
        depth_u8 = depth_norm.astype(np.uint8)

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
        if self._check_no_progress():
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
            try:
                t0 = time.time()
                int_wpts, ts = self._nn_initial_guess(
                    depth_snapshot, sync_state, head)
                self._planner.warm_start_plan(
                    self._esdf, head, tail, int_wpts, ts)
                self._validate_planned_traj()
                self._replan_count += 1
                avg_inf = (np.mean(self._inference_times)
                           if self._inference_times else 0)
                self.get_logger().info(
                    f"[INF] Replan ok #{self._replan_count} "
                    f"cost={self._planner.final_cost:.1f} "
                    f"t={(time.time()-t0)*1000:.0f}ms (nn={avg_inf:.1f}ms)"
                    f"{' short' if shorter else ''}",
                    throttle_duration_sec=1.0)
                ok = True
                break
            except Exception:
                continue

        if not ok:
            self._replan_fail_count += 1
            self.get_logger().warn(
                f"[INF] Replan failed ({self._replan_fail_count}x)")
            if self._replan_fail_count >= 2:
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
