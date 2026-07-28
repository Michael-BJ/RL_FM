#!/usr/bin/env python3
"""
expert_planner_node.py  (v27 — EKF Stable Wait std-based)
==================================================================
FIX v27:
1. _wait_ekf_stable() switched from a min-z strategy to std-based (from the backup manager).
   The min-z strategy is prone to error if the initial EKF drift goes to a very small/negative
   value, making ground_z too low -> cruise_z too low -> the drone flies below target.
   The std-based strategy waits for actual convergence (std < 0.08m for 3 seconds),
   then takes the mean as ground_z.

FIX v26:
1. Arena Bounds Guard: if the drone leaves the arena bounds for > 3 seconds -> auto-disarm.
2. Physics Explosion Guard: if speed > 8 m/s (Gazebo physics exploded) -> disarm immediately.
3. Escape Direction Fix: when out of the arena, bias the escape direction strongly toward the arena center.
4. Ground Z Warning: warn if ground_z > 1.2m (SITL not restarted cleanly).

FIX v25:
1. Z-Axis Stability: hand Z control fully to PX4 (IGNORE_VZ + vel.z=0.0).
2. EKF Stable Wait: replace time.sleep(3.0) with _wait_ekf_stable() so that
   ground_z is truly accurate before takeoff.
3. Altitude Drift Filter: skip a sample if z_err > 0.5m (prevents garbage data
   from the drone dropping/bouncing).
4. No Z-Recovery Loop: removed the recovery logic that caused an infinite loop.
"""
import csv
import datetime
import json
import math
import os
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
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, ParamSetV2
from rcl_interfaces.msg import ParameterValue as RclParameterValue
from rcl_interfaces.msg import ParameterType
from nav_msgs.msg import OccupancyGrid
from pyquaternion import Quaternion
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from esdf_ros2 import ESDF
from min_jerk_planner import MinJerkPlanner, PlannerConfig
from homotopy_frontend import (
    CONTINUITY_COST_FACTOR, compute_mode_signature, mode_signature_to_str)

DISCONTINUITY_THRESHOLD  = 3.0

# 2026-07-14 INPUT CONTRACT MIGRATION (Gemini 2): FIXED-RANGE depth normalization.
# Previously depth was divided by the max valid value PER-FRAME -> absolute scale
# was lost (a wall at 2 m vs 4 m filling the view produced nearly identical u8
# images, right at the most critical moment). Now divided by the constant 10.0 m =
# the physical depth limit of the Orbbec Gemini 2 (deployment camera) = the sim
# camera's far clip (matched in OakD-Lite/model.sdf). MUST be identical to
# nn_inference_node.py.
# Datasets recorded BEFORE this date are not compatible with models trained on
# data using the new convention (and vice versa).
DEPTH_NORM_MAX_M = 10.0
EMERGENCY_STOP_THRESHOLD = 3.0
STUCK_SPEED_THRESHOLD    = 0.15  # (deprecated — _check_stuck is now displacement-based)
STUCK_TIME_THRESHOLD     = 2.0   # seconds -> give up & escape sooner (was 3.0)
STUCK_DRIFT_RESET        = 0.6   # m -> net displacement from the anchor considered "really moving"

# Time budget for replan computation per cycle. batch_plan can take 1-4s/attempt
# at a tight spot; a loop of 22 attempts (2x11 seeds) can block the main thread ~40s
# -> drone hovers long -> altitude goes wrong & stuck-detection is not called. This
# budget breaks the loop early so control returns to navigation/escape quickly.
PLAN_TIME_BUDGET         = 3.0   # seconds (enough for several seeds; 2.0 was too tight)

# v17: time-based emergency stop trigger
EMERGENCY_STOP_PERSIST_TIME = 2.0   # emergency stop > 2 seconds in a row -> escape

# Escape mode
ESCAPE_VELOCITY    = 0.7
ESCAPE_PROBE_DIST  = 1.5
ESCAPE_EXIT_DIST   = 1.2
ESCAPE_MAX_TIME    = 15.0

# ── SAFETY DISTANCES — MEASURED FROM THE CENTER OF MASS (2026-07-24 rewrite) ──
# Before this rewrite, safe_dis/hard_dis were compared directly against
# get_edt_dis, which — because the ESDF inflates obstacles by the body radius —
# is the distance from the drone BODY EDGE, not its center. So every safety
# number carried a HIDDEN +0.30 m you had to add mentally to reason about actual
# clearance (comments even did this; the code never did). Confusing and
# error-prone.
#
# Now: ALL safe_dis/hard_dis constants below are expressed FROM THE CENTER OF
# MASS. The one physical fact that connects the two frames is BODY_RADIUS_M —
# the real ESDF inflation (0.25 m nominal rounds UP to ceil(0.25/0.1)=3 cells =
# 0.30 m at 0.1 m octomap resolution). Conversion, applied at exactly the points
# where we compare against the raw ESDF:
#       clearance_from_center = get_edt_dis(pt) + BODY_RADIUS_M
#       edge_threshold        = center_threshold - BODY_RADIUS_M   (fed to MINCO)
# This is EXPERT-ONLY: the ESDF itself, the homotopy A* frontend (its
# HARD_CLEARANCE/BLOCK_THRESH are calibrated on the inflated grid), has_collision,
# and the inference nodes are all left untouched — only how the expert INTERPRETS
# and sets its own thresholds changes.
BODY_RADIUS_M      = 0.30   # effective body radius = real ESDF inflation (see above)

# Two-tier collision model, now stated from the CENTER OF MASS:
#   clearance >= safe_dis_center            -> free (cost 0)
#   hard_dis_center <= clr < safe_dis_center-> soft cubic penalty (may enter, agile)
#   clearance <  hard_dis_center            -> hard barrier (nearly impenetrable)
# Values chosen 2026-07-24 to MATCH the pre-rewrite EFFECTIVE clearances exactly
# (hard 0.30 body-edge = 0.60 from center; soft 0.35 body-edge = 0.65 from center).
# A tighter set (hard 0.50 / soft 0.55 from center) was tried first — it let the
# drone graze to only 0.14 m body-edge in the R3 lower corridor (tracking error
# ate the reduced buffer) -> physics explosion. Reverted to the conservative
# clearance (this run's decision), KEEPING the center-of-mass semantics + the
# decoupled post-check (both worked: no reject-loops observed). ~0.35 m of
# tracking buffer beyond the physical body at the hard barrier.
HARD_DIS_CENTER    = 0.60   # m from center -> hard barrier (= old 0.30 body-edge)
SAFE_DIS_CENTER    = 0.65   # m from center -> soft-zone start at rest (= old 0.35 body-edge)

# Dynamic safety distance: when trapped (consecutive replan failures), shrink the
# SOFT margin gradually toward the hard barrier so the planner can slip through a
# tight gap instead of escaping — never below the hard barrier (still won't collide).
# Speed-aware margin: the reference path is pushed further from obstacles in
# proportion to speed, to absorb tracking error on turns (the physical drone is
# always slightly outside the reference). All in CENTER-OF-MASS metres.
K_SPEED_MARGIN     = 0.25   # m per (m/s) -> at v=1.0 adds +0.25 m -> 0.90 m soft-start at cruise (= old)
SAFE_DIS_TRAP_STEP = 0.05   # m  -> shrink the soft margin per consecutive replan failure
SAFE_DIS_TRAP_MIN  = HARD_DIS_CENTER   # m  -> floor of the SOFT margin = the hard barrier
# POST-CHECK reject line (2026-07-24): the forward safety guard now rejects a
# trajectory ONLY when a point's center-of-mass clearance drops BELOW this — set
# just under the hard barrier, and DECOUPLED from safe_dis. Previously the
# post-check reused safe_dis (the SOFT comfort zone) as a hard reject line, so it
# threw away exactly the grazing trajectories the optimizer was designed to
# accept -> they fought each other -> near a tight obstacle the drone got a
# reject-loop (documented "trapped, spinning" behavior). Now the optimizer owns
# the soft zone [hard, safe] and the post-check only vetoes what dips below the
# hard barrier — one job each, no overlap.
POST_CHECK_CLEAR_MIN = HARD_DIS_CENTER - 0.05   # m from center (0.55)
ESCAPE_FAIL_COUNT  = 4      # escape only after this many consecutive failures (room to shrink)
# Escape is only useful if the drone is HUGGING an obstacle. In open space a replan
# failure = the planner needs more time/seeds (or the goal is blocked far away) — escape
# just jitters uselessly AND resets the shrink count. If clearance > this value,
# do not escape: let the dynamic safe_dis shrink and keep retrying.
ESCAPE_MIN_CLEAR   = 1.0    # m

# Arena boundary (hard limit) — drone outside this for > OUT_OF_BOUNDS_TIME -> disarm
ARENA_X_MIN_DEFAULT    = -2.0
ARENA_X_MAX_DEFAULT    = 33.0   # poles.pcd X max=34.95, 2m margin
ARENA_Y_MIN_DEFAULT    = -13.0  # poles.pcd Y min=-14.95, 2m margin
ARENA_Y_MAX_DEFAULT    =  13.0  # poles.pcd Y max=+14.95, 2m margin
OUT_OF_BOUNDS_TIME     =  1.5   # seconds before return-home (sped up to avoid hitting walls)
PHYSICS_EXPLOSION_SPD  =  8.0   # m/s  -> Gazebo physics exploded, try to recover
EXPLOSION_GROUND_TIMEOUT  = 30.0  # seconds to wait for the drone to land after disarm
EXPLOSION_GROUND_Z_THR    =  0.6  # m above ground_z to detect "has landed"
EXPLOSION_GROUND_SPD_THR  =  0.5  # m/s speed for "stationary on the ground"
EXPLOSION_MAX_RETRY       =  3    # max recovery attempts before shutdown
GROUND_Z_WARN_THRESH   =  2.0   # m

# Accidental-landing recovery (NO gz reset — also applies in the real world)
# The drone "accidentally landed" if, WHILE flying (TAKEOFF/COLLECTING), it:
#   (a) is auto-disarmed by the PX4 land-detector (armed->False), OR
#   (b) physical z drops near the ground while cruise_z is still far above.
LAND_RECOVER_Z_THR     =  0.35  # m above ground_z -> considered touching the ground
LAND_RECOVER_PERSIST   =  1.0   # seconds condition (b) must persist before recovering
LAND_RECOVER_MAX_RETRY =  5     # max consecutive recoveries before the mission stops

# Altitude recovery — EMERGENCY ONLY. PX4 already locks Z (EKF2_HGT_REF=GPS),
# so normal noise ±0.2-0.3m is NOT an emergency. ALT-RCV is only for a drone that
# truly lost altitude (e.g. fell after clipping an obstacle): large threshold +
# must be PERSISTENT so it is not triggered by a momentary spike.
ALT_RECOVERY_THRESH    =  0.60  # m    -> enter recovery if |z_err| > this value
ALT_RECOVERY_EXIT      =  0.35  # m    -> exit recovery if |z_err| < this value
ALT_RECOVERY_STABLE_T  =  0.8   # seconds stable before resuming navigation
ALT_RECOVERY_ARM_T     =  1.5   # seconds z_err must stay > THRESH before freezing

# Altitude fix at the turnaround point (waypoint): hold XY & wait for small z_err
ALT_FIX_TOL            =  0.15  # m    -> altitude considered ready if |z_err| < this
ALT_FIX_TIMEOUT       =  8.0   # seconds -> maximum wait limit

# Depth sync
DEPTH_MAX_LAG_S = 0.3

# Sample quality filters
SAMPLE_MIN_ESDF   = 0.4
TRAJ_MIN_ESDF     = 0.3   # FIX: 0.2 -> 0.3 (now measured from the drone BODY, see ESDF inflation)
TRAJ_CHECK_POINTS = 10
SAMPLE_MAX_SPEED  = 3.0


def default_rl_policy_path(fname="policy_export_v2.pt"):
    """Resolve the shipped RL policy under model/rl/.

    Robust to both build modes: prefer the installed share dir (populated by
    a copy OR symlink build), then fall back to the in-source tree
    (<pkg>/model/rl/) when running unbuilt. Returns the path even if it does
    not exist yet, so the parameter still has a sensible default.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("fm_planner"),
                         "model", "rl", fname)
        if os.path.isfile(p):
            return p
    except Exception:
        pass
    # source-tree fallback: .../fm_planner/fm_planner/ -> .../fm_planner/model/rl/
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "model", "rl", fname)

# ── FIX ALTITUDE: GPS-AGL cruise trim ────────────────────────────────────────
# Problem: cruise_z is set ONCE in the local EKF frame, then held forever.
# The SITL barometer drifts -> the EKF Z reference drifts with it -> the drone
# PHYSICALLY descends (until it touches the ground) while the EKF z still looks =
# cruise_z, so alt-recovery is NEVER triggered. GPS altitude (raw sensor, not EKF)
# does not follow the baro drift, so it is used as the physical AGL reference to
# trim cruise_z slowly.
# Alt-trim DISABLED: with EKF2_HGT_REF=GPS, PX4 already anchors altitude to GPS so
# cruise_z stays = the stable physical altitude. Recomputing AGL from raw GPS and
# then shifting cruise_z is only double-correcting + injecting noise that triggers
# ALT-RCV. Set True only if the altitude truly drifts due to baro.
ALT_TRIM_ENABLED   = False
ALT_TRIM_PERIOD    = 2.0    # seconds between corrections
# SITL GPS vertical noise ≈ ±0.18m (peak-to-peak ~0.35m). The deadband MUST be
# larger than this noise, otherwise alt-trim chases the noise -> cruise_z
# oscillates -> false z_err > ALT_RECOVERY_THRESH -> ALT-RCV keeps triggering ->
# drone freezes. A long-window median filter + small steps damp the noise before
# it enters the correction.
ALT_TRIM_DEADBAND  = 0.25   # m   -> no correction if |AGL error| < this (> GPS noise)
ALT_TRIM_MAX_STEP  = 0.06   # m   -> maximum correction step per period (smooth)
ALT_TRIM_MAX_TOTAL = 2.0    # m   -> total trim limit from the initial cruise_z
ALT_TRIM_FILTER_N  = 15     # number of GPS samples for the MEDIAN (anti-outlier)

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

class DroneState:
    def __init__(self):
        self.global_pos = np.zeros(3)
        self.global_vel = np.zeros(3)
        self.local_vel  = np.zeros(3)
        self.attitude   = Quaternion()
        self.yaw        = 0.0

    def copy_from(self, other: "DroneState"):
        self.global_pos = other.global_pos.copy()
        self.global_vel = other.global_vel.copy()
        self.local_vel  = other.local_vel.copy()
        self.attitude   = other.attitude
        self.yaw        = other.yaw

class TrajectorySegment:
    def __init__(self):
        self.t_start    = 0.0
        self.state_cmd  = None
        self.cmd_hz     = 100
        self.total_time = 0.0

    def is_valid(self) -> bool:
        return self.state_cmd is not None and len(self.state_cmd) > 0

    def time_remaining(self, t_now: float) -> float:
        if not self.is_valid():
            return 0.0
        return max(0.0, self.total_time - (t_now - self.t_start))

    def invalidate(self):
        self.state_cmd = None

    def sample_at(self, t_now: float):
        if not self.is_valid():
            return None, None, None
        elapsed = t_now - self.t_start
        if elapsed < 0:
            elapsed = 0.0
        if elapsed >= self.total_time:
            last = self.state_cmd[-1]
            return last[0], np.zeros(2), np.zeros(2)
        idx_float = elapsed * self.cmd_hz
        idx_low   = int(idx_float)
        idx_high  = min(idx_low + 1, len(self.state_cmd) - 1)
        alpha     = idx_float - idx_low
        s_low  = self.state_cmd[idx_low]
        s_high = self.state_cmd[idx_high]
        pos = s_low[0] * (1 - alpha) + s_high[0] * alpha
        vel = s_low[1] * (1 - alpha) + s_high[1] * alpha
        acc = s_low[2] * (1 - alpha) + s_high[2] * alpha
        return pos, vel, acc

class ExpertPlannerNode(Node):
    STATE_IDLE       = "IDLE"
    STATE_TAKEOFF    = "TAKEOFF"
    STATE_COLLECTING = "COLLECTING"
    STATE_LANDING    = "LANDING"
    STATE_DONE       = "DONE"

    CSV_HEADER = [
        "id",
        "drone_vel_x", "drone_vel_y", "drone_vel_z",
        "R11", "R12", "R13",
        "R21", "R22", "R23",
        "R31", "R32", "R33",
        "init_pos_x", "init_pos_y", "init_pos_z",
        "init_vel_x", "init_vel_y", "init_vel_z",
        "target_pos_x", "target_pos_y", "target_pos_z",
        "target_vel_x", "target_vel_y", "target_vel_z",
        "wpts1_x", "wpts1_y", "wpts1_z",
        "wpts2_x", "wpts2_y", "wpts2_z",
        "ts1", "ts2", "ts3",
        # MULTI-HOMOTOPY (v1.5, see homotopy_frontend.py): topological
        # side-signature from compute_mode_signature() -- "free" (the straight
        # head->tail line is unobstructed), or a list of +-1 (one per blocked
        # obstacle interval, separated by "_"). Rows with the same `id` but a
        # DIFFERENT `mode_sig` = genuinely different modes (used for analyzing
        # the bimodal-scene ratio, not just waypoint distance).
        "mode_sig",
    ]

    def __init__(self):
        super().__init__("expert_planner_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("data_save_dir",
                               os.path.expanduser("~/training_data"))
        self.declare_parameter("target_samples",  5000)
        self.declare_parameter("goal_x",          15.0)
        self.declare_parameter("target_alt",       1.5)
        self.declare_parameter("use_ekf_stable",   True)
        self.declare_parameter("replan_period",    1.0)
        self.declare_parameter("cmd_hz",         100)
        # 2026-07-14: STAYS 5.0. Was briefly raised to 7.5 (hoping for earlier
        # turn labels) then REVERTED after SITL testing on gauntlet_train2:
        # a 7.5 m target in a dense world (events every ~4 m) lands BEHIND 1-2
        # obstacles at once -> the head->tail line cuts through multiple events
        # -> plan cost explodes (12 -> 33 -> 67) -> "trajectory unsafe" ->
        # escape -> escape ping-pong (3x direction changes in 4 s) -> leaves
        # the arena -> PHYSICS EXPLOSION at x=3. Also, offline sweeps show that
        # longitu_step contributes almost NOTHING to anticipation (divert is
        # ~the same at L=5/7.5/10); the seed continuity from v1.13 is what
        # actually helps. So 7.5 buys little while risking a lot in a dense world.
        self.declare_parameter("longitu_step",     5.0)
        self.declare_parameter("lateral_step",     1.0)
        self.declare_parameter("v_max",            1.0)
        self.declare_parameter("arm_timeout",     30.0)
        self.declare_parameter("connect_timeout", 30.0)
        # safe_dis is now the soft-zone clearance FROM THE CENTER OF MASS (m),
        # not from the body edge (see BODY_RADIUS_M). Default = SAFE_DIS_CENTER.
        self.declare_parameter("safe_dis",         SAFE_DIS_CENTER)
        self.declare_parameter("safe_dis_emergency", 0.3)
        # MULTI-HOMOTOPY (see workspace CLAUDE.md): true = save ALL feasible
        # solutions per scene (left & right) as separate CSV rows sharing the
        # same depth image; false = old behavior (only the lowest-cost
        # solution, unimodal dataset).
        self.declare_parameter("save_all_modes",   True)
        # PILOT (2026-07-08, see CLAUDE.md): true = force 3 consecutive legs
        # (left, right, free) via a per-obstacle virtual wall
        # (homotopy_frontend.forced_sign), then STOP collecting (does not
        # wait for target_samples) — so both sides are actually FLOWN & recorded
        # for each obstacle, not just "found" by A* within 1 scene. Requires
        # use_topo_frontend=true. Default false = old behavior.
        self.declare_parameter("force_lap_alternation", False)
        # 2026-07-11: lap_alternation_loop=true -> do NOT stop after 3
        # legs; REPEAT the cycle (left, right, free, left, ...) until
        # target_samples is reached — for collecting VOLUME with balanced
        # left/right sides, not just the 3-leg validation pilot. Requires
        # force_lap_alternation=true (+ use_topo_frontend=true). Default
        # false = the old 3-leg pilot behavior stays intact.
        self.declare_parameter("lap_alternation_loop", False)
        self.declare_parameter("arena_x_min", ARENA_X_MIN_DEFAULT)
        self.declare_parameter("arena_x_max", ARENA_X_MAX_DEFAULT)
        self.declare_parameter("arena_y_min", ARENA_Y_MIN_DEFAULT)
        self.declare_parameter("arena_y_max", ARENA_Y_MAX_DEFAULT)
        # 2026-07-24: chained turnaround endpoints. The old behavior bounces
        # between exactly two fixed points, (0,0) <-> (goal_x,0) -- every
        # "leg" repeats the identical pair, so consecutive corridor traverses
        # approach every obstacle from the same lateral offset. When true, the
        # turnaround point's Y coordinate is redrawn (uniform random in
        # [-turnaround_y_range, +turnaround_y_range]) each time the drone
        # reaches a turnaround, so leg N+1's target is never the leg N-1
        # point (A->B->C->D..., not A->B->A->B). X still alternates between
        # 0 and goal_x (a single-corridor world only has two ends to traverse
        # between) -- only the entry/exit Y offset varies. Default false =
        # old bounce behavior, unchanged.
        self.declare_parameter("randomize_turnaround_y", False)
        self.declare_parameter("turnaround_y_range", 2.0)

        self._data_dir      = str(self.get_parameter("data_save_dir").value)
        self._target_n      = int(self.get_parameter("target_samples").value)
        self._goal_x        = float(self.get_parameter("goal_x").value)
        self._alt           = float(self.get_parameter("target_alt").value)
        self._use_ekf_stable = bool(self.get_parameter("use_ekf_stable").value)
        self._replan_period = float(self.get_parameter("replan_period").value)
        self._cmd_hz        = int(self.get_parameter("cmd_hz").value)
        self._longitu_step  = float(self.get_parameter("longitu_step").value)
        self._lateral_step  = float(self.get_parameter("lateral_step").value)
        self._v_max         = float(self.get_parameter("v_max").value)
        self._arm_timeout   = float(self.get_parameter("arm_timeout").value)
        self._conn_timeout  = float(self.get_parameter("connect_timeout").value)
        self._safe_dis_norm = float(self.get_parameter("safe_dis").value)
        self._safe_dis_emer = float(self.get_parameter("safe_dis_emergency").value)
        self._save_all_modes = bool(self.get_parameter("save_all_modes").value)
        self._arena_x_min   = float(self.get_parameter("arena_x_min").value)
        self._arena_x_max   = float(self.get_parameter("arena_x_max").value)
        self._arena_y_min   = float(self.get_parameter("arena_y_min").value)
        self._arena_y_max   = float(self.get_parameter("arena_y_max").value)
        self._arena_center  = np.array([
            (self._arena_x_min + self._arena_x_max) / 2.0,
            (self._arena_y_min + self._arena_y_max) / 2.0,
        ])
        self._randomize_turnaround_y = bool(
            self.get_parameter("randomize_turnaround_y").value)
        self._turnaround_y_range = float(
            self.get_parameter("turnaround_y_range").value)
        # Keep the drawn Y at least 1.0m inside the arena wall so the
        # turnaround point itself is never flush against the boundary.
        self._turnaround_y_lo = max(self._arena_y_min + 1.0,
                                    -self._turnaround_y_range)
        self._turnaround_y_hi = min(self._arena_y_max - 1.0,
                                    self._turnaround_y_range)

        # ── Data directory ────────────────────────────────────────────────────
        self._img_dir  = os.path.join(self._data_dir, "depth_img")
        os.makedirs(self._img_dir, exist_ok=True)
        self._csv_path = os.path.join(self._data_dir, "train.csv")
        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.CSV_HEADER)

        # ── Planner config ────────────────────────────────────────────────────
        cfg = PlannerConfig()
        cfg.v_max              = self._v_max
        cfg.T_min              = 0.5
        cfg.T_max              = 5.0
        # safe_dis/hard_dis are stored in CENTER-OF-MASS metres (self._safe_dis_norm,
        # HARD_DIS_CENTER) and converted to the BODY-EDGE frame the MINCO cost
        # actually compares against the (inflated) ESDF. This initial cfg.safe_dis
        # is overwritten every replan with the dynamic+speed-aware value (see
        # _replan_and_record); hard_dis is set once here. clamp >=0 for safety.
        cfg.safe_dis           = max(0.0, self._safe_dis_norm - BODY_RADIUS_M)
        cfg.hard_dis           = max(0.0, HARD_DIS_CENTER - BODY_RADIUS_M)
        cfg.init_wpts_num      = 2
        cfg.init_T             = 2.5
        cfg.delta_t            = 0.1
        # Collision cost tolerance before plan_once() rejects a trajectory.
        # Raised 5.0 -> 20.0: a value of 5.0 rejects a trajectory that only
        # slightly TOUCHES the edge of the safe_dis margin -> consecutive replan
        # failures -> drone freezes & escapes even though the gap is actually
        # passable. With 20.0 the planner accepts a trajectory that grazes the
        # thin margin (still outside the drone body thanks to inflation), so the
        # drone keeps moving and produces a more optimal path.
        cfg.collision_cost_tol = 20.0
        cfg.opt_tol            = 1e-2
        cfg.weights            = [1.0, 1.0, 1.0, 10000.0]
        self._planner          = MinJerkPlanner(cfg)

        # ── Homotopy front-end (2026-07-07) ──────────────────────────────────
        # use_topo_frontend=true (default): batch_plan is monkey-patched into
        # topo_batch_plan — seeds from A* homotopy-class enumeration on the
        # ESDF, NOT 3 hardcoded seeds (straight/±0.6 m). The stock batch_plan
        # is only used as a fallback when the grid isn't ready / A* finds no
        # path / all optimizations fail. false = original 3-seed behavior.
        self.declare_parameter("use_topo_frontend", True)
        self._use_topo_frontend = bool(
            self.get_parameter("use_topo_frontend").value)

        # v1.13 (2026-07-14) CONTINUATION SEED — anticipation (see CONTINUITY_
        # COST_FACTOR in homotopy_frontend.py). On every replan, the remaining
        # trajectory CURRENTLY being flown is offered as an ADDITIONAL seed; if
        # its optimization result is still reasonable, that is what gets flown
        # -> the turn stays put (world coordinates) and is actually executed,
        # instead of being pushed "later" on every replan. Requires
        # use_topo_frontend=true.
        # false = old behavior (replan from scratch every cycle).
        self.declare_parameter("use_plan_continuity", True)
        self._use_plan_continuity = bool(
            self.get_parameter("use_plan_continuity").value)
        if self._use_topo_frontend:
            from homotopy_frontend import patch_planner
            # bounds = arena MINUS a 0.5 m inset: the ESDF grid spans the whole
            # pcd (y +-15); without this bound A* picks the cheap route that
            # goes AROUND THE OUTSIDE of a mirrored pole -> the drone leaves the
            # arena (repeated out-of-bounds + explosion, run_20260707_131700).
            # The inset absorbs the MINCO trajectory width + tracking error;
            # every designed world corridor stays >=1.5 m.
            _inset = 0.5
            patch_planner(self._planner, bounds=(
                self._arena_x_min + _inset, self._arena_x_max - _inset,
                self._arena_y_min + _inset, self._arena_y_max - _inset))
            self.get_logger().info(
                "[Expert] Homotopy front-end ACTIVE — A* topological seed "
                "(old 3-seed only as fallback) | A* bounds: "
                f"X[{self._arena_x_min + _inset:.1f},"
                f"{self._arena_x_max - _inset:.1f}] "
                f"Y[{self._arena_y_min + _inset:.1f},"
                f"{self._arena_y_max - _inset:.1f}]")
        else:
            self.get_logger().info(
                "[Expert] Homotopy front-end OFF — 3-seed hardcoded (stock)")

        # ── Lap-alternation pilot (2026-07-08) ───────────────────────────────
        # Fixed 3 legs: [+1 (force left), -1 (force right), None (free)],
        # applied via homotopy_frontend forced_sign (per-obstacle virtual
        # wall). Legs switch at the turnaround point that ALREADY EXISTS in
        # run_sequence() (toggle traj_direction). After the 3rd leg (free)
        # finishes -> _lap_pilot_done=True -> the main loop stops, it does NOT
        # wait for target_samples (see CLAUDE.md: the goal is to validate the
        # mechanism first, not volume — looping until target_samples would be
        # dominated by non-ambiguous segments, like the gauntlet_train dataset's
        # 9.2%).
        self._force_lap_alternation = bool(
            self.get_parameter("force_lap_alternation").value)
        self._lap_loop = bool(
            self.get_parameter("lap_alternation_loop").value)
        self._lap_phases   = (1, -1, None)
        self._lap_leg_idx  = 0
        self._lap_pilot_done = False
        if self._force_lap_alternation:
            if not self._use_topo_frontend:
                self.get_logger().warn(
                    "[Expert] force_lap_alternation=true but "
                    "use_topo_frontend=false — the virtual wall needs the topo "
                    "frontend, pilot DISABLED.")
                self._force_lap_alternation = False
            else:
                self._planner._topo_frontend.forced_sign = self._lap_phases[0]
                if self._lap_loop:
                    self.get_logger().info(
                        "[Expert] Lap-alternation LOOP active — cycle "
                        "left -> right -> free REPEATS until "
                        f"{self._target_n} samples (instead of stopping at leg 3). "
                        f"Leg 1: forced_sign={self._lap_phases[0]} (left)")
                else:
                    self.get_logger().info(
                        "[Expert] Lap-alternation PILOT active — 3 legs: "
                        "left -> right -> free, then stop (instead of target_samples). "
                        f"Leg 1/3: forced_sign={self._lap_phases[0]} (left)")

        # ── RL seed generator (2026-07-17) ───────────────────────────────────
        # use_rl_seeds=true: on every replan, 2 dual-h seeds (h=-1 / h=+1,
        # structural side) from the frozen SAC policy (rl_seed_adapter.py) are
        # tried FIRST; every converged one enters all_solutions (dataset bimodal
        # by construction, mode_sig dedup still works). If none converges ->
        # fall back to the old batch_plan (topo/3-seed) — existing behavior is
        # unchanged. Default false = no RL at all.
        # rl_policy_path default = policy_export_v2.pt (L4 bidirectional policy,
        # 2026-07-21) — the current best; the older policy_export_ft_side.pt is
        # also shipped and can be selected via rl_policy_path.
        self.declare_parameter("use_rl_seeds", False)
        self.declare_parameter("rl_policy_path", default_rl_policy_path())
        self._rl_seeds = None
        self._rl_try_count  = 0   # replans that attempted an RL seed
        self._rl_ok_count   = 0   # replans where >=1 RL seed converged
        self._rl_dual_count = 0   # replans where BOTH RL seeds converged
        if bool(self.get_parameter("use_rl_seeds").value):
            from rl_seed_adapter import RLSeedAdapter
            _rl_path = str(self.get_parameter("rl_policy_path").value)
            self._rl_seeds = RLSeedAdapter(
                _rl_path, v_max=self._v_max,
                T_min=0.5, T_max=5.0, M=3)
            self.get_logger().info(
                f"[Expert] RL seed generator ACTIVE — policy: {_rl_path}")

        # ── ESDF ──────────────────────────────────────────────────────────────
        # inflation_radius = drone body radius (x500 ≈ 0.25m). Obstacles are
        # fattened by this amount before computing distances, so get_edt_dis
        # effectively measures the distance from the drone BODY EDGE (not its
        # center of mass) to the obstacle. At map resolution 0.1m the inflation
        # rounds to 0.3m (3 cells) — close to the real body radius; safe_dis then
        # becomes the extra margin beyond the body edge.
        # arena_bounds -> virtual wall in the ESDF (see esdf_ros2.py):
        # MINCO/guard/escape consistently treat outside-arena as an obstacle.
        self._esdf = ESDF(inflation_radius=0.25,
                          arena_bounds=(self._arena_x_min, self._arena_x_max,
                                        self._arena_y_min, self._arena_y_max))

        # ── Drone state ───────────────────────────────────────────────────────
        self._connected   = False
        self._armed       = False
        self._mode        = ""
        self._drone_state = DroneState()
        self._ground_z    = 0.0
        self._cruise_z    = None

        # ── Trajectory ────────────────────────────────────────────────────────
        self._traj        = TrajectorySegment()
        self._traj.cmd_hz = self._cmd_hz
        self._traj_lock   = threading.Lock()

        # ── Goal ──────────────────────────────────────────────────────────────
        self._global_target      = np.array([self._goal_x, 0.0])
        self._target_state       = np.zeros((2, 2))
        self._near_global_target = False
        self._reached_target     = False
        self._mission_state      = self.STATE_IDLE

        # ── Recovery / Escape state ───────────────────────────────────────────
        # use_escape=false (default 2026-07-23, user request): the scripted
        # escape maneuver (canned velocity away from the obstacle) is disabled
        # — repeated ping-pong/spin behavior was judged more disruptive than
        # helpful. _enter_escape() still runs, but only resets the stuck/fail
        # bookkeeping and forces a fresh replan from the actual current state
        # (relying on the planner/RL seeds + dynamic safe_dis shrink to find a
        # way out), instead of taking over velocity control.
        self.declare_parameter("use_escape", False)
        self._use_escape           = bool(self.get_parameter("use_escape").value)
        self._replan_fail_count    = 0
        self._stuck_since          = None
        self._last_pos_for_stuck   = None
        self._escape_active        = False
        self._escape_velocity_xy   = np.zeros(2)
        self._escape_start_time    = None

        # v17: TIME-BASED emergency stop tracker
        self._emergency_stop_first_t  = None
        self._emergency_stop_lock     = threading.Lock()

        # v26: arena bounds + physics explosion guard
        self._out_of_bounds_since     = None
        self._out_of_bounds_return    = False   # True -> fly back to start (0,0)

        # v26: altitude recovery — hold XY when z_err is large
        self._alt_recovery_active     = False
        self._alt_recovery_stable_t   = None   # time it started becoming stable again
        self._alt_recovery_arm_t      = None   # v27: when z_err started > THRESH (for persistence)

        # v27: detect "accidental landing" (no gz reset)
        self._near_ground_since       = None   # when physical z started near the ground while flying

        # v27: fix altitude at the turnaround point (waypoint) before navigating
        self._alt_fix_active          = False
        self._alt_fix_xy              = np.zeros(2)

        # ── Sensor + sync state ───────────────────────────────────────────────
        self._latest_depth              = None
        self._latest_depth_drone_state  = None
        self._latest_depth_timestamp    = None
        self._bridge                    = CvBridge()
        self._depth_lock                = threading.Lock()
        self._lag_skip_count = 0
        self._lag_history    = []

        # Sample quality stats
        self._collision_skip_count = 0
        self._traj_unsafe_skip     = 0
        self._velocity_skip_count  = 0
        self._altitude_skip_count  = 0

        # ── Counter ───────────────────────────────────────────────────────────
        self._n_samples = 0
        self._recovery_active = False  # True during explosion recovery so the setpoint stops

        # FIX ALTITUDE: state for GPS-AGL cruise trim
        self._gps_alt_buf      = []     # rolling buffer of GPS altitude (raw, AMSL)
        self._gps_alt_home     = None   # GPS altitude while the drone is on the ground
        self._cruise_z_base    = None   # initial cruise_z before trim (for clamping)
        self._last_alt_trim_t  = 0.0

        # ── ROS interfaces ────────────────────────────────────────────────────
        self._pub_sp = self.create_publisher(
            PositionTarget, "/mavros/setpoint_raw/local", 10)

        # Live viz for RViz: RL seeds + MINCO candidates + targets
        self._pub_viz = self.create_publisher(MarkerArray, "/expert/viz", 1)
        self._viz_planner = MinJerkPlanner(PlannerConfig())
        self._viz_data = None

        self.create_subscription(String, "/px4/state",
                                 self._cb_state, 10)
        self.create_subscription(String, "/px4/sensors",
                                 self._cb_sensors, 10)
        self.create_subscription(
            Image, "/realsense/depth/float32",
            self._cb_depth, _SENSOR_QOS)
        self.create_subscription(
            OccupancyGrid, "/projected_map",
            self._esdf.occupancy_map_cb, 10)

        self._arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self._mode_client   = self.create_client(SetMode,     "/mavros/set_mode")
        self._param_client  = self.create_client(ParamSetV2,  "/mavros/param/set")

        self._cmd_timer    = self.create_timer(
            1.0 / self._cmd_hz, self._publish_cmd)
        self._status_timer = self.create_timer(5.0, self._print_status)
        # FIX ALTITUDE: trim cruise_z from GPS-AGL every ALT_TRIM_PERIOD
        self._alt_trim_timer = self.create_timer(
            ALT_TRIM_PERIOD, self._altitude_trim_cb)

        self._print_header()

    def _print_header(self):
        self.get_logger().info("=" * 62)
        self.get_logger().info("  Expert Planner v26 — Arena Bounds + GPS Height Reference")
        self.get_logger().info("=" * 62)
        self.get_logger().info(f"  v_max              : {self._v_max} m/s")
        self.get_logger().info(
            f"  safe_dis (center)  : {self._safe_dis_norm:.2f} m soft / "
            f"{HARD_DIS_CENTER:.2f} m hard / post-check {POST_CHECK_CLEAR_MIN:.2f} m "
            f"(body radius {BODY_RADIUS_M:.2f} m)")
        self.get_logger().info(f"  ES persist time    : {EMERGENCY_STOP_PERSIST_TIME}s (→ escape)")
        self.get_logger().info(f"  use_escape         : {self._use_escape} "
                               f"({'scripted maneuver' if self._use_escape else 'DISABLED — force replan only'})")
        self.get_logger().info(f"  Sample filters     : drone>{SAMPLE_MIN_ESDF}m, "
                               f"traj>{TRAJ_MIN_ESDF}m, speed<{SAMPLE_MAX_SPEED}m/s, alt_drift<0.5m")
        self.get_logger().info("=" * 62)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _cb_state(self, msg: String):
        try:
            d = json.loads(msg.data)
            self._connected = d.get("connected", False)
            self._armed     = d.get("armed",     False)
            self._mode      = d.get("mode",      "")
        except Exception:
            pass

    def _cb_sensors(self, msg: String):
        try:
            d = json.loads(msg.data)
            x   = float(d.get("local_x", 0))
            y   = float(d.get("local_y", 0))
            z   = float(d.get("local_z", 0))
            vx  = float(d.get("vel_x",  0))
            vy  = float(d.get("vel_y",  0))
            vz  = float(d.get("vel_z",  0))
            yaw = float(d.get("yaw",    0))
            self._drone_state.global_pos = np.array([x, y, z])
            self._drone_state.global_vel = np.array([vx, vy, vz])
            self._drone_state.yaw        = yaw
            qw = math.cos(yaw / 2)
            qz = math.sin(yaw / 2)
            self._drone_state.attitude = Quaternion(qw, 0.0, 0.0, qz)
            self._drone_state.local_vel = (
                self._drone_state.attitude.inverse.rotate(
                    self._drone_state.global_vel))
            # FIX ALTITUDE: take the raw GPS altitude (not EKF) for physical AGL
            gps = d.get("gps")
            if gps and gps.get("altitude_m") is not None:
                alt = float(gps["altitude_m"])
                if math.isfinite(alt):
                    self._gps_alt_buf.append(alt)
                    if len(self._gps_alt_buf) > 20:
                        self._gps_alt_buf.pop(0)
        except Exception:
            pass

    def _cb_depth(self, msg: Image):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            if depth.shape != (480, 640):
                depth = cv2.resize(depth, (640, 480))
            ds_snapshot = DroneState()
            ds_snapshot.copy_from(self._drone_state)
            now = time.time()
            with self._depth_lock:
                self._latest_depth             = depth.copy()
                self._latest_depth_drone_state = ds_snapshot
                self._latest_depth_timestamp   = now
        except Exception:
            pass

    def _path_has_collision(self, start: np.ndarray, end: np.ndarray,
                            n_checks: int = 8) -> bool:
        """Check whether the line start->end passes through a collision area (LOS check)."""
        if not self._esdf.is_ready():
            return False
        for k in range(1, n_checks + 1):
            pt = start + (k / (n_checks + 1)) * (end - start)
            if self._esdf.has_collision(pt):
                return True
        return False

    # ── Sample quality filters ────────────────────────────────────────────────
    def _check_drone_safe(self, drone_state: DroneState) -> bool:
        if not self._esdf.is_ready():
            return True
        drone_pos = drone_state.global_pos[:2]
        return float(self._esdf.get_edt_dis(drone_pos)) > SAMPLE_MIN_ESDF

    def _check_velocity_sane(self, drone_state: DroneState) -> bool:
        speed = float(np.linalg.norm(drone_state.global_vel[:2]))
        return speed < SAMPLE_MAX_SPEED

    def _check_trajectory_safe(self, int_wpts: np.ndarray,
                               ts: np.ndarray,
                               head_pos: np.ndarray,
                               tail_pos: np.ndarray) -> bool:
        if not self._esdf.is_ready():
            return True
        wpts = [head_pos, int_wpts[:2, 0], int_wpts[:2, 1], tail_pos]
        n_segments = len(wpts) - 1
        pts_per_segment = max(2, TRAJ_CHECK_POINTS // n_segments)
        for i in range(n_segments):
            p_start = wpts[i]
            p_end   = wpts[i + 1]
            for t in np.linspace(0, 1, pts_per_segment, endpoint=False):
                sample_pt = p_start + t * (p_end - p_start)
                if float(self._esdf.get_edt_dis(sample_pt)) < TRAJ_MIN_ESDF:
                    return False
        if float(self._esdf.get_edt_dis(wpts[-1])) < TRAJ_MIN_ESDF:
            return False
        return True

    # ── EKF Stability Wait ────────────────────────────────────────────────────
    def _wait_ekf_stable(self, tol: float = 0.08, stable_dur: float = 3.0, timeout: float = 45.0) -> float:
        """
        Wait for the PX4 EKF to converge. The SITL barometer needs 20-30s to stabilize.

        Strategy: wait for the std-dev of z over the last 10 samples to be < tol for stable_dur.
        More accurate than min-z because it detects actual convergence, not just the minimum
        point which can be wrong if the initial EKF drift is very low.
        Ported from expert_manager_node.py (backup).
        """
        self.get_logger().info(
            f"[Expert] Waiting for EKF Z to stabilize (tol={tol}m, dur={stable_dur}s)...")
        t0 = time.time()
        z_history = []
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
                            f"[Expert] ✓ EKF stable. ground_z={z_ground:.3f}m "
                            f"(std={z_std:.4f})")
                        return z_ground
                else:
                    stable_start = None
            elapsed = time.time() - t0
            if int(elapsed) % 5 == 0 and elapsed % 1.0 < 0.15:
                z_std_log = float(np.std(z_history[-10:])) if len(z_history) >= 10 else 999.0
                self.get_logger().info(
                    f"[Expert] EKF Z={z_now:.3f}m std={z_std_log:.4f} ({elapsed:.0f}s/{timeout:.0f}s)...")
            time.sleep(0.1)
        z_ground = float(np.mean(z_history[-10:])) if z_history else 0.0
        self.get_logger().warn(
            f"[Expert] EKF stable timeout — using ground_z={z_ground:.3f}m")
        return z_ground

    # ── FIX ALTITUDE: EKF config + GPS-AGL trim ──────────────────────────────
    def _set_px4_param_int(self, name: str, value: int,
                           timeout: float = 5.0) -> bool:
        """Set a PX4 parameter via MAVROS ParamSetV2 (best-effort)."""
        if not self._param_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(
                f"[Expert] /mavros/param/set not available — {name} skipped")
            return False
        req = ParamSetV2.Request()
        req.param_id = name
        pv = RclParameterValue()
        pv.type          = ParameterType.PARAMETER_INTEGER
        pv.integer_value = int(value)
        req.value = pv
        fut = self._param_client.call_async(req)
        t0  = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                self.get_logger().warn(f"[Expert] Set {name} timeout")
                return False
            time.sleep(0.05)
        res = fut.result()
        ok  = bool(res and res.success)
        if ok:
            self.get_logger().info(f"[Expert] ✓ PX4 param {name} = {value}")
        else:
            self.get_logger().warn(f"[Expert] ✗ Failed to set PX4 param {name}")
        return ok

    def _set_px4_param_float(self, name: str, value: float,
                             timeout: float = 5.0) -> bool:
        """Set a PX4 (double) parameter via MAVROS ParamSetV2 (best-effort)."""
        if not self._param_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(
                f"[Expert] /mavros/param/set not available — {name} skipped")
            return False
        req = ParamSetV2.Request()
        req.param_id = name
        pv = RclParameterValue()
        pv.type         = ParameterType.PARAMETER_DOUBLE
        pv.double_value = float(value)
        req.value = pv
        fut = self._param_client.call_async(req)
        t0  = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                self.get_logger().warn(f"[Expert] Set {name} timeout")
                return False
            time.sleep(0.05)
        res = fut.result()
        ok  = bool(res and res.success)
        if ok:
            self.get_logger().info(f"[Expert] ✓ PX4 param {name} = {value}")
        else:
            self.get_logger().warn(f"[Expert] ✗ Failed to set PX4 param {name}")
        return ok

    def _configure_px4_height_ref(self):
        """
        Make sure the EKF uses GPS as the height reference (EKF2_HGT_REF=1).
        The SITL barometer drifts several meters within minutes; with a baro
        reference the local Z frame drifts too -> the physical drone descends
        even though EKF z is constant.
        No longer depends on manually editing the 4001_gz_x500 airframe file.
        """
        self._set_px4_param_int("EKF2_HGT_REF", 1)   # 1 = GPS

        # Cap the PX4 horizontal speed so the PHYSICAL drone does not shoot to
        # 5–18 m/s when there is a position error (discontinuity/resume recovery)
        # — v_max=1.0 only lives in the planner, it is not enforced on PX4.
        # Without this cap, overshoot in tight corridors + collision impulses
        # trigger a PHYSICS EXPLOSION.
        vmax_phys = self._v_max * 1.5    # hard cap slightly above the planner v_max
        self._set_px4_param_float("MPC_XY_VEL_MAX", vmax_phys)
        self._set_px4_param_float("MPC_XY_CRUISE",  self._v_max)
        self._set_px4_param_float("MPC_XY_VEL_ALL", vmax_phys)

    def _altitude_trim_cb(self):
        """
        Slowly correct cruise_z using AGL from raw GPS (anti baro/EKF drift).
        AGL_gps = gps_alt_now - gps_alt_home. If the physical drone is lower than
        target_alt, cruise_z is raised (and vice versa), in small clamped steps.
        """
        if (not ALT_TRIM_ENABLED
                or self._mission_state != self.STATE_COLLECTING
                or self._cruise_z is None
                or self._cruise_z_base is None
                or self._gps_alt_home is None
                or self._recovery_active
                or self._out_of_bounds_return
                or len(self._gps_alt_buf) < ALT_TRIM_FILTER_N):
            return
        # Long-window MEDIAN -> drop outliers & damp GPS noise before correcting
        agl = float(np.median(self._gps_alt_buf[-ALT_TRIM_FILTER_N:])) - self._gps_alt_home
        err = self._alt - agl                      # >0 -> physical drone too low
        if abs(err) < ALT_TRIM_DEADBAND:
            return
        step = float(np.clip(err * 0.5, -ALT_TRIM_MAX_STEP, ALT_TRIM_MAX_STEP))
        new_cruise = self._cruise_z + step
        # Clamp total trim so it does not run away when GPS noise is large
        new_cruise = float(np.clip(
            new_cruise,
            self._cruise_z_base - ALT_TRIM_MAX_TOTAL,
            self._cruise_z_base + ALT_TRIM_MAX_TOTAL))
        # SAFETY FLOOR: vertical GPS-AGL can have a large bias and push cruise_z
        # down continuously until the drone touches the ground (the main cause of
        # "accidental landing"). Never trim cruise_z below half the target
        # altitude above the ground.
        cruise_floor = self._ground_z + 0.5 * self._alt
        if new_cruise < cruise_floor:
            new_cruise = cruise_floor
        if abs(new_cruise - self._cruise_z) > 1e-4:
            self.get_logger().info(
                f"[Expert] ⇅ Alt-trim: AGL(gps)={agl:.2f}m target={self._alt:.2f}m "
                f"→ cruise_z {self._cruise_z:.2f} → {new_cruise:.2f}",
                throttle_duration_sec=4.0)
            self._cruise_z = new_cruise

    # ── Setpoint publisher (v25 — Full PX4 Z Control) ─────────────────────────
    def _publish_cmd(self):
        if self._recovery_active:
            return  # Stop the setpoint so PX4 leaves OFFBOARD mode during recovery
        msg = PositionTarget()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = "map"
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        # IMPORTANT: IGNORE_VZ so PX4 handles Z fully (prevents oscillation)
        msg.type_mask = (
            PositionTarget.IGNORE_VZ   |
            PositionTarget.IGNORE_AFX  |
            PositionTarget.IGNORE_AFY  |
            PositionTarget.IGNORE_AFZ  |
            PositionTarget.IGNORE_YAW_RATE
        )

        # ── ESCAPE MODE ───────────────────────────────────────────────────────
        if (self._mission_state == self.STATE_COLLECTING
                and self._escape_active
                and self._cruise_z is not None):
            drone_pos  = self._drone_state.global_pos[:2]
            target_pos = drone_pos + self._escape_velocity_xy * 0.5

            msg.position.x = float(target_pos[0])
            msg.position.y = float(target_pos[1])
            msg.position.z = float(self._cruise_z)
            msg.velocity.x = float(self._escape_velocity_xy[0])
            msg.velocity.y = float(self._escape_velocity_xy[1])
            msg.velocity.z = 0.0   # Left to PX4

            vel_xy = self._escape_velocity_xy
            if abs(vel_xy[0]) > 0.05 or abs(vel_xy[1]) > 0.05:
                msg.yaw = math.atan2(vel_xy[1], vel_xy[0])
            else:
                msg.yaw = self._drone_state.yaw

            self._pub_sp.publish(msg)
            return

        # ── ALTITUDE FIX at the turnaround point (waypoint) ──────────────────
        # Hold XY in place & let PX4 return z to cruise_z before continuing
        # navigation (see _fix_altitude_at_waypoint).
        if (self._mission_state == self.STATE_COLLECTING
                and self._alt_fix_active
                and self._cruise_z is not None):
            hold_xy = self._safe_hold_xy(self._alt_fix_xy)
            msg.position.x = float(hold_xy[0])
            msg.position.y = float(hold_xy[1])
            msg.position.z = float(self._cruise_z)
            msg.velocity.x = 0.0
            msg.velocity.y = 0.0
            msg.velocity.z = 0.0
            msg.yaw = float(self._drone_state.yaw)
            self._pub_sp.publish(msg)
            return

        # ── RETURN TO START (out-of-bounds recovery) ─────────────────────────
        # No planner/ESDF — send a setpoint straight to (0,0) so the drone can
        # return from any position outside the map.
        if (self._mission_state == self.STATE_COLLECTING
                and self._out_of_bounds_return
                and self._cruise_z is not None):
            drone_pos = self._drone_state.global_pos[:2]
            home      = np.array([0.0, 0.0])
            dir_home  = home - drone_pos
            dist      = float(np.linalg.norm(dir_home))
            if dist > 0.1:
                dir_home = dir_home / dist
            return_speed = float(np.clip(dist * 0.3, 0.4, 1.5))
            vel = dir_home * return_speed
            pos = drone_pos + dir_home * min(2.0, dist)

            msg.position.x = float(pos[0])
            msg.position.y = float(pos[1])
            msg.position.z = float(self._cruise_z)
            msg.velocity.x = float(vel[0])
            msg.velocity.y = float(vel[1])
            msg.velocity.z = 0.0
            msg.yaw = float(math.atan2(dir_home[1], dir_home[0]))
            self._pub_sp.publish(msg)
            return

        # ── NORMAL MODE ───────────────────────────────────────────────────────
        if (self._mission_state == self.STATE_COLLECTING
                and self._cruise_z is not None):
            t_now = time.time()
            with self._traj_lock:
                pos, vel, acc = self._traj.sample_at(t_now)
            if pos is None:
                pos = self._drone_state.global_pos[:2]
                vel = np.zeros(2)

            drone_pos = self._drone_state.global_pos[:2]
            gap = np.linalg.norm(pos - drone_pos)

            # While alt-rcv is active: the trajectory is intentionally invalid / hold
            # position. The gap between traj_pos and drone_pos is meaningless — just reset the timer.
            if self._alt_recovery_active:
                with self._emergency_stop_lock:
                    self._emergency_stop_first_t = None
            elif gap > EMERGENCY_STOP_THRESHOLD:
                with self._emergency_stop_lock:
                    if self._emergency_stop_first_t is None:
                        self._emergency_stop_first_t = t_now
                duration = t_now - self._emergency_stop_first_t

                self.get_logger().warn(
                    f"[Expert] Emergency stop! gap {gap:.1f}m > thresh {EMERGENCY_STOP_THRESHOLD:.1f}m "
                    f"(persist={duration:.1f}s)",
                    throttle_duration_sec=1.0)

                if duration > EMERGENCY_STOP_PERSIST_TIME:
                    with self._emergency_stop_lock:
                        self._emergency_stop_first_t = None
                    with self._traj_lock:
                        self._traj.invalidate()
                    self.get_logger().warn(
                        f"[Expert] Emergency stop persisted {duration:.1f}s → ESCAPE")
                    self._enter_escape_safe()
                    pos = drone_pos
                    vel = np.zeros(2)
            else:
                with self._emergency_stop_lock:
                    self._emergency_stop_first_t = None

            # ── Altitude recovery ─────────────────────────────────────────────
            # If z_err is large (after a collision), hold the XY position first
            # until the altitude recovers, then continue navigation.
            actual_z  = self._drone_state.global_pos[2]
            z_err_now = self._cruise_z - actual_z

            if abs(z_err_now) > ALT_RECOVERY_THRESH:
                # ARMING: do not freeze immediately. The drone is only frozen if z_err
                # stays > THRESH for ALT_RECOVERY_ARM_T (prevents a noise spike from
                # freezing navigation). During arming, XY navigation keeps running.
                if not self._alt_recovery_active:
                    if self._alt_recovery_arm_t is None:
                        self._alt_recovery_arm_t = t_now
                    elif t_now - self._alt_recovery_arm_t >= ALT_RECOVERY_ARM_T:
                        self._alt_recovery_active   = True
                        self._alt_recovery_stable_t = None
                        self._stuck_since           = None   # reset so the stuck timer does not fire during hold
                        self.get_logger().warn(
                            f"[Expert] ⚠ Alt-recovery EMERGENCY: z_err={z_err_now:+.2f}m "
                            f"persistent — holding XY position until altitude recovers")
                        with self._traj_lock:
                            self._traj.invalidate()
                if self._alt_recovery_active:
                    # Hold the current position, let PX4 return the drone to cruise_z
                    hold_xy = self._drone_state.global_pos[:2]
                    pos = hold_xy
                    vel = np.zeros(2)
                    self._alt_recovery_stable_t = None
            else:
                # z_err recovered below THRESH -> cancel the arming in progress
                self._alt_recovery_arm_t = None
            if (not (abs(z_err_now) > ALT_RECOVERY_THRESH)
                    and self._alt_recovery_active):
                if abs(z_err_now) < ALT_RECOVERY_EXIT:
                    if self._alt_recovery_stable_t is None:
                        self._alt_recovery_stable_t = t_now
                    elif t_now - self._alt_recovery_stable_t >= ALT_RECOVERY_STABLE_T:
                        self._alt_recovery_active   = False
                        self._alt_recovery_stable_t = None
                        self.get_logger().info(
                            f"[Expert] ✓ Alt-recovery done (z_err={z_err_now:+.2f}m) "
                            f"— resume navigation")
                else:
                    self._alt_recovery_stable_t = None
                # Still in cooldown — hold position
                hold_xy = self._safe_hold_xy(self._drone_state.global_pos[:2])
                pos = hold_xy
                vel = np.zeros(2)

            msg.position.x = float(pos[0])
            msg.position.y = float(pos[1])
            msg.position.z = float(self._cruise_z)
            msg.velocity.x = float(vel[0])
            msg.velocity.y = float(vel[1])
            msg.velocity.z = 0.0   # Left to PX4

            if abs(vel[0]) > 0.05 or abs(vel[1]) > 0.05:
                msg.yaw = math.atan2(vel[1], vel[0])
            else:
                msg.yaw = self._drone_state.yaw

        else:
            pos      = self._drone_state.global_pos
            target_z = (self._cruise_z if self._cruise_z is not None
                        else self._alt)

            msg.position.x = float(pos[0])
            msg.position.y = float(pos[1])
            msg.position.z = float(target_z)
            msg.velocity.x = 0.0
            msg.velocity.y = 0.0
            msg.velocity.z = 0.0   # Left to PX4
            msg.yaw        = self._drone_state.yaw

        self._pub_sp.publish(msg)

    # ── Escape mode helpers ───────────────────────────────────────────────────
    def _enter_escape_safe(self):
        if self._escape_active:
            return
        if not self._esdf.is_ready():
            return
        self._enter_escape()

    def _is_out_of_bounds(self, pos: np.ndarray) -> bool:
        return (pos[0] < self._arena_x_min or pos[0] > self._arena_x_max or
                pos[1] < self._arena_y_min or pos[1] > self._arena_y_max)

    def _is_physics_explosion(self) -> bool:
        speed = float(np.linalg.norm(self._drone_state.global_vel))
        return speed > PHYSICS_EXPLOSION_SPD

    def _gz_reset_model(self, world: str = "poles", model: str = "x500_depth_0") -> bool:
        """Reset the Gazebo model position to origin via the gz CLI so the EKF can recover."""
        import subprocess
        req = ('name: "' + model + '" '
               'position: {x: 0.0 y: 0.0 z: 0.3} '
               'orientation: {w: 1.0 x: 0.0 y: 0.0 z: 0.0}')
        cmd = [
            "gz", "service",
            "-s", f"/world/{world}/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req", req,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=5.0)
            ok = r.returncode == 0
            if ok:
                self.get_logger().info(f"[Expert] ✓ Gazebo model reset to origin")
            else:
                self.get_logger().warn(f"[Expert] gz reset failed: {r.stderr.decode()[:80]}")
            return ok
        except Exception as e:
            self.get_logger().warn(f"[Expert] gz reset exception: {e}")
            return False

    def _recover_from_explosion(self) -> bool:
        """
        Recover from a Gazebo physics explosion without restarting SITL.
        Sequence: disarm -> gz reset model to origin -> wait for EKF stable -> re-arm -> OFFBOARD -> takeoff.
        Returns True on success, False if it must shut down.
        """
        self.get_logger().warn("[Expert] ⚡ Recovery: LAND → disarm → Gazebo model reset...")

        # 1. Invalidate the trajectory and STOP setpoint publishing
        self._recovery_active = True
        self._mission_state = self.STATE_IDLE
        self._escape_active        = False
        self._out_of_bounds_return = False
        self._alt_recovery_active  = False
        self._replan_fail_count    = 0
        with self._traj_lock:
            self._traj.invalidate()

        # 1b. Switch to LAND mode so PX4 brings the drone down
        # _recovery_active=True stops the setpoint -> PX4 OFFBOARD watchdog times out
        # -> PX4 automatically falls back to HOLD/LAND mode within ~0.5s
        time.sleep(1.0)  # wait for the OFFBOARD watchdog timeout
        self._set_mode("LAND")
        t_land = time.time()
        while time.time() - t_land < 20.0:
            z   = self._drone_state.global_pos[2]
            spd = float(np.linalg.norm(self._drone_state.global_vel))
            if z < self._ground_z + 1.2 and spd < 2.0:
                self.get_logger().info(
                    f"[Expert] Drone approaching ground (z={z:.2f}m spd={spd:.2f}m/s) — disarm")
                break
            time.sleep(0.3)

        # 1c. Disarm — after the drone has landed/almost landed
        self._arm(False)
        time.sleep(1.0)

        # 2. Reset the Gazebo model to origin, wait for EKF stable
        landed = False
        new_ground_z = None
        for reset_attempt in range(5):
            self._gz_reset_model()
            # Wait for EKF convergence after the teleport (replaces sleep + z check)
            new_ground_z = self._wait_ekf_stable(tol=0.10, stable_dur=3.0, timeout=25.0)
            spd = float(np.linalg.norm(self._drone_state.global_vel))
            # Accept if the EKF is stable at a reasonable altitude
            if abs(new_ground_z) < 2.5 and spd < EXPLOSION_GROUND_SPD_THR:
                landed = True
                break
            self.get_logger().warn(
                f"[Expert] gz reset #{reset_attempt+1} EKF unstable "
                f"(ground_z={new_ground_z:.2f}m spd={spd:.1f}m/s) — retry")
            # Force LAND again so the drone descends before the next reset
            self._set_mode("LAND")
            self._arm(False)
            time.sleep(5.0)

        if not landed:
            self.get_logger().error(
                f"[Expert] Recovery failed after 5 resets "
                f"(z={self._drone_state.global_pos[2]:.1f}m)")
            self._recovery_active = False
            return False

        self.get_logger().info(
            f"[Expert] ✓ Drone on the ground "
            f"(z={self._drone_state.global_pos[2]:.2f}m) — ready to re-arm")

        # 3. Re-calibrate ground_z from the now-stable EKF (not raw pos)
        self._ground_z = new_ground_z
        self._cruise_z = self._ground_z + self._alt
        self._cruise_z_base = self._cruise_z
        # Re-anchor the GPS-AGL reference: the drone is on the ground again
        if len(self._gps_alt_buf) >= 5:
            self._gps_alt_home = float(np.mean(self._gps_alt_buf[-10:]))
        self.get_logger().info(
            f"[Expert] Recovery ground_z={self._ground_z:.3f}m → cruise_z={self._cruise_z:.2f}m")

        # Re-enable setpoint publishing before warm_up
        self._recovery_active = False
        self._warm_up(n=50)

        # 4. Re-arm
        t0 = time.time(); armed = False
        while time.time() - t0 < self._arm_timeout:
            if self._arm(True):
                time.sleep(0.8)
                if self._armed:
                    armed = True
                    break
            time.sleep(2.0)
        if not armed:
            self.get_logger().error("[Expert] Recovery failed: ARM did not succeed")
            return False

        # 5. OFFBOARD
        if not self._set_mode("OFFBOARD"):
            self._arm(False)
            self.get_logger().error("[Expert] Recovery failed: OFFBOARD did not succeed")
            return False
        t0 = time.time()
        while self._mode != "OFFBOARD" and time.time() - t0 < 5.0:
            time.sleep(0.1)

        # 6. Takeoff back to cruise_z
        self._mission_state = self.STATE_TAKEOFF
        self.get_logger().info(f"[Expert] Recovery takeoff → z={self._cruise_z:.2f}m...")
        if not self._wait_altitude(self._cruise_z, tol=0.3, timeout=30.0):
            self.get_logger().warn("[Expert] Recovery takeoff not stable — continuing anyway")
        self.get_logger().info(
            f"[Expert] ✓ Recovery succeeded! z={self._drone_state.global_pos[2]:.2f}m "
            f"| samples={self._n_samples}")

        self._mission_state = self.STATE_COLLECTING
        return True

    # ── v27: detect & recover "accidental landing" (no gz reset) ──────────────
    def _is_accidental_landing(self) -> bool:
        """
        True if a drone that SHOULD be flying has instead touched the ground.
        Two independent signals (no Gazebo visual needed):
          (a) PX4 auto-disarm (armed->False) while the mission is flying, OR
          (b) physical z drops near ground_z and persists for LAND_RECOVER_PERSIST seconds.
        """
        if self._mission_state not in (self.STATE_TAKEOFF, self.STATE_COLLECTING):
            self._near_ground_since = None
            return False
        if self._recovery_active or self._out_of_bounds_return:
            self._near_ground_since = None
            return False

        # (a) PX4 disarmed itself -> it must have landed/entered landed-state
        if not self._armed:
            self.get_logger().warn("[Expert] ⚠ Drone disarmed while flying (PX4 auto-land)")
            self._near_ground_since = None
            return True

        # (b) physical z hugging the ground despite still armed (land-detector not triggered yet)
        z = self._drone_state.global_pos[2]
        if z < self._ground_z + LAND_RECOVER_Z_THR:
            if self._near_ground_since is None:
                self._near_ground_since = time.time()
            elif time.time() - self._near_ground_since > LAND_RECOVER_PERSIST:
                self.get_logger().warn(
                    f"[Expert] ⚠ Drone touched the ground (z={z:.2f}m, "
                    f"ground_z={self._ground_z:.2f}m) while flying")
                self._near_ground_since = None
                return True
        else:
            self._near_ground_since = None
        return False

    def _recover_landing(self) -> bool:
        """
        Lightweight recovery from an accidental landing — NO gz model reset.
        The drone is already at a sane position (on the ground), so it is enough to:
        disarm -> re-anchor ground_z & GPS home -> warm-up -> re-arm -> OFFBOARD ->
        takeoff again -> resume COLLECTING. Works identically in SITL and reality.
        """
        self.get_logger().warn("[Expert] ⤴ Landing recovery: disarm → re-arm → takeoff again")

        # 1. Stop the setpoint & invalidate the trajectory (prevent bogus commands)
        self._recovery_active      = True
        self._mission_state        = self.STATE_IDLE
        self._escape_active        = False
        self._out_of_bounds_return = False
        self._alt_recovery_active  = False
        self._replan_fail_count    = 0
        self._near_ground_since    = None
        with self._traj_lock:
            self._traj.invalidate()

        # 2. Disarm first (deterministic) then wait for EKF stable on the ground
        self._arm(False)
        time.sleep(1.0)
        new_ground_z = self._wait_ekf_stable(tol=0.10, stable_dur=2.0, timeout=20.0)
        if abs(new_ground_z) > 2.5:
            self.get_logger().error(
                f"[Expert] Recovery aborted — ground_z insane ({new_ground_z:.2f}m)")
            self._recovery_active = False
            return False

        # 3. Re-anchor the altitude reference from the current position
        self._ground_z      = new_ground_z
        self._cruise_z      = self._ground_z + self._alt
        self._cruise_z_base = self._cruise_z
        if len(self._gps_alt_buf) >= 5:
            self._gps_alt_home = float(np.mean(self._gps_alt_buf[-10:]))
        self.get_logger().info(
            f"[Expert] Re-anchor ground_z={self._ground_z:.3f}m → cruise_z={self._cruise_z:.2f}m")

        # 4. Re-enable the setpoint -> warm-up -> re-arm -> OFFBOARD
        self._recovery_active = False
        self._warm_up(n=50)

        t0 = time.time(); armed = False
        while time.time() - t0 < self._arm_timeout:
            if self._arm(True):
                time.sleep(0.8)
                if self._armed:
                    armed = True
                    break
            time.sleep(2.0)
        if not armed:
            self.get_logger().error("[Expert] Landing recovery failed: ARM did not succeed")
            return False

        if not self._set_mode("OFFBOARD"):
            self._arm(False)
            self.get_logger().error("[Expert] Landing recovery failed: OFFBOARD did not succeed")
            return False
        t0 = time.time()
        while self._mode != "OFFBOARD" and time.time() - t0 < 5.0:
            time.sleep(0.1)

        # 5. Takeoff again to cruise_z then resume COLLECTING in place
        self._mission_state = self.STATE_TAKEOFF
        self.get_logger().info(f"[Expert] Recovery takeoff again → z={self._cruise_z:.2f}m...")
        if not self._wait_altitude(self._cruise_z, tol=0.3, timeout=30.0):
            self.get_logger().warn("[Expert] Recovery takeoff not stable — continuing anyway")
        self.get_logger().info(
            f"[Expert] ✓ Landing recovery succeeded! z={self._drone_state.global_pos[2]:.2f}m "
            f"| samples={self._n_samples}")
        self._mission_state = self.STATE_COLLECTING
        return True

    def _find_escape_direction(self) -> np.ndarray:
        drone_pos  = self._drone_state.global_pos[:2]
        global_tgt = self._global_target
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        best_score = -np.inf
        best_dir   = np.array([1.0, 0.0])

        to_goal = global_tgt - drone_pos
        to_goal_norm = np.linalg.norm(to_goal)
        to_goal_dir = to_goal / to_goal_norm if to_goal_norm > 1e-3 else np.array([1.0, 0.0])

        to_center = self._arena_center - drone_pos
        to_center_norm = np.linalg.norm(to_center)
        to_center_dir = to_center / to_center_norm if to_center_norm > 1e-3 else np.array([0.0, 0.0])

        # If outside the arena, bias strongly toward the arena center; inside, bias toward the goal
        out = self._is_out_of_bounds(drone_pos)
        center_w = 1.5 if out else 0.1
        goal_w   = 0.1 if out else 0.3

        # v1.11 FIX: escape used to be completely oblivious to
        # force_lap_alternation's forced_sign -- it only optimizes
        # clearance/goal/arena-center, so an escape during a forced leg
        # could (and, per real-flight measurement, did) push the drone to
        # the WRONG side; every replan afterward legitimately continues
        # from that position, so the violation isn't a one-off, it taints
        # the rest of the leg. Same lateral convention as _set_local_target
        # (lat[1]=left/+1, lat[0]=right/-1), using to_goal_dir as the
        # reference axis (closest available proxy to the local flight
        # axis at escape time). Small weight, added only among directions
        # that already pass the safety veto above -- never overrides
        # clearance/out-of-bounds recovery.
        forced_sign = None
        if self._use_topo_frontend:
            _fe = getattr(self._planner, "_topo_frontend", None)
            forced_sign = getattr(_fe, "forced_sign", None) if _fe else None
        forced_dir = None
        if forced_sign is not None:
            lat1 = np.array([-to_goal_dir[1], to_goal_dir[0]])   # left/+1
            forced_dir = lat1 if forced_sign == 1 else -lat1
        forced_w = 0.4

        for angle in angles:
            dir_vec   = np.array([np.cos(angle), np.sin(angle)])
            probe_pt  = drone_pos + dir_vec * ESCAPE_PROBE_DIST
            mid_pt    = drone_pos + dir_vec * (ESCAPE_PROBE_DIST * 0.5)
            esdf_dist = float(self._esdf.get_edt_dis(probe_pt))
            mid_dist  = float(self._esdf.get_edt_dis(mid_pt))
            clearance = min(esdf_dist, mid_dist)
            # Hard-block directions that already collide or are very close to an
            # obstacle — the goal bias must not override physical safety.
            if clearance < self._safe_dis_emer:
                score = clearance - 100.0
            else:
                goal_bias   = float(dir_vec @ to_goal_dir)
                center_bias = float(dir_vec @ to_center_dir)
                score = clearance + goal_w * goal_bias + center_w * center_bias
                if forced_dir is not None:
                    score += forced_w * float(dir_vec @ forced_dir)
            if score > best_score:
                best_score = score
                best_dir   = dir_vec
        return best_dir

    def _check_stuck(self) -> bool:
        # Based on NET DISPLACEMENT, not instantaneous speed. A stuck drone often
        # vibrates (speed occasionally > threshold); using instantaneous speed
        # would keep resetting the timer -> escape never activates -> drone hangs
        # for a long time. Here: if the drone does not move > STUCK_DRIFT_RESET
        # from the anchor point for STUCK_TIME_THRESHOLD seconds -> considered stuck.
        pos = self._drone_state.global_pos[:2].copy()
        now = time.time()
        if self._stuck_since is None or self._last_pos_for_stuck is None:
            self._stuck_since        = now
            self._last_pos_for_stuck = pos
            return False
        drift = float(np.linalg.norm(pos - self._last_pos_for_stuck))
        if drift > STUCK_DRIFT_RESET:        # really moving -> re-anchor
            self._stuck_since        = now
            self._last_pos_for_stuck = pos
            return False
        return (now - self._stuck_since) > STUCK_TIME_THRESHOLD

    def _fix_altitude_at_waypoint(self, tol: float = ALT_FIX_TOL,
                                  timeout: float = ALT_FIX_TIMEOUT):
        """
        Called when the drone reaches the turnaround point (goal / start). Hold the
        XY position in place and let PX4 return the altitude to cruise_z FIRST
        before continuing navigation — so altitude drift is not carried into the
        next leg. Lightly blocking; _publish_cmd (timer) keeps sending the hold
        setpoint via the _alt_fix_active branch.
        """
        if self._cruise_z is None:
            return
        self._alt_fix_xy     = self._drone_state.global_pos[:2].copy()
        self._alt_fix_active = True
        with self._traj_lock:
            self._traj.invalidate()
        self.get_logger().info(
            f"[Expert] ⤓ Fixing altitude at the turnaround point "
            f"(target z={self._cruise_z:.2f}m)...")
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < timeout:
            if abs(self._cruise_z - self._drone_state.global_pos[2]) < tol:
                break
            time.sleep(0.1)
        z_now = self._drone_state.global_pos[2]
        self.get_logger().info(
            f"[Expert] ✓ Altitude ready (z={z_now:.2f}m, "
            f"z_err={self._cruise_z - z_now:+.2f}m) — resume navigation")
        self._alt_fix_active = False
        self._stuck_since    = None   # reset so it is not immediately considered stuck

    def _advance_lap_leg(self):
        """Pilot (force_lap_alternation): called at every turnaround (start<->
        goal), i.e. once per leg. Moves to the next forced side in
        self._lap_phases = (+1, -1, None) and pushes it into the homotopy
        frontend (planner._topo_frontend.forced_sign) so the NEXT leg's
        replans can only ever find/fly that side near each obstacle. After
        the 3rd leg (None = free) finishes: if lap_alternation_loop=true
        (volume collecting), wrap back to leg 1 and keep cycling until
        target_samples; otherwise (pilot) stop collecting — see __init__."""
        self._lap_leg_idx += 1
        if self._lap_leg_idx >= len(self._lap_phases):
            if self._lap_loop:
                # collecting VOLUME: REPEAT the cycle (left, right, free)
                # until target_samples — do NOT stop at leg 3.
                self._lap_leg_idx = 0
                self.get_logger().info(
                    f"[Expert] Lap-alternation cycle complete — REPEATING from "
                    f"leg 1 (loop until {self._target_n} samples; currently "
                    f"{self._n_samples}).")
            else:
                self._lap_pilot_done = True
                self.get_logger().info(
                    f"[Expert] Lap-alternation PILOT complete (3 legs) — "
                    f"stop collecting ({self._n_samples} samples collected).")
                return
        side = self._lap_phases[self._lap_leg_idx]
        self._planner._topo_frontend.forced_sign = side
        side_str = ("left" if side == 1 else "right" if side == -1 else "free")
        self.get_logger().info(
            f"[Expert] Lap-alternation leg {self._lap_leg_idx + 1}/"
            f"{len(self._lap_phases)}: forced_sign={side} ({side_str})")

    def _planner_escape_allowed(self) -> bool:
        """
        Escape due to REPLAN FAILURE is only allowed when the drone is truly hugging
        an obstacle (clearance < ESCAPE_MIN_CLEAR). In open space, a replan failure
        is handled by shrinking safe_dis & retrying — not by an escape that only
        jitters and resets the shrink count.
        """
        if not self._esdf.is_ready():
            return True
        clr = float(self._esdf.get_edt_dis(self._drone_state.global_pos[:2]))
        return clr < ESCAPE_MIN_CLEAR

    def _dynamic_safe_dis(self) -> float:
        """
        Dynamic safety distance (CENTER-OF-MASS metres): the more consecutive replan
        failures (trapped), the smaller the SOFT margin so the planner can slip
        through tight gaps. Never drops below SAFE_DIS_TRAP_MIN (= the hard barrier),
        so the hard anti-collision floor is preserved. The caller converts to the
        body-edge frame before handing it to MINCO.
        """
        shrink = SAFE_DIS_TRAP_STEP * self._replan_fail_count
        return max(SAFE_DIS_TRAP_MIN, self._safe_dis_norm - shrink)

    def _safe_hold_xy(self, xy) -> np.ndarray:
        """
        Obstacle-aware position-hold point. When alt-recovery / alt-fix holds XY,
        if the hold point is too close to an obstacle, push it away along the ESDF
        gradient (direction of increasing distance). Prevents the drone from
        "sitting" right next to a pole — the cause of esdf=0.00 then a PHYSICS
        EXPLOSION during alt-rcv.
        """
        xy = np.asarray(xy, dtype=float)[:2]
        if not self._esdf.is_ready():
            return xy
        d = float(self._esdf.get_edt_dis(xy))
        if d >= self._safe_dis_norm:
            return xy
        g = np.asarray(self._esdf.get_edt_grad(xy), dtype=float)
        gn = float(np.linalg.norm(g))
        if gn < 1e-6:
            return xy
        push = self._safe_dis_norm - d        # push by the amount of missing margin
        return xy + (g / gn) * push

    def _enter_escape(self):
        if not self._use_escape:
            # Escape disabled: skip the scripted-velocity maneuver, just clear
            # the stuck/fail bookkeeping so _check_stuck() re-anchors instead
            # of re-triggering every cycle, and force a fresh replan attempt.
            self._replan_fail_count  = 0
            self._stuck_since        = None
            self._last_pos_for_stuck = None
            with self._traj_lock:
                self._traj.invalidate()
            with self._emergency_stop_lock:
                self._emergency_stop_first_t = None
            self.get_logger().warn(
                "[Expert] Escape disabled (use_escape=false) — forcing fresh "
                "replan instead of a scripted escape maneuver")
            return

        escape_dir = self._find_escape_direction()
        self._escape_velocity_xy = escape_dir * ESCAPE_VELOCITY
        self._escape_active      = True
        self._escape_start_time  = time.time()
        self._replan_fail_count  = 0
        with self._traj_lock:
            self._traj.invalidate()
        with self._emergency_stop_lock:
            self._emergency_stop_first_t = None
        self.get_logger().warn(
            f"[Expert] >>> ESCAPE: vel=({escape_dir[0]:.2f}, "
            f"{escape_dir[1]:.2f}) × {ESCAPE_VELOCITY}m/s")

    def _exit_escape(self):
        if self._escape_active:
            self.get_logger().info("[Expert] <<< Exit escape mode")
            self._escape_active     = False
            self._escape_start_time = None
            self._stuck_since       = None

    def _should_exit_escape(self) -> bool:
        if not self._escape_active:
            return False
        if (self._escape_start_time is not None
                and time.time() - self._escape_start_time > ESCAPE_MAX_TIME):
            self.get_logger().warn("[Expert] Escape timeout — exit")
            return True
        drone_pos = self._drone_state.global_pos[:2]
        esdf_dist = self._esdf.get_edt_dis(drone_pos)
        if esdf_dist > ESCAPE_EXIT_DIST:
            return True
        future_pt = drone_pos + self._escape_velocity_xy * 1.0
        future_dist = self._esdf.get_edt_dis(future_pt)
        if future_dist < esdf_dist * 0.8:
            new_dir = self._find_escape_direction()
            self._escape_velocity_xy = new_dir * ESCAPE_VELOCITY
            self.get_logger().info(
                f"[Expert] Change escape direction: ({new_dir[0]:.2f}, {new_dir[1]:.2f})")
        return False

    # ── Continuation seed (v1.13) ─────────────────────────────────────────────
    def _set_anchor_seed(self, t_ahead: float):
        """Offer the remaining ACTIVE trajectory (still ahead of the head) as
        an additional MINCO seed for this replan — read by topo_batch_plan via
        `frontend.anchor_seed`. None = no continuation (trajectory exhausted/
        invalid/behind the head) -> old behavior (plan from scratch).

        Why: min-jerk always places the turn in the MIDDLE of a segment, while
        each cycle only flies the FIRST ~1 m; replanning from scratch keeps
        pushing that turn "later" over and over -> the drone only diverts once
        the obstacle already fills ~the entire view, then the replan fails ->
        escape. With this seed, the turn stays geometrically in place so it is
        actually executed."""
        fe = getattr(self._planner, "_topo_frontend", None)
        if fe is None:
            return
        fe.anchor_seed = None
        if not (self._use_plan_continuity and self._use_topo_frontend):
            return
        n = int(self._planner.init_wpts_num)          # 2 -> 3 pieces
        t_head = time.time() + t_ahead
        with self._traj_lock:
            if not self._traj.is_valid():
                return
            remain = self._traj.time_remaining(t_head)
            if remain < (n + 1) * self._planner.T_min + 0.2:
                return                                # remaining too short
            head_p, _, _ = self._traj.sample_at(t_head)
            wpts = []
            for k in range(1, n + 1):                 # 1/3 and 2/3 of the remainder
                p, _, _ = self._traj.sample_at(t_head + remain * k / (n + 1))
                if p is None:
                    return
                wpts.append(np.asarray(p)[:2])
        if head_p is None:
            return
        if float(np.linalg.norm(wpts[0] - np.asarray(head_p)[:2])) < 0.3:
            return                                    # w1 not far enough ahead
        int_wpts = np.array([[w[0] for w in wpts], [w[1] for w in wpts]])
        ts_piece = float(np.clip(remain / (n + 1),
                                 self._planner.T_min + 1e-3,
                                 self._planner.T_max - 1e-3))
        fe.anchor_seed = (int_wpts, np.full(n + 1, ts_piece))

    def _clear_anchor_seed(self):
        """Continuity broken (escape / discontinuity / trajectory discarded) ->
        the next replan must truly start from scratch."""
        fe = getattr(self._planner, "_topo_frontend", None)
        if fe is not None:
            fe.anchor_seed = None

    # ── Local target ──────────────────────────────────────────────────────────
    def _set_local_target(self, seed: int = 0, shorter: bool = False,
                          ref_pos: np.ndarray = None):
        if ref_pos is None:
            curr_pos = self._drone_state.global_pos[:2]
        else:
            curr_pos = ref_pos

        global_tgt = self._global_target
        step = 2.0 if shorter else self._longitu_step
        dist = np.linalg.norm(global_tgt - curr_pos)

        fwd = global_tgt - curr_pos
        fwd_norm = np.linalg.norm(fwd)
        fwd = fwd / fwd_norm if fwd_norm > 1e-6 else np.array([1.0, 0.0])
        # lat[0] = right (mode_sig -1), lat[1] = left (mode_sig +1) -- same
        # sign convention as compute_mode_signature()/_signature_for_path()
        # in homotopy_frontend.py (lat = axis_x*rel_y - axis_y*rel_x).
        lat = np.array([[fwd[1], -fwd[0]],
                        [-fwd[1],  fwd[0]]])

        # v1.11 (2026-07-09): read the active lap-alternation forced_sign (if
        # any) so BOTH nudge mechanisms below can respect it -- see fix note.
        forced_sign = None
        if self._use_topo_frontend:
            _fe = getattr(self._planner, "_topo_frontend", None)
            forced_sign = getattr(_fe, "forced_sign", None) if _fe else None

        if dist < step:
            # Already near the global target -> use the global target as the local target.
            # STILL go through the lateral-avoidance loop below: if the goal point hits
            # an obstacle (e.g. the poles-world goal (15,0) is only 1m from the pole at
            # (14,0) r=0.5m, so inside the safe_dis margin), the planner keeps failing &
            # the drone gets stuck. Nudging the local target to the nearest free point
            # keeps the goal reachable (the "reached" check uses dist<1.5 to the global target).
            self._near_global_target = True
            local_tgt = global_tgt.copy()
        else:
            self._near_global_target = False
            if seed > 0:
                jitter = np.random.normal(0, 0.5, 2)
                # v1.11 FIX: this jitter used to be fully isotropic/random,
                # oblivious to force_lap_alternation's forced_sign. During a
                # forced leg, a retry (seed>0, fired whenever the seed=0
                # attempt failed for ANY reason -- not necessarily related to
                # forced_sign) could randomly nudge the local target to the
                # WRONG side; the resulting straight head->target line then
                # legitimately encounters the obstacle from the wrong side,
                # so _BiasedESDF (homotopy_frontend.py) has nothing to
                # correct -- the recorded mode_sig comes out wrong not
                # because the MINCO-level fix failed, but because the RETRY
                # itself silently redefined "correct". Measured across all 8
                # training_worlds pilots (2026-07-09): 43.3% (55/127) of
                # non-free forced-leg samples violated forced_sign, heavily
                # concentrated in denser worlds where seed=0 fails more
                # often (more retries = more chances for random jitter to
                # land on the wrong side). Fix: reflect (not cancel) the
                # jitter's lateral component onto the forced side when it
                # would otherwise oppose forced_sign -- keeps the same
                # search magnitude/diversity (still helps escape a stuck
                # seed=0) without ever proposing a target that contradicts
                # the leg's forced side. Longitudinal jitter (along fwd) is
                # untouched since it doesn't affect side.
                if forced_sign is not None:
                    jit_lon = float(np.dot(jitter, fwd))
                    jit_lat = float(np.dot(jitter, lat[1]))   # + = left/+1
                    if jit_lat * forced_sign < 0:
                        jit_lat = -jit_lat
                    jitter = jit_lon * fwd + jit_lat * lat[1]
                local_tgt = curr_pos + step * fwd + jitter
            else:
                local_tgt = curr_pos + step * fwd

        # v1.11 FIX: when a side is forced, always nudge toward THAT side
        # (never alternate into the wrong one) -- same reasoning as the seed
        # jitter fix above, for the collision-avoidance loop below.
        lat_flag = 0 if forced_sign is None or forced_sign == -1 else 1
        lat_move = self._lateral_step
        max_lat_iter = 16
        n_iter = 0

        # v1.3 (2026-07-08): with the topo front-end active, do NOT nudge
        # local_tgt just because the STRAIGHT line to it is blocked — that
        # pre-resolves the left/right choice deterministically (always
        # tries the same `lat[0]` direction first) BEFORE topo_batch_plan
        # ever runs, so find_paths() is handed a tail_state already off to
        # one side and the straight head->tail line rarely still crosses
        # the obstacle -> the homotopy search sees a "free" (single-class)
        # scene instead of the genuine ambiguity that's actually there.
        # Root-caused via a from-scratch calibration run (box_narrow, 100
        # samples): only 1/100 scenes came back multimodal even after
        # fixing the frontend's own diversification (see homotopy_frontend
        # v1.3) — the frontend was fixed but rarely got the chance to see a
        # blocked line at all. Still nudge if the target POINT itself is in
        # collision (that the topo front-end cannot route around — its
        # tail_state must be a valid goal); only the line-of-sight
        # pre-check is skipped.
        check_path_los = not self._use_topo_frontend
        while (self._esdf.is_ready()
               and (self._esdf.has_collision(local_tgt)
                    or (check_path_los
                        and self._path_has_collision(curr_pos, local_tgt)))
               and n_iter < max_lat_iter):
            local_tgt += lat_move * lat[lat_flag]
            if forced_sign is None:
                lat_flag = 1 - lat_flag
            lat_move  += self._lateral_step
            n_iter    += 1

        # Clamp inside the arena so the local target does not stray far from the navigation area
        margin = self._lateral_step
        local_tgt = np.clip(
            local_tgt,
            [self._arena_x_min + margin, self._arena_y_min + margin],
            [self._arena_x_max - margin, self._arena_y_max - margin],
        )

        goal_dir  = global_tgt - local_tgt
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-3:
            goal_dir = goal_dir / goal_norm

        local_vel = goal_dir * self._v_max * 0.8
        self._target_state[0] = local_tgt
        self._target_state[1] = local_vel

    def _get_drone_state_at(self, t_future: float) -> DroneState:
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

    # ── Form training data ────────────────────────────────────────────────────
    def _form_model_input(self, depth_img, drone_state, plan_init_state, target_state):
        # FIXED-RANGE normalization (see DEPTH_NORM_MAX_M): 0=invalid/no
        # return, 255=beyond range (inf -> 10 m). Absolute scale preserved.
        d = np.nan_to_num(depth_img, nan=0.0,
                          posinf=DEPTH_NORM_MAX_M, neginf=0.0)
        depth_u8 = np.clip(
            d / DEPTH_NORM_MAX_M * 255.0, 0, 255).astype(np.uint8)

        quat            = drone_state.attitude
        drone_local_vel = drone_state.local_vel
        drone_R         = quat.rotation_matrix.reshape(-1)

        init_pos_3d = np.array([
            plan_init_state.global_pos[0],
            plan_init_state.global_pos[1],
            self._cruise_z or self._alt,
        ])
        init_vel_3d = np.array([
            plan_init_state.global_vel[0],
            plan_init_state.global_vel[1],
            0.0,
        ])
        init_pos_body = quat.inverse.rotate(init_pos_3d - drone_state.global_pos)
        init_vel_body = quat.inverse.rotate(init_vel_3d - drone_state.global_vel)

        tgt_pos_3d = np.array([
            target_state[0, 0], target_state[0, 1],
            self._cruise_z or self._alt,
        ])
        tgt_vel_3d = np.array([target_state[1, 0], target_state[1, 1], 0.0])
        tgt_pos_body = quat.inverse.rotate(tgt_pos_3d - drone_state.global_pos)
        tgt_vel_body = quat.inverse.rotate(tgt_vel_3d - drone_state.global_vel)

        motion_info = np.concatenate([
            drone_local_vel, drone_R,
            init_pos_body, init_vel_body,
            tgt_pos_body, tgt_vel_body,
        ]).astype(np.float32)

        return depth_u8, motion_info

    def _form_model_output(self, drone_state, int_wpts):
        quat      = drone_state.attitude
        M_minus_1 = int_wpts.shape[1]
        result    = np.zeros((3, M_minus_1))
        for i in range(M_minus_1):
            wpt_3d       = np.array([
                int_wpts[0, i], int_wpts[1, i],
                self._cruise_z or self._alt,
            ])
            result[:, i] = quat.inverse.rotate(wpt_3d - drone_state.global_pos)
        return result.T.reshape(-1)

    def _save_sample(self, depth_img, drone_state, plan_init_state,
                     target_state, solutions):
        """MULTI-HOMOTOPY (see CLAUDE.md): `solutions` = list
        (int_wpts, ts, mode_sig_str). A single scene can have multiple labels;
        all CSV rows share the SAME id and depth image (the nn_trainer/
        fm_trainer loader reads per row, so a duplicate id is safe).
        `mode_sig_str` (v1.5) = topological side-signature from
        compute_mode_signature() -- see the `mode_sig` column in CSV_HEADER."""
        ts_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        depth_u8, motion_info = self._form_model_input(
            depth_img, drone_state, plan_init_state, target_state)

        img_path = os.path.join(self._img_dir, f"{ts_str}.png")
        cv2.imwrite(img_path, depth_u8)

        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for int_wpts, ts, mode_sig_str in solutions:
                wpts_local = self._form_model_output(drone_state, int_wpts)
                writer.writerow(["t" + ts_str]
                                + motion_info.tolist()
                                + wpts_local.tolist()
                                + ts.tolist()
                                + [mode_sig_str])

        self._n_samples += len(solutions)

    # ── RL seed plan ────────────────────────────────────────────────────────────
    def _try_rl_seed_plan(self, head_state, tail_state):
        """Try the 2 dual-h RL seeds (left/right) BEFORE the old batch_plan.

        Each seed is optimized with a single plan_once (NO random-perturbation
        retry — warm_start_plan's retry replaces the seed with a generic one and
        breaks the dual-h structure). Every converged one enters all_solutions
        with the same format as batch_plan, so the multimodal recording pipeline
        (mode_sig dedup + _check_trajectory_safe) works unchanged; the cheapest
        becomes the executed solution. Returns True if >=1 seed converged;
        False -> the caller falls back to batch_plan (safe fallback).

        PLAN CONTINUITY (2026-07-21): the RL path used to replan from scratch
        every cycle, which re-defers the min-jerk turn each replan -> the drone
        hesitates -> _check_stuck() fires -> ESCAPE (the visible circling). Fix:
        also offer the active-trajectory anchor seed (set by _set_anchor_seed on
        the topo frontend, gated by use_plan_continuity) as an ADDITIONAL
        candidate, and — exactly like topo_batch_plan — fly it instead of the
        cheapest fresh plan when it is still reasonable (cost <=
        CONTINUITY_COST_FACTOR x cheapest). This keeps the turn fixed in world
        coordinates so it actually gets executed. All converged solutions
        (RL seeds + anchor) still reach the dataset via all_solutions.
        """
        try:
            seeds = self._rl_seeds.get_seeds(
                self._esdf, head_state[0], head_state[1],
                self._target_state[0], goal=self._global_target)
        except Exception as ex:
            self.get_logger().warn(
                f"[Expert] RL seed inference failed: {ex}",
                throttle_duration_sec=5.0)
            return False

        self._rl_try_count += 1

        # Build the candidate list: the continuity anchor FIRST (if present &
        # shape-compatible), then the 2 RL dual-h seeds. anchor_seed is set by
        # _set_anchor_seed on the frontend; None when continuity is off / the
        # trajectory was exhausted / invalid -> no anchor (old behavior).
        fe = getattr(self._planner, "_topo_frontend", None)
        anchor = getattr(fe, "anchor_seed", None) if fe is not None else None
        candidates = []   # (is_anchor, int_wpts, ts)
        if anchor is not None and len(seeds) > 0:
            a_w = np.asarray(anchor[0], float)
            a_ts = np.asarray(anchor[1], float)
            if a_w.shape == seeds[0][0].shape and a_ts.shape == seeds[0][1].shape:
                candidates.append((True, a_w, a_ts))
        for int_wpts, ts in seeds:
            candidates.append((False, int_wpts, ts))

        solutions   = []      # all converged (RL seeds + anchor) -> dataset
        best        = None    # cheapest converged
        anchor_sol  = None    # the anchor's optimized result (if it converged)
        rl_converged = 0      # how many of the 2 RL seeds converged
        for is_anchor, int_wpts, ts in candidates:
            try:
                self._planner.read_planning_conditions(
                    self._esdf, head_state, tail_state, int_wpts, ts.copy())
                self._planner.plan_once()
                cost = float(self._planner.weighted_cost.sum())
                sol = (self._planner.int_wpts.copy(),
                       self._planner.ts.copy(), cost)
                solutions.append(sol)
                if best is None or cost < best[2]:
                    best = sol
                if is_anchor:
                    anchor_sol = sol
                else:
                    rl_converged += 1
            except Exception:
                continue

        if best is None:
            return False

        # Prefer continuing the active trajectory when it is still reasonable
        # (<= CONTINUITY_COST_FACTOR x cheapest) — same rule as topo_batch_plan.
        if (anchor_sol is not None
                and anchor_sol[2] <= max(best[2], 1e-6) * CONTINUITY_COST_FACTOR):
            best = anchor_sol

        self._planner.all_solutions = solutions
        self._planner.int_wpts   = best[0].copy()
        self._planner.ts         = best[1].copy()
        self._planner.final_cost = best[2]
        self._rl_ok_count += 1
        if rl_converged == 2:
            self._rl_dual_count += 1
        self._viz_data = {
            "head": head_state.copy(), "tail": tail_state.copy(),
            "seeds": [(iw.copy(), t.copy()) for iw, t in seeds],
            "solutions": [(s[0].copy(), s[1].copy(), s[2]) for s in solutions],
            "best_cost": best[2],
            "target": self._target_state[0].copy(),
        }
        self._publish_viz()
        return True

    # ── Live RViz markers (RL seeds + MINCO candidates + targets) ──────────────
    def _sample_minjerk(self, int_wpts, ts, head, tail):
        try:
            p = self._viz_planner
            p.read_planning_conditions(self._esdf, head, tail,
                                       int_wpts.copy(), ts.copy())
            return p.get_pos_array()
        except Exception:
            return None

    def _publish_viz(self):
        d = self._viz_data
        if d is None:
            return
        z = self._alt
        arr = MarkerArray()
        clr = Marker(); clr.action = Marker.DELETEALL
        arr.markers.append(clr)
        stamp = self.get_clock().now().to_msg()
        idx = [0]

        def add_line(pts2d, rgba, width, ns):
            m = Marker()
            m.header.frame_id = "map"; m.header.stamp = stamp
            m.ns = ns; m.id = idx[0]; idx[0] += 1
            m.type = Marker.LINE_STRIP; m.action = Marker.ADD
            m.scale.x = width
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.pose.orientation.w = 1.0
            for p in pts2d:
                pt = Point(); pt.x = float(p[0]); pt.y = float(p[1]); pt.z = float(z)
                m.points.append(pt)
            arr.markers.append(m)

        def add_sphere(pos2d, rgba, size, ns):
            m = Marker()
            m.header.frame_id = "map"; m.header.stamp = stamp
            m.ns = ns; m.id = idx[0]; idx[0] += 1
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = size
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.pose.orientation.w = 1.0
            m.pose.position.x = float(pos2d[0]); m.pose.position.y = float(pos2d[1])
            m.pose.position.z = float(z)
            arr.markers.append(m)

        # MINIMAL 3-LINE VIEW (2026-07-24, user request for easier analysis):
        # ONLY the two raw RL seeds + the single EXECUTED MINCO trajectory.
        # No non-executed MINCO candidates, no start/target/goal spheres.
        #
        # 2 raw RL seeds (thin): blue = h=-1 (right), orange = h=+1 (left).
        # head -> waypoints -> tail. Populated only when use_rl_seeds=true; the
        # classic/topo path leaves d["seeds"] empty (then only the MINCO line shows).
        seed_cols = [(0.35, 0.85, 1.0, 0.9), (1.0, 0.72, 0.2, 0.9)]  # right, left
        for (iw, t), col in zip(d["seeds"], seed_cols):
            pts = ([d["head"][0]] + [iw[:, j] for j in range(iw.shape[1])]
                   + [d["tail"][0]])
            add_line(pts, col, 0.05, "rl_seed")

        # 1 EXECUTED MINCO trajectory (bright green, thick).
        for sol in d["solutions"]:
            if abs(sol[2] - d["best_cost"]) >= 1e-6:
                continue   # skip everything except the one actually flown
            traj = self._sample_minjerk(sol[0], sol[1], d["head"], d["tail"])
            if traj is not None and len(traj) >= 2:
                add_line(traj, (0.1, 1.0, 0.25, 0.95), 0.11, "minco")
            break
        self._pub_viz.publish(arr)

    # ── Replan + record ───────────────────────────────────────────────────────
    def _replan_and_record(self):
        if not self._esdf.is_ready():
            return

        # Returning to the arena — no replan, _publish_cmd handles motion
        if self._out_of_bounds_return:
            return

        if self._escape_active:
            if self._should_exit_escape():
                self._exit_escape()
            else:
                return

        # Hold replan during altitude recovery — also skip the stuck check
        # because the drone is intentionally still during alt-rcv (speed 0 is not "stuck")
        if self._alt_recovery_active:
            return

        # Hold replan while fixing altitude at the turnaround point
        if self._alt_fix_active:
            return

        if not self._escape_active and self._check_stuck():
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
                    f"[Expert] Skip sample, depth lag {lag*1000:.0f}ms "
                    f"(total skip: {self._lag_skip_count})",
                    throttle_duration_sec=2.0)
                return
            depth_snapshot = self._latest_depth.copy()
            sync_drone_state = DroneState()
            sync_drone_state.copy_from(self._latest_depth_drone_state)

        self._lag_history.append(lag)
        if len(self._lag_history) > 100:
            self._lag_history.pop(0)

        dist = np.linalg.norm(
            sync_drone_state.global_pos[:2] - self._global_target)
        if dist < 0.8:
            self._reached_target = True
            return

        # DYNAMIC safety distance — shrinks gradually when trapped.
        # All CENTER-OF-MASS metres here; convert to the body-edge frame the MINCO
        # collision cost compares against the (inflated) ESDF before handing it over.
        safe_dis_dyn = self._dynamic_safe_dis()
        # Speed-aware margin: add distance proportional to the current speed so the
        # reference path moves away from obstacles (absorbs tracking error on turns).
        speed_now = float(np.linalg.norm(sync_drone_state.global_vel[:2]))
        safe_dis_eff = safe_dis_dyn + K_SPEED_MARGIN * speed_now       # center-of-mass
        self._planner.safe_dis = max(0.0, safe_dis_eff - BODY_RADIUS_M)  # -> body edge
        use_emergency_safe = (safe_dis_dyn < self._safe_dis_norm - 1e-6)

        plan_time_ahead = 1.0
        head_state    = None
        plan_init_state = None
        ok = False

        # v1.13: continuation seed from the active trajectory (see _set_anchor_seed)
        self._set_anchor_seed(plan_time_ahead)

        plan_start = time.time()
        budget_hit = False
        for use_shorter in [False, True]:
            for seed in range(11):
                # Break the loop if the time budget is exceeded -> do not block the
                # main thread (navigation/escape/stuck) too long at a hard spot.
                if time.time() - plan_start > PLAN_TIME_BUDGET:
                    budget_hit = True
                    break
                self._set_local_target(
                    seed, shorter=use_shorter,
                    ref_pos=sync_drone_state.global_pos[:2])

                plan_init_state = self._get_drone_state_at(plan_time_ahead)
                head_state    = np.zeros((3, 2))
                head_state[0] = plan_init_state.global_pos[:2]
                head_state[1] = plan_init_state.global_vel[:2]

                tail_state    = np.zeros((3, 2))
                tail_state[0] = self._target_state[0]
                tail_state[1] = self._target_state[1]

                try:
                    t0 = time.time()
                    # RL seeds first (if enabled); fall back to batch_plan when
                    # no RL seed converges. Default (no RL) -> batch_plan only.
                    used_rl = (self._rl_seeds is not None
                               and self._try_rl_seed_plan(head_state,
                                                          tail_state))
                    if not used_rl:
                        self._planner.batch_plan(
                            self._esdf, head_state, tail_state)
                    dt = time.time() - t0
                    rl_tag    = " RL" if used_rl else ""
                    short_tag = " short" if use_shorter else ""
                    emer_tag  = f" sd={safe_dis_dyn:.2f}" if use_emergency_safe else ""
                    self.get_logger().info(
                        f"[Expert] Replan ✓ cost={self._planner.final_cost:.1f} "
                        f"t={dt*1000:.0f}ms seed={seed}{rl_tag}{short_tag}{emer_tag}",
                        throttle_duration_sec=1.0)
                    ok = True
                    break
                except Exception:
                    pass
            if ok or budget_hit:
                break

        if not ok:
            self._replan_fail_count += 1
            self.get_logger().warn(
                f"[Expert] Replan failed ({self._replan_fail_count}x in a row"
                f"{', budget' if budget_hit else ''})")
            if (self._replan_fail_count >= ESCAPE_FAIL_COUNT
                    and self._planner_escape_allowed()):
                self._enter_escape()
            return

        self._replan_fail_count = 0

        drone_pos_now = self._drone_state.global_pos[:2]
        discontinuity = float(np.linalg.norm(head_state[0] - drone_pos_now))

        if discontinuity > DISCONTINUITY_THRESHOLD:
            self.get_logger().warn(
                f"[Expert] Discontinuity {discontinuity:.1f}m — replan from actual")
            # v1.13: the head moved far -> the old trajectory's continuation is
            # no longer relevant (could be behind the head). Replan from scratch.
            self._clear_anchor_seed()
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
            for re_seed in range(5):
                try:
                    self._planner.batch_plan(self._esdf, head_actual, tail_state)
                    replan_ok = True
                    plan_init_state = DroneState()
                    plan_init_state.global_pos = np.append(
                        drone_pos_now, self._cruise_z or self._alt)
                    plan_init_state.global_vel = np.append(v_now, 0.0)
                    plan_init_state.attitude   = sync_drone_state.attitude
                    plan_init_state.local_vel  = sync_drone_state.local_vel
                    plan_init_state.yaw        = sync_drone_state.yaw
                    head_state = head_actual
                    break
                except Exception:
                    pass

            if not replan_ok:
                return

        new_traj_array = self._planner.get_full_state_cmd(hz=self._cmd_hz)
        total_time     = float(np.sum(self._planner.ts))

        # Live RViz viz for the classic/topo path (_try_rl_seed_plan already
        # publishes its own when used_rl=true -- skip here to avoid a
        # redundant double-publish of the same replan).
        if not used_rl:
            self._viz_data = {
                "head": head_state.copy(), "tail": tail_state.copy(),
                "seeds": [],
                "solutions": [(s[0].copy(), s[1].copy(), s[2]) for s in
                             getattr(self._planner, "all_solutions", [])],
                "best_cost": float(self._planner.final_cost),
                "target": self._target_state[0].copy(),
            }
            self._publish_viz()

        # Race-condition guard: alt-rcv can trigger DURING a slow replan computation.
        if self._alt_recovery_active:
            return

        # Forward safety guard (2026-07-24 rewrite): reject the trajectory only if a
        # point's CENTER-OF-MASS clearance dips below POST_CHECK_CLEAR_MIN — a single
        # threshold set just under the hard barrier, DECOUPLED from safe_dis. The old
        # guard reused safe_dis (the SOFT comfort zone) as a hard reject line — and a
        # stricter one for the first 0.5s — so it threw away precisely the grazing
        # trajectories the optimizer's soft cost was built to accept; near a tight
        # obstacle that produced a reject-after-reject loop (the "trapped/spinning"
        # behavior). Now the optimizer owns the soft zone and the post-check only
        # vetoes points that dip below the hard barrier, uniformly over the whole
        # trajectory. clearance_center = get_edt_dis (body edge) + BODY_RADIUS_M.
        if self._esdf.is_ready() and len(new_traj_array) > 0:
            step = max(1, int(self._cmd_hz * 0.1))   # check every 0.1s of the trajectory
            for i in range(0, len(new_traj_array), step):
                pt_xy = new_traj_array[i][0]
                clr_center = float(self._esdf.get_edt_dis(pt_xy)) + BODY_RADIUS_M
                if clr_center < POST_CHECK_CLEAR_MIN:
                    self._replan_fail_count += 1
                    self.get_logger().warn(
                        f"[Expert] ✗ Trajectory rejected: point #{i} "
                        f"clearance(center)={clr_center:.2f}m "
                        f"< {POST_CHECK_CLEAR_MIN:.2f}m. fail={self._replan_fail_count}",
                        throttle_duration_sec=1.0)
                    if (self._replan_fail_count >= ESCAPE_FAIL_COUNT
                            and self._planner_escape_allowed()):
                        self._enter_escape()
                    return

        with self._traj_lock:
            self._traj.t_start    = time.time()
            self._traj.state_cmd  = new_traj_array
            self._traj.total_time = total_time
            self._traj.cmd_hz     = self._cmd_hz

        with self._emergency_stop_lock:
            self._emergency_stop_first_t = None

        # ── SAMPLE QUALITY FILTERS ────────────────────────────────────────────
        if self._n_samples >= self._target_n:
            return

        if not self._check_drone_safe(sync_drone_state):
            self._collision_skip_count += 1
            self.get_logger().warn(
                f"[Expert] ✗ Skip: drone near obstacle/void. "
                f"Total: {self._collision_skip_count}",
                throttle_duration_sec=2.0)
            return

        if not self._check_velocity_sane(sync_drone_state):
            self._velocity_skip_count += 1
            speed = float(np.linalg.norm(sync_drone_state.global_vel[:2]))
            self.get_logger().warn(
                f"[Expert] ✗ Skip: velocity {speed:.2f}m/s. "
                f"Total: {self._velocity_skip_count}",
                throttle_duration_sec=2.0)
            return

        if not self._check_trajectory_safe(
                self._planner.int_wpts, self._planner.ts,
                head_state[0], self._target_state[0]):
            self._traj_unsafe_skip += 1
            self.get_logger().warn(
                f"[Expert] ✗ Skip: trajectory unsafe/void. "
                f"Total: {self._traj_unsafe_skip}",
                throttle_duration_sec=2.0)
            return

        # Altitude Drift Filter: skip if the drone dropped/bounced > 0.5m
        if self._cruise_z is not None:
            z_err = abs(sync_drone_state.global_pos[2] - self._cruise_z)
            if z_err > 0.5:
                self._altitude_skip_count += 1
                self.get_logger().warn(
                    f"[Expert] ✗ Skip: altitude drift {z_err:.2f}m (> 0.5m). "
                    f"Total: {self._altitude_skip_count}",
                    throttle_duration_sec=2.0)
                return

        # ── MULTI-HOMOTOPY (see CLAUDE.md): collect all feasible modes ──
        # The best solution (the one the drone executes) always becomes the
        # first row. v1.5: dedup by TOPOLOGICAL SIGNATURE (compute_mode_signature),
        # not Euclidean waypoint distance < 0.5m. Reason: waypoint distance
        # cannot distinguish "same side, just shifted" (previously slipped
        # through as a "new mode" whenever it happened to be > 0.5m) from
        # "genuinely opposite side" (previously could get merged whenever it
        # happened to be < 0.5m in a narrow gap). Two rows with the same `id`
        # are considered different modes ONLY if `mode_sig` differs -- that is
        # the correct definition, and it is auditable in the CSV.
        def _path_xy(wpts):
            return [head_state[0], wpts[:2, 0], wpts[:2, 1],
                    self._target_state[0]]

        best_sig = compute_mode_signature(
            self._esdf, head_state[0], self._target_state[0],
            _path_xy(self._planner.int_wpts))
        solutions = [(self._planner.int_wpts.copy(), self._planner.ts.copy(),
                     mode_signature_to_str(best_sig))]
        seen_sigs = {best_sig}
        if self._save_all_modes:
            for wpts_k, ts_k, _cost_k in getattr(
                    self._planner, "all_solutions", []):
                sig_k = compute_mode_signature(
                    self._esdf, head_state[0], self._target_state[0],
                    _path_xy(wpts_k))
                if sig_k is None or sig_k in seen_sigs:
                    continue
                if not self._check_trajectory_safe(
                        wpts_k, ts_k, head_state[0], self._target_state[0]):
                    continue
                seen_sigs.add(sig_k)
                solutions.append((wpts_k.copy(), ts_k.copy(),
                                  mode_signature_to_str(sig_k)))

        self._save_sample(
            depth_snapshot,
            sync_drone_state,
            plan_init_state,
            self._target_state.copy(),
            solutions,
        )

        if self._n_samples % 50 == 0:
            avg_lag = (1000 * sum(self._lag_history) / len(self._lag_history)
                       if self._lag_history else 0)
            self.get_logger().info(
                f"[Expert] ✓ {self._n_samples}/{self._target_n} samples "
                f"| avg lag {avg_lag:.0f}ms "
                f"| skip: lag={self._lag_skip_count} "
                f"col={self._collision_skip_count} "
                f"traj={self._traj_unsafe_skip} "
                f"vel={self._velocity_skip_count} "
                f"alt={self._altitude_skip_count}")

    # ── Status ────────────────────────────────────────────────────────────────
    def _print_status(self):
        pos = self._drone_state.global_pos
        vel = self._drone_state.global_vel
        speed = float(np.linalg.norm(vel[:2]))
        esc = (" [ESCAPE]" if self._escape_active
               else " [RETURN-HOME]" if self._out_of_bounds_return
               else " [ALT-RCV]" if self._alt_recovery_active
               else "")
        z_err = ""
        if self._cruise_z is not None:
            z_err_val = self._cruise_z - pos[2]
            z_err = f" | z_err={z_err_val:+.2f} cruise={self._cruise_z:.2f}"
        esdf_info = ""
        if self._esdf.is_ready():
            esdf_dist = float(self._esdf.get_edt_dis(pos[:2]))
            esdf_info = f" | esdf={esdf_dist:.2f}"

        es_info = ""
        with self._emergency_stop_lock:
            if self._emergency_stop_first_t is not None:
                persist = time.time() - self._emergency_stop_first_t
                es_info = f" | ES={persist:.1f}s"

        self.get_logger().info(
            f"[Expert] state={self._mission_state}{esc} | "
            f"samples={self._n_samples}/{self._target_n} | "
            f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}){z_err}{esdf_info}{es_info} | "
            f"speed={speed:.2f}m/s | "
            f"goal=({self._global_target[0]:.1f},{self._global_target[1]:.1f})")

    # ── Service helpers ───────────────────────────────────────────────────────
    def _call_srv(self, client, request, timeout=8.0):
        fut = client.call_async(request)
        t0  = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                return None
            time.sleep(0.05)
        return fut.result()

    def _set_mode(self, mode: str) -> bool:
        if not self._mode_client.wait_for_service(timeout_sec=3.0):
            return False
        req = SetMode.Request(); req.custom_mode = mode
        res = self._call_srv(self._mode_client, req)
        ok  = bool(res and res.mode_sent)
        if ok:
            self.get_logger().info(f"[MODE] → {mode} ✓")
        return ok

    def _arm(self, value: bool) -> bool:
        if not self._arming_client.wait_for_service(timeout_sec=3.0):
            return False
        req = CommandBool.Request(); req.value = value
        res = self._call_srv(self._arming_client, req)
        ok  = bool(res and res.success)
        if ok:
            self.get_logger().info(
                f"[ARM] {'ARMED' if value else 'DISARMED'} ✓")
        return ok

    def _wait_altitude(self, target_z: float, tol=0.2, timeout=30.0) -> bool:
        t0 = time.time()
        t_stable_start = None
        STABLE_DURATION = 2.0
        while time.time() - t0 < timeout:
            z = self._drone_state.global_pos[2]
            vz = abs(self._drone_state.global_vel[2])
            if abs(z - target_z) < tol and vz < 0.2:
                if t_stable_start is None:
                    t_stable_start = time.time()
                elif time.time() - t_stable_start >= STABLE_DURATION:
                    return True
            else:
                t_stable_start = None
            time.sleep(0.1)
        return False

    def _warm_up(self, n: int = 50):
        self.get_logger().info(f"[Expert] Warm-up: {n} setpoints @ {self._cmd_hz}Hz")
        time.sleep(n / self._cmd_hz + 0.5)

    # ── Main sequence ─────────────────────────────────────────────────────────
    def run_sequence(self):
        self.get_logger().info("[Expert] Waiting for MAVROS...")
        t0 = time.time()
        while not self._connected:
            if time.time() - t0 > self._conn_timeout:
                rclpy.shutdown(); return
            time.sleep(0.5)

        self.get_logger().info("[Expert] Waiting for depth image...")
        t0 = time.time()
        while self._latest_depth is None:
            if time.time() - t0 > 30.0:
                rclpy.shutdown(); return
            time.sleep(0.5)

        self.get_logger().info("[Expert] Waiting for ESDF...")
        t0 = time.time()
        while not self._esdf.is_ready():
            if time.time() - t0 > 60.0:
                rclpy.shutdown(); return
            time.sleep(1.0)
        self.get_logger().info("[Expert] ESDF ready ✓")

        # FIX ALTITUDE #1: force the EKF to use the GPS height reference BEFORE
        # calibration, so barometer drift does not shift the local Z frame during the mission.
        self._configure_px4_height_ref()
        time.sleep(2.0)  # give the EKF time to re-anchor the height reference

        if self._use_ekf_stable:
            self._ground_z = self._wait_ekf_stable()
        else:
            self._ground_z = self._drone_state.global_pos[2]
            self.get_logger().info("[Expert] EKF stable wait skipped (use_ekf_stable=false)")

        # FIX ALTITUDE #2: record the GPS altitude while still on the ground as the
        # physical AGL reference (does not follow baro/EKF drift).
        if len(self._gps_alt_buf) >= 5:
            self._gps_alt_home = float(np.mean(self._gps_alt_buf[-10:]))
            self.get_logger().info(
                f"[Expert] GPS alt home = {self._gps_alt_home:.2f}m (AGL reference)")
        else:
            self._gps_alt_home = None
            self.get_logger().warn(
                "[Expert] GPS altitude not available — GPS alt-trim disabled")

        self.get_logger().info(
            f"[Expert] Ground Z detected: {self._ground_z:.3f}m → "
            f"takeoff target = {self._alt}m (z = {self._ground_z + self._alt:.3f}m)")

        self._warm_up(n=50)

        self.get_logger().info("[Expert] ARM drone...")
        t0 = time.time(); armed = False
        while time.time() - t0 < self._arm_timeout:
            if self._arm(True):
                time.sleep(0.8)
                if self._armed: armed = True; break
            time.sleep(2.0)
        if not armed:
            rclpy.shutdown(); return

        if not self._set_mode("OFFBOARD"):
            self._arm(False); rclpy.shutdown(); return
        t0 = time.time()
        while self._mode != "OFFBOARD" and time.time() - t0 < 5.0:
            time.sleep(0.1)

        self._mission_state = self.STATE_TAKEOFF
        takeoff_z = self._ground_z + self._alt  # EKF frame = ground + physical target
        self.get_logger().info(f"[Expert] Takeoff to z={takeoff_z:.2f}m (physical ≈{self._alt:.1f}m)...")
        self._cruise_z      = takeoff_z
        self._cruise_z_base = takeoff_z   # clamp reference for GPS alt-trim

        if not self._wait_altitude(takeoff_z, tol=0.2, timeout=30.0):
            self.get_logger().warn("[Expert] Takeoff not stable — continuing anyway")
        self.get_logger().info(
            f"[Expert] ✓ Takeoff! z={self._drone_state.global_pos[2]:.3f}m")
        time.sleep(1.0)

        self._mission_state = self.STATE_COLLECTING
        self.get_logger().info(
            f"[Expert] Start data collection — target {self._target_n} samples")

        first_ok = False
        first_plan_deadline = time.time() + 20.0  # max 20s for the first plan
        for first_seed in range(10):
            if time.time() > first_plan_deadline:
                break
            self._set_local_target(first_seed)
            head0    = np.zeros((3, 2))
            head0[0] = self._drone_state.global_pos[:2]
            tail0    = np.zeros((3, 2))
            tail0[0] = self._target_state[0]
            tail0[1] = self._target_state[1]
            try:
                self._planner.batch_plan(self._esdf, head0, tail0)
                first_traj = self._planner.get_full_state_cmd(hz=self._cmd_hz)
                with self._traj_lock:
                    self._traj.t_start    = time.time()
                    self._traj.state_cmd  = first_traj
                    self._traj.total_time = float(np.sum(self._planner.ts))
                    self._traj.cmd_hz     = self._cmd_hz
                self.get_logger().info(
                    f"[Expert] First plan ✓ seed={first_seed} ({len(first_traj)} samples, "
                    f"{self._traj.total_time:.2f}s)")
                first_ok = True
                break
            except Exception:
                pass

        if not first_ok:
            self.get_logger().error("[Expert] First plan failed — entering escape toward the goal")
            self._enter_escape()

        last_replan_t     = time.time()
        traj_direction    = 1
        explosion_retries = 0
        landing_retries   = 0

        while (self._n_samples < self._target_n and rclpy.ok()
               and not self._lap_pilot_done):
            now = time.time()

            # ── Physics explosion guard (with auto-recovery) ───────────────────
            if self._is_physics_explosion():
                pos3 = self._drone_state.global_pos
                spd  = float(np.linalg.norm(self._drone_state.global_vel))
                explosion_retries += 1
                self.get_logger().error(
                    f"[Expert] ⚡ PHYSICS EXPLOSION! speed={spd:.1f}m/s "
                    f"pos=({pos3[0]:.1f},{pos3[1]:.1f},{pos3[2]:.1f}) "
                    f"— recovery #{explosion_retries}/{EXPLOSION_MAX_RETRY}")

                if (explosion_retries <= EXPLOSION_MAX_RETRY
                        and self._recover_from_explosion()):
                    self._set_local_target(0)
                    last_replan_t = time.time()
                    self.get_logger().info(
                        f"[Expert] ✓ Resume collecting after recovery "
                        f"({self._n_samples} samples)")
                    continue

                # Recovery failed or too many retries -> shutdown
                self.get_logger().warn(
                    f"[Expert] Mission stopped — {self._n_samples} samples saved. "
                    f"Restart SITL before the next run.")
                rclpy.shutdown()
                return

            # ── v27: Accidental-landing guard (no gz reset) ────────────────────
            if self._is_accidental_landing():
                landing_retries += 1
                self.get_logger().error(
                    f"[Expert] ⤴ Accidental landing detected "
                    f"— recovery #{landing_retries}/{LAND_RECOVER_MAX_RETRY}")
                if (landing_retries <= LAND_RECOVER_MAX_RETRY
                        and self._recover_landing()):
                    self._set_local_target(0)
                    last_replan_t = time.time()
                    self.get_logger().info(
                        f"[Expert] ✓ Resume collecting after landing recovery "
                        f"({self._n_samples} samples)")
                    continue
                self.get_logger().warn(
                    f"[Expert] Landing recovery failed/too frequent "
                    f"— mission stopped ({self._n_samples} samples saved).")
                rclpy.shutdown()
                return

            # ── v26: Arena bounds guard ────────────────────────────────────────
            pos3 = self._drone_state.global_pos
            if self._is_out_of_bounds(pos3):
                if self._out_of_bounds_since is None:
                    self._out_of_bounds_since = now
                    self.get_logger().warn(
                        f"[Expert] ⚠ Drone left the arena! "
                        f"pos=({pos3[0]:.1f},{pos3[1]:.1f}) "
                        f"bounds X:[{self._arena_x_min},{self._arena_x_max}] "
                        f"Y:[{self._arena_y_min},{self._arena_y_max}]")
                elif now - self._out_of_bounds_since > OUT_OF_BOUNDS_TIME:
                    if not self._out_of_bounds_return:
                        self._out_of_bounds_return = True
                        self._escape_active = False
                        self._stuck_since   = None
                        with self._traj_lock:
                            self._traj.invalidate()
                        self.get_logger().warn(
                            f"[Expert] ⚠ Out-of-bounds {OUT_OF_BOUNDS_TIME}s → "
                            f"returning to start (0,0)")
            else:
                self._out_of_bounds_since = None
                if self._out_of_bounds_return:
                    self._out_of_bounds_return = False
                    self.get_logger().info(
                        f"[Expert] ✓ Back in the arena — resume navigation")

            if now - last_replan_t >= self._replan_period:
                last_replan_t = now
                self._replan_and_record()

            pos2d = pos3[:2]
            dist  = np.linalg.norm(pos2d - self._global_target)

            if not self._escape_active and (dist < 1.5 or self._reached_target):
                self._reached_target = False
                # USER IDEA: fix altitude first at the turnaround point before
                # navigating to the next target, so drift does not accumulate.
                self._fix_altitude_at_waypoint()
                next_y = 0.0
                if self._randomize_turnaround_y:
                    next_y = float(np.random.uniform(
                        self._turnaround_y_lo, self._turnaround_y_hi))
                if traj_direction == 1:
                    self._global_target = np.array([0.0, next_y])
                    traj_direction      = -1
                else:
                    self._global_target = np.array([self._goal_x, next_y])
                    traj_direction      = 1
                self._near_global_target = False
                self.get_logger().info(
                    f"[Expert] New goal: ({self._global_target[0]:.1f},"
                    f"{self._global_target[1]:.1f}) | "
                    f"samples={self._n_samples}")
                if self._force_lap_alternation:
                    self._advance_lap_leg()

            time.sleep(0.05)

        if not rclpy.ok():
            return

        self._mission_state = self.STATE_LANDING
        self._cruise_z      = None
        self.get_logger().info("[Expert] Landing...")

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

        avg_lag = (1000 * sum(self._lag_history) / len(self._lag_history)
                   if self._lag_history else 0)

        self.get_logger().info("=" * 62)
        self.get_logger().info("  EXPERT PLANNER DONE ✓")
        self.get_logger().info(f"  Samples accepted   : {self._n_samples}")
        self.get_logger().info(f"  Avg depth lag      : {avg_lag:.0f} ms")
        self.get_logger().info(f"  Skipped (lag)      : {self._lag_skip_count}")
        self.get_logger().info(f"  Skipped (collision): {self._collision_skip_count}")
        self.get_logger().info(f"  Skipped (traj)     : {self._traj_unsafe_skip}")
        self.get_logger().info(f"  Skipped (velocity) : {self._velocity_skip_count}")
        self.get_logger().info(f"  Skipped (altitude) : {self._altitude_skip_count}")
        total_skip = (self._lag_skip_count + self._collision_skip_count
                      + self._traj_unsafe_skip + self._velocity_skip_count
                      + self._altitude_skip_count)
        total_attempted = self._n_samples + total_skip
        if total_attempted > 0:
            accept_rate = 100 * self._n_samples / total_attempted
            self.get_logger().info(f"  Accept rate        : {accept_rate:.1f}%")
        # v1.12 (2026-07-10): surface how often the topo front-end had to drop
        # to the stock 3-seed fallback. A forced-leg fallback is the pathway
        # the lap-alternation audit traced ~40% wrong-side samples to; the
        # fallback now runs on the biased ESDF (see homotopy_frontend.py) but
        # a high count still means the forced side was frequently unreachable
        # -> re-check that world/scene rather than trusting the forced labels.
        fe = getattr(self._planner, "_topo_frontend", None)
        if fe is not None:
            fb = getattr(fe, "fallback_count", 0)
            fbf = getattr(fe, "fallback_forced_count", 0)
            self.get_logger().info(
                f"  Topo fallback      : {fb} total ({fbf} during forced legs)")
        if self._rl_seeds is not None and self._rl_try_count > 0:
            self.get_logger().info(
                f"  RL seeds           : {self._rl_ok_count}/"
                f"{self._rl_try_count} replans converged "
                f"({self._rl_dual_count} dual-h/both sides)")
        self.get_logger().info("=" * 62)

        if rclpy.ok():
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ExpertPlannerNode()
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
