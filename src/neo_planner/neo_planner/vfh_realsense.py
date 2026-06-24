"""
vfh_obstacle_avoidance_node.py
===============================
VFH (Vector Field Histogram) Obstacle Avoidance Node for drone.
Camera: SIYI A8 Mini, fixed front-facing mount.

Drone size  : 49 cm x 49 cm
              robot_radius = 0.346 m  (half the diagonal: (49*sqrt(2))/2 = 34.6 cm)
              min_valley_width = 3    (matches the 9-sector FOV capacity)

Subscribe : /realsense/depth/float32 (sensor_msgs/Image, 32FC1, meters — from realsense_d455_node)
            /vfh/target_angle        (std_msgs/Float32, degrees, 0=front CW)

Publish   : /vfh/cmd_vel              (geometry_msgs/Twist — empty, velocity from another node)
            /vfh/obstacles             (std_msgs/String, JSON obstacle direction map)
            /vfh/movement_direction    (std_msgs/String, JSON: state, obstacle, steer_deg)
            /vfh/visual                (sensor_msgs/Image, BGR debug overlay)

This node focuses on three main outputs (read by other nodes via /vfh/movement_direction):
    state             : 'move' | 'avoid' | 'stop'
    obstacle          : obstacle position relative to the drone
    steer_deg         : target rotation angle of the drone to avoid the obstacle
                        range: -40° (full left) to +40° (full right)
                        negative = turn left, positive = turn right

Unit note: all distance parameters use MILLIMETERS (mm) matching the original
uint16 from the RealSense SDK — there is no conversion to meters.

─────────────────────────────────────────────────────────────────────────────
Topic graph (full pipeline):

  realsense_d455_node
      │
      ├─ /realsense/depth/float32  (32FC1, m)   ──► vfh_obstacle_avoidance_node
      ├─ /realsense/depth/float32  (32FC1, m)   ──► (bolt_measurement / depth_node)
      └─ /realsense/depth/stats    (String/JSON) ──► (debug / monitoring)

  vfh_obstacle_avoidance_node
      ├─ /vfh/movement_direction  ──► flight controller node
      ├─ /vfh/obstacles ──► monitoring
      └─ /vfh/visual    ──► rviz / rqt_image_view
─────────────────────────────────────────────────────────────────────────────

Parameters:
    n_sectors       (int)   : number of angular sectors            default: 36
    threshold       (float) : obstacle density threshold 0-1       default: 0.45
    smooth_window   (int)   : moving-average half-window           default: 2
    d_max           (float) : max detection distance (mm)          default: 5000.0
    d_min           (float) : min distance, closer is skipped      default: 300.0
    min_valley_width(int)   : min sectors for a valid valley       default: 3
    robot_radius    (float) : drone radius for safety (mm)         default: 346.0
    safety_dist     (float) : extra safety distance (mm)           default: 500.0
    target_angle    (float) : default goal direction (deg)         default: 0.0
    frame_skip      (int)   : process 1 of every N frames          default: 1
    publish_visual  (bool)  : publish the debug overlay image      default: true
    show_debug      (bool)  : show a local OpenCV window           default: false
    hfov_deg        (float) : camera horizontal FOV (deg)          default: 88.0

Coordinate convention (SIYI A8 Mini, top-down view):
    0°   = straight ahead of the drone
    negative = left, positive = right
    steering_deg range: -40° to +40° (follows the camera FOV)

Usage:
    ros2 run com_px_to_jetson vfh_obstacle_avoidance_node

Set the target angle at runtime:
    ros2 topic pub /vfh/target_angle std_msgs/Float32 "data: 0.0"
"""

import json
import os
import time
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

HAS_DISPLAY = bool(
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
)

_DEPTH_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

_CMD_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)


# ---------------------------------------------------------------------------
# VFH Core — pure NumPy, no ROS dependency
# ---------------------------------------------------------------------------

class VFH:
    """
    Vector Field Histogram obstacle avoidance (classic).

    Pipeline:
      1. build_polar_histogram  — POD(k) with robot enlargement
      2. smooth_histogram       — moving average
      3. to_binary              — single threshold
      4. find_valleys           — free-sector run >= min_valley_width
      5. select_best_valley     — score = angle_err - width*0.5
      6. compute_cmd            — state: move/avoid/stop
      7. lateral_proximity_penalty — FOV edge correction

    Coordinates: 0=front, positive=right, negative=left.
    Drone: 49cmx49cm, robot_radius=346mm.
    """

    def __init__(
        self,
        n_sectors:        int   = 36,
        threshold:        float = 0.45,
        smooth_window:    int   = 2,
        d_max:            float = 5000.0,
        d_min:            float = 300.0,
        min_valley_width: int   = 3,
        robot_radius:     float = 346.0,
        safety_dist:      float = 500.0,
        hfov_deg:         float = 88.0,
    ) -> None:
        self.n_sectors        = n_sectors
        self.threshold        = threshold
        self.smooth_window    = smooth_window
        self.d_max            = d_max
        self.d_min            = d_min
        self.min_valley_width = min_valley_width
        self.robot_radius     = robot_radius
        self.safety_dist      = safety_dist
        self.hfov_deg         = hfov_deg
        self.sector_size_deg  = 360.0 / n_sectors

        half = hfov_deg / 2.0
        self.fov_mask = np.zeros(n_sectors, dtype=bool)
        for k in range(n_sectors):
            angle = k * self.sector_size_deg
            if angle > 180.0:
                angle -= 360.0
            if abs(angle) <= half:
                self.fov_mask[k] = True

        valid_indices = np.where(self.fov_mask)[0]
        self.fov_sector_right = int(valid_indices[-1])
        left_valid = [k for k in range(n_sectors)
                      if self.fov_mask[k] and k > n_sectors // 2]
        self.fov_sector_left = int(left_valid[0]) if left_valid else 0

    # ------------------------------------------------------------------ #
    # Step 1 — Polar histogram with robot enlargement                    #
    # ------------------------------------------------------------------ #

    def build_polar_histogram(self, depth_map: np.ndarray) -> np.ndarray:
        """
        POD(k) = sum[(1 - d/d_max)^2] per sector, with enlargement:
            arcsin((robot_radius + safety_dist) / d) sectors to the left/right.
        Vertical ROI 15%-85% so tree trunks are read.
        """
        h, w      = depth_map.shape
        hist      = np.zeros(self.n_sectors, dtype=np.float32)
        hfov_rad  = math.radians(self.hfov_deg)
        row_start = int(h * 0.15)
        row_end   = int(h * 0.85)

        col_indices = np.arange(w, dtype=np.float32)
        angles_deg  = np.degrees((col_indices / w - 0.5) * hfov_rad)
        col_sectors = (
            np.round(angles_deg / self.sector_size_deg).astype(int) % self.n_sectors
        )

        depth_roi  = depth_map[row_start:row_end, :]
        valid_mask = (
            np.isfinite(depth_roi)
            & (depth_roi > self.d_min)
            & (depth_roi < self.d_max)
        )
        weight = np.where(
            valid_mask,
            (1.0 - depth_roi / self.d_max) ** 2,
            0.0,
        ).astype(np.float32)

        enlarge_r = self.robot_radius + self.safety_dist
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(valid_mask, enlarge_r / depth_roi, 0.0)
        enlarge_sec = np.round(
            np.degrees(np.arcsin(np.clip(ratio, 0.0, 1.0))) / self.sector_size_deg
        ).astype(int)

        for col in range(w):
            k       = col_sectors[col]
            col_w   = weight[:, col]
            col_m   = valid_mask[:, col]
            total_w = col_w.sum()
            if total_w == 0.0:
                continue
            max_e = int(enlarge_sec[:, col][col_m].max()) if col_m.any() else 0
            for dk in range(-max_e, max_e + 1):
                tk = (k + dk) % self.n_sectors
                if not self.fov_mask[tk]:
                    continue
                fade = 1.0 - abs(dk) / (max_e + 1) if max_e > 0 else 1.0
                hist[tk] += total_w * fade

        hist[~self.fov_mask] = 0.0
        mx = hist.max()
        if mx > 0:
            hist /= mx
        return hist

    # ------------------------------------------------------------------ #
    # Step 2 — Smooth histogram                                           #
    # ------------------------------------------------------------------ #

    def smooth_histogram(self, hist: np.ndarray) -> np.ndarray:
        w = self.smooth_window
        if w == 0:
            s = hist.copy(); s[~self.fov_mask] = 0.0; return s
        smoothed = np.zeros_like(hist)
        n = len(hist)
        for i in range(n):
            if not self.fov_mask[i]:
                continue
            total = 0.0; count = 0
            for d in range(-w, w + 1):
                j = (i + d) % n
                if self.fov_mask[j]:
                    total += hist[j]; count += 1
            smoothed[i] = total / count if count > 0 else 0.0
        return smoothed

    # ------------------------------------------------------------------ #
    # Step 3 — Binary histogram (single threshold)                        #
    # ------------------------------------------------------------------ #

    def to_binary(self, smoothed: np.ndarray) -> np.ndarray:
        """1 = blocked (>= threshold), 0 = free. Blind sectors always 0."""
        binary = (smoothed >= self.threshold).astype(np.int8)
        binary[~self.fov_mask] = 0
        return binary

    # ------------------------------------------------------------------ #
    # Step 4 — Valley detection                                           #
    # ------------------------------------------------------------------ #

    def find_valleys(self, binary: np.ndarray) -> list:
        """Find contiguous free (0) runs >= min_valley_width within the FOV."""
        n       = len(binary)
        valleys = []
        valid_ordered = sorted(
            [k for k in range(n) if self.fov_mask[k]],
            key=lambda k: (k * self.sector_size_deg
                           if k * self.sector_size_deg <= 180
                           else k * self.sector_size_deg - 360)
        )

        run_start_pos = None
        for pos, k in enumerate(valid_ordered):
            if binary[k] == 0:
                if run_start_pos is None:
                    run_start_pos = pos
            else:
                if run_start_pos is not None:
                    if pos - run_start_pos >= self.min_valley_width:
                        valleys.append(self._make_valley(valid_ordered, run_start_pos, pos - 1))
                    run_start_pos = None
        if run_start_pos is not None:
            if len(valid_ordered) - run_start_pos >= self.min_valley_width:
                valleys.append(self._make_valley(valid_ordered, run_start_pos, len(valid_ordered) - 1))
        return valleys

    def _make_valley(self, valid_ordered, start_pos, end_pos):
        def sa(k):
            a = k * self.sector_size_deg
            return a - 360.0 if a > 180.0 else a
        mid_k = valid_ordered[round((start_pos + end_pos) / 2.0)]
        return {
            "start_idx":  valid_ordered[start_pos],
            "end_idx":    valid_ordered[end_pos],
            "width":      end_pos - start_pos + 1,
            "center_deg": round(sa(mid_k), 1),
        }

    # ------------------------------------------------------------------ #
    # Step 5 — Select best valley                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = (a - b) % 360.0
        return diff - 360.0 if diff > 180.0 else diff

    def select_best_valley(
        self,
        valleys:       list,
        target_deg:    float,
        smoothed_hist: np.ndarray,
        robot_heading: float = 0.0,
    ):
        """
        Classic VFH: score = angle_err - width * 0.5
        The valley with the smallest score is selected.
        """
        if not valleys:
            return None

        best = None; best_score = float("inf"); best_cand = 0.0
        for v in valleys:
            margin      = max(1, self.min_valley_width // 2)
            half_clamp  = max(0.0, (v["width"] * self.sector_size_deg / 2)
                                   - margin * self.sector_size_deg)
            raw_diff    = self._angle_diff(target_deg, v["center_deg"])
            clamped     = max(-half_clamp, min(half_clamp, raw_diff))
            candidate   = v["center_deg"] + clamped

            angle_err = abs(self._angle_diff(target_deg, candidate))
            score     = angle_err - v["width"] * 0.5
            if score < best_score:
                best_score = score; best = dict(v); best_cand = candidate

        if best is None:
            return None
        best["steering_deg"] = round(best_cand, 1)
        return best

    # ------------------------------------------------------------------ #
    # Step 6 — Compute command                                            #
    # ------------------------------------------------------------------ #

    def compute_cmd(self, smoothed_hist, binary_hist, best_valley):
        peak_k = -1; peak_val = 0.0
        for k in range(self.n_sectors):
            if self.fov_mask[k] and smoothed_hist[k] >= self.threshold:
                if smoothed_hist[k] > peak_val:
                    peak_val = smoothed_hist[k]; peak_k = k
        if peak_k == -1:
            obs_dir = "none"
        else:
            pa = peak_k * self.sector_size_deg
            if pa > 180.0: pa -= 360.0
            obs_dir = self._angle_to_label(pa % 360.0)

        ld = rd = 0.0
        for k in range(self.n_sectors):
            if not self.fov_mask[k]: continue
            a = k * self.sector_size_deg
            if a > 180.0: a -= 360.0
            if a < 0: ld += float(smoothed_hist[k])
            elif a > 0: rd += float(smoothed_hist[k])
        if ld == 0.0 and rd == 0.0: avoid_dir = "none"
        elif ld < rd:                avoid_dir = "left"
        else:                        avoid_dir = "right"

        state = "stop" if best_valley is None else ("avoid" if obs_dir != "none" else "move")
        return {"state": state, "obstacle_direction": obs_dir, "avoid_direction": avoid_dir}

    def lateral_proximity_penalty(self, depth_map: np.ndarray, steer_deg: float) -> float:
        """Correct steering by ±20° if an obstacle is very close at the FOV edge."""
        h, w     = depth_map.shape
        row_s    = int(h * 0.30); row_e = int(h * 0.70)
        col_edge = max(1, int(w * 0.10))
        danger   = self.robot_radius + self.safety_dist + 200.0
        delta    = 0.0
        for zone, sign in [(depth_map[row_s:row_e, :col_edge], 1.0),
                           (depth_map[row_s:row_e, w - col_edge:], -1.0)]:
            v = zone[(np.isfinite(zone)) & (zone > self.d_min) & (zone < self.d_max)]
            if v.size > 0:
                mn = float(v.min())
                if mn < danger:
                    delta += sign * (1.0 - mn / danger) * 20.0
        return delta

    # ------------------------------------------------------------------ #
    # Full VFH pipeline                                                   #
    # ------------------------------------------------------------------ #

    def run(self, depth_map: np.ndarray, target_deg: float = 0.0,
            robot_heading: float = 0.0) -> dict:
        """Classic VFH pipeline: PHD -> Smooth -> Binary -> Valleys -> select -> cmd."""
        raw_hist      = self.build_polar_histogram(depth_map)
        smoothed_hist = self.smooth_histogram(raw_hist)
        binary_hist   = self.to_binary(smoothed_hist)
        valleys       = self.find_valleys(binary_hist)
        best_valley   = self.select_best_valley(valleys, target_deg, smoothed_hist, robot_heading)
        cmd           = self.compute_cmd(smoothed_hist, binary_hist, best_valley)

        steer_deg = best_valley.get("steering_deg", 0.0) if best_valley else 0.0
        lat_delta = self.lateral_proximity_penalty(depth_map, steer_deg)
        if abs(lat_delta) > 0.5:
            steer_deg = float(np.clip(steer_deg + lat_delta, -40.0, 40.0))

        return {
            "raw_hist":      raw_hist.tolist(),
            "smoothed_hist": smoothed_hist.tolist(),
            "binary_hist":   binary_hist.tolist(),
            "valleys":       valleys,
            "best_valley":   best_valley,
            "cmd":           cmd,
            "target_deg":    target_deg,
            "steer_deg":     round(steer_deg, 1),
            "lateral_delta": round(lat_delta, 1),
        }


    _DIRECTION_LABELS = [
        (  0,  22, "front"),
        ( 23,  67, "front-right"),
        ( 68, 112, "right"),
        (113, 157, "rear-right"),
        (158, 202, "rear"),
        (203, 247, "rear-left"),
        (248, 292, "left"),
        (293, 337, "front-left"),
        (338, 360, "front"),
    ]

    def _angle_to_label(self, deg: float) -> str:
        deg = deg % 360.0
        for lo, hi, label in self._DIRECTION_LABELS:
            if lo <= deg <= hi:
                return label
        return "front"

    def extract_obstacle_map(
        self,
        smoothed_hist: np.ndarray,
        binary_hist:   np.ndarray,
    ) -> list[dict]:
        n      = len(binary_hist)
        groups = []

        extended_bin  = np.concatenate([binary_hist,  binary_hist])
        extended_smo  = np.concatenate([smoothed_hist, smoothed_hist])

        run_start = None
        for i in range(2 * n):
            if extended_bin[i] == 1:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    indices = [j % n for j in range(run_start, i)]
                    seen_idx = set()
                    unique_idx = []
                    for idx in indices:
                        if idx not in seen_idx:
                            seen_idx.add(idx)
                            unique_idx.append(idx)

                    if unique_idx:
                        densities    = [float(extended_smo[j]) for j in range(run_start, i)]
                        peak_density = max(densities)
                        mean_density = sum(densities) / len(densities)

                        start_deg  = (run_start % n) * self.sector_size_deg
                        end_deg    = ((i - 1) % n)   * self.sector_size_deg
                        center_deg = ((run_start + (i - run_start) / 2) % n) * self.sector_size_deg
                        width_deg  = len(unique_idx) * self.sector_size_deg

                        is_critical = (0 in unique_idx)

                        groups.append({
                            "angle_start_deg":  round(start_deg,  1),
                            "angle_end_deg":    round(end_deg,    1),
                            "angle_center_deg": round(center_deg, 1),
                            "width_deg":        round(width_deg,  1),
                            "peak_density":     round(peak_density, 3),
                            "mean_density":     round(mean_density, 3),
                            "sector_indices":   unique_idx,
                            "relative_pos":     self._angle_to_label(center_deg),
                            "is_critical":      is_critical,
                        })
                    run_start = None

        if run_start is not None:
            indices = [j % n for j in range(run_start, 2 * n)]
            seen_idx = set()
            unique_idx = []
            for idx in indices:
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    unique_idx.append(idx)
            if unique_idx:
                densities    = [float(extended_smo[j]) for j in range(run_start, 2 * n)]
                peak_density = max(densities)
                mean_density = sum(densities) / len(densities)
                start_deg    = (run_start % n) * self.sector_size_deg
                end_deg      = ((2 * n - 1) % n) * self.sector_size_deg
                center_deg   = ((run_start + (2 * n - run_start) / 2) % n) * self.sector_size_deg
                width_deg    = len(unique_idx) * self.sector_size_deg
                is_critical  = (0 in unique_idx)
                groups.append({
                    "angle_start_deg":  round(start_deg,  1),
                    "angle_end_deg":    round(end_deg,    1),
                    "angle_center_deg": round(center_deg, 1),
                    "width_deg":        round(width_deg,  1),
                    "peak_density":     round(peak_density, 3),
                    "mean_density":     round(mean_density, 3),
                    "sector_indices":   unique_idx,
                    "relative_pos":     self._angle_to_label(center_deg),
                    "is_critical":      is_critical,
                })

        seen_keys = set()
        unique_groups = []
        for g in groups:
            key = tuple(sorted(g["sector_indices"]))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_groups.append(g)

        unique_groups.sort(key=lambda g: g["peak_density"], reverse=True)
        return unique_groups


# ---------------------------------------------------------------------------
# Debug Visualiser
# ---------------------------------------------------------------------------

class VFHVisualiser:
    """Renders VFH state as a BGR debug image."""

    def __init__(self, width: int = 960, height: int = 540) -> None:
        self.W = width
        self.H = height

    def render(
        self,
        depth_bgr:    np.ndarray,
        vfh_result:   dict,
        n_sectors:    int,
        threshold:    float,
    ) -> np.ndarray:
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        half   = self.W // 2

        if depth_bgr is not None:
            thumb = cv2.resize(depth_bgr, (half, self.H))
            canvas[:, :half] = thumb

        # ── scale all dimensions proportionally to the window size ──
        # Base reference: 640x400. Scale factor s = H/400
        s       = self.H / 400.0

        vx       = half
        smoothed = np.array(vfh_result["smoothed_hist"])
        binary   = np.array(vfh_result["binary_hist"])
        n        = n_sectors
        bar_w    = max(1, (self.W - half - 20) // n)
        chart_h  = int(120 * s)
        base_y   = int(20 * s) + chart_h

        # ── histogram bars ──
        for i in range(n):
            bh      = int(smoothed[i] * chart_h)
            x       = vx + 10 + i * bar_w
            blocked = binary[i] == 1
            colour  = (50, 50, 220) if blocked else (50, 180, 80)
            cv2.rectangle(
                canvas,
                (x, base_y - bh),
                (x + bar_w - 1, base_y),
                colour, -1,
            )

        # ── threshold line ──
        ty = base_y - int(threshold * chart_h)
        cv2.line(canvas, (vx + 10, ty), (self.W - 10, ty), (180, 100, 220), 1)
        cv2.putText(canvas, f"thresh={threshold:.2f}", (vx + 12, ty - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47 * s, (180, 100, 220), 1)

        # ── x-axis labels ──
        sector_size = 360.0 / n
        for deg in [0, 90, 180, 270]:
            idx = int(round(deg / sector_size)) % n
            x   = vx + 10 + idx * bar_w
            cv2.putText(canvas, f"{deg}", (x, base_y + int(14 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40 * s, (160, 160, 160), 1)

        # ── polar diagram ──
        cx  = vx + (self.W - vx) // 2
        cy  = base_y + int(20 * s) + int(108 * s)
        rad = int(108 * s)

        cv2.circle(canvas, (cx, cy), rad,      (60, 60, 60), 1)
        cv2.circle(canvas, (cx, cy), rad // 2, (40, 40, 40), 1)

        for i in range(n):
            ang_deg = i * sector_size - 90
            ang_rad = math.radians(ang_deg)
            density = smoothed[i]
            r_bar   = int(density * rad)
            x3 = int(cx + r_bar * math.cos(ang_rad))
            y3 = int(cy + r_bar * math.sin(ang_rad))
            blocked = binary[i] == 1
            col = (50, 50, 200) if blocked else (50, 160, 60)
            cv2.line(canvas, (cx, cy), (x3, y3), col, 2)

        # ── valley markers ──
        for v in vfh_result.get("valleys", []):
            vang = math.radians(v["center_deg"] - 90)
            vx2  = int(cx + (rad + int(10 * s)) * math.cos(vang))
            vy2  = int(cy + (rad + int(10 * s)) * math.sin(vang))
            cv2.circle(canvas, (vx2, vy2), int(5 * s), (0, 255, 180), -1)

        # ── best valley arrow ──
        bv = vfh_result.get("best_valley")
        if bv:
            sang = math.radians(bv["steering_deg"] - 90)
            sx2  = int(cx + (rad + int(18 * s)) * math.cos(sang))
            sy2  = int(cy + (rad + int(18 * s)) * math.sin(sang))
            cv2.arrowedLine(canvas, (cx, cy), (sx2, sy2), (0, 220, 255), 2, tipLength=0.25)

        cv2.circle(canvas, (cx, cy), int(10 * s), (200, 200, 200), -1)
        cv2.circle(canvas, (cx, cy), int(10 * s), (255, 255, 255),  1)

        # ── bottom info bar ──
        cmd   = vfh_result.get("cmd", {})
        state = cmd.get("state", "?")
        obs   = cmd.get("obstacle_direction", "none")
        bv_deg = bv["steering_deg"] if bv else float("nan")

        state_col = {
            "move":  (80, 200, 80),
            "avoid": (50, 160, 220),
            "stop":  (50, 50, 220),
        }.get(state, (200, 200, 200))

        bar_h  = int(108 * s)
        info_y = self.H - bar_h
        cv2.rectangle(canvas, (0, info_y - 4), (self.W, self.H), (20, 20, 20), -1)
        cv2.putText(canvas, f"STATE: {state.upper()}",
                    (8, info_y + int(18 * s)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68 * s, state_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"obstacle={obs}  steer={bv_deg:.1f}deg",
                    (8, info_y + int(40 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.51 * s, (200, 200, 200), 1)

        n_blocked = int(np.sum(binary))
        n_valleys = len(vfh_result.get("valleys", []))
        target_str = f"{vfh_result.get('target_deg', 0.0):.1f}"
        cv2.putText(canvas,
                    f"blocked={n_blocked}/{n}  valleys={n_valleys}  target={target_str}deg",
                    (8, info_y + int(60 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.51 * s, (150, 150, 150), 1)
        cv2.putText(canvas,
                    f"robot_r=0.346m  d_min=300mm  d_max=5000mm",
                    (8, info_y + int(80 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (100, 100, 100), 1)

        # ── divider ──
        cv2.line(canvas, (half, 0), (half, self.H), (60, 60, 60), 1)

        return canvas


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class VFHObstacleAvoidanceNode(Node):

    def __init__(self) -> None:
        super().__init__("vfh_obstacle_avoidance")

        # ── parameters ──────────────────────────────────────────────────
        self.declare_parameter("n_sectors",        36)
        self.declare_parameter("threshold",        0.45)
        self.declare_parameter("smooth_window",    2)
        self.declare_parameter("d_max",            5000.0)
        self.declare_parameter("d_min",            300.0)
        self.declare_parameter("min_valley_width", 3)
        self.declare_parameter("target_angle",     0.0)
        self.declare_parameter("frame_skip",       1)
        self.declare_parameter("publish_visual",   True)
        self.declare_parameter("show_debug",       True)
        self.declare_parameter("robot_radius",     346.0)
        self.declare_parameter("safety_dist",      500.0)
        self.declare_parameter("hfov_deg",         88.0)
        self.declare_parameter("depth_source",     "float32")

        p = lambda name, t: self.get_parameter(name).get_parameter_value().__getattribute__(
            {"int": "integer_value", "float": "double_value",
             "bool": "bool_value", "str": "string_value"}[t]
        )

        n_sectors        = p("n_sectors",        "int")
        threshold        = p("threshold",        "float")
        smooth_window    = p("smooth_window",    "int")
        d_max            = p("d_max",            "float")
        d_min            = p("d_min",            "float")
        min_valley_width = p("min_valley_width", "int")
        self._target_angle   = p("target_angle", "float")
        self._frame_skip     = max(1, p("frame_skip", "int"))
        self._publish_visual = p("publish_visual", "bool")
        self._show_debug     = p("show_debug",     "bool")
        robot_radius         = p("robot_radius",   "float")
        safety_dist          = p("safety_dist",    "float")
        hfov_deg             = p("hfov_deg",       "float")
        self._depth_source   = p("depth_source",   "str")

        self._enable_display = HAS_DISPLAY and self._show_debug

        # ── VFH core + visualiser ────────────────────────────────────────
        self._vfh = VFH(
            n_sectors        = n_sectors,
            threshold        = threshold,
            smooth_window    = smooth_window,
            d_max            = d_max,
            d_min            = d_min,
            min_valley_width = min_valley_width,
            robot_radius     = robot_radius,
            safety_dist      = safety_dist,
            hfov_deg         = hfov_deg,
        )
        self._vis = VFHVisualiser(width=960, height=540)

        # ── state ────────────────────────────────────────────────────────
        self._skip_counter:   int               = 0
        self._last_depth_bgr: np.ndarray | None = None
        self._last_result:    dict | None       = None

        # ── ROS interfaces ───────────────────────────────────────────────
        self._bridge = CvBridge()

        # Subscribe to /realsense/depth/float32 (32FC1, meters) from realsense_d455_node
        if self._depth_source == "float32":
            depth_topic = "/realsense/depth/float32"
            depth_enc   = "32FC1"
        elif self._depth_source == "uint16":
            depth_topic = "/realsense/depth/z16"
            depth_enc   = "16UC1"
        else:
            # Depth Anything V2
            depth_topic = "/camera/depth_map"
            depth_enc   = "32FC1"

        self._depth_encoding = depth_enc
        self._sub_depth = self.create_subscription(
            Image, depth_topic,
            self._depth_callback, _DEPTH_QOS
        )
        self._sub_target = self.create_subscription(
            Float32, "/vfh/target_angle", self._target_callback, 10
        )

        self._pub_cmd       = self.create_publisher(Twist,  "/vfh/cmd_vel",   _CMD_QOS)
        self._pub_obstacles = self.create_publisher(String, "/vfh/obstacles", 10)
        self._pub_debug     = self.create_publisher(String, "/vfh/movement_direction", 10)
        self._pub_visual    = self.create_publisher(Image,  "/vfh/visual",    _DEPTH_QOS)

        # ── startup log ──────────────────────────────────────────────────
        self.get_logger().info("=" * 60)
        self.get_logger().info("VFH Obstacle Avoidance Node  [com_px_to_jetson]")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Depth source   : {self._depth_source}")
        self.get_logger().info(f"  Subscribe      : {depth_topic}  ({depth_enc})")
        self.get_logger().info(f"  Unit           : {'mm (no conversion)' if self._depth_source == 'uint16' else 'meter'}")
        self.get_logger().info(f"  Drone size     : 49cm x 49cm")
        self.get_logger().info(f"  robot_radius   : {robot_radius:.0f} mm")
        self.get_logger().info(f"  safety_dist    : {safety_dist:.0f} mm")
        self.get_logger().info(f"  min_valley_w   : {min_valley_width} sectors = {min_valley_width * 360.0 / n_sectors:.0f} deg")
        self.get_logger().info(f"  n_sectors      : {n_sectors}  threshold={threshold}  smooth={smooth_window}")
        self.get_logger().info(f"  d_max={d_max:.0f}mm  d_min={d_min:.0f}mm  hfov={hfov_deg}deg")
        valid_sectors = int(self._vfh.fov_mask.sum())
        self.get_logger().info(f"  FOV sectors    : {valid_sectors}/{n_sectors} valid")
        self.get_logger().info("-" * 60)
        self.get_logger().info("  Publishing:")
        self.get_logger().info("    /vfh/cmd_vel   /vfh/obstacles   /vfh/movement_direction   /vfh/visual")
        self.get_logger().info("=" * 60)

    # ── callbacks ────────────────────────────────────────────────────────

    def _target_callback(self, msg: Float32) -> None:
        self._target_angle = float(msg.data) % 360.0
        self.get_logger().info(
            f"[VFH] target_angle → {self._target_angle:.1f}°",
            throttle_duration_sec=1.0,
        )

    def _depth_callback(self, msg: Image) -> None:
        """
        Main callback — supports two formats:
            16UC1  (uint16, mm)   <- /realsense/depth/z16  from realsense_d455_node
            32FC1  (float32, m)   <- /camera/depth_map     from depth_node
        """
        self._skip_counter += 1
        if self._skip_counter % self._frame_skip != 0:
            return

        try:
            if self._depth_source == "float32":
                # /realsense/depth/float32 — 32FC1, values in METERS
                # Convert to mm since the whole VFH works in mm
                raw_m   = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
                depth_m = raw_m * 1000.0            # meters -> mm
                depth_m[raw_m == 0.0] = np.nan      # no-data -> NaN
            elif self._depth_source == "uint16":
                # /realsense/depth/z16 — 16UC1, values in mm
                raw     = self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
                depth_m = raw.astype(np.float32)    # stays mm
                depth_m[raw == 0]     = np.nan
                depth_m[raw == 65535] = np.nan
            else:
                # /camera/depth_map — 32FC1, values in meters -> convert to mm
                raw_m   = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
                depth_m = raw_m * 1000.0
        except Exception as exc:
            self.get_logger().error(f"cv_bridge error: {exc}")
            return

        try:
            result = self._vfh.run(
                depth_map     = depth_m,
                target_deg    = self._target_angle,
                robot_heading = 0.0,   # the drone is always at 0° orientation (facing East)
            )
        except Exception as exc:
            self.get_logger().error(f"VFH error: {exc}")
            return

        self._last_result = result
        now = self.get_clock().now().to_msg()

        self._pub_cmd.publish(Twist())

        # obstacle map
        obstacle_groups = self._vfh.extract_obstacle_map(
            smoothed_hist = np.array(result["smoothed_hist"]),
            binary_hist   = np.array(result["binary_hist"]),
        )
        obs_msg      = String()
        obs_msg.data = json.dumps({
            "stamp":        f"{now.sec}.{now.nanosec:09d}",
            "n_obstacles":  len(obstacle_groups),
            "has_critical": any(g["is_critical"] for g in obstacle_groups),
            "obstacles":    obstacle_groups,
        })
        self._pub_obstacles.publish(obs_msg)

        # debug JSON
        _bv = result.get("best_valley")
        debug_msg      = String()
        debug_msg.data = json.dumps({
            "stamp":         f"{now.sec}.{now.nanosec:09d}",
            "target_deg":    result["target_deg"],
            "state":         result["cmd"]["state"],
            "obstacle":      result["cmd"]["obstacle_direction"],
            "avoid_toward":  result["cmd"]["avoid_direction"],
            # steer_deg already includes the lateral proximity penalty
            "steer_deg":     result.get("steer_deg",
                                round(_bv["steering_deg"], 1) if _bv else None),
            "lateral_delta": result.get("lateral_delta", 0.0),
            "valley_center": round(_bv["center_deg"], 1) if _bv else None,
            "n_valleys":     len(result["valleys"]),
            "n_blocked":     int(np.sum(result["binary_hist"])),
            "depth_source":  self._depth_source,
            "depth_topic":   {"float32": "/realsense/depth/float32",
                              "uint16":  "/realsense/depth/z16"}.get(
                              self._depth_source, "/camera/depth_map"),
        })
        self._pub_debug.publish(debug_msg)

        # visual
        if self._publish_visual or self._enable_display:
            depth_bgr = self._colorize_depth(depth_m)
            self._last_depth_bgr = depth_bgr
            vis = self._vis.render(
                depth_bgr  = depth_bgr,
                vfh_result = result,
                n_sectors  = self._vfh.n_sectors,
                threshold  = self._vfh.threshold,
            )
            if self._publish_visual:
                try:
                    vis_msg = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
                    vis_msg.header.stamp    = now
                    vis_msg.header.frame_id = msg.header.frame_id
                    self._pub_visual.publish(vis_msg)
                except Exception as exc:
                    self.get_logger().error(f"Publish visual error: {exc}")

            if self._enable_display:
                cv2.namedWindow("VFH Obstacle Avoidance", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("VFH Obstacle Avoidance", 960, 540)
                cv2.imshow("VFH Obstacle Avoidance", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    rclpy.shutdown()

        cmd = result["cmd"]
        bv  = result.get("best_valley")
        steer_str = f"{bv['steering_deg']:.1f}" if bv else "None"
        self.get_logger().info(
            f"[VFH] "
            f"State: {cmd['state']} | "
            f"Obstacle: {cmd['obstacle_direction']} | "
            f"Avoid toward: {cmd['avoid_direction']} | "
            f"Steer: {steer_str} deg",
            throttle_duration_sec=0.5,
        )

    @staticmethod
    def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
        # Ideal D455 range: 600mm–6000mm
        # Pixels < 600mm = full red, > 6000mm = full blue
        D_MIN_MM = 600.0
        D_MAX_MM = 6000.0
        valid_mask = np.isfinite(depth_m) & (depth_m > 0)
        if not valid_mask.any():
            return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
        norm    = ((depth_m - D_MIN_MM) / (D_MAX_MM - D_MIN_MM) * 255).clip(0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(255 - norm, cv2.COLORMAP_JET)
        colored[~valid_mask] = 0
        return colored

    def destroy_node(self) -> None:
        if self._enable_display:
            cv2.destroyAllWindows()
        try:
            self._pub_cmd.publish(Twist())
        except Exception:
            pass
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = VFHObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
