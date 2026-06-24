#!/usr/bin/env python3
"""
gz_depth_viewer_node.py

Subscribes to /realsense/depth/float32 then publishes a colorized depth image
(JET colormap) to a ROS2 topic so it can be viewed in rqt_image_view or rviz2
without needing an OpenCV GUI.

Topics Published:
  /depth_viz/image   sensor_msgs/Image  (BGR8) — colorized depth + overlay
  /depth_viz/debug   sensor_msgs/Image  (BGR8) — full canvas + histogram
"""

import json
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

COLORMAPS = [
    ("JET",     cv2.COLORMAP_JET),
    ("TURBO",   cv2.COLORMAP_TURBO),
    ("MAGMA",   cv2.COLORMAP_MAGMA),
    ("INFERNO", cv2.COLORMAP_INFERNO),
    ("PLASMA",  cv2.COLORMAP_PLASMA),
]

H_OUT  = 480
W_HIST = 280


class GzDepthViewerNode(Node):

    def __init__(self):
        super().__init__("gz_depth_viewer_node")

        self.declare_parameter("max_depth_m",  10.0)
        self.declare_parameter("cmap_index",   0)
        self.declare_parameter("input_topic",  "/realsense/depth/float32")

        self.max_depth  = self.get_parameter("max_depth_m").value
        self.cmap_idx   = int(self.get_parameter("cmap_index").value)
        input_topic     = self.get_parameter("input_topic").value

        self.bridge       = CvBridge()
        self.frame_count  = 0
        self.fps_val      = 0.0
        self._fps_t       = time.time()
        self._fps_n       = 0
        self._last_stats: dict = {}

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_img   = self.create_subscription(
            Image, input_topic, self._on_image, sensor_qos
        )
        self.sub_stats = self.create_subscription(
            String, "/realsense/depth/stats", self._on_stats, 10
        )

        self.pub_img   = self.create_publisher(Image, "/depth_viz/image", sensor_qos)
        self.pub_debug = self.create_publisher(Image, "/depth_viz/debug", sensor_qos)

        self.get_logger().info(
            f"GzDepthViewerNode active\n"
            f"  input  : {input_topic}\n"
            f"  output : /depth_viz/image  (colorized depth)\n"
            f"           /depth_viz/debug  (canvas + histogram)\n"
            f"  view   : ros2 run rqt_image_view rqt_image_view /depth_viz/image"
        )

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _on_image(self, msg: Image):
        try:
            f32 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().warn(str(e), throttle_duration_sec=2.0)
            return

        self.frame_count += 1
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self.fps_val = self._fps_n / (now - self._fps_t)
            self._fps_n  = 0
            self._fps_t  = now

        stamp = msg.header.stamp
        _, cmap = COLORMAPS[self.cmap_idx]

        # ── /depth_viz/image — colorized depth panel only ──
        depth_panel = self._colorize_depth(f32, cmap)
        msg_img = self.bridge.cv2_to_imgmsg(depth_panel, encoding="bgr8")
        msg_img.header.stamp    = stamp
        msg_img.header.frame_id = "camera_depth_frame"
        self.pub_img.publish(msg_img)

        # ── /depth_viz/debug — panel + histogram ──
        hist_panel = self._build_histogram(f32, cmap)
        canvas     = np.hstack([depth_panel, hist_panel])
        msg_dbg    = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        msg_dbg.header.stamp    = stamp
        msg_dbg.header.frame_id = "camera_depth_frame"
        self.pub_debug.publish(msg_dbg)

        # Log stats every 60 frames
        if self.frame_count % 60 == 0:
            # valid pixels: finite, positive, and within sensor range
            valid = f32[np.isfinite(f32) & (f32 > 0.0)]
            inf_pct  = (~np.isfinite(f32)).sum() * 100 / f32.size
            zero_pct = (f32 <= 0.0).sum() * 100 / f32.size
            self.get_logger().info(
                f"[#{self.frame_count}] fps={self.fps_val:.1f} | "
                f"mean={valid.mean():.2f}m  max={valid.max():.2f}m  "
                f"inf(beyond-range)={inf_pct:.1f}%  zero(invalid)={zero_pct:.1f}%"
                if valid.size else
                f"[#{self.frame_count}] fps={self.fps_val:.1f} | "
                f"inf={inf_pct:.1f}%  zero={zero_pct:.1f}%"
            )

    def _on_stats(self, msg: String):
        try:
            self._last_stats = json.loads(msg.data)
        except Exception:
            pass

    # ── depth colorizer ───────────────────────────────────────────────────────
    def _colorize_depth(self, f32: np.ndarray, cmap) -> np.ndarray:
        h, w   = f32.shape
        near, far = 0.2, self.max_depth

        # Clamp inf to max_depth so it can be colorized as "farthest"
        f32_vis = np.where(np.isfinite(f32), f32, far)

        norm   = np.clip((f32_vis - near) / (far - near), 0.0, 1.0)
        norm8  = (norm * 255).astype(np.uint8)
        panel  = cv2.applyColorMap(255 - norm8, cmap)  # invert: near=red
        panel[f32 <= 0.0]          = [10, 10, 10]      # zero/invalid -> black
        panel[~np.isfinite(f32)]   = [80, 30, 30]      # inf (beyond range) -> dark blue (BGR)

        w_out = int(H_OUT * w / h)
        panel = cv2.resize(panel, (w_out, H_OUT))

        cmap_name = COLORMAPS[self.cmap_idx][0]

        # header
        cv2.rectangle(panel, (0, 0), (w_out, 28), (15, 15, 15), -1)
        cv2.putText(panel, f"Depth  [{cmap_name}]  {near:.1f}m-{far:.0f}m",
                    (8, 19), cv2.FONT_HERSHEY_DUPLEX, 0.52, (80, 220, 120), 1, cv2.LINE_AA)
        cv2.putText(panel, f"FPS:{self.fps_val:.1f}  #{self.frame_count}",
                    (w_out - 140, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        # bottom stats bar
        valid    = f32[np.isfinite(f32) & (f32 > 0.0)]
        inf_pct  = (~np.isfinite(f32)).sum() * 100 / f32.size
        zero_pct = (f32 <= 0.0).sum() * 100 / f32.size
        if valid.size > 0:
            info = (f"  min:{valid.min():.2f}m  max:{valid.max():.2f}m  "
                    f"mean:{valid.mean():.2f}m  "
                    f"inf(far):{inf_pct:.0f}%  zero:{zero_pct:.0f}%")
        elif inf_pct > 50:
            info = f"  All pixels beyond range (inf={inf_pct:.0f}%) — no obstacle"
        else:
            info = "  waiting for data from Gazebo..."
        cv2.rectangle(panel, (0, H_OUT - 22), (w_out, H_OUT), (15, 15, 15), -1)
        cv2.putText(panel, info, (4, H_OUT - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)

        # color scale bar
        bar_h = H_OUT - 60
        scale = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
        scale_col = cv2.applyColorMap(scale, cmap)
        scale_col = cv2.resize(scale_col, (12, bar_h))
        panel[30:30 + bar_h, 8:20] = scale_col
        cv2.putText(panel, f"{far:.0f}m", (22, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
        cv2.putText(panel, f"{near:.1f}m", (22, 30 + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

        return panel

    # ── histogram panel ───────────────────────────────────────────────────────
    def _build_histogram(self, f32: np.ndarray, cmap) -> np.ndarray:
        panel = np.full((H_OUT, W_HIST, 3), 18, dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (W_HIST, 28), (15, 15, 15), -1)
        cv2.putText(panel, "Distance Histogram", (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 200, 80), 1)

        valid = f32[np.isfinite(f32) & (f32 > 0.2) & (f32 < self.max_depth)]
        if valid.size == 0:
            cv2.putText(panel, "waiting for data...", (16, H_OUT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)
            return panel

        BINS     = 32
        plot_top = 36
        plot_bot = H_OUT - 80
        plot_h   = plot_bot - plot_top
        bar_w    = (W_HIST - 16) / BINS

        hist, _ = np.histogram(valid, bins=BINS, range=(0.2, self.max_depth))
        hist_n  = hist / (hist.max() + 1e-6)

        for i, v in enumerate(hist_n):
            bh    = int(v * plot_h)
            x0    = int(8 + i * bar_w)
            x1    = int(8 + (i + 1) * bar_w) - 1
            ratio = i / BINS
            color = (int(20 + 200 * (1 - ratio)), 80, int(20 + 200 * ratio))
            cv2.rectangle(panel, (x0, plot_bot - bh), (x1, plot_bot), color, -1)

        # x-axis
        for frac, label in [(0, "0.2m"),
                            (0.5, f"{self.max_depth / 2:.1f}m"),
                            (1.0, f"{self.max_depth:.0f}m")]:
            x = int(8 + frac * (W_HIST - 16))
            cv2.putText(panel, label, (max(0, x - 12), plot_bot + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110, 110, 110), 1)

        # percentile lines
        for pct, col in [(25, (60, 180, 60)), (50, (255, 220, 0)), (75, (60, 120, 255))]:
            val = float(np.percentile(valid, pct))
            xp  = int(8 + (val - 0.2) / (self.max_depth - 0.2) * (W_HIST - 16))
            xp  = max(9, min(W_HIST - 9, xp))
            cv2.line(panel, (xp, plot_top), (xp, plot_bot), col, 1)
            cv2.putText(panel, f"p{pct}", (xp + 2, plot_top + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, col, 1)

        # stats text
        lines = [
            f"count : {valid.size:,} px",
            f"min   : {valid.min():.2f} m",
            f"max   : {valid.max():.2f} m",
            f"mean  : {valid.mean():.2f} m",
            f"std   : {valid.std():.2f} m",
            f"median: {float(np.median(valid)):.2f} m",
        ]
        for i, line in enumerate(lines):
            cv2.putText(panel, line, (8, plot_bot + 28 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)

        return panel


# ── entry point ───────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = GzDepthViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
