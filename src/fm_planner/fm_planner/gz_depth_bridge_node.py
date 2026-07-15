#!/usr/bin/env python3
"""
gz_depth_bridge_node.py

Receives a depth image from Gazebo (via ros_gz_bridge) and forwards it to the
topics used by the obstacle-avoidance / VFH pipeline:

  Input  (from ros_gz_bridge):
    /depth_camera              sensor_msgs/Image  (32FC1, meters)

  Output (compatible with realsense_d455_node):
    /realsense/depth/float32   sensor_msgs/Image  (32FC1, meters)
    /realsense/depth/z16       sensor_msgs/Image  (16UC1, mm)
    /realsense/depth/camera_info  sensor_msgs/CameraInfo
    /realsense/depth/stats        std_msgs/String  (JSON)

Camera intrinsics match the Intel RealSense D455 (hfov~88deg, 640x480).
"""

import json
import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

# ─── Intel RealSense D455 intrinsics (640×480, HFOV≈88°) ────────────────────
_W    = 640
_H    = 480
_HFOV = math.radians(88.0)                       # D455 HFOV
_FX   = _W / (2.0 * math.tan(_HFOV / 2.0))      # ≈ 332
_FY   = _FX
_CX   = _W / 2.0
_CY   = _H / 2.0


class GzDepthBridgeNode(Node):

    def __init__(self):
        super().__init__("gz_depth_bridge_node")

        self.declare_parameter("input_topic", "/depth_camera")
        self.declare_parameter("frame_id",    "camera_depth_frame")

        input_topic    = self.get_parameter("input_topic").value
        self.frame_id  = self.get_parameter("frame_id").value

        self.bridge       = CvBridge()
        self.frame_count  = 0
        self._cam_info    = self._make_camera_info()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_f32   = self.create_publisher(Image,      "/realsense/depth/float32",     sensor_qos)
        self.pub_z16   = self.create_publisher(Image,      "/realsense/depth/z16",         sensor_qos)
        self.pub_info  = self.create_publisher(CameraInfo, "/realsense/depth/camera_info", sensor_qos)
        self.pub_stats = self.create_publisher(String,     "/realsense/depth/stats",       10)

        self.sub = self.create_subscription(Image, input_topic, self._on_image, sensor_qos)

        self.get_logger().info(
            f"GzDepthBridgeNode active | input: {input_topic} | frame: {self.frame_id}"
        )

    # ── static camera info from SDF parameters ───────────────────────────────
    def _make_camera_info(self) -> CameraInfo:
        info = CameraInfo()
        info.width  = _W
        info.height = _H
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [_FX,  0.0, _CX,
                  0.0,  _FY, _CY,
                  0.0,  0.0, 1.0]
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [_FX,  0.0, _CX, 0.0,
                  0.0,  _FY, _CY, 0.0,
                  0.0,  0.0, 1.0, 0.0]
        return info

    # ── main callback ──────────────────────────────────────────────────────────
    def _on_image(self, msg: Image):
        try:
            # Gazebo gz sends depth as 32FC1 (float32, meters)
            f32 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().warn(f"imgmsg_to_cv2 failed: {e}", throttle_duration_sec=2.0)
            return

        stamp = msg.header.stamp

        # Convert to Z16 (uint16, mm) — invalid pixels (0 / negative) stay 0
        z16 = np.clip(f32 * 1000.0, 0, 65535).astype(np.uint16)
        z16[f32 <= 0.0] = 0

        # Publish Float32
        msg_f32 = self.bridge.cv2_to_imgmsg(f32, encoding="32FC1")
        msg_f32.header.stamp    = stamp
        msg_f32.header.frame_id = self.frame_id
        self.pub_f32.publish(msg_f32)

        # Publish Z16
        msg_z16 = self.bridge.cv2_to_imgmsg(z16, encoding="16UC1")
        msg_z16.header.stamp    = stamp
        msg_z16.header.frame_id = self.frame_id
        self.pub_z16.publish(msg_z16)

        # Publish CameraInfo
        self._cam_info.header.stamp    = stamp
        self._cam_info.header.frame_id = self.frame_id
        self.pub_info.publish(self._cam_info)

        # Stats every 30 frames
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self._publish_stats(f32, z16)

    # ── JSON stats ────────────────────────────────────────────────────────────
    def _publish_stats(self, f32: np.ndarray, z16: np.ndarray):
        vf    = f32[f32 > 0.0]
        vz    = z16[z16 > 0]
        total = f32.size

        stats = {
            "frame":      self.frame_count,
            "source":     "gazebo_x500_depth",
            "resolution": f"{_W}x{_H}",
            "float32": {
                "unit":        "meter",
                "valid_px":    int(vf.size),
                "invalid_pct": round((total - vf.size) / total * 100, 2),
                "min":         round(float(vf.min()),  3) if vf.size else 0,
                "max":         round(float(vf.max()),  3) if vf.size else 0,
                "mean":        round(float(vf.mean()), 3) if vf.size else 0,
                "std":         round(float(vf.std()),  3) if vf.size else 0,
            },
            "z16": {
                "unit":     "mm",
                "valid_px": int(vz.size),
                "mean":     round(float(vz.mean()), 1) if vz.size else 0,
                "std":      round(float(vz.std()),  1) if vz.size else 0,
            },
        }
        msg = String()
        msg.data = json.dumps(stats)
        self.pub_stats.publish(msg)

        self.get_logger().info(
            f"[gz_depth #{self.frame_count}] "
            f"mean={stats['float32']['mean']:.3f}m  "
            f"invalid={stats['float32']['invalid_pct']:.1f}%"
        )


# ── entry point ───────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = GzDepthBridgeNode()
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
