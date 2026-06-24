#!/usr/bin/env python3
"""
publish_pcd_node.py
===================
Reads a .pcd file (ground-truth map) and publishes it as a PointCloud2 to a
ROS2 topic periodically.

This replaces the role of a mapping session — instead of recording a map from
the drone camera, we directly use the ground-truth map from a .pcd file that was
converted from map.npz or from a Gazebo plugin.

Why this matters for expert planning:
  - The expert planner must know the WHOLE map (not just what the camera sees)
    so the MINCO optimizer can avoid all obstacles
  - The .pcd file contains the complete point cloud of the entire environment
  - The OctoMap server subscribes to this topic and builds the projected_map
  - The projected_map is then used by the ESDF for collision checking

Topics:
  Publish: /pointcloud/map  (sensor_msgs/PointCloud2)

Parameters:
  pcd_path   : path to the .pcd file               default: ~/maps/map.pcd
  frame_id   : point cloud reference frame          default: map
  pub_hz     : publish rate (Hz)                    default: 2.0

Usage:
  ros2 run nn_planner publish_pcd_node \\
    --ros-args -p pcd_path:=/home/michael/maps/bricks.pcd
"""

import os
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def read_pcd(path: str) -> np.ndarray:
    """
    Read a .pcd file and return an Nx3 float32 array (x, y, z).

    Supports formats:
      - DATA ascii   : each line contains "x y z"
      - DATA binary  : packed float32 (x, y, z) per point

    .pcd format:
      # .PCD v0.7
      FIELDS x y z
      SIZE 4 4 4
      TYPE F F F
      COUNT 1 1 1
      WIDTH N
      HEIGHT 1
      VIEWPOINT 0 0 0 0 0 0 1
      POINTS N
      DATA ascii
      x1 y1 z1
      x2 y2 z2
      ...
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f".pcd file not found: {path}")

    points = []
    data_type = "ascii"
    n_points  = 0
    fields    = []
    sizes     = []
    header_done = False

    with open(path, "rb") as f:
        # Read the header line by line
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                break
            try:
                line = line_bytes.decode("utf-8", errors="ignore").strip()
            except Exception:
                break

            if line.startswith("#"):
                continue
            if line.lower().startswith("fields"):
                fields = line.split()[1:]
            elif line.lower().startswith("size"):
                sizes = [int(x) for x in line.split()[1:]]
            elif line.lower().startswith("points"):
                n_points = int(line.split()[1])
            elif line.lower().startswith("data"):
                data_type = line.split()[1].lower()
                header_done = True
                # After the DATA line, the data content begins
                break

        if not header_done:
            raise ValueError("Invalid .pcd header — DATA line not found")

        # Make sure x, y, z fields exist
        if not all(f in fields for f in ["x", "y", "z"]):
            raise ValueError(f"x/y/z fields not found in header: {fields}")

        x_idx = fields.index("x")
        y_idx = fields.index("y")
        z_idx = fields.index("z")

        if data_type == "ascii":
            # ASCII format: read line by line
            for _ in range(n_points):
                line_bytes = f.readline()
                if not line_bytes:
                    break
                parts = line_bytes.decode("utf-8", errors="ignore").strip().split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[x_idx])
                        y = float(parts[y_idx])
                        z = float(parts[z_idx])
                        if all(np.isfinite([x, y, z])):
                            points.append([x, y, z])
                    except (ValueError, IndexError):
                        continue

        elif data_type == "binary":
            # Binary format: each point = n_fields * 4 bytes (float32)
            point_size = sum(sizes)
            raw = f.read(n_points * point_size)
            for i in range(n_points):
                offset = i * point_size
                chunk  = raw[offset:offset + point_size]
                if len(chunk) < point_size:
                    break
                vals = struct.unpack(f"{len(sizes)}f", chunk)
                x, y, z = vals[x_idx], vals[y_idx], vals[z_idx]
                if all(np.isfinite([x, y, z])):
                    points.append([x, y, z])
        else:
            raise ValueError(f"Unknown DATA format: {data_type}")

    if not points:
        raise ValueError(f"No valid points could be read from: {path}")

    return np.array(points, dtype=np.float32)


def make_pointcloud2(pts: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """
    Convert an Nx3 float32 array to sensor_msgs/PointCloud2.

    Format: XYZ float32, unordered (HEIGHT=1, WIDTH=N).
    """
    msg             = PointCloud2()
    msg.header      = Header()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id

    msg.height     = 1
    msg.width      = len(pts)
    msg.is_dense   = True
    msg.is_bigendian = False

    msg.fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 12          # 3 x float32 = 12 bytes per point
    msg.row_step   = 12 * len(pts)
    msg.data       = pts.tobytes()

    return msg


class PublishPcdNode(Node):
    """
    Node that reads a .pcd file and publishes it as a PointCloud2.

    The OctoMap server subscribes to /pointcloud/map and builds an OctoMap +
    projected_map (2D OccupancyGrid) from this point cloud.
    """

    def __init__(self):
        super().__init__("publish_pcd_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("pcd_path",
                               os.path.expanduser("~/maps/map.pcd"))
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("pub_hz",    2.0)

        self._pcd_path = str(self.get_parameter("pcd_path").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._pub_hz   = float(self.get_parameter("pub_hz").value)

        # ── Load .pcd ─────────────────────────────────────────────────────────
        self.get_logger().info(f"[PCD] Reading file: {self._pcd_path}")
        try:
            self._pts = read_pcd(self._pcd_path)
            self.get_logger().info(
                f"[PCD] OK {len(self._pts):,} points loaded | "
                f"bounds X:[{self._pts[:,0].min():.2f},{self._pts[:,0].max():.2f}] "
                f"Y:[{self._pts[:,1].min():.2f},{self._pts[:,1].max():.2f}] "
                f"Z:[{self._pts[:,2].min():.2f},{self._pts[:,2].max():.2f}]")
        except Exception as e:
            self.get_logger().error(f"[PCD] Failed to read .pcd: {e}")
            raise

        # ── Publisher ─────────────────────────────────────────────────────────
        self._pub = self.create_publisher(
            PointCloud2, "/pointcloud/map", 10)

        # Publish once immediately at startup
        self._publish_once()

        # ── Timer ─────────────────────────────────────────────────────────────
        self._timer = self.create_timer(
            1.0 / self._pub_hz, self._publish_once)

        self.get_logger().info(
            f"[PCD] Node active — publishing /pointcloud/map "
            f"@ {self._pub_hz} Hz | frame_id={self._frame_id}")

    def _publish_once(self):
        msg = make_pointcloud2(
            self._pts,
            self._frame_id,
            self.get_clock().now().to_msg(),
        )
        self._pub.publish(msg)
        self.get_logger().info(
            f"[PCD] Published {len(self._pts):,} points → /pointcloud/map",
            throttle_duration_sec=10.0)


def main(args=None):
    rclpy.init(args=args)
    node = PublishPcdNode()
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
