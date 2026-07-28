#!/usr/bin/env python3
"""
depth_to_pointcloud_node.py  (v2 — FIX TF time sync)

Converts a depth image + camera_info into a PointCloud2 to be consumed by
octomap_server. Does not need depth_image_proc — the conversion is done
manually with NumPy.

v2 FIX (FIX "Message Filter dropping message ... earlier than transform cache"):
  Previously the cloud header was copied from the depth image (carrying Gazebo
  SIM-TIME), while the TF odom->base_link from MAVROS uses WALL-CLOCK.
  As a result octomap tf2 could not transform -> ALL clouds dropped ->
  /projected_map never published -> ESDF not ready -> drone stuck.

  FIX: stamp the cloud with THIS NODE's clock (self.get_clock().now()), instead
  of copying the sim-time from Gazebo. This way the cloud uses the SAME time base
  as the MAVROS TF. Robust for two cases:
    - Without use_sim_time (default): now() = wall-clock = same as MAVROS TF
    - With use_sim_time on all nodes: now() = sim-time = consistent too
  frame_id is preserved (camera_depth_frame).

Input:
  /realsense/depth/float32      sensor_msgs/Image      (32FC1, meters)
  /realsense/depth/camera_info  sensor_msgs/CameraInfo

Output:
  /realsense/depth/points       sensor_msgs/PointCloud2
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


class DepthToPointcloudNode(Node):

    def __init__(self):
        super().__init__("depth_to_pointcloud_node")

        # 2026-07-26: default 0.15 -> 0.5. Returns closer than 0.5 m are almost
        # always near-field noise / self-view and, at 0.15 m, burned SPURIOUS
        # occupied cells into the octomap right at the drone -> esdf=0.00 at its
        # own cell -> stuck / chaotic escape. The launches override this too.
        self.declare_parameter("min_range",   0.5)
        self.declare_parameter("max_range",   4.0)
        self.declare_parameter("skip_pixels", 2)    # downsample: 1 point per N pixels

        self.min_range  = self.get_parameter("min_range").value
        self.max_range  = self.get_parameter("max_range").value
        self.skip       = max(1, int(self.get_parameter("skip_pixels").value))

        self.bridge    = CvBridge()
        self.cam_info  = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_cloud = self.create_publisher(
            PointCloud2, "/realsense/depth/points", sensor_qos
        )

        self.sub_info = self.create_subscription(
            CameraInfo, "/realsense/depth/camera_info", self._on_info, sensor_qos
        )
        self.sub_depth = self.create_subscription(
            Image, "/realsense/depth/float32", self._on_depth, sensor_qos
        )

        self.get_logger().info(
            f"DepthToPointcloudNode v2 active | skip={self.skip} | "
            f"range=[{self.min_range}, {self.max_range}]m | "
            f"cloud stamped with node clock (FIX TF sync)"
        )

    def _on_info(self, msg: CameraInfo):
        self.cam_info = msg

    def _on_depth(self, msg: Image):
        if self.cam_info is None:
            return

        # Stamp NOW, before the back-projection below. The stamp still has to be
        # this node's clock (see the v2 note in the module docstring — the Gazebo
        # image carries sim-time while the MAVROS TF is wall-clock), but taking
        # it at ARRIVAL instead of after ~76k points of numpy work removes that
        # processing time from the pose octomap transforms the cloud with. Any
        # residual latency still smears every point by v*L in world coordinates,
        # which is where stale phantom voxels come from — see STUCK_MAP_RESET_*
        # in fm_inference_base.py for the recovery path.
        stamp = self.get_clock().now().to_msg()

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().warn(f"imgmsg_to_cv2 failed: {e}", throttle_duration_sec=2.0)
            return

        fx = self.cam_info.k[0]
        fy = self.cam_info.k[4]
        cx = self.cam_info.k[2]
        cy = self.cam_info.k[5]

        h, w = depth.shape
        skip = self.skip

        v_idx, u_idx = np.mgrid[0:h:skip, 0:w:skip]
        z = depth[v_idx, u_idx]

        valid = (z >= self.min_range) & (z <= self.max_range) & np.isfinite(z)

        z_v = z[valid].astype(np.float32)
        u_v = u_idx[valid].astype(np.float32)
        v_v = v_idx[valid].astype(np.float32)

        # Back-project to 3D camera coordinates (optical: z=forward, x=right, y=down)
        x = (u_v - cx) / fx * z_v
        y = (v_v - cy) / fy * z_v

        points = np.column_stack([x, y, z_v]).astype(np.float32)

        # Skip empty clouds: octomap would shout "No data to copy" and there is no
        # point processing 0 points (e.g. when facing an open area).
        if len(points) == 0:
            return

        # FIX: stamp with THIS NODE's clock, not msg.header (which is Gazebo sim-time)
        frame_id = msg.header.frame_id
        self.pub_cloud.publish(self._to_pointcloud2(points, frame_id, stamp))

    def _to_pointcloud2(self, points: np.ndarray, frame_id: str,
                        stamp) -> PointCloud2:
        msg = PointCloud2()
        # FIX TF SYNC: node clock, not the depth image's sim-time stamp. Captured
        # at image arrival by the caller (not here) — see _on_depth.
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        msg.height     = 1
        msg.width      = len(points)
        msg.is_dense   = False
        msg.is_bigendian = False
        msg.fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step   = 12 * len(points)
        msg.data       = points.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointcloudNode()
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
