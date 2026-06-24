#!/usr/bin/env python3
"""
px4_sensor_reader.py
====================
ROS2 node for reading sensor data from PX4 via MAVROS.
MAVROS is launched automatically when the node starts.

    ros2 run com_px_to_jetson px4_sensor_reader

Connection: Jetson Nano <-> PX4/PX6 via USB-C to USB-A

Topics published by this node:
    /px4/state     (std_msgs/String, JSON) — PX4 state for other nodes
    /px4/sensors   (std_msgs/String, JSON) — full sensor snapshot (optional)

Topics subscribed from MAVROS:
    /mavros/imu/data, /mavros/imu/mag, /mavros/imu/static_pressure,
    /mavros/imu/temperature_imu, /mavros/global_position/global,
    /mavros/local_position/pose, /mavros/local_position/velocity_local,
    /mavros/battery, /mavros/state, /mavros/altitude

FIXES applied (v2):
    [FIX-1] Terrain NaN -> safe_float_str
    [FIX-2] Battery percentage inconsistency
    [FIX-3] is_mavros_active() publisher count check
    [FIX-4] Log file handle leak
    [FIX-5] TimeoutExpired not caught
    [FIX-6] import signal moved to top-level
    [FIX-7] Python 3.10+ type hints incompatible with Jetson Nano
    [FIX-8] GPS fallback log missing

[FIX-9] Publish local_x and local_y in /px4/sensors snapshot
        so that other nodes (takeoff_land_test, etc.) can read
        full 3D position without subscribing to MAVROS directly.

[NEW] Publish /px4/state and /px4/sensors as JSON
      so other nodes (control_command, etc.) don't need to
      subscribe directly to MAVROS topics.
"""

import glob
import json
import math
import os
import signal
import subprocess
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Imu, MagneticField, FluidPressure, Temperature, NavSatFix, BatteryState
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, Altitude, StatusText
from mavros_msgs.msg import HomePosition
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Auto-detect FCU URL: USB if available, SITL if not
# ---------------------------------------------------------------------------
def detect_fcu_url() -> str:
    candidates = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
    if candidates:
        return "{0}:57600".format(candidates[0])
    return "udp://:14540@127.0.0.1:14580"  # SITL default


# ---------------------------------------------------------------------------
# Check if MAVROS is already running
# ---------------------------------------------------------------------------
def is_mavros_active() -> bool:
    for topic in ("/mavros/state", "/uas1/state"):
        try:
            result = subprocess.run(
                ["ros2", "topic", "info", topic],
                capture_output=True, text=True, timeout=3,
            )
            output = result.stdout
            if "Publisher count:" in output and "Publisher count: 0" not in output:
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# QoS Profiles
# ---------------------------------------------------------------------------
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


# ---------------------------------------------------------------------------
# Helper: Quaternion -> Euler angles
# ---------------------------------------------------------------------------
def quaternion_to_euler_deg(q) -> Tuple[float, float, float]:
    x, y, z, w = q.x, q.y, q.z, q.w
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ---------------------------------------------------------------------------
# Helper: safe float format
# ---------------------------------------------------------------------------
def safe_float_str(value, fmt=".2f", unit="", fallback="N/A") -> str:
    try:
        if math.isnan(value) or math.isinf(value):
            return fallback
        return ("{0:" + fmt + "}{1}").format(value, unit)
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Main Node
# ---------------------------------------------------------------------------
class PX4SensorReader(Node):

    def __init__(self):
        super().__init__("px4_sensor_reader")

        # -- parameter fcu_url --
        # Default: auto-detect USB, fallback to SITL UDP
        self.declare_parameter("fcu_url", detect_fcu_url())
        fcu_url = self.get_parameter("fcu_url").get_parameter_value().string_value

        # -- parameter auto_launch_mavros --
        # Set False when MAVROS is already launched externally (e.g. mavros_only.launch.py)
        # to prevent px4_sensor_reader from spawning a second conflicting MAVROS instance.
        self.declare_parameter("auto_launch_mavros", True)
        auto_launch_mavros = self.get_parameter("auto_launch_mavros").get_parameter_value().bool_value

        # -- sensor data storage --
        self.imu_data         = None  # type: Optional[Imu]
        self.mag_data         = None  # type: Optional[MagneticField]
        self.gps_data         = None  # type: Optional[NavSatFix]
        self.local_pose       = None  # type: Optional[PoseStamped]
        self.local_velocity   = None  # type: Optional[TwistStamped]
        self.altitude_data    = None  # type: Optional[Altitude]
        self.battery_status   = None  # type: Optional[BatteryState]
        self.vehicle_state    = None  # type: Optional[State]
        self.baro_pressure    = None  # type: Optional[FluidPressure]
        self.imu_temperature  = None  # type: Optional[Temperature]
        self.home_position    = None  # type: Optional[HomePosition]
        self._last_px4_status = ""

        # -- MAVROS process --
        self._mavros_proc     = None  # type: Optional[subprocess.Popen]
        self._mavros_log_file = None
        if auto_launch_mavros:
            self._launch_mavros_if_needed(fcu_url)
        else:
            self.get_logger().info("auto_launch_mavros=False — MAVROS managed externally.")

        # -- publishers ----------------------------------------------------------
        # /px4/state   : compact state for other nodes (control_command, etc.)
        # /px4/sensors : full sensor snapshot (optional, for monitoring)
        self._pub_state   = self.create_publisher(String, "/px4/state",   10)
        self._pub_sensors = self.create_publisher(String, "/px4/sensors", 10)

        # -- MAVROS subscribers --
        self.create_subscription(Imu,           "/mavros/imu/data",                      self._cb_imu,           SENSOR_QOS)
        self.create_subscription(MagneticField, "/mavros/imu/mag",                       self._cb_mag,           SENSOR_QOS)
        self.create_subscription(FluidPressure, "/mavros/imu/static_pressure",           self._cb_baro,          SENSOR_QOS)
        self.create_subscription(Temperature,   "/mavros/imu/temperature_imu",           self._cb_temperature,   SENSOR_QOS)
        self.create_subscription(NavSatFix,     "/mavros/global_position/global",        self._cb_gps,           SENSOR_QOS)
        self.create_subscription(PoseStamped,   "/mavros/local_position/pose",           self._cb_local_pose,    SENSOR_QOS)
        self.create_subscription(TwistStamped,  "/mavros/local_position/velocity_local", self._cb_local_velocity,SENSOR_QOS)
        self.create_subscription(BatteryState,  "/mavros/battery",                       self._cb_battery,       SENSOR_QOS)
        self.create_subscription(State,         "/mavros/state",                         self._cb_state,         STATE_QOS)
        self.create_subscription(Altitude,      "/mavros/altitude",                      self._cb_altitude,      SENSOR_QOS)
        self.create_subscription(HomePosition,  "/mavros/home_position/home",            self._cb_home,          STATE_QOS)
        self.create_subscription(StatusText,    "/mavros/statustext/recv",               self._cb_statustext,    SENSOR_QOS)

        # -- timers --
        self.create_timer(1.0,  self._timer_log)        # terminal log at 1 Hz
        self.create_timer(0.1,  self._timer_publish)    # publish /px4/* at 10 Hz

        self.get_logger().info("PX4SensorReader started.")
        self.get_logger().info("  Publishing: /px4/state  /px4/sensors")
        self.get_logger().info("  local_x, local_y, local_z all included in /px4/sensors")

    # -- MAVROS launch --------------------------------------------------------

    def _launch_mavros_if_needed(self, fcu_url: str):
        if is_mavros_active():
            self.get_logger().info("MAVROS already running, skipping auto-launch.")
            return
        self.get_logger().info("Auto-launching MAVROS  FCU: {0}".format(fcu_url))
        log_path = os.path.expanduser("~/.ros/mavros_auto.log")
        self._mavros_log_file = open(log_path, "w")
        self._mavros_proc = subprocess.Popen(
            [
                "ros2", "launch", "mavros", "px4.launch",
                "fcu_url:={0}".format(fcu_url),
            ],
            stdout=self._mavros_log_file,
            stderr=self._mavros_log_file,
            preexec_fn=os.setsid,
        )
        self.get_logger().info("MAVROS PID={0}".format(self._mavros_proc.pid))
        time.sleep(4)

    def destroy_node(self):
        if self._mavros_proc is not None and self._mavros_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._mavros_proc.pid), signal.SIGTERM)
                self._mavros_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._mavros_proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
        if self._mavros_log_file is not None:
            try:
                self._mavros_log_file.close()
            except Exception:
                pass
        super().destroy_node()

    # -- sensor callbacks -----------------------------------------------------

    def _cb_imu(self, msg):            self.imu_data        = msg
    def _cb_mag(self, msg):            self.mag_data        = msg
    def _cb_baro(self, msg):           self.baro_pressure   = msg
    def _cb_temperature(self, msg):    self.imu_temperature = msg
    def _cb_gps(self, msg):            self.gps_data        = msg
    def _cb_local_pose(self, msg):     self.local_pose      = msg
    def _cb_local_velocity(self, msg): self.local_velocity  = msg
    def _cb_battery(self, msg):        self.battery_status  = msg
    def _cb_altitude(self, msg):       self.altitude_data   = msg
    def _cb_home(self, msg):           self.home_position   = msg

    def _cb_statustext(self, msg):
        self._last_px4_status = msg.text
        sev = msg.severity
        if sev <= 4:
            self.get_logger().warn("[PX4] {0}".format(msg.text))
        else:
            self.get_logger().info("[PX4] {0}".format(msg.text))

    def _cb_state(self, msg):
        """Receive /mavros/state -> store + re-publish to /px4/state."""
        prev = self.vehicle_state
        self.vehicle_state = msg

        # log state changes
        if prev is None or prev.connected != msg.connected:
            self.get_logger().info(
                "MAVROS {0}".format("CONNECTED" if msg.connected else "DISCONNECTED")
            )
        if prev is None or prev.armed != msg.armed:
            self.get_logger().info(
                "PX4 {0}  mode={1}".format(
                    "ARMED" if msg.armed else "DISARMED", msg.mode
                )
            )
        if prev is None or prev.mode != msg.mode:
            self.get_logger().info("PX4 mode -> {0}".format(msg.mode))

        # re-publish to /px4/state
        self._publish_state()

    # -- publishers -----------------------------------------------------------

    def _publish_state(self):
        """
        Publish PX4 state to /px4/state (JSON).
        Called on every /mavros/state change AND every 10Hz timer tick.

        Format:
        {
            "connected": bool,
            "armed":     bool,
            "mode":      str,
            "guided":    bool,
            "manual":    bool
        }
        """
        if self.vehicle_state is None:
            return

        state_dict = {
            "connected": self.vehicle_state.connected,
            "armed":     self.vehicle_state.armed,
            "mode":      self.vehicle_state.mode,
            "guided":    self.vehicle_state.guided,
            "manual":    self.vehicle_state.manual_input,
        }
        msg = String()
        msg.data = json.dumps(state_dict)
        self._pub_state.publish(msg)

    def _timer_publish(self):
        """10 Hz: publish /px4/state and /px4/sensors."""
        # state
        self._publish_state()

        # full sensor snapshot
        snapshot = self.get_sensor_snapshot()
        if snapshot:
            msg = String()
            msg.data = json.dumps(snapshot)
            self._pub_sensors.publish(msg)

    # -- terminal log at 1 Hz -------------------------------------------------

    def _timer_log(self):
        sep = "=" * 60
        if self.vehicle_state is None or not self.vehicle_state.connected:
            self.get_logger().warn("Waiting for MAVROS connection...")
            return

        self.get_logger().info(sep)
        self.get_logger().info(
            "Mode: {0}  Armed: {1}  Guided: {2}".format(
                self.vehicle_state.mode,
                self.vehicle_state.armed,
                self.vehicle_state.guided,
            )
        )

        # -- IMU attitude -------------------------------------------------------
        if self.imu_data:
            r, p, y = quaternion_to_euler_deg(self.imu_data.orientation)
            level_warn = ""
            if abs(r) > 10.0 or abs(p) > 10.0:
                level_warn = "  WARNING: NOT LEVEL"
            self.get_logger().info(
                "[IMU]  Roll={0:+.1f}  Pitch={1:+.1f}  Yaw={2:+.1f} deg{3}".format(
                    r, p, y, level_warn
                )
            )

        # -- Local position (ENU) -----------------------------------------------
        if self.local_pose:
            x = self.local_pose.pose.position.x
            y = self.local_pose.pose.position.y
            z = self.local_pose.pose.position.z
            self.get_logger().info(
                "[POS]  x={0:.3f}m  y={1:.3f}m  z={2:.3f}m  (ENU local frame)".format(x, y, z)
            )

        # -- Magnetometer (compass) ---------------------------------------------
        if self.mag_data:
            mx = self.mag_data.magnetic_field.x
            my = self.mag_data.magnetic_field.y
            mz = self.mag_data.magnetic_field.z
            strength = math.sqrt(mx*mx + my*my + mz*mz) * 1e6  # T -> uT
            heading = math.degrees(math.atan2(my, mx)) % 360.0
            mag_warn = ""
            if strength < 5.0:
                mag_warn = "  WARNING: TOO WEAK (normal: 25-65 uT)"
            self.get_logger().info(
                "[MAG]  Heading={0:.1f} deg  strength={1:.2f}uT  "
                "x={2:.2e}  y={3:.2e}  z={4:.2e} T{5}".format(
                    heading, strength, mx, my, mz, mag_warn
                )
            )
        else:
            self.get_logger().warn("[MAG]  No data <- may cause 'Yaw estimate error'")

        # -- GPS ----------------------------------------------------------------
        if self.gps_data:
            fix_map = {-1: "NO FIX", 0: "FIX", 1: "SBAS", 2: "GBAS"}
            fix = fix_map.get(self.gps_data.status.status, "?")
            self.get_logger().info(
                "[GPS]  Lat={0:.7f}  Lon={1:.7f}  Alt={2:.2f}m  Fix={3}".format(
                    self.gps_data.latitude,
                    self.gps_data.longitude,
                    self.gps_data.altitude,
                    fix,
                )
            )
        else:
            self.get_logger().warn("[GPS]  No data")

        # -- Velocity (EKF velocity estimate) -----------------------------------
        if self.local_velocity:
            vx = self.local_velocity.twist.linear.x
            vy = self.local_velocity.twist.linear.y
            vz = self.local_velocity.twist.linear.z
            spd = math.sqrt(vx*vx + vy*vy + vz*vz)
            ekf_warn = ""
            if not (self.vehicle_state and self.vehicle_state.armed) and spd > 0.15:
                ekf_warn = "  WARNING: EKF not yet converged"
            self.get_logger().info(
                "[VEL]  vx={0:+.2f}  vy={1:+.2f}  vz={2:+.2f}  speed={3:.2f} m/s{4}".format(
                    vx, vy, vz, spd, ekf_warn
                )
            )

        # -- Altitude -----------------------------------------------------------
        if self.altitude_data:
            self.get_logger().info(
                "[ALT]  AMSL={0}  Rel={1}  Terrain={2}".format(
                    safe_float_str(self.altitude_data.amsl,     unit="m"),
                    safe_float_str(self.altitude_data.relative, unit="m"),
                    safe_float_str(self.altitude_data.terrain,  unit="m"),
                )
            )

        # -- Home position ------------------------------------------------------
        if self.home_position:
            self.get_logger().info(
                "[HOME] Lat={0:.7f}  Lon={1:.7f}  Alt={2:.2f}m  SET".format(
                    self.home_position.geo.latitude,
                    self.home_position.geo.longitude,
                    self.home_position.geo.altitude,
                )
            )
        else:
            self.get_logger().warn("[HOME] Not set <- RTL will not work")

        # -- Battery ------------------------------------------------------------
        if self.battery_status:
            raw = self.battery_status.percentage
            if raw < 0:
                self.get_logger().info("[BAT]  No battery")
            else:
                pct = raw * 100.0
                bat_warn = "  WARNING: LOW" if pct < 20.0 else ""
                self.get_logger().info(
                    "[BAT]  {0:.2f}V  {1:.1f}%{2}".format(
                        self.battery_status.voltage, pct, bat_warn
                    )
                )

        # -- Last PX4 status message --------------------------------------------
        if self._last_px4_status:
            self.get_logger().info("[PX4]  {0}".format(self._last_px4_status))

        self.get_logger().info(sep)

    # -- snapshot -------------------------------------------------------------

    def get_sensor_snapshot(self) -> dict:
        snapshot = {
            "connected": self.vehicle_state.connected if self.vehicle_state else False,
            "mode":      self.vehicle_state.mode      if self.vehicle_state else "UNKNOWN",
            "armed":     self.vehicle_state.armed     if self.vehicle_state else False,
        }

        if self.imu_data:
            r, p, y = quaternion_to_euler_deg(self.imu_data.orientation)
            snapshot["imu"] = {
                "roll_deg": r, "pitch_deg": p, "yaw_deg": y,
                "accel_x": self.imu_data.linear_acceleration.x,
                "accel_y": self.imu_data.linear_acceleration.y,
                "accel_z": self.imu_data.linear_acceleration.z,
            }

        if self.gps_data:
            snapshot["gps"] = {
                "latitude":   self.gps_data.latitude,
                "longitude":  self.gps_data.longitude,
                "altitude_m": self.gps_data.altitude,
                "fix_status": self.gps_data.status.status,
            }

        if self.altitude_data:
            terrain = self.altitude_data.terrain
            snapshot["altitude"] = {
                "amsl_m":     self.altitude_data.amsl,
                "relative_m": self.altitude_data.relative,
                "terrain_m":  None if math.isnan(terrain) else terrain,
            }

        if self.local_velocity:
            vx = self.local_velocity.twist.linear.x
            vy = self.local_velocity.twist.linear.y
            vz = self.local_velocity.twist.linear.z
            snapshot["velocity"] = {
                "vx": vx, "vy": vy, "vz": vz,
                "speed": math.sqrt(vx*vx + vy*vy + vz*vz),
            }

        if self.mag_data:
            mx = self.mag_data.magnetic_field.x
            my = self.mag_data.magnetic_field.y
            mz = self.mag_data.magnetic_field.z
            snapshot["mag"] = {
                "strength_uT": math.sqrt(mx*mx + my*my + mz*mz) * 1e6,
                "heading_deg": math.degrees(math.atan2(my, mx)) % 360.0,
            }

        # [FIX-9] Publish full 3D local position (x, y, z) in ENU frame.
        if self.local_pose:
            snapshot["local_x"] = self.local_pose.pose.position.x
            snapshot["local_y"] = self.local_pose.pose.position.y
            snapshot["local_z"] = self.local_pose.pose.position.z

            # Actual drone yaw in radians (ENU frame: 0=East, π/2=North)
            # Needed by control_command to convert the VFH camera frame
            # -> world frame during the AVOIDING phase.
            q = self.local_pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            snapshot["yaw"] = math.atan2(siny_cosp, cosy_cosp)  # radian

        # Velocity flat at the snapshot root — needed by control_command for
        # _ground_speed() and _wait_until_stopped()
        if self.local_velocity:
            snapshot["vel_x"] = self.local_velocity.twist.linear.x
            snapshot["vel_y"] = self.local_velocity.twist.linear.y
            snapshot["vel_z"] = self.local_velocity.twist.linear.z

        if self.battery_status:
            raw = self.battery_status.percentage
            snapshot["battery"] = {
                "voltage_V":          self.battery_status.voltage,
                "percentage_raw":     None if raw < 0.0 else raw,
                "percentage_display": None if raw < 0.0 else round(raw * 100.0, 1),
            }

        return snapshot


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = PX4SensorReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()