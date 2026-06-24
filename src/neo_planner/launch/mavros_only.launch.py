#!/usr/bin/env python3
"""
mavros_only.launch.py
=====================
Runs MAVROS + px4_sensor_reader only.
Run this after sitl.launch.py is already running.

The mission is run separately in another terminal:
    ros2 launch nn_planner mission.launch.py          <- mission A->B
    ros2 launch nn_planner mission.launch.py mission:=takeoff_test

Usage:
    ros2 launch nn_planner mavros_only.launch.py
    ros2 launch nn_planner mavros_only.launch.py fcu_url:=/dev/ttyACM0:57600
"""

import os
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_MAVROS_CFG = "/opt/ros/jazzy/share/mavros/launch/px4_config.yaml"
_LOCAL_PLG  = os.path.join(
    os.path.dirname(__file__), "px4_pluginlists.yaml"
)


def _kill_mavros(context):
    """Kill stale MAVROS + px4_sensor_reader and clean up DDS shared memory."""
    import time, glob
    try:
        subprocess.run(["pkill", "-9", "-f", "mavros_node"],   capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "px4_sensor_reader"], capture_output=True)
    except Exception:
        pass
    time.sleep(2.0)
    # Remove stale FastRTPS segments so MAVROS does not crash (SIGABRT)
    for shm in glob.glob("/dev/shm/fastrtps_*"):
        try:
            os.remove(shm)
        except Exception:
            pass
    subprocess.run(["ros2", "daemon", "stop"], capture_output=True)
    time.sleep(1.0)
    subprocess.run(["ros2", "daemon", "start"], capture_output=True)
    time.sleep(1.0)
    return []


def generate_launch_description():

    arg_fcu_url = DeclareLaunchArgument(
        "fcu_url", default_value="udp://:14540@127.0.0.1:14580",
        description="MAVLink FCU URL — SITL: udp://:14540@127.0.0.1:14580 | USB: /dev/ttyACM0:57600",
    )

    fcu_url = LaunchConfiguration("fcu_url")

    # ── 0. Clean up old MAVROS processes ─────────────────────────────────────
    kill_old = OpaqueFunction(function=_kill_mavros)

    # ── 1. MAVROS ─────────────────────────────────────────────────────────────
    # IMPORTANT: Do not set name="mavros" — this causes the container node to get
    # a /mavros/mavros/ prefix so the topic paths become double-nested and MAVROS
    # crashes (RCLError: incompatible type) on restart.
    # Let MAVROS use its own default name.
    mavros_node = Node(
        package="mavros",
        executable="mavros_node",
        namespace="mavros",
        output="screen",
        parameters=[
            {"fcu_url":             fcu_url},
            {"gcs_url":             ""},
            {"target_system_id":    1},
            {"target_component_id": 1},
            {"fcu_protocol":        "v2.0"},
            _MAVROS_CFG,
            _LOCAL_PLG,
        ],
    )

    # ── 2. PX4 Sensor Reader ──────────────────────────────────────────────────
    # Delay 10 seconds so MAVROS finishes connecting to PX4 (6s is not enough on some systems).
    sensor_reader = TimerAction(
        period=10.0,
        actions=[
            Node(
                package="neo_planner",
                executable="px4_sensor_reader",
                name="px4_sensor_reader",
                output="screen",
                parameters=[{"fcu_url": fcu_url, "auto_launch_mavros": False}],
            ),
        ],
    )

    return LaunchDescription([
        arg_fcu_url,
        kill_old,
        mavros_node,
        sensor_reader,
    ])
