#!/usr/bin/env python3
"""
sitl.launch.py
==============
Runs PX4 SITL + Gazebo server + Gazebo GUI only.
Camera, VFH, MAVROS, and the mission are run in separate terminals.

Usage:
    ros2 launch drone sitl.launch.py
    ros2 launch drone sitl.launch.py headless:=true
    ros2 launch drone sitl.launch.py vehicle:=x500_depth world:=obstacles
"""

import os
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration


def _kill_existing():
    import time, signal

    # Stage 1: SIGTERM first to all related processes
    soft_patterns = [
        "gz sim", "gz server", "gzserver", "gzclient", "ruby.*gazebo",
        "px4", "micro_xrce_dds",
        "mavros", "expert_planner_node", "camera_vfh",
        "publish_pcd_node", "octomap_server", "mavros_tf_broadcaster",
    ]
    for pattern in soft_patterns:
        subprocess.run(["pkill", "-f", pattern], capture_output=True, timeout=5)

    time.sleep(2.0)

    # Stage 2: SIGKILL for whatever is still alive
    for pattern in soft_patterns:
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True, timeout=5)

    # Stage 3: Clean up Gazebo/PX4 sockets/ports that may still be locked
    for port_arg in ["11345/tcp", "18570/udp", "14580/udp", "8888/udp"]:
        try:
            subprocess.run(["fuser", "-k", port_arg], capture_output=True, timeout=3)
        except Exception:
            pass

    # Stage 3b: Remove stale FastDDS/FastRTPS shared memory.
    # These segments are not removed automatically when a process is SIGKILLed,
    # so MAVROS crashes (SIGABRT) on restart due to a publisher name conflict.
    import glob
    for shm in glob.glob("/dev/shm/fastrtps_*"):
        try:
            os.remove(shm)
        except Exception:
            pass
    # Restart the ros2 daemon so the participant list is clean (with a timeout)
    try:
        subprocess.run(["ros2", "daemon", "stop"], capture_output=True, timeout=8)
    except Exception:
        pass
    time.sleep(1.0)
    try:
        subprocess.run(["ros2", "daemon", "start"], capture_output=True, timeout=8)
    except Exception:
        pass

    # Stage 3c: Remove the PX4 unix socket so a new instance can start
    import glob as _glob
    for _sock in _glob.glob("/tmp/px4-sock-*") + _glob.glob("/tmp/px4_lock-*"):
        try:
            os.remove(_sock)
        except Exception:
            pass

    # Stage 4: Remove PX4 state files so the default airframe parameters are
    # re-read (including EKF2_HGT_REF=1 newly added to 4001_gz_x500)
    px4_build = os.path.join(os.path.expanduser("~/PX4-Autopilot"), "build/px4_sitl_default")
    for fpath in [
        # parameters stored in rootfs/
        os.path.join(px4_build, "rootfs/parameters.bson"),
        os.path.join(px4_build, "rootfs/parameters_backup.bson"),
        os.path.join(px4_build, "rootfs/dataman"),
        # some PX4 versions store these in gz_bridge/
        os.path.join(px4_build, "src/modules/simulation/gz_bridge/parameters.bson"),
        os.path.join(px4_build, "src/modules/simulation/gz_bridge/parameters_backup.bson"),
        os.path.join(px4_build, "src/modules/simulation/gz_bridge/dataman"),
        os.path.join(px4_build, "src/modules/simulation/gz_bridge/.lock"),
    ]:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                print(f"[sitl.launch] Removed: {os.path.basename(fpath)}")
            except Exception:
                pass

    time.sleep(1.0)
    print("[sitl.launch] OK All old processes cleaned up")


_PX4_DIR = os.path.expanduser("~/PX4-Autopilot")
_PX4_BIN = os.path.join(_PX4_DIR, "build/px4_sitl_default/bin/px4")
_PX4_CWD = os.path.join(
    _PX4_DIR, "build/px4_sitl_default/src/modules/simulation/gz_bridge"
)


def _launch_px4_gz(context, vehicle_arg, headless_arg, world_arg):
    _kill_existing()

    vehicle  = vehicle_arg.perform(context)
    headless = headless_arg.perform(context).lower() == "true"
    world    = world_arg.perform(context)

    sim_model = vehicle if vehicle.startswith("gz_") else "gz_" + vehicle

    _px4_worlds = os.path.join(_PX4_DIR, "Tools/simulation/gz/worlds")
    _existing_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    _gz_resource_path = ":".join(filter(None, [_px4_worlds, _existing_resource_path]))

    _nvidia_egl_lib = "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"
    _use_nvidia_egl = os.path.exists(_nvidia_egl_lib)
    _egl_extra = {
        # Suppress Mesa DRI2 probe warnings — Mesa is still loaded by GLVND but
        # warning-level logs are muted. EGL_LOG_LEVEL=fatal shows only fatal errors.
        "EGL_LOG_LEVEL":                  "fatal",
        "__EGL_VENDOR_LIBRARY_FILENAMES": _nvidia_egl_lib,
        "__GLX_VENDOR_LIBRARY_NAME":      "nvidia",
    } if _use_nvidia_egl else {}

    px4_env = {
        **os.environ,
        "GZ_CONFIG_PATH":       "/usr/share/gz",
        "GZ_SIM_RESOURCE_PATH": _gz_resource_path,
        "PX4_SIM_MODEL":        sim_model,
        "PX4_GZ_WORLD":         world,
        "HEADLESS":             "1",
        **_egl_extra,
    }

    gz_env = {
        **os.environ,
        "GZ_CONFIG_PATH":       "/usr/share/gz",
        "GZ_SIM_RESOURCE_PATH": _gz_resource_path,
        **_egl_extra,
    }

    actions = [
        ExecuteProcess(
            cmd=[_PX4_BIN],
            cwd=_PX4_CWD,
            name="px4_sitl",
            output="screen",
            env=px4_env,
        ),
    ]

    if not headless:
        # Delay 12 seconds so the Gazebo server finishes loading the world
        actions.append(
            TimerAction(
                period=12.0,
                actions=[
                    ExecuteProcess(
                        cmd=["gz", "sim", "-g"],
                        name="gazebo_gui",
                        output="screen",
                        env=gz_env,
                    ),
                ],
            )
        )

    return actions


def generate_launch_description():

    arg_vehicle = DeclareLaunchArgument(
        "vehicle", default_value="x500_depth",
        description="Drone model: x500, x500_depth, x500_windy, etc.",
    )
    arg_headless = DeclareLaunchArgument(
        "headless", default_value="false",
        description="true = run without the Gazebo GUI window",
    )
    arg_world = DeclareLaunchArgument(
        "world", default_value="default",
        description=(
            "Gazebo world name without the .sdf extension. "
            "Available: default, forest, forest1, bricks, bricks2, lawn, obstacles, "
            "poles, poles_uniform, baylands, windy, rand_world_10/12/14/15/16/18/20/24"
        ),
    )

    return LaunchDescription([
        arg_vehicle,
        arg_headless,
        arg_world,
        OpaqueFunction(
            function=_launch_px4_gz,
            args=[
                LaunchConfiguration("vehicle"),
                LaunchConfiguration("headless"),
                LaunchConfiguration("world"),
            ],
        ),
    ])
