# NEO-Planner (ROS 2 / PX4)

**NEO-Planner** deployment package for obstacle-avoiding drone navigation,
running on **ROS 2 Jazzy** with **PX4 SITL + Gazebo (gz sim) + MAVROS**.

NEO-Planner is a *learning-based local planner*: a neural network learns to map
**depth image + drone state + target** directly into a safe **minimum-jerk
trajectory**, imitating an **expert planner** (MINCO optimization + ESDF). This
package provides both sides:

1. **Dataset collection (expert)** — flies the drone with the expert planner and
   records input→trajectory pairs.
2. **Inference (NN)** — deploys the trained network to fly in unknown
   environments.

---

## Features

- Expert planner based on **MINCO** (minimum-control-effort trajectory) with a
  collision cost from an **ESDF** (Euclidean Signed Distance Field).
- **Two-tier collision potential**: a hard barrier (anti-collision) + a soft
  margin (stays agile, can slip through gaps).
- **Speed-aware margin** & **physical PX4 speed cap** to absorb tracking error
  and prevent *physics explosions* in the simulator.
- Depth camera + **classic VFH** pipeline as a comparison/helper.
- Complete mission state machine: EKF-stable wait, takeoff, replanning, escape
  mode, altitude recovery, arena bounds, post-collision recovery.
- GPS height reference (`EKF2_HGT_REF`) → stable altitude with no baro drift.

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │  PX4 SITL + Gazebo (gz sim)   [sitl.launch]   │
                 │   - x500_depth drone dynamics                 │
                 │   - depth camera  → /depth_camera             │
                 └───────────────┬──────────────────────────────┘
                                 │ uXRCE-DDS / MAVLink
                 ┌───────────────▼──────────────┐
                 │  MAVROS        [mavros_only]  │  state, setpoint_raw
                 └───────────────┬──────────────┘
        depth ───────────────────┼───────────────── /pointcloud/map (.pcd)
   [camera_vfh]                  │                    [publish_pcd]
   gz_depth_bridge               │                         │
   vfh_obstacle_avoidance        │                    octomap_server
                                 │                         │ /projected_map
                 ┌───────────────▼─────────────────────────▼────┐
                 │  expert_planner_node  /  neo_planner_node     │
                 │   ESDF → MINCO optimizer → trajectory         │
                 │   → /mavros/setpoint_raw/local                │
                 └───────────────────────────────────────────────┘
```

---

## Prerequisites

- Ubuntu + **ROS 2 Jazzy**
- **PX4-Autopilot** (SITL, target `px4_sitl_default`) at `~/PX4-Autopilot`
- **Gazebo (gz sim)** + `ros_gz_bridge`
- **MAVROS** (`ros-jazzy-mavros`) + geomagnetic dataset
- `octomap_server`, `tf2_ros`, `cv_bridge`, `numpy`, `opencv`, `scipy`
- (inference) **PyTorch**
- A ground-truth obstacle `.pcd` file for the world in use (e.g. `poles.pcd`)

---

## Build

```bash
cd ~/neo_planner_ws
colcon build --symlink-install
source install/setup.bash
```

> `--symlink-install` makes edits to existing `.py`/`.launch.py` files take effect
> immediately without a rebuild (just restart the node). A rebuild is only needed
> when adding new files or changing `setup.py`/`package.xml`.

---

## Usage

Run each command in a separate terminal, in order (wait for the previous one to
be ready).

### A. Dataset collection (expert)
```bash
ros2 launch neo_planner sitl.launch.py world:=poles
ros2 launch neo_planner mavros_only.launch.py
ros2 launch neo_planner camera_vfh.launch.py
ros2 launch neo_planner expert_planning.launch.py \
    pcd_path:=$HOME/poles.pcd target_samples:=500
```
The dataset is saved to `~/training_data/run_<timestamp>/`. Add `headless:=true`
to `sitl.launch.py` to run without the GUI.

### B. Inference (NN)
```bash
ros2 launch neo_planner sitl.launch.py world:=forest1
ros2 launch neo_planner mavros_only.launch.py
ros2 launch neo_planner camera_vfh.launch.py
ros2 launch neo_planner nn_planning_unknown.launch.py \
    model_path:=$HOME/training_nn_planner/saved_net_01/planner_net.pth \
    goal_x:=20.0
```

Available worlds: `default, forest, forest1, bricks, bricks2, lawn, obstacles,
poles, poles_uniform, baylands, windy, rand_world_*`.

---

## Key parameters

See [CLAUDE.md](CLAUDE.md) for the full list. In brief:

| Param | Default | Meaning |
|---|---|---|
| `goal_x` | 20.0 m | Target X (drone shuttles (0,0)↔(goal_x,0)) |
| `target_alt` | 2.0 m | Cruise altitude |
| `v_max` | 1.0 m/s | Max speed (planner) |
| `safe_dis` | 0.5 m | Soft margin to obstacles |
| `target_samples` | 5000 | Target number of samples |

---

## Notes

- See [CLAUDE.md](CLAUDE.md) for architecture details, the two-tier collision
  model, the full parameter list, and the fix history (anti-stuck &
  anti-collision).
- Latest test result (`poles` world, 500 samples): completed fully **500/500**,
  **0 physics explosions**, **0 collision skips**, stable altitude (std ~5 cm).

---

## References

The approach and implementation in this package are based on:

1. **NEO-Planner** — a *learning-based local planner* imitated from a MINCO+ESDF
   expert (IROS 2025). Source of the ported modules `min_jerk_planner.py`,
   `esdf_ros2.py`, and `expert_planner_node.py`.

2. Z. Wang, X. Zhou, C. Xu, and F. Gao,
   **"Geometrically Constrained Trajectory Optimization for Multicopters,"**
   *IEEE Transactions on Robotics (T-RO)*, vol. 38, no. 5, pp. 3259–3278, 2022.
   — the **MINCO** (Minimum-Control trajectory) framework used by the expert
   planner. DOI: 10.1109/TRO.2022.3160022 · arXiv:2103.00190

3. J. Borenstein and Y. Koren,
   **"The Vector Field Histogram — Fast Obstacle Avoidance for Mobile Robots,"**
   *IEEE Transactions on Robotics and Automation*, vol. 7, no. 3, 1991.
   — basis of the **VFH** module (`vfh_realsense.py`).

> Note: complete the NEO-Planner citation (authors/venue) according to the paper
> you are referencing; the IROS 2025 placeholder above follows the comments in
> the source code.
