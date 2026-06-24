# CLAUDE.md — Working guide for `neo_planner`

This document is for Claude (and developers) when working in this ROS 2 package.
It covers: program overview, how to build/run, key parameters, and a log of the
changes that have been made.

---

## 1. Program overview

`neo_planner` is a ROS 2 package (Jazzy, `ament_python`) for deploying
**NEO-Planner** on a PX4 drone. There are two pipelines:

1. **Expert data collection** — an expert planner (MINCO minimum-jerk + ESDF)
   flies the drone back and forth between `(0,0) ↔ (goal_x, 0)` across a pole
   field, recording `(depth image, state, target) → trajectory` pairs as a
   dataset to train a neural network.
2. **NN inference** — a neural network (`PlannerNet`, PyTorch) replaces the expert
   planner for navigation in unknown environments.

The stack runs on top of **PX4 SITL + Gazebo (gz sim) + MAVROS**.

### Main nodes (entry points in `setup.py`)
| Executable | File | Purpose |
|---|---|---|
| `expert_planner_node` | `expert_planner_node.py` | Expert planner + mission state machine + dataset recorder |
| `neo_planner_node` | `neo_planner_node.py` | NN inference (deployment) |
| `nn_inference_node` | `nn_inference_node.py` | Standalone NN inference |
| `publish_pcd_node` | `publish_pcd_node.py` | Publish `.pcd` ground truth → `/pointcloud/map` |
| `px4_sensor_reader` | `px4_sensor_reader.py` | Read PX4 state (pos/vel/yaw) via uXRCE/MAVROS |
| `vfh_obstacle_avoidance_node` | `vfh_realsense.py` | Classic VFH from the depth camera |
| `gz_depth_bridge_node` / `gz_depth_viewer_node` | — | Bridge & visualize Gazebo depth → ROS 2 |
| `mavros_tf_broadcaster_node` | — | TF `odom→base_link` from MAVROS |

### Supporting modules (not nodes)
- `min_jerk_planner.py` — `MinJerkPlanner` (MINCO) + `PlannerConfig`. Core trajectory optimization.
- `esdf_ros2.py` — `ESDF` from `/projected_map` (octomap) + drone-body radius inflation.

---

## 2. Build & run

**The workspace uses `colcon build --symlink-install`.** Implications:

- ✅ **Editing the contents of existing `.py` / `.launch.py` / `.yaml` files takes
  effect immediately, NO rebuild needed.** Just restart the affected node/launch.
- ❌ **A `colcon build` is only required** when: adding a new file, adding/changing
  an entry point in `setup.py`, modifying `package.xml`, or adding msg/srv.

```bash
cd ~/neo_planner_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash      # usually enough; rebuild only when needed (see above)
```

### 4-terminal workflow (see `command.txt`)

**Dataset collection (expert):**
```bash
ros2 launch neo_planner sitl.launch.py world:=poles          # +headless:=true for no GUI
ros2 launch neo_planner mavros_only.launch.py
ros2 launch neo_planner camera_vfh.launch.py
ros2 launch neo_planner expert_planning.launch.py \
    pcd_path:=/home/michael/poles.pcd target_samples:=500
```

**Inference:**
```bash
ros2 launch neo_planner sitl.launch.py world:=forest1
ros2 launch neo_planner mavros_only.launch.py
ros2 launch neo_planner camera_vfh.launch.py
ros2 launch neo_planner nn_planning_unknown.launch.py \
    model_path:=.../planner_net.pth goal_x:=20.0
```

Order matters: wait for PX4 "Ready for takeoff", then MAVROS "Got HEARTBEAT",
then the camera, then the planner. `sitl.launch.py` has a `_kill_existing` that
automatically cleans up old processes + FastDDS shm + PX4 state before starting.

> After a `PHYSICS EXPLOSION` exceeds `EXPLOSION_MAX_RETRY`, **restart SITL**
> before the next run.

---

## 3. Key parameters

### `expert_planning.launch.py` (launch arguments)
| Param | Default | Notes |
|---|---|---|
| `pcd_path` | (required) | Ground-truth obstacle `.pcd` file |
| `target_samples` | 5000 | Target number of samples (the command uses 500) |
| `goal_x` | **20.0** | Goal X coordinate (m) — *changed from 15.0* |
| `target_alt` | **2.0** | Cruise altitude (m) — *changed from 1.5* |
| `replan_period` | 1.0 | Replan period (s) |
| `cmd_hz` | 100 | Setpoint publish rate to MAVROS |
| `v_max` | 1.0 | Planner max speed (m/s) |
| `safe_dis` | 0.5 | SOFT body→obstacle margin (m) |
| `use_ekf_stable` | true | Wait for EKF Z to converge before takeoff |
| `arena_x_min/max` | -2.0 / 23.0 | X navigation bounds |
| `arena_y_min/max` | -6.0 / 6.0 | Y navigation bounds |
| `occ_min_z/max_z` | 1.0 / 3.0 | Octomap 2D projection band (poles) |
| `data_save_dir` | auto | `~/training_data/run_<timestamp>/` |

### `PlannerConfig` (`min_jerk_planner.py`)
| Param | Value | Notes |
|---|---|---|
| `v_max` | 1.0 | Soft penalty (not a hard constraint) |
| `safe_dis` | 0.5 | SOFT margin; dynamically overridden by the node |
| `hard_dis` | **0.30** | *NEW* — HARD drone-body barrier |
| `w_hard_rel` | **20.0** | *NEW* — relative weight of the hard barrier |
| `weights` | [1, 1, 1, 10000] | [energy, time, feasibility, collision] |
| `delta_t` | 0.1 | Cost sampling interval |
| `collision_cost_tol` | 10.0 | Plan-rejection threshold |
| `T_min / T_max` | 0.5 / 5.0 | Duration per piece |
| `init_wpts_num` | 2 | → 3 pieces, 3 homotopy classes (straight/left/right) |

### Safety constants (`expert_planner_node.py`)
| Constant | Value | Notes |
|---|---|---|
| `SAFE_DIS_TRAP_MIN` | **0.30** | *changed from 0.20* — floor of dynamic margin = `hard_dis` |
| `SAFE_DIS_TRAP_STEP` | 0.05 | Margin shrink per failed replan |
| `K_SPEED_MARGIN` | **0.25** | *NEW* — speed-aware margin: `safe_dis_eff = safe_dis_dyn + 0.25·\|v\|` |
| `TRAJ_MIN_ESDF` | 0.3 | Min distance for each trajectory point |
| `ESCAPE_FAIL_COUNT` | 4 | Consecutive replan failures before escape |
| `PLAN_TIME_BUDGET` | 3.0 | Replan time budget (s) |
| `EXPLOSION_MAX_RETRY` | 3 | Max recoveries before the mission stops |
| ESDF `inflation_radius` | 0.25 | x500 body radius (octomap res 0.1 → ~0.3 m) |

### PX4 parameters set by the node at startup (`_configure_px4_height_ref`)
| PX4 param | Value | Reason |
|---|---|---|
| `EKF2_HGT_REF` | 1 (GPS) | Prevent SITL baro drift → stable altitude |
| `MPC_XY_VEL_MAX` | **1.5** | *NEW* — physical speed cap (v_max·1.5) |
| `MPC_XY_CRUISE` | **1.0** | *NEW* — cruise = planner v_max |
| `MPC_XY_VEL_ALL` | **1.5** | *NEW* — master cap |

> `v_max=1.0` only lives in the planner. Without the MPC caps above, PX4 is allowed
> to fly up to ~12 m/s while chasing a position error → overshoot in tight gaps +
> Gazebo collision impulses trigger a `PHYSICS EXPLOSION`.

### VFH (`camera_vfh.launch.py`)
`hfov_deg=88`, `d_min=300mm`, `d_max=4000mm`, `threshold=0.45`,
`robot_radius=346mm`, `safety_dist=500mm`.

---

## 4. Collision model (important for tuning)

The planner's collision cost is a **two-tier potential** (`add_sampled_cost`/`_grad_CT`):

```
edt ≥ safe_dis            → cost 0           (open space: free/smooth motion)
hard_dis ≤ edt < safe_dis → (safe_dis-edt)³  (soft margin: can be entered → agile)
edt < hard_dis            → w_hard·(hard_dis-edt)²  (hard barrier: nearly impenetrable)
```

Effective profile (per sample, weight 10000, tol 10): `edt ≥ 0.35` accepted,
`edt ≤ 0.30` rejected. → the drone is not rigid in open space, yet does not collide.

---

## 5. Changes that have been made (debugging session)

Goal: make the expert pipeline reach 500 samples **without getting stuck and
without collisions**.

1. **`_set_local_target` anti-deadlock near the goal** (`expert_planner_node.py`).
   Previously, when `dist < longitu_step` the target was *hard-snapped* to the goal
   and then `return`ed, skipping lateral avoidance. If the goal collided with an
   obstacle (e.g. goal (15,0) only 1 m from the pole at (14,0)) → `batch_plan` kept
   failing → stuck. Now the goal still goes through the lateral-avoidance loop →
   nudged to the nearest free point.

2. **Two-tier collision cost** (`min_jerk_planner.py`). Added `hard_dis`,
   `w_hard_rel`, a hard-barrier term in the cost + gradient. Prevents the
   trajectory from hugging obstacles (the soft cubic is too weak near the boundary).

3. **`SAFE_DIS_TRAP_MIN` 0.20 → 0.30** — the dynamic-margin floor must not vanish.

4. **PX4 speed cap** — `MPC_XY_VEL_MAX/CRUISE/VEL_ALL` set at startup.

5. **Speed-aware margin** (`K_SPEED_MARGIN`) — the reference path moves away from
   poles in proportion to speed, absorbing tracking error on turns.

6. **`_safe_hold_xy` (obstacle-aware alt-recovery/alt-fix)** — before "holding XY",
   move away from poles along the ESDF gradient. Prevents `esdf=0.00` during
   alt-recovery.

7. **Launch defaults** — `goal_x` 15→20, `target_alt` 1.5→2.0.

### Measured results (poles, 500 samples)
| Metric | Before | After |
|---|---|---|
| Samples | stuck at 14 / stopped at 463 | **500/500** |
| PHYSICS EXPLOSION | 4 | **0** |
| Collision / velocity skips | 31 / 3 | **0 / 0** |
| Max physical speed | 8–18 m/s | **1.57 m/s** |
| Min actual `esdf` | 0.00 m | **0.50 m** |
| Altitude | — | mean 1.96 m, std **5 cm**, no drift |

---

## 6. Gotchas

- **A goal at y≈0 is prone to poles.** Poles in the `poles` world near the y=0
  axis: x≈3.58, 9.66, 14.0, 20.82 (radius 0.5 m). A `goal_x` that lands ~1 m from
  one of them creates a tight turnaround. The `_set_local_target` fix handles it,
  but pick a `goal_x` with clearance if you want a wide margin.
- **Planner `v_max` ≠ physical limit.** Always check `MPC_XY_*` if speed goes wild.
- **Restart the node, not a rebuild,** for `.py` changes (symlink-install).
- The position reported at a `PHYSICS EXPLOSION` is **post-impulse**, not the
  contact point — find the contact point from the low `esdf` a few lines earlier.
