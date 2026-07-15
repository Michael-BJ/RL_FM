# CLAUDE.md — Working guide for `fm_planner`

This document is for Claude (and developers) when working in this ROS 2 package.
It covers: program overview, how to build/run, key parameters, and a log of the
changes that have been made.

---

## 1. Program overview

`fm_planner` is a ROS 2 package (Jazzy, `ament_python`) for deploying
**NEO-Planner** and **FM-Planner** on a PX4 drone. There are two pipelines:

1. **Expert data collection** — an expert planner (MINCO minimum-jerk + ESDF)
   flies the drone back and forth between `(0,0) ↔ (goal_x, 0)` across a pole
   field, recording `(depth image, state, target) → trajectory` pairs as a
   dataset to train a neural network.
2. **NN/FM inference** — a neural network (`PlannerNet`, PyTorch) or the FM
   (Flow Matching) model replaces the expert planner for navigation in unknown
   environments.

The stack runs on top of **PX4 SITL + Gazebo (gz sim) + MAVROS**.

### Main nodes (entry points in `setup.py`)
| Executable | File | Purpose |
|---|---|---|
| `expert_planner_node` | `expert_planner_node.py` | Expert planner + mission state machine + dataset recorder |
| `legacy_inference_node` | `legacy_inference_node.py` | NN inference (deployment) — legacy mirror of the expert pipeline, superseded by `nn_inference_node`/`fm_inference_node` |
| `nn_inference_node` | `nn_inference_node.py` | Standalone NN inference |
| `fm_inference_node` | `fm_inference_node.py` | FM (flow-matching) inference — samples K candidates per replan, ranks by ESDF clearance, warm-starts MINCO with the best; inherits all guards from `nn_inference_node` |
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

> ⚠️ **VERIFIED 2026-07-05: the claim below is currently FALSE.** The files in
> `install/fm_planner/lib/python3.12/site-packages/fm_planner/` are **copies,
> not symlinks** (the last build was apparently done without
> `--symlink-install`). Editing a source `.py` does NOT reach the running node
> until you either rebuild (`colcon build --packages-select fm_planner`) or
> copy the file into the install tree manually. This bit us once: the MPC-cap
> fix was silently absent from a test run. Check with
> `ls -la install/.../site-packages/fm_planner/` before trusting an edit.

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
ros2 launch fm_planner sitl.launch.py world:=poles          # +headless:=true for no GUI
ros2 launch fm_planner mavros_only.launch.py
ros2 launch fm_planner camera_vfh.launch.py
ros2 launch fm_planner expert_planning.launch.py \
    pcd_path:=/home/michael/poles.pcd target_samples:=500
```

**Inference:**
```bash
ros2 launch fm_planner sitl.launch.py world:=forest1
ros2 launch fm_planner mavros_only.launch.py
ros2 launch fm_planner camera_vfh.launch.py
ros2 launch fm_planner nn_planning_unknown.launch.py \
    model_path:=.../planner_net.pth goal_x:=20.0
```

**Inference (FM planner)** — same first 3 terminals, then:
```bash
ros2 launch fm_planner fm_planning_unknown.launch.py \
    model_path:=/home/michael/saved_net/fm/run_XXXX/fm_planner_XXXX.pth \
    goal_x:=20.0 K:=8 n_steps:=2
```
`K` = candidates per replan, `n_steps` = Euler steps (`.pth` backend only; the
`.onnx` graph has n_steps baked in at export). The FM checkpoint format is
`{state_dict, norm_mean, norm_std}` (see `fm_model.py`, copied from
`~/Training_model_FM_planner/` — keep the two copies in sync).

Order matters: wait for PX4 "Ready for takeoff", then MAVROS "Got HEARTBEAT",
then the camera, then the planner. `sitl.launch.py` has a `_kill_existing` that
automatically cleans up old processes + FastDDS shm + PX4 state before starting.

> After a `PHYSICS EXPLOSION` exceeds `EXPLOSION_MAX_RETRY`, **restart SITL**
> before the next run.

> **EKF/state-estimator divergence (2026-07-07, gauntlet_train run 8,
> sample 3497):** distinct from `PHYSICS EXPLOSION` — no collision, position
> drifts *monotonically* away from origin across several accidental-landing
> recoveries (`ground_z` reference itself shifts each retry: -0.09 → -0.17 →
> -0.68 m; eventual position exceeded even the `.pcd` ground-truth bounds,
> y=16 vs world y=±15). "Landing recovery succeeded" can fire against an
> already-corrupted reference (e.g. `z=-0.29m` logged as success) — the
> in-node recovery loop cannot fix a bad MAVROS/EKF estimate, it only trusts
> it. Symptom to watch: `left the arena` recurring at *growing* distance
> rather than a single graze-and-return. **Fix: kill the whole SITL stack
> and relaunch fresh** (same as the `PHYSICS EXPLOSION` gotcha) — do not
> wait for the retry budget (`recovery #N/5`) to resolve it, since the
> trend was worsening, not converging. Samples recorded before the episode
> stay valid (`_n_samples` freezes during the glitch, confirmed no bad CSV
> rows); the restarted run starts a new `run_<timestamp>` folder — merge
> CSVs/`depth_img` manually later if you want one combined dataset.

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
| `arena_x_min/max` | -2.0 / **25.0** | X navigation bounds — *raised from 23.0 to allow `goal_x` up to ~23* |
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

---

## 7. Session changes 2026-07-05 — NN/FM inference (multimodal experiment)

Context: multimodal dataset `~/training_data/run_20260705_172250` (5000 rows,
ratio 1.51), new models `~/saved_net/nn/run_20260705_182409` and
`~/saved_net/fm/run_20260705_190248`. Baseline & test results saved in
`~/monitor_logs/`. New monitoring tool: `fm_planner/flight_monitor.py`
(passive node, run directly with `python3 -u`, no build needed — records
which side each wide box is passed on (left/right), ground-truth clearance,
escapes, PNG plot).

### 7a. `nn_inference_node.py` — PX4 MPC speed cap (fix physics explosion)

- Added `_set_px4_param_float()`; in `run_sequence()` (after
  `EKF2_HGT_REF`) the node now sets `MPC_XY_VEL_MAX=v_max*1.5`,
  `MPC_XY_CRUISE=v_max`, `MPC_XY_VEL_ALL=v_max*1.5` — mirroring the expert
  (`_configure_px4_height_ref`). Previously ONLY the expert installed this
  cap; the inference node did not → PX4 was free to chase the setpoint at
  ~12 m/s during an escape/tracking-gap near an obstacle → Gazebo contact
  impulse → PHYSICS EXPLOSION (exactly what took down FM twice in the
  2026-07-05 baseline test).
- Automatically inherited by `fm_inference_node`. Check the startup log:
  `ok PX4 param MPC_XY_VEL_MAX = 1.5`.

### 7b. `fm_inference_node.py` — mode persistence (anti left/right oscillation)

- Problem: every replan, FM resamples K candidates from fresh noise WITH NO
  memory; on an ambiguous scene the left/right scores are nearly tied →
  the decision can flip between replans (observed: the drone crossed the
  box's plane 5 times within a single lap).
- Fix: `_current_side()` measures the lateral offset of the ACTIVE
  trajectory (0.8-1.6 s ahead) relative to the head→tail axis;
  `_rank_by_clearance()` gives a `MODE_PERSISTENCE_BONUS=0.3` m bonus to
  candidates on the same side once the drone has committed
  (`|lat| > MODE_COMMIT_MIN_LAT=0.2`). Effect: the mode only switches if the
  other side has >0.3 m more clearance (hysteresis), not because of sampling
  noise.
- Result: double-crossing disappeared in the wide_test single-pass test
  (3/3 boxes committed once, one consistent side).

### 7c. `fm_inference_node.py` — bimodal early-steering (pre-octomap anticipation)

- Problem (found in the single-pass test): the final trajectory is driven by
  an ESDF/octomap-based collision cost that lags BEHIND the camera;
  `_set_local_target` also keeps the target straight ahead until the map
  flags a collision. As a result, even though the FM candidate spread already
  "knows" a wide box is there as soon as the camera sees it (~4 m,
  `max_range`), the drone still goes head-on until ~1 m and gets rescued by
  the guard/escape (4 escapes at the first box).
- Fix: `_update_steer_bias()` — if the candidates split left+right
  (`FM_BIMODAL_SIDE_LAT`) with a spread ≥ `FM_BIMODAL_MIN_SPREAD`,
  set `self._steer_bias` toward the BEST candidate's side (capped at
  `FM_TARGET_BIAS_MAX=1.5` m); decays by `FM_BIAS_DECAY=0.5`/replan once the
  split disappears. `_set_local_target()` is overridden to shift the lateral
  target by the bias (checked collision-free; falls back to half the bias).
  Log: `[FM] Bimodal spread=..m -> steer LEFT/RIGHT` + a `[FM] lat spread=..`
  diagnostic every 2 s.
- This is the channel that carries FM's anticipatory information to the
  planner (the core research idea: "the system knows BEFORE the view is
  fully blocked") — the NN cannot imitate this because its output is a
  single point, with no spread.
- **Empirical tuning result (3 wide_test single-pass runs, 2026-07-05):** the
  lateral spread of model `run_20260705_190248`'s candidates turned out to be
  only **0.2-0.7 m and mostly ONE side** (consistent with the dataset: only
  5% different-side scenes) — the initial 1.0/0.3 thresholds never fired.
  Final values: `FM_BIMODAL_SIDE_LAT=0.15`, `FM_BIMODAL_MIN_SPREAD=0.5`. With
  these the trigger fires in ambiguous zones and the side decision is
  consistent with the final path (RIGHT→passes right, LEFT→passes left), BUT
  the resulting bias is small (±0.3 m) → not yet enough to prevent escapes
  (6-7/run). **Bottleneck = the model's sample spread, not the pipeline.**
  Suggested next step: mirror-y augmentation in `fm_trainer.py` (every scene
  contributes both sides → widens the sample spread on ambiguous scenes)
  then retrain FM; alternative: increase the bias gain (use the full lateral
  offset of the best candidate, not 0.5*spread).

### 7d. Fairness & ablation (prepared 2026-07-05, retrain POSTPONED at the
### user's request)

- **Ablation toggles in `fm_inference_node`** (parameter + launch arg,
  default `false`): `use_mode_persistence`, `use_bimodal_steering`.
  `false,false` = **FM-minimal** — the inference pipeline is identical to
  `nn_inference` except for the candidate generator (sample K + rank by
  clearance) → the official "model swap only" comparison configuration.
  `true` = the FM+ variant, to show system value that is only possible with
  a generative model.
- **`--mirror_aug` in BOTH `fm_trainer.py` and `nn_trainer.py`**
  (`~/Training_model_FM_planner`): mirror-y augmentation of 50% of the
  train set (horizontal depth flip; negate the lateral y component of
  motion/label; rotmat R'=SRS; the FM normalizer is fit on train+mirror).
  Applied to BOTH trainers so the training data stays identical (fairness).
  **Not yet run** — the command is ready:
  `python3 {fm,nn}_trainer.py --data_dir ~/training_data/run_20260705_172250
   --out_dir ~/saved_net/{fm,nn} --epochs 100 --batch_size 32 --lr 3e-4
   --train_split 0.8 --num_workers 8 --mirror_aug`

### 7e. Updated `command.txt` + test notes

- `command.txt`: FM inference block (launch already exists), baseline test +
  5-terminal monitoring block, latest model path.
- Unknown-environment evaluation protocol: **single-pass**
  (`auto_reverse:=false`, `flight_monitor --laps 1`) on the held-out
  `wide_test` world; the primary metric = commit distance at the FIRST
  encounter (box x=5.5), not travel time.
