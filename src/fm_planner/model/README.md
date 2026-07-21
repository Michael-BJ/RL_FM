# `model/` — trained model files for `fm_planner`

Central place for the model artifacts the package uses, split by model type.

```
model/
├── rl/    RL seed-generator policies (TorchScript .pt) — SHIPPED with the package
└── fm/    FM (flow-matching) checkpoints (.pth / .onnx) — drop your chosen run here
```

## `rl/` — RL policies (used by data collection)

TorchScript policies exported from the SAC gym (`rl_seed_adapter.py --export`).
They are consumed by `expert_planner_node` when collecting with `use_rl_seeds:=true`.

| File | What it is |
|---|---|
| `policy_export_v2.pt` | L4 bidirectional policy (2026-07-21) — **the default** |
| `policy_export_ft_side.pt` | older single-direction policy (`sac_v1_ft_side`) |

`expert_planner_node` resolves the default via `default_rl_policy_path()`:
it looks first in the installed `share/fm_planner/model/rl/`, then falls back
to this source directory. Override with `rl_policy_path:=<abs path>`.

```bash
# collect using the default (v2) policy
ros2 launch fm_planner expert_planning.launch.py \
    pcd_path:=.../world.pcd goal_x:=20.0 use_rl_seeds:=true

# collect using the older policy
ros2 launch fm_planner expert_planning.launch.py \
    pcd_path:=.../world.pcd goal_x:=20.0 use_rl_seeds:=true \
    rl_policy_path:=$PWD/src/fm_planner/model/rl/policy_export_ft_side.pt
```

Re-export after retraining:
```bash
python3 fm_planner/rl_seed_adapter.py \
    --export ~/rl_gym_v1.0/rl_gym/runs/<run>/final.zip \
    --out model/rl/policy_export_v2.pt
```

## `fm/` — FM checkpoints (used by inference)

Destination for the FM checkpoint you want to fly. FM checkpoints are large and
run-tagged, so they are NOT auto-shipped; copy the run's `.pth` (or `.onnx`
+ `.onnx.data`) here and point the launch at it explicitly:

```bash
cp ~/saved_net/fm/run_<ts>/fm_planner_<ts>.pth model/fm/
ros2 launch fm_planner fm_planning_unknown.launch.py \
    model_path:=$PWD/src/fm_planner/model/fm/fm_planner_<ts>.pth goal_x:=20.0
```
