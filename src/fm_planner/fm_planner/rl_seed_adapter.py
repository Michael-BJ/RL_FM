#!/usr/bin/env python3
# rl_seed_adapter.py — v1.0
"""
rl_seed_adapter.py
=============================================================================
BRIDGE gym -> the real FM pipeline: a frozen RL policy as a SEED GENERATOR
inside expert_planner_node (Gazebo/hardware), REPLACING (or complementing)
homotopy_frontend as the source of multi-homotopy seeds.

WHY IT MATTERS (discussion 2026-07-15): the gym dataset uses an ESDF crop,
whereas the FM is conditioned on the DEPTH IMAGE. To keep the FM dataset
depth-based:
  - expert_planner_node runs normally in Gazebo (the depth image is recorded
    as O as usual),
  - the seed for MINCO comes from the RL policy (dual-h -> 2 structurally
    symmetric seeds per replanning event),
  - MINCO optimizes each seed; EVERY converged one is recorded (fix argmin).
  Result: the dataset (depth, Q*, t̄*) is bimodal by construction, without the
  wall-off skew bug, directly compatible with fm_model.py.

This adapter has NO gym/SB3 dependency at inference time — only torch is needed
(the policy is exported via export_policy() below, MLP+CNN forward done
manually) so it stays light on a Jetson. It consumes esdf_ros2.ESDF live
(occupancy_2d + esdf_map API identical to StaticESDF, so _build_crop is
identical).

Integration in expert_planner_node.py (sketch):
    from rl_seed_adapter import RLSeedAdapter
    self.rl_seeds = RLSeedAdapter("policy_export.pt")
    ...
    seeds = self.rl_seeds.get_seeds(esdf, pos, vel, local_target)
    for (int_wpts, ts) in seeds:            # h=-1 and h=+1
        planner.warm_start_plan(esdf, head, tail, int_wpts, ts)
        ... record EVERY converged one (not argmin) ...

Export the policy after training:
    python3 rl_seed_adapter.py --export runs/sac_v1/final.zip \
        --out policy_export.pt

CHANGELOG
  v1.0  (2026-07-15)  Initial release.
"""

import math

import numpy as np

CROP = 64
CROP_RES = 0.1
ESDF_CLIP_M = 3.0
UNKNOWN_VAL = -1.0


def _heading(vel, pos, goal):
    if np.linalg.norm(vel) > 0.1:
        return math.atan2(vel[1], vel[0])
    d = goal - pos
    return math.atan2(d[1], d[0])


def build_crop_from_esdf(esdf, pos, vel, goal):
    """Egocentric crop from the LIVE esdf (esdf_ros2.ESDF OR StaticESDF).
    In the real stack, unknown cells come from octomap occupancy -1; here the
    fallback treats every in-bounds cell as known (octomap projected_map does
    treat -1 as free for the EDT — consistent with esdf_ros2)."""
    res = esdf.map_resolution
    yaw = _heading(vel, pos, goal)
    cy, sy = math.cos(yaw), math.sin(yaw)
    lin = (np.arange(CROP) - CROP / 2 + 0.5) * CROP_RES
    XB, YB = np.meshgrid(lin, lin, indexing="xy")
    XW = pos[0] + cy * XB - sy * YB
    YW = pos[1] + sy * XB + cy * YB
    rows = ((YW - esdf.map_origin.y) / res).astype(int)
    cols = ((XW - esdf.map_origin.x) / res).astype(int)
    inb = ((rows >= 0) & (rows < esdf.map_height) &
           (cols >= 0) & (cols < esdf.map_width))
    crop = np.full((CROP, CROP), UNKNOWN_VAL, dtype=np.float32)
    crop[inb] = np.clip(esdf.esdf_map[rows[inb], cols[inb]] / ESDF_CLIP_M,
                        0, 1).astype(np.float32)
    occl = float((crop == UNKNOWN_VAL).mean())
    return crop[None], occl, yaw


class RLSeedAdapter:
    """Load an exported policy (TorchScript) and produce dual-h seeds."""

    def __init__(self, export_path: str, v_max: float = 1.0,
                 T_min: float = 0.5, T_max: float = 5.0,
                 q_scale: float = 3.0, tau_scale: float = 3.0,
                 M: int = 3):
        import torch
        self.torch = torch
        self.net = torch.jit.load(export_path, map_location="cpu").eval()
        self.v_max, self.T_min, self.T_max = v_max, T_min, T_max
        self.q_scale, self.tau_scale, self.M = q_scale, tau_scale, M

    def _obs_vec(self, pos, vel, goal, occl, h, yaw):
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rwb = np.array([[cy, sy], [-sy, cy]])
        d = goal - pos
        gd = np.linalg.norm(d)
        gb = Rwb @ (d / max(gd, 1e-6))
        vb = Rwb @ vel
        return np.array([gb[0], gb[1], min(gd / 20.0, 1.0),
                         vb[0], vb[1], occl, h, 0.0], dtype=np.float32)
        # progress_frac is not available in the node -> 0.0 (a neutral column;
        # if desired, compute it from the episode-start distance in the node)

    def get_seeds(self, esdf, pos, vel, local_target, goal=None):
        """-> [(int_wpts(D,M-1), ts(M)), ...] for h=-1 and h=+1."""
        torch = self.torch
        goal = local_target if goal is None else np.asarray(goal, float)
        pos = np.asarray(pos, float)
        vel = np.asarray(vel, float)
        tgt = np.asarray(local_target, float)
        crop, occl, yaw = build_crop_from_esdf(esdf, pos, vel, goal)
        seeds = []
        n = self.M - 1
        d = tgt - pos
        step = d / (n + 1)
        base = np.stack([pos + step * (i + 1) for i in range(n)])
        for h in (-1.0, +1.0):
            vec = self._obs_vec(pos, vel, goal, occl, h, yaw)
            with torch.no_grad():
                a = self.net(torch.from_numpy(crop[None]),
                             torch.from_numpy(vec[None]))[0].numpy()
            a = np.clip(a, -1, 1)
            dq = (a[:2 * n] * self.q_scale).reshape(n, 2)
            int_wpts = (base + dq).T
            tau = a[2 * n:] * self.tau_scale
            ts = ((self.T_max - self.T_min) / (1 + np.exp(-tau)) + self.T_min)
            seeds.append((int_wpts, ts))
        return seeds


# ── export an SB3 policy -> TorchScript deterministic actor ──────────────────
def export_policy(sb3_zip: str, out_path: str):
    import torch
    from stable_baselines3 import SAC

    model = SAC.load(sb3_zip, device="cpu")
    fe = model.policy.actor.features_extractor
    latent = model.policy.actor.latent_pi
    mu = model.policy.actor.mu

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = fe.cnn
            self.cnn_head = fe.cnn_head
            self.vec_mlp = fe.vec_mlp
            self.latent = latent
            self.mu = mu

        def forward(self, crop, vec):
            z = torch.cat([self.cnn_head(self.cnn(crop)),
                           self.vec_mlp(vec)], dim=1)
            return torch.tanh(self.mu(self.latent(z)))

    actor = Actor().eval()
    ex_crop = torch.zeros(1, 1, CROP, CROP)
    ex_vec = torch.zeros(1, 8)
    ts = torch.jit.trace(actor, (ex_crop, ex_vec))
    ts.save(out_path)
    print("policy exported ->", out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=str, required=True)
    ap.add_argument("--out", type=str, default="policy_export.pt")
    a = ap.parse_args()
    export_policy(a.export, a.out)
