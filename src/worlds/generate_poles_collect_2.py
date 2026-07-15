#!/usr/bin/env python3
"""Generate poles_collect_2 — a hand-placed poles-style collecting world
matching the user's sketch (2026-07-11):

  - a MELINTANG wall (sx=1, sy=3) directly in front of the drone at START;
  - a pair of poles just after it (a gate) + a SECOND melintang wall at x~13;
  - a head-on 1x1 box further on; off-axis poles/boxes as fillers;
  - corridor 0..23  (goal_x = 23).

Every gap is verified with a grid-clearance BOTTLENECK audit: the widest
start->goal path must keep clearance >= R_MIN (so the hole between obstacles
is comfortably drone-passable). Positions were nudged until it passes.
"""
import os, math
import numpy as np
from scipy.ndimage import label

GOAL_X = 23.0
R_MIN  = 0.65          # min clearance-to-surface on the best path (~1.3 m gap)
R_GOOD = 0.75          # comfortable target (~1.5 m gap)

def B(x, y, sx, sy, h, tag=None):
    return dict(type="box", x=x, y=y, sx=sx, sy=sy, h=h, tag=tag)
def C(x, y, h=3.9, r=0.5, tag=None):
    return dict(type="cyl", x=x, y=y, r=r, h=h, tag=tag)

# ── hand-placed layout (matches the sketch; gaps tuned to pass the audit) ─────
OBST = [
    B(4.5,  0.0, 1.0, 3.0, 3.6, tag="wall"),   # front wall (head-on at start)
    B(13.0, 0.0, 1.0, 3.0, 3.6, tag="wall"),   # second melintang wall
    B(17.6, -0.3, 1.0, 1.0, 3.36),             # head-on box
    # gate poles just after the front wall
    C(7.6,  2.6),
    C(8.1, -2.2),
    # off-axis fillers — upper
    C(9.4,  4.6),
    B(12.4, 4.4, 1.0, 1.0, 6.26),
    B(16.4, 3.1, 1.0, 1.0, 3.36),
    C(20.2, 2.1),
    # off-axis fillers — lower
    C(6.4, -4.4),
    C(10.4, -3.9),
    C(12.9, -3.9),
    B(16.9, -4.1, 1.0, 1.0, 6.26),
    C(20.4, -3.6),
]

# ── grid clearance + bottleneck audit ────────────────────────────────────────
def clearance_grid(obs, res=0.1):
    xs = np.arange(-1.0, GOAL_X + 1.0, res)
    ys = np.arange(-5.6, 5.6, res)
    X, Y = np.meshgrid(xs, ys)
    D = np.full(X.shape, 9.9)
    for o in obs:
        if o["type"] == "box":
            dx = np.maximum(np.abs(X - o["x"]) - o["sx"] / 2, 0)
            dy = np.maximum(np.abs(Y - o["y"]) - o["sy"] / 2, 0)
            d = np.hypot(dx, dy)
        else:
            d = np.maximum(np.hypot(X - o["x"], Y - o["y"]) - o["r"], 0)
        D = np.minimum(D, d)
    return xs, ys, D

def cell(xs, ys, x, y):
    return int(np.argmin(np.abs(ys - y))), int(np.argmin(np.abs(xs - x)))

def bottleneck(xs, ys, D):
    si = cell(xs, ys, 0.0, 0.0)
    gi = cell(xs, ys, GOAL_X, 0.0)
    best = 0.0
    for R in np.arange(0.95, 0.34, -0.05):
        lab, _ = label(D >= R)
        ls, lg = lab[si], lab[gi]
        if ls != 0 and ls == lg:
            best = round(float(R), 2)
            break
    # locate the tightest point on that corridor (approx): min D along goal-line band
    return best

def tight_pairs(obs, thr=1.5):
    out = []
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            a, b = obs[i], obs[j]
            ha = a["r"] if a["type"] == "cyl" else math.hypot(a["sx"]/2, a["sy"]/2)
            hb = b["r"] if b["type"] == "cyl" else math.hypot(b["sx"]/2, b["sy"]/2)
            gap = math.hypot(a["x"]-b["x"], a["y"]-b["y"]) - ha - hb
            if gap < thr:
                out.append((round(gap, 2), a, b))
    return sorted(out)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_wide_worlds import write_sdf, write_pcd
    wdir = os.path.dirname(os.path.abspath(__file__))

    xs, ys, D = clearance_grid(OBST)
    bn = bottleneck(xs, ys, D)
    gap_m = 2 * bn
    print(f"obstacles={len(OBST)}  goal_x={GOAL_X}")
    print(f"BOTTLENECK best-path clearance = {bn:.2f} m  (~gap {gap_m:.2f} m)  "
          f"-> {'GOOD' if bn>=R_GOOD else 'ok' if bn>=R_MIN else 'TOO TIGHT'}")
    print("tightest obstacle pairs (edge-gap < 1.5 m):")
    for gap, a, b in tight_pairs(OBST)[:8]:
        print(f"   gap={gap:.2f} m : ({a['x']:.1f},{a['y']:+.1f}) <-> ({b['x']:.1f},{b['y']:+.1f})")

    if bn >= R_MIN:
        write_sdf("poles_collect_2", OBST, os.path.join(wdir, "poles_collect_2.sdf"))
        write_pcd(OBST, os.path.join(wdir, "poles_collect_2.pcd"))
        print("PASSABLE -> files written ✓")
    else:
        print("!! bottleneck below R_MIN — widen the tightest pair before writing.")
