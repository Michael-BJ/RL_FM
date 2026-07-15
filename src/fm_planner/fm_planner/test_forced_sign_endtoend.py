#!/usr/bin/env python3
"""test_forced_sign_endtoend.py (2026-07-10) — A/B end-to-end demo of the
v1.12 fallback-bias fix, using the REAL MinJerkPlanner (not a spy).

Reproduces the lap-alternation audit metric OFFLINE: across many forced-side
scenes, what fraction of the FLOWN (best) trajectory lands on the WRONG side?
  OLD = fallback runs on the RAW esdf (pre-fix behavior)
  NEW = fallback runs on the biased esdf (v1.12, production)
Only the esdf handed to the stock 3-seed fallback differs; the normal
topo/MINCO path is identical, so this isolates exactly the fix.

Run:  python3 test_forced_sign_endtoend.py
"""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from min_jerk_planner import MinJerkPlanner, PlannerConfig
import homotopy_frontend as H
from homotopy_frontend import (topo_batch_plan, HomotopyFrontEnd,
                               compute_mode_signature)

RES = 0.1
OX, OY = -2.0, -6.0
ARENA = (-1.5, 15.5, -5.5, 5.5)   # A* bounds (arena minus inset), like expert


class _Origin:
    x, y = OX, OY


class MockESDF:
    """Distance-transform ESDF with the same query surface as esdf_ros2.ESDF."""
    map_resolution = RES
    map_origin = _Origin()

    def __init__(self, boxes):
        rows = int(round(12.0 / RES))   # y in [-6, 6]
        cols = int(round(26.0 / RES))   # x in [-2, 24]
        occ = np.zeros((rows, cols), bool)
        for (x0, x1, y0, y1) in boxes:
            c0 = int(round((x0 - OX) / RES)); c1 = int(round((x1 - OX) / RES))
            r0 = int(round((y0 - OY) / RES)); r1 = int(round((y1 - OY) / RES))
            occ[max(r0, 0):r1, max(c0, 0):c1] = True
        self.esdf_map = ndimage.distance_transform_edt(~occ) * RES
        self.esdf_grad_y, self.esdf_grad_x = np.gradient(self.esdf_map, RES)

    def is_ready(self):
        return True

    def _idx(self, pos):
        return (int((pos[1] - OY) / RES), int((pos[0] - OX) / RES))

    def _ok(self, r, c):
        return 0 <= r < self.esdf_map.shape[0] and 0 <= c < self.esdf_map.shape[1]

    def get_edt_dis(self, pos):
        r, c = self._idx(pos)
        return float(self.esdf_map[r, c]) if self._ok(r, c) else 10000.0

    def get_edt_grad(self, pos):
        r, c = self._idx(pos)
        if not self._ok(r, c):
            return [0.0, 0.0]
        return [float(self.esdf_grad_x[r, c]), float(self.esdf_grad_y[r, c])]

    def has_collision(self, pos, thr=0.3):
        return self.get_edt_dis(pos) < thr


def make_planner():
    cfg = PlannerConfig()
    cfg.v_max, cfg.T_min, cfg.T_max = 1.0, 0.5, 5.0
    cfg.safe_dis, cfg.init_wpts_num, cfg.init_T = 0.4, 2, 2.5
    cfg.delta_t, cfg.collision_cost_tol, cfg.opt_tol = 0.1, 20.0, 1e-2
    cfg.weights = [1.0, 1.0, 1.0, 10000.0]
    return MinJerkPlanner(cfg)


def flown_sig(esdf, planner, head_xy, tail_xy):
    """Side-signature of the trajectory the drone would actually fly (best
    solution), judged against the REAL esdf -- exactly like the recorder."""
    w = planner.int_wpts
    path = [head_xy, w[:2, 0], w[:2, 1], tail_xy]
    return compute_mode_signature(esdf, head_xy, tail_xy, path)


def classify(sig, forced):
    if sig is None:
        return "fail"
    nonfree = [s for s in sig if s in (1, -1)]
    if not nonfree:
        return "free"          # box not actually blocking the straight line
    return "correct" if all(s == forced for s in nonfree) else "wrong"


def run_one(mode, esdf, planner, fe, head_state, tail_state, head_xy, tail_xy):
    """mode='new' -> fallback uses whatever map it's given (biased);
       mode='old' -> fallback forced onto the RAW esdf (pre-fix)."""
    stock = planner._stock_batch_plan
    if mode == "new":
        fb = stock
    else:
        fb = lambda _m, h, t: stock(esdf, h, t)   # ignore biased map -> raw
    try:
        topo_batch_plan(planner, esdf, head_state, tail_state,
                        frontend=fe, fallback=fb)
    except Exception:
        return "fail"
    return classify(flown_sig(esdf, planner, head_xy, tail_xy), fe.forced_sign)


def main():
    planner = make_planner()
    planner._stock_batch_plan = planner.batch_plan   # keep the real stock
    fe = HomotopyFrontEnd(bounds=ARENA)

    head_xy = np.array([0.0, 0.0])
    tail_xy = np.array([16.0, 0.0])
    head_state = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    tail_state = np.array([[16.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    N = 120
    # Two forcing policies:
    #   tight = always force the smaller-gap (hardest) side -> adversarial
    #           worst case, maximal fallback + OLD pull to the cheap side.
    #   alt   = alternate +1/-1 regardless of geometry -> what the real
    #           lap-alternation pilot actually does (leg1 +1, leg2 -1).
    for policy in ("tight", "alt"):
        rng = np.random.default_rng(0)   # same scenes for both policies
        tally = {m: {k: 0 for k in ("correct", "wrong", "free", "fail")}
                 for m in ("old", "new")}
        fb_old = fb_new = 0
        for i in range(N):
            half_w = rng.uniform(0.7, 1.4)
            cx = rng.uniform(6.0, 10.0)
            cy = rng.uniform(-1.2, 1.2)
            box = (cx - 0.25, cx + 0.25, cy - half_w, cy + half_w)
            esdf = MockESDF([box])
            gap_left = ARENA[3] - (cy + half_w)     # toward +y (left, +1)
            gap_right = (cy - half_w) - ARENA[2]    # toward -y (right, -1)
            if min(gap_left, gap_right) < 0.9:      # forced side must be passable
                continue
            if policy == "tight":
                forced = 1 if gap_left <= gap_right else -1
            else:
                forced = 1 if i % 2 == 0 else -1
            for mode in ("old", "new"):
                fe.forced_sign = forced
                fb0 = fe.fallback_count
                res = run_one(mode, esdf, planner, fe, head_state, tail_state,
                              head_xy, tail_xy)
                tally[mode][res] += 1
                fired = fe.fallback_count - fb0
                if mode == "old":
                    fb_old += fired
                else:
                    fb_new += fired
            fe.forced_sign = None

        def pct(m, k, _t=tally):
            tot = sum(_t[m].values())
            return 100.0 * _t[m][k] / tot if tot else 0.0

        title = ("FORCE THE TIGHTEST SIDE (worst case)" if policy == "tight"
                 else "FORCE ALTERNATING +1/-1 (like the real pilot)")
        print(f"\n===== {title} =====")
        print(f"Forced scenes: {sum(tally['old'].values())} | "
              f"fallback fired OLD {fb_old}x = NEW {fb_new}x "
              f"(only the esdf differs)")
        hdr = f"{'code':4} {'correct':>9} {'WRONG':>8} {'free':>6} {'fail':>6}"
        print(hdr); print("-" * len(hdr))
        for m in ("old", "new"):
            t = tally[m]
            print(f"{m:4} {t['correct']:>4} {pct(m,'correct'):>3.0f}%"
                  f" {t['wrong']:>4} {pct(m,'wrong'):>3.0f}%"
                  f" {t['free']:>5} {t['fail']:>6}")
        wo, wn = pct("old", "wrong"), pct("new", "wrong")
        print(f"Wrong-side flown: OLD {wo:.0f}%  ->  NEW {wn:.0f}%"
              f"   (down {wo - wn:.0f} points)")


if __name__ == "__main__":
    main()
