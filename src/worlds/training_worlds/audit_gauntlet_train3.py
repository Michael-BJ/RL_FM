#!/usr/bin/env python3
# audit_gauntlet_train3.py — v1.0 (2026-07-21)
"""
Audit laik-lewat + multimodalitas untuk world GAUNTLET_TRAIN3 (tanpa ROS).
Ambang IDENTIK audit_gauntlet_train.py / audit_gauntlet_train2.py, plus target
jarak fungsional user 1.8 m:

  1. PASSABILITY  : jalur start->goal dengan clearance >= 0.65 m DAN >= 0.80 m.
  2. PINCH AUDIT  : jarak antar-event >= 1.80 m (target user; batas absolut 1.30).
  3. MULTIMODAL   : SEMUA 6 event >= 2 kelas homotopi (kiri/kanan) — nol
                    threading (beda kompleksitas dari gauntlet_train).

Jalankan:  python3 audit_gauntlet_train3.py
Output  :  laporan konsol + gauntlet_train3_layout.png (denah + jalur A*).
"""
import itertools
import os
import sys

import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "fm_planner", "fm_planner"))
sys.path.insert(0, os.path.join(_HERE, ".."))
from homotopy_frontend import HomotopyFrontEnd  # noqa: E402

from generate_wide_worlds import GAUNTLET_TRAIN3  # noqa: E402

RES        = 0.05
X_RANGE    = (-3.0, 34.0)
Y_RANGE    = (-7.0, 7.0)
ARENA_X    = (-2.0, 33.0)
ARENA_Y    = (-6.0, 6.0)
START      = np.array([0.0, 0.0])
GOAL       = np.array([30.0, 0.0])
NEED_HARD  = 0.65
NEED_FREE  = 0.80
PAIR_MIN   = 1.30
GAP_TARGET = 1.80          # target fungsional user (jarak minimum antar-obstacle)
A_BOUNDS   = (-1.5, 32.5, -5.5, 5.5)

ROW_OF = {0: "H1", 1: "H1", 2: "H1",
          3: "H2", 4: "H2", 5: "H2",
          6: "H3", 7: "H3",
          8: "H4",
          9: "H5", 10: "H5", 11: "H5",
          12: "H6", 13: "H6", 14: "H6"}

# SEMUA event bimodal (min 2 kelas kiri/kanan).
EVENTS = [("H1 box tengah 2.2",       (2.0, 0.0),  (6.0, 0.0),  2),
          ("H2 silinder gemuk 2.0",   (7.0, 0.0),  (11.0, 0.0), 2),
          ("H3 pintu 2.2 (off -1.4)", (11.5, 0.0), (15.5, 0.0), 2),
          ("H4 dinding solid 4.6",    (16.0, 0.0), (20.0, 0.0), 2),
          ("H5 box tengah 3.0",       (20.5, 0.0), (24.5, 0.0), 2),
          ("H6 silinder gemuk 2.4",   (25.0, 0.0), (29.0, 0.0), 2)]


def build_edt():
    xs = np.arange(X_RANGE[0] + RES / 2, X_RANGE[1], RES)
    ys = np.arange(Y_RANGE[0] + RES / 2, Y_RANGE[1], RES)
    XX, YY = np.meshgrid(xs, ys)
    occ = np.zeros(XX.shape, bool)
    for ob in GAUNTLET_TRAIN3:
        if ob["type"] == "box":
            occ |= ((np.abs(XX - ob["x"]) <= ob["sx"] / 2)
                    & (np.abs(YY - ob["y"]) <= ob["sy"] / 2))
        else:
            occ |= (XX - ob["x"]) ** 2 + (YY - ob["y"]) ** 2 <= ob["r"] ** 2
    occ |= ((XX < ARENA_X[0]) | (XX > ARENA_X[1])
            | (YY < ARENA_Y[0]) | (YY > ARENA_Y[1]))
    return ndimage.distance_transform_edt(~occ) * RES, xs, ys


class _Origin:
    x, y = X_RANGE[0] + RES / 2, Y_RANGE[0] + RES / 2


class _MockESDF:
    map_resolution = RES
    map_origin = _Origin()

    def __init__(self, edt):
        self.esdf_map = edt


def surface_gap(a, b):
    def rect_dist(ob, px, py):
        dx = max(abs(px - ob["x"]) - ob["sx"] / 2, 0.0)
        dy = max(abs(py - ob["y"]) - ob["sy"] / 2, 0.0)
        return np.hypot(dx, dy)

    if a["type"] == "cyl" and b["type"] == "cyl":
        return np.hypot(a["x"] - b["x"], a["y"] - b["y"]) - a["r"] - b["r"]
    if a["type"] == "box" and b["type"] == "box":
        dx = max(abs(a["x"] - b["x"]) - (a["sx"] + b["sx"]) / 2, 0.0)
        dy = max(abs(a["y"] - b["y"]) - (a["sy"] + b["sy"]) / 2, 0.0)
        return np.hypot(dx, dy)
    box, cyl = (a, b) if a["type"] == "box" else (b, a)
    return rect_dist(box, cyl["x"], cyl["y"]) - cyl["r"]


def min_edt_along(edt, xs, ys, path_xy):
    ii = np.clip(np.round((path_xy[:, 1] - (Y_RANGE[0] + RES / 2)) / RES)
                 .astype(int), 0, len(ys) - 1)
    jj = np.clip(np.round((path_xy[:, 0] - (X_RANGE[0] + RES / 2)) / RES)
                 .astype(int), 0, len(xs) - 1)
    return float(edt[ii, jj].min())


def main():
    edt, xs, ys = build_edt()
    ok = True

    # ── pinch antar-event (target user 1.8 m) ────────────────────────────────
    print(f"── Pinch audit (antar-event < {GAP_TARGET:.2f} m = FAIL) ──")
    worst = (np.inf, None)
    for i, j in itertools.combinations(range(len(GAUNTLET_TRAIN3)), 2):
        if ROW_OF[i] == ROW_OF[j]:
            continue
        g = surface_gap(GAUNTLET_TRAIN3[i], GAUNTLET_TRAIN3[j])
        if g < worst[0]:
            worst = (g, (i, j))
        if g < GAP_TARGET:
            ok = False
            print(f"  FAIL: #{i}({ROW_OF[i]}) vs #{j}({ROW_OF[j]}) "
                  f"gap {g:.2f} m")
    print(f"  tersempit antar-event: {worst[0]:.2f} m "
          f"(#{worst[1][0]} vs #{worst[1][1]}) -> "
          f"{'OK (>= 1.8 m)' if worst[0] >= GAP_TARGET else 'FAIL'}")

    # ── passage dalam event (box/silinder tengah <-> tiang, dua sisi pintu) ───
    print("── Lebar passage dalam event (celah L/R yang harus dilewati) ──")
    worst_pass = (np.inf, None)
    for row in ("H1", "H2", "H3", "H5", "H6"):
        idx = [k for k, v in ROW_OF.items() if v == row]
        for i, j in itertools.combinations(idx, 2):
            g = surface_gap(GAUNTLET_TRAIN3[i], GAUNTLET_TRAIN3[j])
            if g < worst_pass[0]:
                worst_pass = (g, (row, i, j))
    g = worst_pass[0]
    print(f"  passage tersempit: {g:.2f} m ({worst_pass[1]}) -> "
          f"{'OK (>= 1.8 m)' if g >= GAP_TARGET else 'FAIL'}")
    if g < GAP_TARGET:
        ok = False

    # ── passability start->goal ──────────────────────────────────────────────
    esdf = _MockESDF(edt)
    for need, label in [(NEED_HARD, "0.65 m: planner menerima"),
                        (NEED_FREE, "0.80 m: bebas penalti")]:
        fe = HomotopyFrontEnd(hard_clearance=need, max_classes=3,
                              bounds=A_BOUNDS)
        paths = fe.find_paths(esdf, START, GOAL)
        tag = f"{len(paths)} jalur" if paths else "TIDAK ADA JALUR"
        print(f"── Passability @ clearance >= {need:.2f} m [{label}]: {tag}")
        for p, sig in paths:
            print(f"    sig={sig}  min-EDT {min_edt_along(edt, xs, ys, p):.2f} m")
        if not paths:
            ok = False

    # ── multimodalitas per event ─────────────────────────────────────────────
    print("── Kelas homotopi per event (harus 2 = bimodal) ──")
    fe = HomotopyFrontEnd(bounds=A_BOUNDS)
    all_paths = []
    for name, s, g, need_k in EVENTS:
        paths = fe.find_paths(esdf, np.array(s), np.array(g))
        sigs = [sig for _, sig in paths]
        flag = "OK " if len(paths) >= need_k else "!!  (kurang)"
        if len(paths) < need_k:
            ok = False
        print(f"  {flag} {name:26s} {len(paths)} kelas "
              f"(min {need_k})  sigs={sigs}")
        all_paths.extend(paths)

    # ── denah PNG ────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        fig, ax = plt.subplots(figsize=(16, 6))
        for ob in GAUNTLET_TRAIN3:
            if ob["type"] == "box":
                ax.add_patch(Rectangle((ob["x"] - ob["sx"] / 2,
                                        ob["y"] - ob["sy"] / 2),
                                       ob["sx"], ob["sy"],
                                       color="#b03030" if ob["sy"] >= 1.2
                                       else "#5570a8"))
            else:
                ax.add_patch(Circle((ob["x"], ob["y"]), ob["r"],
                                    color="#b03030" if ob["r"] >= 0.6
                                    else "#8a5a2b"))
        for p, sig in all_paths:
            ax.plot(p[:, 0], p[:, 1], lw=1.2, alpha=0.7)
        ax.axhline(6.0, color="k", ls="--", lw=0.8)
        ax.axhline(-6.0, color="k", ls="--", lw=0.8)
        ax.plot(*START, "g^", ms=12, label="start (0,0)")
        ax.plot(*GOAL, "r*", ms=16, label="goal (30,0)")
        for name, s, g, _k in EVENTS:
            ax.annotate(name.split()[0], (s[0] + (g[0] - s[0]) / 2, 5.3),
                        ha="center", fontsize=9)
        ax.set_xlim(-2, 33); ax.set_ylim(-6.5, 6.5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title("gauntlet_train3 — 6 event BIMODAL (kiri/kanan), "
                     "koridor 0..30, goal_x:=30, arena_x_max:=33")
        ax.legend(loc="lower right")
        out = os.path.join(_HERE, "gauntlet_train3_layout.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"[png] {out}")
    except ImportError:
        print("[png] matplotlib tidak ada — denah dilewati")

    print("\n=== AUDIT", "LULUS ===" if ok else "GAGAL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
