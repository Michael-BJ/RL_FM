#!/usr/bin/env python3
# audit_maze1.py — v1.0 (2026-07-11)
"""
Audit laik-lewat + multimodalitas untuk world MAZE1 (held-out, tanpa ROS).
Ambang IDENTIK audit_test01.py / audit_gauntlet_train.py (world uji harus
setara sulitnya — beda hasil antar world murni soal generalisasi model):

  1. PASSABILITY  : jalur start->goal dengan clearance >= 0.65 m (planner
                    menerima) DAN >= 0.80 m (bebas penalti).
  2. PINCH AUDIT  : pasangan obstacle antar-baris >= 1.30 m (batas absolut
                    planner). MAZE1 tak punya tiang pengisi (pelajaran
                    gauntlet_train: pengisi = chokepoint tak-disengaja) jadi
                    tak ada ambang GAP_TARGET pengisi -> arena.
  3. MULTIMODAL   : gate BIMODAL (G1/G3/G5) >= 2 kelas homotopi; CHICANE
                    (G2/G4, pintu tunggal off-center) = 1 kelas by design
                    (forced weave, BUKAN event ambigu — jangan dianggap bug).

Koridor 0..24 -> goal_x:=24 (arena default, TANPA override).
Jalankan:  python3 audit_maze1.py
Output  :  laporan konsol + maze1_layout.png (denah + jalur A%2A per event).
"""
import itertools
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "fm_planner", "fm_planner"))
from homotopy_frontend import HomotopyFrontEnd  # noqa: E402

from generate_wide_worlds import MAZE1  # noqa: E402

RES        = 0.05
X_RANGE    = (-3.0, 27.0)
Y_RANGE    = (-7.0, 7.0)
ARENA_X    = (-2.0, 25.0)  # = arena inference/expert default (arena_x_max 25)
ARENA_Y    = (-6.0, 6.0)
START      = np.array([0.0, 0.0])
GOAL       = np.array([24.0, 0.0])
NEED_HARD  = 0.65          # inflasi 0.3 + edt terima 0.35
NEED_FREE  = 0.80          # inflasi 0.3 + safe_dis 0.5 (bebas penalti)
PAIR_MIN   = 1.30          # 2 x NEED_HARD (batas absolut)
A_BOUNDS   = (-1.5, 24.5, -5.5, 5.5)   # bounds A* frontend (arena - inset 0.5)

# Baris/event MAZE1 (indeks mengikuti urutan list MAZE1):
ROW_OF = {0: "G1", 1: "G1", 2: "G1",
          3: "G2", 4: "G2",
          5: "G3", 6: "G3", 7: "G3",
          8: "G4", 9: "G4",
          10: "G5", 11: "G5", 12: "G5"}

# (nama, start lokal, goal lokal, min kelas). Chicane sengaja 1 (forced weave).
EVENTS = [("G1 bimodal kotak 2.6m", (2.0, 0.0),  (6.5, 0.0),  2),
          ("G2 chicane pintu-atas", (6.5, 0.0),  (10.5, 0.0), 1),
          ("G3 bimodal kotak 3.0m", (10.5, 0.0), (14.5, 0.0), 2),
          ("G4 chicane pintu-bawah",(14.5, 0.0), (18.5, 0.0), 1),
          ("G5 bimodal kotak 2.6m", (18.5, 0.0), (23.0, 0.0), 2)]


def build_edt():
    xs = np.arange(X_RANGE[0] + RES / 2, X_RANGE[1], RES)
    ys = np.arange(Y_RANGE[0] + RES / 2, Y_RANGE[1], RES)
    XX, YY = np.meshgrid(xs, ys)               # (rows=y, cols=x)
    occ = np.zeros(XX.shape, bool)
    for ob in MAZE1:
        if ob["type"] == "box":
            occ |= ((np.abs(XX - ob["x"]) <= ob["sx"] / 2)
                    & (np.abs(YY - ob["y"]) <= ob["sy"] / 2))
        else:
            occ |= (XX - ob["x"]) ** 2 + (YY - ob["y"]) ** 2 <= ob["r"] ** 2
    # Dinding virtual arena — identik esdf_ros2 arena_bounds
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

    # ── 2. pinch audit antar-baris ──────────────────────────────────────────
    print(f"── Pinch audit (antar-baris < {PAIR_MIN:.2f} m = FAIL; pintu "
          f"dalam-baris dinilai oleh passability, bukan di sini) ──")
    worst = (np.inf, None)
    for i, j in itertools.combinations(range(len(MAZE1)), 2):
        if ROW_OF[i] == ROW_OF[j]:
            continue
        g = surface_gap(MAZE1[i], MAZE1[j])
        if g < worst[0]:
            worst = (g, (i, j))
        if g < PAIR_MIN:
            ok = False
            print(f"  FAIL: obstacle #{i}({ROW_OF[i]}) vs #{j}({ROW_OF[j]}) "
                  f"gap {g:.2f} m (ambang {PAIR_MIN:.2f} m)")
    print(f"  tersempit antar-baris: {worst[0]:.2f} m "
          f"(#{worst[1][0]} vs #{worst[1][1]}) -> "
          f"{'OK' if worst[0] >= PAIR_MIN else 'FAIL'}")

    # ── 1. passability start->goal ──────────────────────────────────────────
    esdf = _MockESDF(edt)
    for need, label in [(NEED_HARD, "WAJIB (0.65 m: planner menerima)"),
                        (NEED_FREE, "WAJIB (0.80 m: bebas penalti)")]:
        fe = HomotopyFrontEnd(hard_clearance=need, max_classes=3,
                              bounds=A_BOUNDS)
        paths = fe.find_paths(esdf, START, GOAL)
        tag = f"{len(paths)} jalur" if paths else "TIDAK ADA JALUR"
        print(f"── Passability @ clearance >= {need:.2f} m [{label}]: {tag}")
        for p, sig in paths:
            print(f"    sig={sig}  min-EDT sepanjang jalur "
                  f"{min_edt_along(edt, xs, ys, p):.2f} m")
        if not paths:
            ok = False

    # ── 3. multimodalitas per event ─────────────────────────────────────────
    print("── Kelas homotopi per event (jendela lokal, frontend default) ──")
    fe = HomotopyFrontEnd(bounds=A_BOUNDS)
    all_paths = []
    for name, s, g, need_k in EVENTS:
        paths = fe.find_paths(esdf, np.array(s), np.array(g))
        sigs = [sig for _, sig in paths]
        flag = "OK " if len(paths) >= need_k else "!!  (kurang)"
        if len(paths) < need_k:
            ok = False
        print(f"  {flag} {name:24s} {len(paths)} kelas "
              f"(min {need_k})  sigs={sigs}")
        all_paths.extend(paths)

    # ── denah PNG ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        fig, ax = plt.subplots(figsize=(15, 6))
        for ob in MAZE1:
            if ob["type"] == "box":
                ax.add_patch(Rectangle((ob["x"] - ob["sx"] / 2,
                                        ob["y"] - ob["sy"] / 2),
                                       ob["sx"], ob["sy"],
                                       color="#b03030" if ob["sy"] >= 1.2
                                       else "#5570a8"))
            else:
                ax.add_patch(Circle((ob["x"], ob["y"]), ob["r"],
                                    color="#8a5a2b"))
        for p, sig in all_paths:
            ax.plot(p[:, 0], p[:, 1], lw=1.3, alpha=0.75)
        ax.axhline(6.0, color="k", ls="--", lw=0.8)
        ax.axhline(-6.0, color="k", ls="--", lw=0.8)
        ax.plot(*START, "g^", ms=12, label="start (0,0)")
        ax.plot(*GOAL, "r*", ms=16, label="goal (24,0)")
        for name, s, g, _k in EVENTS:
            ax.annotate(name.split()[0], (s[0] + (g[0] - s[0]) / 2, 5.3),
                        ha="center", fontsize=9)
        ax.set_xlim(-2, 25); ax.set_ylim(-6.5, 6.5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title("maze1 (held-out) — koridor 0..24, goal_x:=24 "
                     "(bimodal G1/G3/G5 + chicane G2/G4; garis = jalur A*)")
        ax.legend(loc="lower right")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "maze1_layout.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"[png] {out}")
    except ImportError:
        print("[png] matplotlib tidak ada — denah dilewati")

    print("\n=== AUDIT", "LULUS ===" if ok else "GAGAL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
