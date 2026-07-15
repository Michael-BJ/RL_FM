#!/usr/bin/env python3
"""
flight_monitor.py — NN/FM planner flight-test monitor in ambiguity worlds
=========================================================================
PASSIVE monitoring node (does not control the drone, no colcon build needed —
run it directly with python3). Used as the 5th terminal when testing
nn_inference / fm_inference in the poles_wide / wide_test worlds.

What it records:
  - Drone trajectory from /px4/sensors (JSON) -> traj.csv + PNG plot.
  - Mode decision at EVERY wide box (ambiguous obstacle on y=0):
    side passed LEFT(y+) / RIGHT(y-) + clearance while crossing -> crossings.csv.
  - Events from the planner node's /rosout: Replan ok/failed, ESCAPE, collision /
    look-ahead guard, tracking gap, no progress -> events.csv.
  - Per-lap statistics: duration, path length, minimum clearance, escape count.
  - Ground-truth obstacles taken directly from worlds/generate_wide_worlds.py
    (POLES_WIDE / WIDE_TEST) so clearance is computed exactly, not from the ESDF.

Symptoms being looked for (multimodal hypothesis):
  - NN (mode averaging)  : heads for the CENTER of the wide box -> frequent
    guard/escape, crossings with small clearance, slow/failed laps.
  - FM (commits to one mode): clean left OR right crossing, rare escapes,
    side distribution varies lap to lap.

Usage (terminal 5, after the planner is running):
  python3 src/fm_planner/fm_planner/flight_monitor.py \
      --world wide_test --goal-x 20.0 --label fm
  # stop: Ctrl+C (or automatically after --laps N)
Output: ~/monitor_logs/<label>_<world>_<timestamp>/
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Log
from std_msgs.msg import String

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "worlds")))
from generate_wide_worlds import (                       # noqa: E402
    POLES_WIDE, WIDE_TEST, WALL_GAUNTLET, GAUNTLET_TRAIN, GAUNTLET_TRAIN2,
    TEST01, DECISIVE_GAP, MAZE1,
)

sys.path.insert(0, os.path.abspath(
    os.path.join(_THIS_DIR, "..", "..", "worlds", "training_worlds")))
from generate_training_worlds import WORLDS as TRAINING_WORLDS  # noqa: E402

WORLDS = {"poles_wide": POLES_WIDE, "wide_test": WIDE_TEST,
          "wall_gauntlet": WALL_GAUNTLET, "gauntlet_train": GAUNTLET_TRAIN,
          "gauntlet_train2": GAUNTLET_TRAIN2,
          "decisive_gap": DECISIVE_GAP, "maze1": MAZE1,
          "test01": TEST01, **TRAINING_WORLDS}

# "Ambiguity generator" box: width >= 1.2 m and (nearly) on the y=0 axis
WIDE_BOX_MIN_SY = 1.2
WIDE_BOX_MAX_Y  = 0.5

MIN_FLIGHT_Z   = 0.8    # samples below this are ignored (not airborne yet)
GOAL_REACH_DIS = 1.6    # lap-complete detection (the planner node uses 1.5)

# Planner node log pattern (nn_inference_node / fm_inference_node / legacy_inference_node)
LOG_PATTERNS = [
    ("escape",      ">>> ESCAPE"),
    ("exit_escape", "<<< Exit escape"),
    ("guard_pos",   "Collision guard"),
    ("guard_ahead", "Look-ahead guard"),
    ("gap",         "tracking gap"),
    ("no_progress", "No progress"),
    ("replan_ok",   "Replan ok"),
    ("replan_fail", "Replan failed"),
    ("done",        "DONE"),
]
PLANNER_NODE_NAMES = ("nn_inference_node", "fm_inference_node",
                      "legacy_inference_node")


def obstacle_clearance(x, y, ob):
    """Exact horizontal distance from point (x,y) to the obstacle surface (m)."""
    if ob["type"] == "box":
        dx = max(abs(x - ob["x"]) - ob["sx"] / 2.0, 0.0)
        dy = max(abs(y - ob["y"]) - ob["sy"] / 2.0, 0.0)
        return math.hypot(dx, dy)
    return max(math.hypot(x - ob["x"], y - ob["y"]) - ob["r"], 0.0)


def _find_wide_obstacles(obstacles):
    """Obstacles that force a binary left/right decision at y=0: a single
    wide box/cylinder centered on the axis, OR a pair of cylinders
    straddling y=0 at (nearly) the same x that together block the middle
    (twin_pole_gate — no single object is centered, but the PAIR is).
    Returned as box-shaped dicts (x, y=0.0, sy=full width) so the existing
    crossing/plotting code (which only knows about "sy") needs no changes."""
    wide = [dict(x=ob["x"], y=0.0,
                 sy=(ob["sy"] if ob["type"] == "box" else 2 * ob["r"]))
            for ob in obstacles
            if (ob["sy"] if ob["type"] == "box" else 2 * ob["r"])
            >= WIDE_BOX_MIN_SY and abs(ob["y"]) <= WIDE_BOX_MAX_Y]
    cyls = [ob for ob in obstacles if ob["type"] == "cyl"]
    used = set()
    for i, a in enumerate(cyls):
        if i in used:
            continue
        for j, b in enumerate(cyls):
            if j <= i or j in used:
                continue
            if abs(a["x"] - b["x"]) < 0.05 and a["y"] * b["y"] < 0:
                half = max(abs(a["y"]), abs(b["y"])) + max(a["r"], b["r"])
                wide.append(dict(x=a["x"], y=0.0, sy=2 * half))
                used.add(i); used.add(j)
                break
    return wide


class FlightMonitor(Node):
    def __init__(self, args):
        super().__init__("flight_monitor")
        self._obstacles = WORLDS[args.world]
        self._wide_boxes = _find_wide_obstacles(self._obstacles)
        self._goal_x    = args.goal_x
        self._max_laps  = args.laps
        self._label     = args.label
        self._world     = args.world

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._out_dir = os.path.expanduser(
            args.out_dir or f"~/monitor_logs/{args.label}_{args.world}_{stamp}")
        os.makedirs(self._out_dir, exist_ok=True)

        # Trajectory state
        self._t0        = time.time()
        self._prev      = None            # (t, x, y) last valid sample
        self._pos       = None            # (x, y, z) latest (for events)
        self._traj      = []              # (t, x, y, z, lap)
        self._crossings = []              # dict per wide-box crossing
        self._events    = []              # dict per log event
        self._counts    = {k: 0 for k, _ in LOG_PATTERNS}
        self._detected_node = None

        # Lap state
        self._lap        = 1
        self._direction  = +1             # +1: heading to goal_x, -1: returning to 0
        self._lap_start  = None
        self._lap_len    = 0.0
        self._lap_minclr = math.inf
        self._lap_esc    = 0
        self._laps_done  = []             # dict per completed lap
        self._min_clear_all = math.inf

        self.create_subscription(String, "/px4/sensors", self._cb_sensors, 20)
        self.create_subscription(Log, "/rosout", self._cb_rosout, 50)
        self.create_timer(5.0, self._print_status)

        box_str = ", ".join(
            f"x={b['x']:.1f}(width {b['sy']:.1f}m)" for b in self._wide_boxes)
        print(f"[MON] world={args.world}  goal_x={args.goal_x}  "
              f"out={self._out_dir}")
        print(f"[MON] monitored wide boxes: {box_str}")
        print("[MON] waiting for /px4/sensors and /rosout data ... (Ctrl+C to "
              "stop & write the report)")

    # ── Trajectory & lap/crossing detection ──────────────────────────────────

    def _cb_sensors(self, msg):
        try:
            d = json.loads(msg.data)
            x = float(d["local_x"]); y = float(d["local_y"])
            z = float(d["local_z"])
        except Exception:
            return
        t = time.time() - self._t0
        self._pos = (x, y, z)
        if z < MIN_FLIGHT_Z:
            self._prev = None
            return

        if self._lap_start is None:
            self._lap_start = t
            print(f"[MON] flight detected — lap 1 starts (t={t:.1f}s)")

        clear = min(obstacle_clearance(x, y, ob) for ob in self._obstacles)
        self._min_clear_all = min(self._min_clear_all, clear)
        self._lap_minclr    = min(self._lap_minclr, clear)
        self._traj.append((round(t, 3), round(x, 3), round(y, 3),
                           round(z, 3), self._lap))

        if self._prev is not None:
            tp, xp, yp = self._prev
            self._lap_len += math.hypot(x - xp, y - yp)
            # Crossing of the plane x = wide-box center -> record side & clearance
            # (half-open so a point EXACTLY on the plane isn't counted twice)
            for ob in self._wide_boxes:
                if (xp < ob["x"] <= x) or (x <= ob["x"] < xp):
                    frac = (ob["x"] - xp) / (x - xp)
                    yc = yp + frac * (y - yp)
                    edge = abs(yc - ob["y"]) - ob["sy"] / 2.0
                    side = "LEFT(y+)" if yc > ob["y"] else "RIGHT(y-)"
                    if edge < 0.15:
                        side = "CENTER!"   # grazing/inside the footprint -> mode averaging
                    self._crossings.append(dict(
                        t=round(t, 2), lap=self._lap, direction=self._direction,
                        box_x=ob["x"], box_w=ob["sy"], y_cross=round(yc, 2),
                        side=side, clearance=round(max(edge, 0.0), 2)))
                    print(f"[MON] lap{self._lap} box x={ob['x']:.1f} "
                          f"(w={ob['sy']:.1f}m): {side}  y={yc:+.2f}  "
                          f"clear={max(edge, 0):.2f}m")
        self._prev = (t, x, y)

        # Lap complete?
        tgt = (self._goal_x, 0.0) if self._direction > 0 else (0.0, 0.0)
        if math.hypot(x - tgt[0], y - tgt[1]) < GOAL_REACH_DIS:
            self._finish_lap(t)

    def _finish_lap(self, t):
        dur = t - self._lap_start if self._lap_start is not None else 0.0
        self._laps_done.append(dict(
            lap=self._lap, direction="forward" if self._direction > 0 else "return",
            duration=round(dur, 1), length=round(self._lap_len, 1),
            min_clear=round(self._lap_minclr, 2)
            if self._lap_minclr < math.inf else None,
            escape=self._lap_esc))
        print(f"[MON] ✓ lap {self._lap} complete: {dur:.1f}s  "
              f"path={self._lap_len:.1f}m  min_clear={self._lap_minclr:.2f}m  "
              f"escape={self._lap_esc}")
        self._lap       += 1
        self._direction *= -1
        self._lap_start  = t
        self._lap_len    = 0.0
        self._lap_minclr = math.inf
        self._lap_esc    = 0
        if self._max_laps and len(self._laps_done) >= self._max_laps:
            print(f"[MON] {self._max_laps} laps reached — done.")
            raise KeyboardInterrupt

    # ── Planner event log ─────────────────────────────────────────────────────

    def _cb_rosout(self, msg):
        if not any(n in msg.name for n in PLANNER_NODE_NAMES):
            return
        if self._detected_node is None:
            self._detected_node = msg.name
            print(f"[MON] planner node detected: {msg.name}")
        for key, pat in LOG_PATTERNS:
            if pat in msg.msg:
                self._counts[key] += 1
                if key == "escape":
                    self._lap_esc += 1
                if key != "replan_ok":   # replan_ok is too noisy to log individually
                    x, y, z = self._pos if self._pos else (None, None, None)
                    self._events.append(dict(
                        t=round(time.time() - self._t0, 2), lap=self._lap,
                        event=key, x=x, y=y, msg=msg.msg[:120]))
                break

    # ── Periodic status ───────────────────────────────────────────────────────

    def _print_status(self):
        if self._pos is None:
            return
        x, y, z = self._pos
        c = self._counts
        print(f"[MON] lap{self._lap} {'->' if self._direction > 0 else '<-'} "
              f"pos=({x:.1f},{y:.1f},{z:.1f})  "
              f"replan {c['replan_ok']}ok/{c['replan_fail']}fail  "
              f"escape={c['escape']}  guard={c['guard_pos']+c['guard_ahead']}  "
              f"min_clear={self._min_clear_all:.2f}m")

    # ── Final report ──────────────────────────────────────────────────────────

    def finalize(self):
        out = self._out_dir
        with open(os.path.join(out, "traj.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "z", "lap"])
            w.writerows(self._traj)
        for name, rows in (("crossings.csv", self._crossings),
                           ("events.csv", self._events),
                           ("laps.csv", self._laps_done)):
            with open(os.path.join(out, name), "w", newline="") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

        lines = self._summary_lines()
        report = "\n".join(lines)
        with open(os.path.join(out, "summary.txt"), "w") as f:
            f.write(report + "\n")
        print("\n" + report)
        try:
            self._plot(os.path.join(out, "traj_plot.png"))
            print(f"\n[MON] plot   : {out}/traj_plot.png")
        except Exception as e:
            print(f"\n[MON] plot failed ({e}) — CSVs are still saved")
        print(f"[MON] report : {out}/summary.txt")

    def _summary_lines(self):
        c = self._counts
        L = ["=" * 64,
             f"  MONITOR REPORT — {self._label} @ {self._world}"
             + (f"  (node: {self._detected_node})" if self._detected_node else ""),
             "=" * 64,
             f"Laps complete    : {len(self._laps_done)}",
             f"Replan           : {c['replan_ok']} ok / {c['replan_fail']} failed",
             f"Escape           : {c['escape']}  (guard pos={c['guard_pos']}, "
             f"look-ahead={c['guard_ahead']}, gap={c['gap']}, "
             f"no-progress={c['no_progress']})",
             f"Min clearance    : {self._min_clear_all:.2f} m (ground truth)"
             if self._min_clear_all < math.inf else "Min clearance    : -",
             "",
             "── Mode decision per wide box ──"]
        for ob in self._wide_boxes:
            cr = [c_ for c_ in self._crossings if c_["box_x"] == ob["x"]]
            n = {"LEFT(y+)": 0, "RIGHT(y-)": 0, "CENTER!": 0}
            for c_ in cr:
                n[c_["side"]] += 1
            det = ", ".join(f"lap{c_['lap']}:{c_['side'][:2]}" for c_ in cr)
            L.append(f"  box x={ob['x']:4.1f} width={ob['sy']:.1f}m : "
                     f"{n['LEFT(y+)']}x left, {n['RIGHT(y-)']}x right, "
                     f"{n['CENTER!']}x CENTER  [{det}]")
        if self._laps_done:
            L.append("")
            L.append("── Per lap ──")
            for lp in self._laps_done:
                L.append(f"  lap {lp['lap']:2d} ({lp['direction']:7s}): "
                         f"{lp['duration']:6.1f}s  path={lp['length']:5.1f}m  "
                         f"min_clear={lp['min_clear']}m  escape={lp['escape']}")
        L.append("=" * 64)
        return L

    def _plot(self, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        wide_xs = {round(b["x"], 1) for b in self._wide_boxes}
        fig, ax = plt.subplots(figsize=(13, 6))
        for ob in self._obstacles:
            if ob["type"] == "box":
                wide = round(ob["x"], 1) in wide_xs
                ax.add_patch(Rectangle(
                    (ob["x"] - ob["sx"] / 2, ob["y"] - ob["sy"] / 2),
                    ob["sx"], ob["sy"],
                    color="#c0392b" if wide else "#5d7fa3", alpha=0.85))
            else:
                ax.add_patch(Circle((ob["x"], ob["y"]), ob["r"],
                                    color="#8d5a2b", alpha=0.85))
        if self._traj:
            arr = np.array([(r[1], r[2], r[4]) for r in self._traj])
            n_laps = int(arr[:, 2].max())
            cmap = plt.get_cmap("viridis", max(n_laps, 2))
            for lap in range(1, n_laps + 1):
                seg = arr[arr[:, 2] == lap]
                if len(seg) > 1:
                    ax.plot(seg[:, 0], seg[:, 1], lw=1.4,
                            color=cmap(lap - 1), label=f"lap {lap}")
        esc = [(e["x"], e["y"]) for e in self._events
               if e["event"] == "escape" and e["x"] is not None]
        if esc:
            ex, ey = zip(*esc)
            ax.plot(ex, ey, "rx", ms=10, mew=2.5, label="escape")
        ax.set_xlim(-2, self._goal_x + 3)
        ax.set_ylim(-7, 7)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{self._label} @ {self._world} — "
                     f"{len(self._laps_done)} lap, "
                     f"{self._counts['escape']} escape")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", required=True, choices=sorted(WORLDS.keys()))
    ap.add_argument("--goal-x", type=float, default=20.0)
    ap.add_argument("--label", default="run",
                    help="run label, e.g. nn / fm (used for folder name & title)")
    ap.add_argument("--laps", type=int, default=0,
                    help="stop automatically after N laps (0 = until Ctrl+C)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    rclpy.init()
    node = FlightMonitor(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
