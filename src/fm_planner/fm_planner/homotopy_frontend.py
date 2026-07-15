#!/usr/bin/env python3
# homotopy_frontend.py — v1.1 — Topological multi-homotopy seed generator for the expert planner
"""
homotopy_frontend.py — Topological front-end for expert data collection
==========================================================================
CHANGELOG
  v1.9 (2026-07-08)  REVERTED the v1.6/v1.7 per-seed optimizer constraint.
                     Removed constraint_for_seed(), the LinearConstraint
                     import, and LATERAL_BOUND_MARGIN_M; topo_batch_plan()
                     now calls planner.plan_once() unconstrained (plain
                     L-BFGS-B, same solver/path as every other call site in
                     the codebase) again, same as v1.5.
                     Why: the 3-run real-flight comparison on box_narrow
                     (300 samples each, same world/target) never showed
                     v1.7's SLSQP+constraint fix beating the ORIGINAL
                     unconstrained v1.5 baseline on the metric it was built
                     for. Bimodal rate: v1.5 6.0% (17/283) vs v1.7 4.2%
                     (12/288), z=1.00 -- not statistically distinguishable.
                     Meanwhile the SEPARATE, still-unexplained sign-skew
                     issue (A* preferring side -1) got WORSE and became
                     statistically significant under v1.7 (3:20, p=0.0005)
                     vs v1.5's 14:7 (p=0.19, not significant) -- the
                     "fix" for one problem correlated with a clearer
                     regression on an adjacent one. On top of that, SLSQP's
                     lack of an implicit trust region (unlike L-BFGS-B's
                     line search) needed a bolted-on tau safety bound
                     ([-30,30]) after it was observed pushing tau to
                     +-hundreds and overflowing map_tau2T() -- a failure
                     mode plain L-BFGS-B never had. Net: strictly more
                     moving parts (a second solver, a hand-derived linear
                     constraint, a numerical safety patch) for a benefit
                     that never cleared statistical noise. v1.5's
                     unconstrained seeding + v1.3's hard wall-off diversity
                     (still active, unaffected by this revert) already
                     covers the bimodal-discovery job; the seed-collapse
                     failure mode v1.6/v1.7 targeted was real but rare
                     enough that fixing it this way cost more than it
                     returned. plan_once() in min_jerk_planner.py is back
                     to L-BFGS-B-only (no bounds/constraints params).
  v1.8 (2026-07-08)  Lap-level forced side (pilot: see expert_planner_node.py
                     force_lap_alternation). Per-replan enumeration (v1.0-v1.7)
                     only answers "does THIS scene have >1 valid side" — it
                     says nothing about whether both sides are ever actually
                     FLOWN (closed-loop, Gazebo-simulated) across a mission,
                     and the flown side is always whichever the optimizer
                     found cheaper (measured biased toward -1, see v1.7 note).
                     find_paths() now takes an optional `forced_sign`: before
                     diversification starts, wall off the OPPOSITE side of
                     every straight-line blocked interval (reuses
                     _wall_off_side(), same mechanism v1.3 already uses
                     mid-loop) so that call can only ever discover
                     `forced_sign`'s class -- the drone has no alternative to
                     fly. topo_batch_plan() reads this from
                     frontend.forced_sign (mutable attribute, default None =
                     unchanged v1.7 behavior) so no existing call site
                     (planner.batch_plan(esdf, head, tail)) needs to change.
                     patch_planner() now stashes the frontend instance on
                     planner._topo_frontend so the node's mission loop can
                     reach it. Local-per-obstacle (not a corridor-wide split):
                     using the whole-corridor half would make asymmetric
                     single-door events (e.g. gauntlet_train E2/E5, doors on
                     opposite sides) unreachable whenever their only real
                     door lands on the forced-closed half.
  v1.7 (2026-07-08)  Fix a real regression in v1.6's bounds_for_seed(),
                     found by A/B'ing v1.6 in an actual box_narrow flight
                     (run_20260708_195555) against the v1.5 baseline
                     (run_20260708_193548): bimodal rate FELL 6.0% -> 1.5%,
                     and every recorded single-signature scene came back
                     '-1' (never '+1'). Root cause: bounds_for_seed()
                     approximated "lateral offset from the head->tail
                     line" as the RAW y-coordinate, valid only when
                     head->tail is near-horizontal -- but real flight
                     often has head_y far from tail_y (drift from prior
                     escapes/nudges), so the bound constrained the wrong
                     region and silently broke convergence for whichever
                     side it got wrong. Replaced with constraint_for_seed()
                     -- an EXACT scipy LinearConstraint on the true
                     axis-relative lateral offset (same formula
                     _signature_for_path() uses to define the signature),
                     via plan_once(constraints=...) / method='SLSQP'
                     instead of an L-BFGS-B box bound. See
                     constraint_for_seed()'s own docstring for the full
                     derivation. bounds_for_seed() is removed (no longer
                     used from any call site).
  v1.6 (2026-07-08)  Two fixes motivated by mode_sig analysis of a real
                     300-sample box_narrow run (run_20260708_193548, first
                     collection with v1.5's mode_sig labeling active):
                     6.0% genuinely bimodal scenes, but a further 7.4%
                     "unimodal-nonfree" -- the straight head->tail line WAS
                     blocked, yet only one side ever reached the dataset.
                     (1) ENUM_LOOKAHEAD_M: topo_batch_plan() now queries
                     find_paths() over an extended, purely collinear
                     lookahead instead of the real (short) tail_state --
                     the real local target (longitu_step=5m in
                     expert_planner_node.py) frequently lands before or
                     even inside the very obstacle we want the frontend to
                     see, so it read as a "free" straight line. Only the
                     ENUMERATION query is extended; seeds_from_paths()'s
                     new `arc_len` parameter truncates back down to the
                     real segment, so the actual seed/execution/recorded
                     trajectory is unaffected -- only which classes get
                     DISCOVERED changes. (2) bounds_for_seed(): plan_once()
                     now accepts optional L-BFGS-B `bounds`; each seed gets
                     bounded to its own side of the obstacle (small margin)
                     so the optimizer can no longer collapse two genuinely
                     different A* seeds onto the same final side just
                     because one side is locally cheaper -- the other root
                     cause of "found 2 classes, recorded 1 signature".
                     Both fixes are seed-local and cost/tolerance-neutral;
                     the drone's flown trajectory selection logic
                     (cheapest tol-passing solution) is unchanged.
  v1.5 (2026-07-08)  Add compute_mode_signature() / mode_signature_to_str():
                     the recorder (expert_planner_node.py) was deduplicating
                     "is this a new mode?" using a 0.5m Euclidean waypoint
                     distance -- a HEURISTIC that cannot tell "genuinely
                     different side" from "same side, drifted >0.5m" (and
                     can equally mis-merge two different sides that happen
                     to stay close near a thin gap). This exposes the exact
                     side-signature logic find_paths() already uses for its
                     OWN seed diversification, as a reusable function over
                     any final path (topo-seeded or stock-fallback), so the
                     recorder can dedup/label by topology instead of
                     distance, and the resulting CSV (`mode_sig` column) can
                     actually be audited for genuine bimodality.
  v1.4 (2026-07-08)  Stop cost-gating which alternate homotopy classes reach
                     the dataset. Root cause (measured on box_narrow via a
                     direct offline repro): find_paths() reliably proposes
                     BOTH sides with real geometric clearance (~1.0-1.4m),
                     but topo_batch_plan's own collision_cost_tol (hard
                     reject inside plan_once) and COST_WINDOW (soft filter,
                     both keyed on weighted_cost which is dominated 10000x
                     by the collision term) were silently discarding a
                     genuinely-valid alternate side because MINCO's L-BFGS-B
                     landed in a locally worse configuration on it (only 2
                     interior waypoints -> little room to both stay smooth
                     AND hug the far side as tightly as the near side).
                     plan_once() already runs the FULL optimization before
                     that check, so the discarded result is still a valid,
                     smoothed, dynamically-consistent trajectory -- just not
                     good enough to fly RIGHT NOW. Fix: catch that specific
                     rejection and keep the result as a dataset-only
                     candidate (never flown), and stop applying COST_WINDOW
                     to what reaches planner.all_solutions at all. The
                     existing downstream _check_trajectory_safe() in
                     expert_planner_node.py (ESDF distance along the
                     trajectory, not weighted cost) is the real safety gate
                     for what actually reaches the CSV now. The flown
                     solution (planner.int_wpts/ts) is unaffected -- still
                     only ever the cheapest solution that passed
                     collision_cost_tol properly, via the same fallback path
                     as before if none did.
  v1.2 (2026-07-07)  + time budget: A* diversification stops after
                     ASTAR_BUDGET_S and optimizing the remaining seeds is
                     skipped after TOTAL_BUDGET_S (>=1 seed is always
                     optimized). Reason: in a narrow pocket (R2 lower
                     corridor, run_20260707_160427) find_paths 5 rounds + 3x
                     MINCO blew through the node's replan budget (3 s) ->
                     repeated "Replan failed (budget)" -> the drone tracked
                     a stale trajectory -> escape ping-pong -> PHYSICS
                     EXPLOSION. A slow replan is more dangerous than an
                     incomplete replan.
  v1.1 (2026-07-07)  + `bounds` (x_min, x_max, y_min, y_max): cells outside
                     the bounds are blocked for A*. MUST be filled with the
                     expert's arena bounds (with an ~0.5 m inset): the ESDF
                     grid spans the full pcd extent (y +-15) while the arena
                     is only y +-6 — without this A* picks the
                     "penalty-free" route that goes AROUND THE OUTSIDE of
                     the mirror pole and the drone exits the arena (observed
                     in run_20260707_131700: repeated out-of-bounds at
                     y 6.0-6.9 near R3/R5).
  v1.0 (2026-07-06)  Initial version. A* over the expert's ESDF grid +
                     iterative corridor-penalty diversification + side-
                     signature dedup + cost-window filter. Drop-in via
                     patch_planner() — no edits needed inside
                     expert_planner_node.py beyond 2 lines in __init__.

WHY THIS FILE EXISTS (research context)
  The expert's batch_plan uses 3 hardcoded seeds (straight, ±0.6 m lateral).
  Result measured 2026-07-05/06: dataset run_20260705_172250 has only 164/5000
  (3.3%) different-side scenes, and the FM checkpoint trained on it is
  effectively UNIMODAL (two-sided candidate fraction ~3% in flight).
  Mirror-y augmentation balances sides ACROSS scenes (mirrored depth is a
  DIFFERENT observation) — it cannot give one observation two labels.
  Only the expert can: it must find BOTH detours for the SAME scene.

  This module replaces the fixed-offset seed generation with homotopy-class
  enumeration on the ground-truth ESDF (the expert runs offline — it is
  allowed to be expensive). Approach follows established topological planning
  practice (topo-PRM in RAPTOR, Zhou et al., T-RO 2021; TEB parallel-topology
  planning, Roesmann et al.; penalty-based k-diverse-paths heuristics).
  It is deliberately NOT claimed as a contribution — it is the *teacher*.
  The paper sentence it supports:
    "For data collection we replace the expert's three fixed initializations
     with a topological front-end [cit], ensuring the dataset contains every
     cost-competitive homotopy class per scene."

WHAT IT DOES per replanning event
  1. A* from head to tail on the ESDF occupancy/clearance grid
     (soft clearance-aware edge costs -> paths keep sensible margins).
  2. Accept the path, compute its SIDE SIGNATURE: the sign of the path's
     lateral offset at each interval where the straight head->tail line is
     blocked (one sign per blocked obstacle interval -> e.g. (+1,) = left).
  3. Paint an additive penalty over the accepted path's corridor and rerun
     A* -> the next path is pushed into a different homotopy class.
  4. Repeat until signatures repeat / max_iters. Deduplicate by signature.
  5. Downsample each path into MINCO seeds (init_wpts_num interior waypoints
     by arc length + length-based time allocation) and hand every seed to the
     planner via read_planning_conditions + plan_once.
  6. Keep EVERY seed that MINCO managed to optimize — including ones that
     exceeded collision_cost_tol (still fully smoothed/optimized, just not
     flyable right now) — in planner.all_solutions, uncapped by cost (see
     v1.4). The recorder's own _check_trajectory_safe() (ESDF distance) is
     the real filter for what reaches the dataset. The DRONE still flies
     the single best (cheapest, tol-passing) solution — only the DATASET
     gets richer.

INTEGRATION (2 lines in expert_planner_node.py __init__, after
self._planner and self._esdf exist; every existing batch_plan call site
— including the fallback semantics — keeps working untouched):

    from homotopy_frontend import patch_planner
    patch_planner(self._planner)            # add use_topo=False to disable

  To A/B against the stock expert, launch-param it:
    self.declare_parameter("use_topo_frontend", True)
    if bool(self.get_parameter("use_topo_frontend").value):
        patch_planner(self._planner)

FAIRNESS NOTE (methodology decision 2026-07-05 still holds)
  Only the TEACHER changes. Deployment planners (NEO 3-seed baseline,
  nn_inference_node, fm_inference_node) are untouched; NN and FM both train
  on the same new dataset. Official comparison stays "only swap the model".

SELF-TEST (no ROS needed):
    python3 homotopy_frontend.py
  Builds a synthetic ESDF with one 2.6 m wide box dead ahead and asserts the
  front-end finds BOTH side classes (left & right) — the exact capability the
  3-seed expert lacks for obstacles wider than ~1.2 m.
"""
import heapq
import logging
import math
import time

import numpy as np

_LOG = logging.getLogger("homotopy_frontend")


def _log_fallback(frontend, forced_sign, reason):
    """v1.12: record + surface a drop to the stock 3-seed fallback. A fallback
    during a FORCED leg is the pathology the lap-alternation audit could not
    see (it was silent): warn every time so it is grep-able in the launch log
    ('[homotopy] FALLBACK'). Non-forced fallbacks are ordinary (grid warming
    up, degenerate scene) -> count silently, log only the first + every 50th
    to avoid spamming a normal run. Counters live on the frontend instance so
    a run can also be summarized post-hoc."""
    if frontend is None:
        return
    frontend.fallback_count = getattr(frontend, "fallback_count", 0) + 1
    if forced_sign is not None:
        frontend.fallback_forced_count = getattr(
            frontend, "fallback_forced_count", 0) + 1
        _LOG.warning(
            "[homotopy] FALLBACK during FORCED leg (sign=%+d): %s -> stock "
            "3-seed on biased ESDF (wrong side repelled, not free). "
            "forced-fallbacks so far: %d",
            forced_sign, reason, frontend.fallback_forced_count)
    elif (frontend.fallback_count == 1
          or frontend.fallback_count % 50 == 0):
        _LOG.info("[homotopy] FALLBACK (%s), total: %d",
                  reason, frontend.fallback_count)

# ── Tunables ────────────────────────────────────────────────────────────────
MAX_CLASSES        = 3      # keep at most this many distinct homotopy classes
MAX_SEARCH_ITERS   = 5      # A* diversification rounds (>= MAX_CLASSES)
SOFT_CLEARANCE     = 0.6    # m — below this, edge cost grows (path keeps margin)
CLEARANCE_WEIGHT   = 3.0    # strength of the soft-clearance edge cost
HARD_CLEARANCE     = 0.15   # m — cells with edt below this are untraversable
CORRIDOR_RADIUS    = 0.8    # m — penalty brush radius around an accepted path
CORRIDOR_PENALTY   = 6.0    # additive per-cell penalty (in "meters" of cost)
BLOCK_THRESH       = 0.35   # m — straight-line edt below this marks a blocked
                            #     interval (used for the side signature)
MIN_SIG_GAP_M      = 0.8    # merge blocked intervals closer than this
WALL_X_MARGIN      = 1.0    # m — v1.3 hard wall-off: x-padding around a
                            #     blocked interval when hard-blocking the
                            #     already-explored side (see find_paths)
PATH_SPEED_FRAC    = 0.7    # seed time allocation assumes 0.7 * v_max cruise
GOAL_TOL_CELLS     = 1      # A* goal tolerance (cells)
ASTAR_BUDGET_S     = 0.7    # s — stop the A* diversification round after this
TOTAL_BUDGET_S     = 1.5    # s — skip optimizing the remaining seeds after
                            # this. Must be well below the node's replan
                            # budget (3.0 s): a single hard plan_once MINCO
                            # call can take 1-4 s and CANNOT be cut off
                            # mid-way — leave headroom (lesson from
                            # explosion #2, R6 return).
ENUM_LOOKAHEAD_M   = 9.0    # m — v1.6: minimum distance for the A* frontend
                            # QUERY (not the execution tail_state). Measured
                            # directly on box_narrow (run_20260708_193548,
                            # 300 samples): 7.4% "unimodal-nonfree" scenes
                            # (the straight head->tail line WAS blocked but
                            # only one side got recorded) — the root cause is
                            # PARTLY that the tail_state (longitu_step=5m in
                            # expert_planner_node.py) often lands BEFORE or
                            # even INSIDE the obstacle, so find_paths() sees
                            # the straight line as already "free" even though
                            # the drone hasn't passed the obstacle yet.
                            # topo_batch_plan() queries A* to an extended
                            # point (collinear, same direction) so
                            # find_paths() sees the obstacle & both its
                            # sides; seeds_from_paths(arc_len=...) truncates
                            # it back down to the ORIGINAL tail_state distance
                            # before it becomes a MINCO seed — the drone
                            # still only ever flies the same short segment,
                            # only the ENUMERATION is extended.
# v1.13 (2026-07-14) — CONTINUATION SEED (anti "late turn").
# Measured problem (offline harness, real planner+frontend, box 1.6-5.6 m):
# every replan builds a plan from SCRATCH, and min-jerk always places the
# turn in the MIDDLE of the segment (lowest energy). What actually gets
# flown each cycle is only the FIRST ~1 m -- which is still straight. The
# next replan starts over from scratch -> the turn gets pushed "later"
# again, over and over. Result: the drone only starts deviating |y|>=0.3 m
# at 0.7-1.7 m from the obstacle, by which point the obstacle already
# covers ~100% of the HFOV (= the complaint "it keeps going forward until
# vision is blocked, then turns"), then replan fails -> escape.
# Fix: offer the remaining trajectory that is CURRENTLY being flown as an
# ADDITIONAL seed (not a replacement) on every replan; if its optimized
# result is still reasonable (<= CONTINUITY_COST_FACTOR x the cheapest),
# that's what gets flown. Effect: the turn STAYS WHERE IT IS in world
# coordinates, so it actually gets executed. Measured offline: divert
# 2.7-3.75 m (36-81% HFOV) vs 0.7-1.7 m (100%) -- without changing the
# local target (the scene stays ambiguous -> the label stays multimodal)
# and without changing the replan period.
CONTINUITY_COST_FACTOR = 2.5
# Note on safe_dis (TESTED, NOT adopted): raising safe_dis 0.5 -> 1.0
# (+ collision_cost_tol 10 -> 100) adds even more anticipation (divert
# 1.76 -> 2.74 m on a 2.6 m box), BUT on the inflated ESDF (0.25 m, as in
# the real system) it BLOCKS threading a 2.0 m gap: edt at the gate center
# 0.70 < safe_dis 1.0 -> the whole gate gets penalized -> 4x replan
# failures -> the drone refuses to enter (exactly the "spinning in place
# at the R5 gate" bug, CLAUDE.md). safe_dis STAYS 0.5. If pursuing this
# direction further, safe_dis needs to be ADAPTIVE (bounded by the local
# corridor width), not a global constant.

_SQRT2 = math.sqrt(2.0)
_NBRS = [(-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2),
         (0, -1, 1.0),                   (0, 1, 1.0),
         (1, -1, _SQRT2),  (1, 0, 1.0),  (1, 1, _SQRT2)]


class HomotopyFrontEnd:
    """Enumerates cost-competitive homotopy classes on an ESDF grid and turns
    each into a MINCO warm-start seed (int_wpts, ts)."""

    def __init__(self,
                 max_classes=MAX_CLASSES,
                 max_iters=MAX_SEARCH_ITERS,
                 soft_clearance=SOFT_CLEARANCE,
                 clearance_weight=CLEARANCE_WEIGHT,
                 hard_clearance=HARD_CLEARANCE,
                 corridor_radius=CORRIDOR_RADIUS,
                 corridor_penalty=CORRIDOR_PENALTY,
                 bounds=None):
        self.max_classes      = int(max_classes)
        self.max_iters        = int(max_iters)
        self.soft_clearance   = float(soft_clearance)
        self.clearance_weight = float(clearance_weight)
        self.hard_clearance   = float(hard_clearance)
        self.corridor_radius  = float(corridor_radius)
        self.corridor_penalty = float(corridor_penalty)
        # (x_min, x_max, y_min, y_max) world — cells outside are blocked for A*.
        # None = no bounds (self-test). Fill with the expert's arena MINUS an
        # inset ~0.5 m so the MINCO trajectory/tracking error doesn't touch the bounds.
        self.bounds = tuple(float(b) for b in bounds) if bounds else None
        # v1.8: lap-level forced side, set/cleared by the mission loop
        # (expert_planner_node.py) between legs. None = unchanged v1.7
        # behavior (free enumeration). +-1 = every find_paths() call this
        # frontend serves only ever returns that side's class.
        self.forced_sign = None
        # v1.12 diagnostics (2026-07-10): count how often topo_batch_plan had
        # to drop to the stock 3-seed fallback, split by whether a forced_sign
        # was active at the time. The lap-alternation audit (2026-07-09) could
        # not see this at all -- the fallback was silent -- so ~40% wrong-side
        # forced-leg samples had no traceable origin. These make the leak
        # observable (logged live + inspectable post-run).
        self.fallback_count        = 0   # total fallbacks
        self.fallback_forced_count = 0   # fallbacks while forced_sign active

    # ── public API ──────────────────────────────────────────────────────────

    def find_paths(self, esdf, start_xy, goal_xy, forced_sign=None):
        """-> list of (path_xy [N,2] world coords, signature tuple).
        One entry per distinct homotopy class, best-first. Empty list if the
        grid is unusable or no path exists.

        `forced_sign` (v1.8, +-1 or None): if given, wall off the OPPOSITE
        side of every straight-line blocked interval before diversification
        starts, so this call can only ever discover `forced_sign`'s class.
        Local per-obstacle (same x-window as _wall_off_side), not a
        corridor-wide split -- see v1.8 CHANGELOG note for why."""
        grid = getattr(esdf, "esdf_map", None)
        if grid is None or getattr(esdf, "map_origin", None) is None:
            return []
        res = float(esdf.map_resolution)
        ox  = float(esdf.map_origin.x)
        oy  = float(esdf.map_origin.y)
        edt = np.asarray(grid, dtype=np.float64)          # (rows, cols), meters
        rows, cols = edt.shape

        def to_cell(p):
            r = int(round((p[1] - oy) / res))
            c = int(round((p[0] - ox) / res))
            return (min(max(r, 0), rows - 1), min(max(c, 0), cols - 1))

        def to_world(rc):
            return np.array([ox + rc[1] * res, oy + rc[0] * res])

        start = to_cell(start_xy)
        goal  = to_cell(goal_xy)

        # Base traversal cost field: 0 in open space, grows inside the soft
        # margin. hard-blocked cells get +inf — EXCEPT we never block the
        # start/goal cells themselves (the drone may legitimately be inside
        # the soft margin when replanning; mirrors the node's adaptive
        # safe_dis logic).
        soft = np.maximum(0.0, self.soft_clearance - edt) / self.soft_clearance
        base = self.clearance_weight * soft
        blocked = edt < self.hard_clearance
        if self.bounds is not None:
            bx0, bx1, by0, by1 = self.bounds
            ys_ax = oy + np.arange(rows) * res
            xs_ax = ox + np.arange(cols) * res
            blocked |= ((ys_ax < by0) | (ys_ax > by1))[:, None]
            blocked |= ((xs_ax < bx0) | (xs_ax > bx1))[None, :]
        blocked[start] = False
        blocked[goal]  = False

        # Blocked-interval detection depends only on the STRAIGHT line +
        # edt, not on any particular path -> compute once, reuse every round.
        axis, L, fr, groups = self._blocked_intervals(
            edt, res, ox, oy, start_xy, goal_xy)

        # v1.8: lap-level forcing -- wall off the side NOT wanted this lap,
        # around every obstacle actually on the straight line, before any
        # A* round runs. groups==[] (straight line already free) -> nothing
        # to force, forced_sign is simply moot for this scene.
        if forced_sign is not None and groups:
            for g in groups:
                self._wall_off_side(blocked, res, ox, oy, start_xy, axis,
                                    fr, L, g, sign=-forced_sign)
            blocked[start] = False
            blocked[goal]  = False

        extra = np.zeros_like(edt)                        # corridor penalties
        found, sigs = [], set()
        t0 = time.monotonic()
        for _ in range(self.max_iters):
            path_rc = self._astar(base, extra, blocked, start, goal, res)
            if path_rc is None:
                break
            path_xy = np.array([to_world(rc) for rc in path_rc])
            sig = self._signature_for_path(axis, fr, groups, start_xy, path_xy)
            if sig not in sigs:
                sigs.add(sig)
                found.append((path_xy, sig))
                if len(found) >= self.max_classes:
                    break
            # v1.3 (2026-07-08): HARD-block the side just taken, not just a
            # soft corridor penalty. Root cause fixed: for an obstacle in
            # open space (no confining wall/pole nearby), a soft penalty
            # along the accepted path only discourages reusing that EXACT
            # corridor — drifting a bit further out on the SAME side stays
            # cheaper than actually crossing to the other side, so the
            # search never finds the second class (measured: 0/100 samples
            # multimodal in a from-scratch calibration run on an isolated
            # box). Hard-blocking the whole half-plane on that side, in a
            # local x-window around the obstacle, makes the same side
            # physically unusable next round -> forces a genuinely
            # different homotopy class instead of a wider lateral shift.
            for g, s in zip(groups, sig):
                self._wall_off_side(blocked, res, ox, oy, start_xy,
                                    axis, fr, L, g, s)
            blocked[start] = False
            blocked[goal]  = False
            # Soft penalty kept as a secondary tie-breaker (helps separate
            # further sub-variants once MAX_CLASSES > 2 on the SAME side).
            self._paint_corridor(extra, path_rc, res)
            # v1.2: a slow replan is more dangerous than an incomplete one —
            # the classes already found are still used, the rest are skipped.
            if time.monotonic() - t0 > ASTAR_BUDGET_S:
                break
        return found

    def seeds_from_paths(self, paths, head_state, tail_state, planner,
                         arc_len=None):
        """-> list of (int_wpts (D, n), ts (n+1,), sig) MINCO seeds, one per
        path. Waypoints by arc length at fractions i/(n+1) of `arc_len` (or
        the full path length if `arc_len` is None); time allocation from
        that same span at PATH_SPEED_FRAC * v_max, clipped to [T_min, T_max].

        `arc_len` (v1.6): lets the caller enumerate homotopy classes over a
        path that runs FURTHER than the real head->tail segment (see
        topo_batch_plan's ENUM_LOOKAHEAD_M) while still producing seeds
        sized for the real, shorter segment MINCO will actually optimize --
        pass the real head->tail distance here so waypoints/timing stay on
        that segment, using only the SHAPE (which side) from the longer
        path. None reproduces the pre-v1.6 behavior exactly (span = full
        path length)."""
        n = int(planner.init_wpts_num)
        seeds = []
        for path_xy, sig in paths:
            seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            total = float(cum[-1])
            if total < 1e-6:
                continue
            span = min(float(arc_len), total) if arc_len is not None else total
            if span < 1e-6:
                continue
            wpts = []
            for i in range(1, n + 1):
                s = span * i / (n + 1)
                j = int(np.searchsorted(cum, s, side="right")) - 1
                j = min(max(j, 0), len(seg) - 1)
                a = (s - cum[j]) / max(seg[j], 1e-9)
                wpts.append(path_xy[j] * (1 - a) + path_xy[j + 1] * a)
            int_wpts = np.array(wpts).T                    # (D, n)
            v = max(PATH_SPEED_FRAC * float(planner.v_max), 1e-3)
            t_piece = (span / v) / (n + 1)
            t_piece = float(np.clip(t_piece,
                                    planner.T_min + 1e-3,
                                    planner.T_max - 1e-3))
            ts = t_piece * np.ones(n + 1)
            ts[0]  = min(ts[0] * 1.5, planner.T_max - 1e-3)
            ts[-1] = min(ts[-1] * 1.5, planner.T_max - 1e-3)
            seeds.append((int_wpts, ts, sig))
        return seeds

    # ── internals ───────────────────────────────────────────────────────────

    def _astar(self, base, extra, blocked, start, goal, res):
        rows, cols = base.shape
        gscore = np.full(base.shape, np.inf)
        gscore[start] = 0.0
        came = {}
        h0 = math.hypot(goal[0] - start[0], goal[1] - start[1])
        heap = [(h0, start)]
        closed = np.zeros(base.shape, dtype=bool)
        while heap:
            _, cur = heapq.heappop(heap)
            if closed[cur]:
                continue
            closed[cur] = True
            if (abs(cur[0] - goal[0]) <= GOAL_TOL_CELLS
                    and abs(cur[1] - goal[1]) <= GOAL_TOL_CELLS):
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            for dr, dc, step in _NBRS:
                r, c = cur[0] + dr, cur[1] + dc
                if r < 0 or r >= rows or c < 0 or c >= cols:
                    continue
                if blocked[r, c] or closed[r, c]:
                    continue
                cost = step * res * (1.0 + base[r, c] + extra[r, c])
                ng = gscore[cur] + cost
                if ng < gscore[r, c]:
                    gscore[r, c] = ng
                    came[(r, c)] = cur
                    h = math.hypot(goal[0] - r, goal[1] - c) * res
                    heapq.heappush(heap, (ng + h, (r, c)))
        return None

    def _paint_corridor(self, extra, path_rc, res):
        rad = max(1, int(round(self.corridor_radius / res)))
        rows, cols = extra.shape
        for (r, c) in path_rc:
            r0, r1 = max(0, r - rad), min(rows, r + rad + 1)
            c0, c1 = max(0, c - rad), min(cols, c + rad + 1)
            extra[r0:r1, c0:c1] = np.maximum(
                extra[r0:r1, c0:c1], self.corridor_penalty)

    @staticmethod
    def _blocked_intervals(edt, res, ox, oy, start_xy, goal_xy):
        """Sample the STRAIGHT start->goal line; return (axis_unit, L, fr,
        groups) where `groups` is a list of index arrays into `fr` — one per
        contiguous run where edt < BLOCK_THRESH (merged if the gap between
        runs is < MIN_SIG_GAP_M). This depends only on the straight line +
        edt (NOT on any particular path), so it is computed ONCE per
        find_paths() call and reused both for the per-path side signature
        and to know exactly where to wall off the already-explored side."""
        axis = np.asarray(goal_xy, dtype=float) - np.asarray(start_xy, float)
        L = float(np.linalg.norm(axis))
        if L < 1e-6:
            return axis, L, None, []
        axis = axis / L
        n_samp = max(int(L / res), 2)
        fr = np.linspace(0.0, 1.0, n_samp)
        pts = np.asarray(start_xy) + fr[:, None] * axis[None, :] * L
        rows, cols = edt.shape
        rr = np.clip(np.round((pts[:, 1] - oy) / res).astype(int), 0, rows - 1)
        cc = np.clip(np.round((pts[:, 0] - ox) / res).astype(int), 0, cols - 1)
        is_blocked = edt[rr, cc] < BLOCK_THRESH
        if not is_blocked.any():
            return axis, L, fr, []
        # merge blocked runs closer than MIN_SIG_GAP_M into one interval
        idx = np.where(is_blocked)[0]
        gaps = np.where(np.diff(idx) * (L / n_samp) > MIN_SIG_GAP_M)[0]
        groups = np.split(idx, gaps + 1)
        return axis, L, fr, groups

    @staticmethod
    def _signature_for_path(axis, fr, groups, start_xy, path_xy):
        """Side signature: for each blocked interval (from
        _blocked_intervals), the sign of `path_xy`'s lateral offset at that
        interval's arc-fraction midpoint. ('free',) if there are no blocked
        intervals (straight line already clear -> single trivial class)."""
        if not groups:
            return ("free",)
        seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = max(float(cum[-1]), 1e-9)
        signs = []
        for g in groups:
            f_mid = float(fr[g].mean())
            s = total * f_mid
            j = int(np.searchsorted(cum, s, side="right")) - 1
            j = min(max(j, 0), len(seg) - 1)
            a = (s - cum[j]) / max(seg[j], 1e-9)
            p = path_xy[j] * (1 - a) + path_xy[j + 1] * a
            rel = p - np.asarray(start_xy)
            lat = axis[0] * rel[1] - axis[1] * rel[0]   # + = left of axis
            signs.append(1 if lat >= 0.0 else -1)
        return tuple(signs)

    def _wall_off_side(self, blocked, res, ox, oy, start_xy, axis, fr, L,
                       group, sign, x_margin=WALL_X_MARGIN):
        """Hard-block the region on `sign`'s side of this blocked interval,
        in a local x-window around it (+-x_margin along the flight axis).
        See the v1.3 note at the find_paths() call site for why this
        replaces relying on the soft corridor-penalty alone."""
        f0, f1 = float(fr[group[0]]), float(fr[group[-1]])
        p0 = np.asarray(start_xy) + f0 * axis * L
        p1 = np.asarray(start_xy) + f1 * axis * L
        x_lo, x_hi = sorted([p0[0], p1[0]])
        x_lo -= x_margin
        x_hi += x_margin
        rows, cols = blocked.shape
        ys_ax = oy + np.arange(rows) * res
        xs_ax = ox + np.arange(cols) * res
        in_band = (xs_ax >= x_lo) & (xs_ax <= x_hi)
        Y, X = np.meshgrid(ys_ax, xs_ax, indexing="ij")
        lat = axis[0] * (Y - start_xy[1]) - axis[1] * (X - start_xy[0])
        same_side = (lat * sign) > 0.0
        blocked |= same_side & in_band[None, :]


def compute_mode_signature(esdf, head_xy, tail_xy, path_xy):
    """Topological side-signature of an arbitrary FINAL path (e.g. a MINCO
    trajectory's [head, interior waypoints..., tail] polyline), against the
    same head->tail blocked-interval definition find_paths() uses to build
    signatures for its A* seeds.

    This is the single source of truth for "is this the same mode as that
    other solution" (see v1.5 CHANGELOG / expert_planner_node.py recorder):
    two solutions are the same mode iff this signature is equal, regardless
    of how far apart their waypoints land in meters. Works whether the
    seed came from the topo front-end OR the stock 3-seed fallback, since
    it only looks at the resulting path, not how it was generated.

    Returns:
      - a tuple of +-1 (one per blocked obstacle interval on the straight
        head->tail line), or ("free",) if that line has no blocked interval
      - None if the ESDF grid is not ready/usable
    """
    grid = getattr(esdf, "esdf_map", None)
    if grid is None or getattr(esdf, "map_origin", None) is None:
        return None
    res = float(esdf.map_resolution)
    ox  = float(esdf.map_origin.x)
    oy  = float(esdf.map_origin.y)
    edt = np.asarray(grid, dtype=np.float64)
    head_xy = np.asarray(head_xy, dtype=float)
    tail_xy = np.asarray(tail_xy, dtype=float)
    path_xy = np.asarray(path_xy, dtype=float)
    axis, L, fr, groups = HomotopyFrontEnd._blocked_intervals(
        edt, res, ox, oy, head_xy, tail_xy)
    if fr is None:
        return None
    return HomotopyFrontEnd._signature_for_path(axis, fr, groups, head_xy, path_xy)


def mode_signature_to_str(sig):
    """CSV-friendly encoding of a compute_mode_signature() result."""
    if sig is None:
        return "nogrid"
    if sig == ("free",):
        return "free"
    return "_".join(str(s) for s in sig)


class _BiasedESDF:
    """v1.10 (2026-07-09) — read-only wrapper around a real ESDF: same
    query interface (get_edt_dis/get_edt_grad/is_ready/map_resolution/
    map_origin/esdf_map), but with distance zeroed (= solid obstacle) on
    the side OPPOSITE `forced_sign`, in the same local x-window around
    each head->tail blocked interval that _wall_off_side() already uses
    for A*.

    WHY THIS EXISTS: forced_sign (v1.8) only walls off the A* grid used to
    propose a SEED -- it does not touch the real ESDF that MinJerkPlanner's
    collision cost (self.map.get_edt_dis/get_edt_grad) samples from during
    plan_once(). Measured on real flight (box_narrow, force_lap_alternation
    pilot, 2026-07-09, run_20260709_101942): with v1.9's plan_once()
    unconstrained, 3/3 non-free samples in the forced legs recorded the
    WRONG mode_sig (100% violation) -- the seed started on the forced side
    but the optimizer, seeing the unbiased real ESDF, drifted straight back
    to the cheaper side, exactly the v1.6 root cause ("nothing in
    unconstrained L-BFGS-B stops a waypoint from sliding across the axis").
    v1.7's fix for that was a LinearConstraint + SLSQP; v1.9 reverted it
    (data showed it wasn't worth its fragility for NORMAL per-scene
    diversification). This is a narrower, L-BFGS-B-compatible fix scoped
    ONLY to the deliberate forced_sign case: bias the actual cost surface
    instead of constraining the optimizer, so plain unconstrained L-BFGS-B
    (see homotopy_frontend.py CHANGELOG v1.9 / min_jerk_planner.py
    plan_once()) is physically repelled from the wrong side, the same way
    it already avoids any other obstacle. Reuses the exact wall-off
    geometry _wall_off_side() defines (same WALL_X_MARGIN, same
    lateral-offset formula compute_mode_signature() itself is defined by)
    so "the side MINCO is pushed away from" and "the side mode_sig calls a
    violation" are provably the same side.

    Only the OPTIMIZATION sees this biased grid (passed as `map` to
    planner.read_planning_conditions). Seed search (find_paths) and the
    recorder's own compute_mode_signature()/_check_trajectory_safe() keep
    using the REAL esdf, so what gets labeled/accepted into the dataset is
    always judged against the true environment, never the artificial wall.
    """

    def __init__(self, base_esdf, forced_sign, head_xy, tail_xy):
        self._base           = base_esdf
        self.map_resolution  = base_esdf.map_resolution
        self.map_origin      = base_esdf.map_origin
        self.esdf_map        = np.asarray(base_esdf.esdf_map, dtype=np.float64).copy()

        res = self.map_resolution
        ox, oy = float(self.map_origin.x), float(self.map_origin.y)
        head_xy = np.asarray(head_xy, dtype=float)
        tail_xy = np.asarray(tail_xy, dtype=float)
        axis, L, fr, groups = HomotopyFrontEnd._blocked_intervals(
            self.esdf_map, res, ox, oy, head_xy, tail_xy)

        if groups:
            rows, cols = self.esdf_map.shape
            ys_ax = oy + np.arange(rows) * res
            xs_ax = ox + np.arange(cols) * res
            Y, X  = np.meshgrid(ys_ax, xs_ax, indexing="ij")
            lat   = axis[0] * (Y - head_xy[1]) - axis[1] * (X - head_xy[0])
            wrong_side = (lat * (-float(forced_sign))) > 0.0
            for g in groups:
                f0, f1 = float(fr[g[0]]), float(fr[g[-1]])
                p0 = head_xy + f0 * axis * L
                p1 = head_xy + f1 * axis * L
                x_lo, x_hi = sorted([p0[0], p1[0]])
                x_lo -= WALL_X_MARGIN
                x_hi += WALL_X_MARGIN
                in_band = (xs_ax >= x_lo) & (xs_ax <= x_hi)
                self.esdf_map[wrong_side & in_band[None, :]] = 0.0

        # Recompute the gradient from the BIASED grid (not the base one) so
        # the collision gradient near the new wall points the right way —
        # same convention as esdf_ros2.py's own ESDF (spacing=map_resolution
        # so |grad| is per-meter, not per-cell).
        self.esdf_grad_y, self.esdf_grad_x = np.gradient(self.esdf_map, res)

    def is_ready(self):
        return self._base.is_ready()

    def _world_to_index(self, pos):
        row = int((pos[1] - self.map_origin.y) / self.map_resolution)
        col = int((pos[0] - self.map_origin.x) / self.map_resolution)
        return row, col

    def _in_bounds(self, row, col):
        rows, cols = self.esdf_map.shape
        return 0 <= row < rows and 0 <= col < cols

    def get_edt_dis(self, pos) -> float:
        row, col = self._world_to_index(pos)
        if not self._in_bounds(row, col):
            return 10000.0
        return float(self.esdf_map[row, col])

    def get_edt_grad(self, pos):
        row, col = self._world_to_index(pos)
        if not self._in_bounds(row, col):
            return [0.0, 0.0]
        return [float(self.esdf_grad_x[row, col]),
                float(self.esdf_grad_y[row, col])]


def topo_batch_plan(planner, esdf, head_state, tail_state,
                    frontend=None, fallback=None):
    """Drop-in replacement for MinJerkPlanner.batch_plan (same signature after
    binding `planner`): topological seeds -> optimize each -> every result
    (including collision_cost_tol-rejected ones, see v1.4) goes into
    planner.all_solutions; the cheapest tol-passing one goes to
    planner.int_wpts / planner.ts / planner.final_cost (the drone flies it).
    Falls back to the stock 3-seed batch_plan when the grid is unusable, no
    path is found, or nothing passes collision_cost_tol."""
    frontend = frontend or HomotopyFrontEnd()
    fallback = fallback or planner.batch_plan

    t0 = time.monotonic()
    head_xy = np.asarray(head_state[0], float)
    tail_xy = np.asarray(tail_state[0], float)
    real_dist = float(np.linalg.norm(tail_xy - head_xy))
    # v1.6: query the A* frontend over an EXTENDED lookahead so it can see
    # an obstacle the real (short, safety-tuned) tail_state lands before or
    # inside of -- see ENUM_LOOKAHEAD_M. Purely collinear extension (same
    # direction, longer distance), so the side-signature convention is
    # unaffected. seeds_from_paths(arc_len=real_dist) truncates back down
    # to the real segment for the actual MINCO seed/execution.
    if 1e-6 < real_dist < ENUM_LOOKAHEAD_M:
        enum_xy = head_xy + (tail_xy - head_xy) * (ENUM_LOOKAHEAD_M / real_dist)
    else:
        enum_xy = tail_xy
    # v1.8: mutable attribute, not a call-site arg -- lets the mission loop
    # force a side for a whole lap without touching planner.batch_plan(...)
    # call sites (still the stock 3-arg signature everywhere).
    forced_sign = getattr(frontend, "forced_sign", None)

    # v1.10: forced_sign only steered the A* SEED via find_paths() below --
    # plan_once() is unconstrained (v1.9), so nothing stops MINCO drifting
    # the seed back across the axis if that's cheaper (measured on real
    # flight: 100% of forced-leg non-free samples came back the WRONG side,
    # see _BiasedESDF docstring). Bias the REAL optimization surface too,
    # over the actual (short) head->tail segment MINCO will sample -- not
    # the extended enum_xy lookahead, which is only for seed shaping.
    # v1.12 (2026-07-10): built HERE, before the seed check, so the two
    # fallbacks below can ALSO hand it to the stock 3-seed batch_plan. The
    # lap-alternation audit (2026-07-09, 12 runs) traced ~40% of forced-leg
    # wrong-side samples to exactly this fallback: A* finds no forced-side
    # path in a dense scene -> empty seeds -> stock batch_plan used to run on
    # the RAW esdf with zero forced_sign awareness and flew the cheaper
    # (wrong) side, silently. Passing opt_esdf makes the wrong side a solid
    # obstacle for the fallback's collision cost too, so it is repelled the
    # same way -- or fails outright (better than a mislabeled sample; the
    # recorder still judges the flown result against the REAL esdf). When
    # forced_sign is None, opt_esdf IS esdf -> fallback behavior unchanged.
    opt_esdf = (_BiasedESDF(esdf, forced_sign, head_xy, tail_xy)
               if forced_sign is not None else esdf)

    paths = frontend.find_paths(esdf, head_xy, enum_xy, forced_sign=forced_sign)
    seeds = frontend.seeds_from_paths(paths, head_state, tail_state, planner,
                                      arc_len=real_dist)
    if not seeds:
        _log_fallback(frontend, forced_sign, "no topo seed")
        return fallback(opt_esdf, head_state, tail_state)

    # v1.13: CONTINUATION seed (the remaining trajectory currently being
    # flown; set every replan by expert_planner_node._set_anchor_seed). Put
    # FIRST so it always gets a turn to be optimized even if the time budget
    # runs out -- ">=1 seed always optimized" (v1.2) then guarantees
    # continuity instead of a random A* seed. The A* enumeration is STILL
    # optimized afterward (the multimodal label set is not reduced). See
    # CONTINUITY_COST_FACTOR.
    anchor = getattr(frontend, "anchor_seed", None)
    anchor_i = None
    if anchor is not None:
        a_w, a_ts = anchor
        a_w = np.asarray(a_w, float)
        a_ts = np.asarray(a_ts, float)
        if a_w.shape == seeds[0][0].shape and a_ts.shape == seeds[0][1].shape:
            seeds = [(a_w, a_ts, None)] + list(seeds)
            anchor_i = 0

    solutions    = []   # passed collision_cost_tol -> flyable
    dataset_only = []   # optimized/smoothed but over collision_cost_tol ->
                        # never flown, kept only as an extra multimodal
                        # label (v1.4 — see CHANGELOG); _check_trajectory_safe()
                        # downstream is the real gate for these.
    anchor_sol = None   # v1.13: the optimized solution from the continuation seed
    for i, (int_wpts, ts, _seed_sig) in enumerate(seeds):
        # v1.2: >=1 seed is always optimized; the rest are skipped once the
        # total elapsed time exceeds TOTAL_BUDGET_S (keep replan < node budget).
        if solutions and time.monotonic() - t0 > TOTAL_BUDGET_S:
            break
        try:
            planner.read_planning_conditions(
                opt_esdf, head_state, tail_state, int_wpts, ts.copy())
            # v1.9 (2026-07-08): unconstrained again -- see CHANGELOG for why
            # the v1.6/v1.7 per-seed LinearConstraint was reverted. v1.10:
            # `opt_esdf` carries the forced_sign wall (if active) into the
            # actual collision cost, not just the A* seed -- see above.
            planner.plan_once()
            sol = (planner.int_wpts.copy(), planner.ts.copy(),
                   float(planner.weighted_cost.sum()))
            solutions.append(sol)
            if i == anchor_i:
                anchor_sol = sol
        except ValueError:
            # collision_cost_tol rejection: plan_once() already completed the
            # full MINCO optimization before this check (see plan_once()),
            # so int_wpts/ts is still a smoothed, dynamically-consistent
            # result — just not safe enough to EXECUTE right now.
            dataset_only.append((planner.int_wpts.copy(), planner.ts.copy(),
                                 float(planner.weighted_cost.sum())))
        except Exception:
            continue
    if not solutions:
        _log_fallback(frontend, forced_sign, "no tol-passing solution")
        return fallback(opt_esdf, head_state, tail_state)

    solutions.sort(key=lambda s: s[2])
    best = solutions[0]

    # v1.13: if continuing the trajectory currently being flown is still
    # REASONABLE (cost <= CONTINUITY_COST_FACTOR x the cheapest), fly THAT
    # instead of the cheapest fresh plan. Without this, every replan
    # re-picks the min-jerk "turn later" solution (always slightly cheaper
    # because it postpones the maneuver), so the turn never actually gets
    # executed. The 1.5x threshold still lets the drone switch plans if
    # continuing becomes much more expensive (e.g. the map changed / the
    # other side is clearly better).
    if (anchor_sol is not None
            and anchor_sol[2] <= max(best[2], 1e-6) * CONTINUITY_COST_FACTOR):
        best = anchor_sol

    # v1.4: no cost-window filtering on what reaches the dataset — every
    # converged solution AND every collision_cost_tol-rejected-but-optimized
    # one is offered to the recorder as-is. The drone still only ever flies
    # `best` (cheapest solution that passed collision_cost_tol).
    planner.all_solutions = solutions + dataset_only
    planner.int_wpts   = best[0].copy()
    planner.ts         = best[1].copy()
    planner.final_cost = best[2]
    planner.get_coeffs(planner.int_wpts, planner.ts)


def patch_planner(planner, frontend=None, bounds=None):
    """Monkey-patch planner.batch_plan -> topo_batch_plan (stock batch_plan is
    kept as the fallback). All existing call sites in expert_planner_node.py
    keep working untouched. Returns the unpatch function.
    `bounds` (x_min, x_max, y_min, y_max): arena bounds for A* (v1.1) —
    only used when `frontend` is not given."""
    stock = planner.batch_plan
    fe = frontend or HomotopyFrontEnd(bounds=bounds)
    # v1.8: expose the frontend instance so the mission loop can set
    # planner._topo_frontend.forced_sign = +-1/None between legs.
    planner._topo_frontend = fe

    def _patched(esdf, head_state, tail_state):
        return topo_batch_plan(planner, esdf, head_state, tail_state,
                               frontend=fe, fallback=stock)

    planner.batch_plan = _patched

    def unpatch():
        planner.batch_plan = stock
    return unpatch


# ── Self-test: 2.6 m box dead ahead -> must find BOTH side classes ──────────
if __name__ == "__main__":
    try:
        from scipy import ndimage
    except ImportError:
        raise SystemExit("self-test needs scipy (pip install scipy)")

    class _Origin:
        x, y = -2.0, -6.0

    class _MockESDF:
        map_resolution = 0.1
        map_origin = _Origin()

        def __init__(self):
            rows = int(12.0 / self.map_resolution)   # y in [-6, 6]
            cols = int(26.0 / self.map_resolution)   # x in [-2, 24]
            occ = np.zeros((rows, cols), bool)
            # 2.6 m wide, 0.4 m deep box centered at (8, 0) — wider than the
            # 3-seed expert's ~1.2 m ceiling.
            x0, x1 = 7.8, 8.2
            y0, y1 = -1.3, 1.3
            c0 = int((x0 - self.map_origin.x) / self.map_resolution)
            c1 = int((x1 - self.map_origin.x) / self.map_resolution)
            r0 = int((y0 - self.map_origin.y) / self.map_resolution)
            r1 = int((y1 - self.map_origin.y) / self.map_resolution)
            occ[r0:r1, c0:c1] = True
            self.esdf_map = (ndimage.distance_transform_edt(~occ)
                             * self.map_resolution)

    esdf = _MockESDF()
    fe = HomotopyFrontEnd()
    paths = fe.find_paths(esdf, np.array([0.0, 0.0]), np.array([16.0, 0.0]))
    print(f"[self-test] classes found: {len(paths)} "
          f"signatures: {[sig for _, sig in paths]}")
    sides = {sig[0] for _, sig in paths if sig != ('free',)}
    assert {+1, -1} <= sides, (
        f"FAILED: expected both left (+1) and right (-1) classes, got {sides}")

    class _MockPlanner:
        init_wpts_num, v_max, T_min, T_max = 2, 1.0, 0.5, 5.0

    seeds = fe.seeds_from_paths(paths, None, None, _MockPlanner())
    for w, t, sig in seeds:
        print(f"[self-test] sig={sig} wpts=\n{np.round(w, 2)} ts={np.round(t, 2)}")
    print("[self-test] OK — both homotopy classes recovered for a 2.6 m box.")
