"""PNS-native demo builder: best-first search over policy actions, no oracle gate.

For every ratsnest pin of a net, a lazy best-first search expands *policy actions*
(start_route + make_line / make_via over the same candidate pool the policy sees)
with the real PNS router as the transition model and engine checkpoint/restore as
backtracking. Feasibility is exact by construction; failed actions are pruned
branches. Priority f = LAMBDA * (#actions) + Euclidean distance to the nearest
unconnected same-net copper (+ via penalty), so the search prefers "one make_line
straight to a pad" and only spends waypoints when PNS cannot do it directly.

Walkaround only by default (--shove adds shove variants, accepted only when the
DRC violation count does not grow). No finish, no rip-up. The successful action
list per board is saved for a later replay pass that records (obs, masks, action).
"""
from __future__ import annotations

import argparse, glob, heapq, json, math, os, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CADET = HERE.parents[1]
sys.path.insert(0, str(CADET)); sys.path.insert(0, str(HERE))
from cadet.paths import use_pcbworld  # noqa: E402
REPO = use_pcbworld()   # the PCBWorld checkout

from pcb_world.core.action_schema import (  # noqa: E402
    ACT_NET_SELECT, ACT_START_ROUTE, ACT_NET_END, ACT_MAKE_LINE, ACT_MAKE_VIA, ACTION_NAMES,
)
from pcb_world.core.indexed_obs import is_indexed, arrays_to_dict  # noqa: E402
from pcb_world.vec.candidate_pool import DIRECTIONAL_DISTANCE_PRESETS, _DIRECTIONS  # noqa: E402

import numpy as np
from numba import njit
from executor import ObsOracle, _CachedOracle  # optimistic grid (margin 0) built from the obs

MODE_WALK, MODE_SHOVE = 2, 1
_INF = np.iinfo(np.int64).max


@njit(cache=True)
def _dist_field(blocked, via_ok, goal, via_cost):
    """Multi-source Dijkstra distance (1/10 cell units: orth 10, diag 14, via via_cost) from goal cells."""
    L, ny, nx = blocked.shape
    n = L * ny * nx
    dist = np.full(n, _INF, np.int64)
    heap = [(np.int64(0), np.int64(0))]; heap.pop()
    for l in range(L):
        for y in range(ny):
            for x in range(nx):
                if goal[l, y, x]:
                    i = (l * ny + y) * nx + x; dist[i] = 0; heapq.heappush(heap, (np.int64(0), np.int64(i)))
    while len(heap) > 0:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        l = i // (ny * nx); r = i % (ny * nx); y = r // nx; x = r % nx
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy = y + dy; xx = x + dx
                if yy < 0 or yy >= ny or xx < 0 or xx >= nx or blocked[l, yy, xx]:
                    continue
                if dy != 0 and dx != 0:
                    if blocked[l, y, xx] or blocked[l, yy, x]:
                        continue
                    nd = d + 14
                else:
                    nd = d + 10
                j = (l * ny + yy) * nx + xx
                if nd < dist[j]:
                    dist[j] = nd; heapq.heappush(heap, (nd, np.int64(j)))
        if via_ok[y, x]:
            for l2 in range(L):
                if l2 == l:
                    continue
                if blocked[l2, y, x]:
                    continue
                j = (l2 * ny + y) * nx + x; nd = d + via_cost
                if nd < dist[j]:
                    dist[j] = nd; heapq.heappush(heap, (nd, np.int64(j)))
    return dist.reshape((L, ny, nx))

LAMBDA = 2.0        # mm of heuristic distance one extra action is worth
ALPHA = 1.3         # max realized wirelength per mm of heuristic progress (walkaround detour cap)
SLACK = 1.0         # mm of free detour per hop
VIA_PENALTY = 1.0   # mm-equivalent cost of a make_via edge
SHOVE_PENALTY = 0.5


def _key(x, y, l):
    return (round(float(x), 3), round(float(y), 3), int(l))


def _obs(o):
    if is_indexed(o):
        d = arrays_to_dict(o)
        for k in ("_aug", "_masks"):
            if k in o: d[k] = o[k]
        return d
    return o


def _ratsnest(obs, nid):
    return obs.get("routing_geometry", {}).get(f"net_{nid}", {}).get("points", [])


def _wirelength(obs):
    return sum(math.hypot(t["p1"]["xy"][0] - t["p2"]["xy"][0], t["p1"]["xy"][1] - t["p2"]["xy"][1])
               for g in obs.get("routing_geometry", {}).values() for t in g.get("tracks", {}).values())


def _dangling(obs):
    return sum(1 for v in (obs.get("drc_violations") or []) if isinstance(v, dict)
               and any(k in str(v.get("error_type", "")).lower() for k in ("unconnected end", "not connected")))


def _drc_real(obs):
    """DRC violations that are not the 'unconnected items' bookkeeping of open ratsnest / a mid-route dangling end
    ('Track has unconnected end', 'Via is not connected ...' — transient while a pin is being routed)."""
    return sum(1 for v in (obs.get("drc_violations") or []) if isinstance(v, dict)
               and not any(k in str(v.get("error_type", "")).lower() for k in ("unconnected", "missing connection", "not connected")))


def net_points(obs, nid, nl=2):
    """All (x, y, human_layer) copper anchor keys of a net + union-find clusters.

    Thru pads and vias bridge layers; tracks join their endpoints. Returns
    (keys, find) where find(k) -> cluster root.
    """
    parent = {}
    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for pad in obs["board_static"]["nets"][f"net_{nid}"]["pads"].values():
        px, py = pad["center"]["xy"]; pl = int(pad["layer"])
        ks = [_key(px, py, l) for l in (range(1, nl + 1) if pl == 0 else (pl,))]
        for k in ks: union(ks[0], k)
    g = obs.get("routing_geometry", {}).get(f"net_{nid}", {})
    for t in g.get("tracks", {}).values():
        union(_key(*t["p1"]["xy"], t["layer"]), _key(*t["p2"]["xy"], t["layer"]))
    for v in g.get("vias", {}).values():
        vx, vy = v["center"]["xy"]
        ks = [_key(vx, vy, l) for l in range(int(v.get("layer_start", 1)), int(v.get("layer_end", nl)) + 1)]
        for k in ks: union(ks[0], k)
    return list(parent.keys()), find


class Searcher:
    def __init__(self, env, *, budget=120, shove=False, log=print, trace=False, use_oracle=True, oracle_step=0.2):
        self.env, self.budget, self.shove, self.log, self.trace = env, budget, shove, log, trace
        self.use_oracle, self.oracle_step, self._orc, self._field = use_oracle, oracle_step, None, None
        self.pin_trace = False; self.short_first = True; self.cap = True; self.verify_pins = True; self.board_time_cap = 0.0
        # direct-hop modes (mode, priority penalty); strict_drc: every mode must keep the real DRC count (default = legacy:
        # walkaround unchecked, shove checked). The generator sets all three router modes + strict_drc.
        self.direct_modes = [(MODE_WALK, 0.0)] + ([(MODE_SHOVE, SHOVE_PENALTY)] if shove else []); self.strict_drc = False
        self.skip_noncand = False      # generator: drop directional edges that are not pool candidates before queuing
        self.pcb = env.env
        self.rungs = DIRECTIONAL_DISTANCE_PRESETS.get(str(env._aug.get("directional_candidates")) if getattr(env, "_aug", None) else "multi_resolution", (0.2, 1.0, 5.0, 25.0))

    # ---- env helpers
    def cand_index(self, x, y, layer=None):
        k2 = (round(float(x), 3), round(float(y), 3))
        for i, (cx, cy, cl) in enumerate(self.env.cand_mm_list):
            if (round(cx, 3), round(cy, 3)) == k2 and (layer is None or cl == layer):
                return i
        return None

    def step(self, at, ptr, mode=0):
        # PNS keeps a residual via-placement toggle after a failed make_via; a linear replay never has
        # it, so clear it before every action to keep search-found hops reproducible.
        eng = getattr(self.pcb, "_engine", None)
        if eng is not None and at in (ACT_START_ROUTE, ACT_MAKE_LINE, ACT_MAKE_VIA):
            try:
                if eng.is_placing_via():
                    eng.toggle_via()
            except Exception:
                pass
        obs, rew, term, trunc, info = self.env.step((int(at), int(ptr), int(mode)))
        self.n_env_steps += 1
        o = _obs(obs)
        if self.trace:
            rh = o["router_head"]
            print(f"      {ACTION_NAMES[at]} ptr={ptr} mode={mode} -> {info.get('action_class')} routing={rh.get('is_routing')}")
        return o, term, trunc, info

    def snapshot(self):
        return (self.pcb.checkpoint(), self.env.snapshot_mcts_state(obs_cache=True))

    def restore(self, snap):
        l1, l2 = snap
        self.pcb.restore(l1, incremental=False)
        if not self.env.restore_mcts_state(l2):
            raw = self.pcb._get_obs(); aug = self.env._inject_aug(raw); self.env._refresh_cache(aug)
        obs = self.env._last_obs
        self.env._attach_masks(obs)
        return _obs(obs)

    @staticmethod
    def new_head(o, o2, nid, head, at, target):
        """Where the committed hop actually ended: newest via centre (make_via, on the new layer)
        or the free end of the newly added track segments; falls back to the requested target."""
        g1 = o.get("routing_geometry", {}).get(f"net_{nid}", {}); g2 = o2.get("routing_geometry", {}).get(f"net_{nid}", {})
        tx, ty, tl = target
        if at == ACT_MAKE_VIA:
            old = {(round(v["center"]["xy"][0], 3), round(v["center"]["xy"][1], 3)) for v in g1.get("vias", {}).values()}
            new = [v for v in g2.get("vias", {}).values() if (round(v["center"]["xy"][0], 3), round(v["center"]["xy"][1], 3)) not in old]
            if new:
                v = min(new, key=lambda v: math.hypot(v["center"]["xy"][0] - tx, v["center"]["xy"][1] - ty))
                return (float(v["center"]["xy"][0]), float(v["center"]["xy"][1]), tl)
            return target
        def segs(g):
            return {(round(t["p1"]["xy"][0], 3), round(t["p1"]["xy"][1], 3), round(t["p2"]["xy"][0], 3), round(t["p2"]["xy"][1], 3), int(t["layer"])) for t in g.get("tracks", {}).values()}
        new = segs(g2) - segs(g1)
        if not new:
            return target
        cnt = Counter()
        for x1, y1, x2, y2, l in new:
            cnt[(x1, y1, l)] += 1; cnt[(x2, y2, l)] += 1
        hk = (round(head[0], 3), round(head[1], 3), int(head[2]))
        free = [k for k, c in cnt.items() if c == 1 and k != hk]
        if not free:
            return target
        k = min(free, key=lambda k: math.hypot(k[0] - tx, k[1] - ty))
        return (float(k[0]), float(k[1]), int(k[2]))

    # ---- geometry helpers
    def goals(self, obs, nid, src_key):
        keys, find = net_points(obs, nid, self.nl)
        root = find(src_key) if src_key in keys else None
        return [k for k in keys if root is None or find(k) != root]

    def h_euclid(self, x, y, layer, goals):
        best = None
        for gx, gy, gl in goals:
            d = math.hypot(gx - x, gy - y) + (0.0 if gl == layer else VIA_PENALTY + 0.5)
            if best is None or d < best: best = d
        return 1e9 if best is None else best

    def h(self, x, y, layer, goals):
        f = self._field
        if f is None:
            return self.h_euclid(x, y, layer, goals)
        orc = self._orc; cy, cx = orc.cell(x, y)
        if not (0 <= cy < orc.orc.ny and 0 <= cx < orc.orc.nx):
            return 1e9
        d = f[layer - 1, cy, cx]
        if d == _INF:
            return 1000.0 + self.h_euclid(x, y, layer, goals)   # unreachable on the grid: try last
        return d * orc.step / 10.0

    def build_field(self, obs, nid, goals):
        """Distance-to-goal field on the optimistic grid (margin 0) for this pin; None if the oracle is off."""
        if not self.use_oracle:
            self._field = None; return
        if self._orc is None:
            self._orc = ObsOracle(obs, step=self.oracle_step, via_cost=1500)
        else:
            self._orc.resync(obs)
        o = self._orc.orc; r = self._orc.rules[nid]
        blocked, via_ok = o._blocked_maps(nid, r["clearance"], r["width"], r["via_dia"])
        goal = np.zeros_like(blocked)
        for gx, gy, gl in goals:
            cy, cx = self._orc.cell(gx, gy)
            if 0 <= cy < o.ny and 0 <= cx < o.nx:
                goal[gl - 1, cy, cx] = True; blocked[gl - 1, cy, cx] = False
        self._field = _dist_field(blocked, via_ok, goal, np.int64(o.via_cost))

    # ---- one pin
    def route_pin(self, obs, nid, src):
        """Best-first search from anchor ``src`` (x, y, layer) until a ratsnest edge of ``nid`` closes.

        Returns (obs, actions, expansions, ok). ``actions`` = list of executed
        (action_type, x, y, layer, mode) that led to the connection (root->goal)."""
        rats0 = len(_ratsnest(obs, nid))
        goals = self.goals(obs, nid, _key(*src))
        if not goals:
            return obs, [], 0, False
        self.build_field(obs, nid, goals)
        root_snap = self.snapshot()
        diag = Counter(); h_src = self.h(src[0], src[1], src[2], goals)
        budget = self.budget if h_src < 1000.0 else min(self.budget, 12)   # grid says unreachable: a few direct tries only
        # node: (snap, obs, g, head(x,y,layer), actions)
        nodes = [(root_snap, obs, 0, src, [])]
        seq = 0; heap = []; visited = {(round(src[0], 2), round(src[1], 2), src[2])}
        def push_edges(ni):
            nonlocal seq
            snap, o, g, head, acts = nodes[ni]
            hx, hy, hl = head
            edges = []
            # direct targets: same-net copper outside the head's cluster, on this layer (or thru)
            gl_here = [(x, y, l) for x, y, l in goals if l == hl]
            gl_here.sort(key=lambda p: math.hypot(p[0] - hx, p[1] - hy))
            for x, y, l in gl_here[:3]:
                for mode, pen in self.direct_modes:
                    edges.append((LAMBDA * (g + 1) + pen, ACT_MAKE_LINE, x, y, hl, mode))
            other_layer_goal = any(l != hl for _, _, l in goals)
            for d in self.rungs:
                for dx, dy in _DIRECTIONS:
                    x, y = hx + dx * d, hy + dy * d
                    if self.skip_noncand and o["router_head"].get("is_routing") and self.cand_index(x, y, hl) is None:
                        continue                      # not a pool point (off-board / filtered): would only burn budget
                        # (directional candidates exist only while a session is open — never skip at a closed-session node)
                    if (round(x, 2), round(y, 2), hl) not in visited:
                        edges.append((LAMBDA * (g + 1) + self.h(x, y, hl, goals), ACT_MAKE_LINE, x, y, hl, MODE_WALK))
                    if other_layer_goal:
                        nl2 = 2 if hl == 1 else 1
                        if (round(x, 2), round(y, 2), nl2) not in visited:
                            edges.append((LAMBDA * (g + 1) + VIA_PENALTY + self.h(x, y, nl2, goals), ACT_MAKE_VIA, x, y, nl2, MODE_WALK))
            for f, at, x, y, l, mode in edges:
                seq += 1; heapq.heappush(heap, (f, seq, ni, at, x, y, l, mode))
        push_edges(0)
        expansions = 0; cur = 0    # index of the node whose state the env currently holds
        result = None
        while heap and expansions < budget:
            f, _, ni, at, x, y, l, mode = heapq.heappop(heap)
            snap, o, g, head, acts = nodes[ni]
            if ni != cur:
                o = self.restore(snap); cur = ni
            # open the session at the node's head
            if not o["router_head"].get("is_routing"):
                if not o["_masks"]["action"][ACT_START_ROUTE]:
                    break
                si = self.cand_index(head[0], head[1], head[2])
                if si is None:
                    diag["no_start_cand"] += 1; cur = -1; continue
                o, term, trunc, info = self.step(ACT_START_ROUTE, si)
                if not info.get("action_success") or not o["router_head"].get("is_routing"):
                    diag["start_fail"] += 1; cur = -1; continue
            ti = self.cand_index(x, y, head[2])
            expansions += 1
            if ti is None or not o["_masks"]["action"][at]:
                diag["no_target_cand" if ti is None else "masked"] += 1; cur = -1; continue
            drc_before = _drc_real(o); wl_before = _wirelength(o); h_before = self.h(head[0], head[1], head[2], goals); dang_before = _dangling(o)
            o2, term, trunc, info = self.step(at, ti, mode)
            ok = info.get("action_class") == "valid_effective" and ((mode == MODE_WALK and not self.strict_drc) or _drc_real(o2) <= drc_before)
            if ok and len(_ratsnest(o2, nid)) < rats0 and _dangling(o2) > dang_before:
                ok = False; diag["overshoot"] += 1     # connected mid-way and overshot: a dangling tail would stay
            connected_now = len(_ratsnest(o2, nid)) < rats0
            req = (at, x, y, head[2], mode)     # the action as issued: requested target on the head's layer
            if (term or trunc) and ok and connected_now:
                # the board's last connection terminates the episode — that is success, not failure
                self.term = True
                result = (o2, acts + [(ACT_START_ROUTE, head[0], head[1], head[2], 0), req]); break
            if not ok or term or trunc:
                diag[str(info.get("action_class")) + ("/via" if at == ACT_MAKE_VIA else "")] += 1; cur = -1; continue
            x, y, l = self.new_head(o, o2, nid, head, at, (x, y, l))
            h_after = 0.0 if connected_now else self.h(x, y, l, goals)
            gain = max(0.0, min(h_before, 1e6) - min(h_after, 1e6))
            if self.cap and _wirelength(o2) - wl_before > ALPHA * gain + SLACK:
                diag["detour"] += 1; cur = -1; continue          # walkaround wandered: it would wall in later pins
            diag["ok"] += 1
            new_acts = acts + [(ACT_START_ROUTE, head[0], head[1], head[2], 0), req]
            if connected_now:
                result = (o2, new_acts); break
            key = (round(x, 2), round(y, 2), l)
            if key in visited:
                cur = -1; continue
            visited.add(key)
            nodes.append((self.snapshot(), o2, g + 1, (x, y, l), new_acts)); cur = len(nodes) - 1
            push_edges(cur)
        if result is not None and self.verify_pins:
            o = self.restore(root_snap); acts = result[1]; good = True
            for at, x, y, l, mode in acts:
                if not o["router_head"].get("is_routing") and at != ACT_START_ROUTE:
                    good = False; break
                ptr = self.cand_index(x, y, l)
                if ptr is None:
                    good = False; break
                o, term, trunc, info = self.step(at, ptr, mode)
                if not info.get("action_success"):
                    good = False; break
            if good and len(_ratsnest(o, nid)) >= rats0:
                good = False
            if not good:
                diag["verify_fail"] += 1; result = None; self.term = False
            else:
                result = (o, acts)
                if term or trunc: self.term = True
        if self.pin_trace and (result is None or expansions > 20):
            print(f"    pin net={nid} src={tuple(round(v, 1) for v in src)} goals={len(goals)} h_src={h_src:.1f} exp={expansions} nodes={len(nodes)} {'OK' if result else 'FAIL'} {dict(diag)}")
        if result is None:
            # restore the pin's root BEFORE releasing handles: a released handle restores nothing
            # (documented), which left every failed branch's stubs on the board.
            o = self.restore(root_snap)
        for snap, *_ in nodes:          # nodes[0] holds root_snap
            try: snap[0].release()
            except Exception: pass
        if result is None:
            return o, [], expansions, False
        return result[0], result[1], expansions, True

    # ---- one board
    def run(self, board):
        t0 = time.time(); env = self.env
        obs = _obs(env.reset()[0]); self.n_env_steps = 0; self._orc = None; self._field = None; self._dang = 0; self.term = False
        bs = obs["board_static"]; self.nl = max(int(bs.get("copper_layers", 2)), 2)
        nets = list(env.sorted_net_codes); pins0 = sum(len(_ratsnest(obs, n)) for n in nets) // 2
        demo = []; per_pin_len = Counter(); exp_total = 0; fails = 0; term = trunc = False
        net_status = {}
        order = sorted(nets, key=lambda n: self.net_mst_len(obs, n)) if self.short_first else nets
        timed_out = False
        for nid in order:
            if term or trunc: break
            if self.board_time_cap and time.time() - t0 > self.board_time_cap:
                timed_out = True; break          # leave the remaining nets unrouted (partial board)
            if not _ratsnest(obs, nid) or nid in (obs.get("closed_nets") or []):
                continue
            idx = nets.index(nid); nv = obs["_masks"]["net_valid"]
            if idx >= len(nv) or not nv[idx]:
                continue
            obs, term, trunc, _ = self.step(ACT_NET_SELECT, idx); demo.append((ACT_NET_SELECT, nid, 0, 0, 0))
            tried = set()
            while _ratsnest(obs, nid) and not (term or trunc):
                if self.board_time_cap and time.time() - t0 > self.board_time_cap:
                    timed_out = True; break
                src = self.pick_source(obs, nid, tried)
                if src is None:
                    fails += 1; break
                self.cap = True
                obs, acts, exp, ok = self.route_pin(obs, nid, src)
                exp_total += exp
                if not ok:                       # second chance without the detour cap
                    self.cap = False
                    obs, acts, exp, ok = self.route_pin(obs, nid, src)
                    exp_total += exp; self.cap = True
                if ok:
                    demo.extend(acts); per_pin_len[len(acts) // 2] += 1; tried = set()
                    if self.term:
                        term = True
                    if self.pin_trace:
                        dang = [(round(v["x_mm"], 2), round(v["y_mm"], 2)) for v in (obs.get("drc_violations") or []) if "unconnected end" in str(v.get("error_type", ""))]
                        if len(dang) > getattr(self, "_dang", 0):
                            print(f"    DANGLING after pin net={nid}: {dang[-3:]} pin actions={[(ACTION_NAMES[a[0]], round(a[1], 2), round(a[2], 2), a[3]) for a in acts]}")
                        self._dang = len(dang)
                else:
                    tried.add((round(src[0], 2), round(src[1], 2)))
                    if len(tried) >= 3:
                        fails += 1; break
            net_status[nid] = "ok" if not _ratsnest(obs, nid) else "partial"
            if obs["_masks"]["action"][ACT_NET_END] and not (term or trunc):
                obs, term, trunc, _ = self.step(ACT_NET_END, 0); demo.append((ACT_NET_END, 0, 0, 0, 0))
        unrouted = sum(len(_ratsnest(obs, n)) for n in nets) // 2
        drc = len(obs.get("drc_violations") or []); drc_real = _drc_real(obs)
        drc_types = dict(Counter(str(v.get("error_type")) for v in (obs.get("drc_violations") or []) if isinstance(v, dict)))
        at_counts = Counter(ACTION_NAMES[a[0]] for a in demo); mode_counts = Counter(a[4] for a in demo if a[0] in (ACT_MAKE_LINE, ACT_MAKE_VIA))
        return dict(board=os.path.basename(board), nets=len(nets), pins=pins0, unrouted_pins=unrouted, pin_fail=fails,
                    demo_actions=len(demo), actions_by_type=dict(at_counts), modes={"walk": mode_counts[MODE_WALK], "shove": mode_counts[MODE_SHOVE]},
                    pin_actions_hist={str(k): v for k, v in sorted(per_pin_len.items())},
                    env_steps=self.n_env_steps, expansions=exp_total, drc=drc, drc_real=drc_real, drc_types=drc_types,
                    success=unrouted == 0, clean_pass=unrouted == 0 and drc == 0,
                    wirelength=round(_wirelength(obs), 2), wall_s=round(time.time() - t0, 1), timed_out=timed_out, demo=demo, net_status=net_status)

    @staticmethod
    def net_mst_len(obs, nid):
        pts = [tuple(p["center"]["xy"]) for p in obs["board_static"]["nets"][f"net_{nid}"]["pads"].values()]
        if len(pts) < 2:
            return 0.0
        inn = {0}; total = 0.0
        while len(inn) < len(pts):
            d, j = min((math.hypot(pts[i][0] - pts[k][0], pts[i][1] - pts[k][1]), k) for i in inn for k in range(len(pts)) if k not in inn)
            inn.add(j); total += d
        return total

    def pick_source(self, obs, nid, tried):
        """Ratsnest anchor of the smallest cluster (lone pin first) that is a start candidate."""
        keys, find = net_points(obs, nid, self.nl)
        size = Counter(find(k) for k in keys)
        best = None
        for p in _ratsnest(obs, nid):
            x, y = p["xy"]; l = int(p.get("layer", 0))
            if (round(x, 2), round(y, 2)) in tried:
                continue
            for cand_l in ((l,) if l > 0 else (1, 2)):
                if self.cand_index(x, y, cand_l) is None:
                    continue
                k = _key(x, y, cand_l); sz = size.get(find(k), 1) if k in keys else 1
                if best is None or sz < best[0]:
                    best = (sz, (x, y, cand_l))
                break
        return None if best is None else best[1]


_CKPT_ARGS = {}


def build_env(board, ckpt, max_steps=8192, obs_format="json", preset=None, offboard_mask=None):
    from configs.loader.schema import RLEnvConfig
    from methods.rl_agent.wrappers.factory import make_decoder_env
    if ckpt not in _CKPT_ARGS:          # read the base arguments once per process
        from cadet.base_args import load_base_args
        _CKPT_ARGS[ckpt] = load_base_args(ckpt)
    a = _CKPT_ARGS[ckpt]
    kw = RLEnvConfig.from_checkpoint(a, max_steps).to_pool_kwargs()
    kw["keep_routing_fraction"] = None; kw["obs_format"] = obs_format
    if preset:
        kw["directional_candidates"] = preset
    if offboard_mask is not None and "offboard_mask" in kw:
        kw["offboard_mask"] = bool(offboard_mask)
    return make_decoder_env(board, seed=0, policy_net_select=True, **kw), kw


CKPT = str(CADET / "cadet/configs/base_args/generation.json")   # base arguments (cadet/base_args.py); a trainer checkpoint also works
VAL = str(CADET / "var/datasets/routedemos/val")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default=VAL); ap.add_argument("--n", type=int, default=2); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ckpt", default=CKPT); ap.add_argument("--out", default=str(CADET / "var/results/search/search.jsonl"))
    ap.add_argument("--budget", type=int, default=120); ap.add_argument("--shove", action="store_true"); ap.add_argument("--trace", action="store_true"); ap.add_argument("--no-oracle", action="store_true"); ap.add_argument("--pin-trace", action="store_true"); ap.add_argument("--no-short-first", action="store_true"); ap.add_argument("--no-verify", action="store_true"); ap.add_argument("--board-time-cap", type=float, default=0.0); ap.add_argument("--skip-done", action="store_true"); ap.add_argument("--oracle-step", type=float, default=0.2)
    ap.add_argument("--demo-dir", default=str(CADET / "var/results/search/demos"))
    ap.add_argument("--list", default=None, help="text file of board paths (one per line); --start/--n slice it instead of --boards")
    a = ap.parse_args()
    if a.list:
        boards = [l.strip() for l in open(a.list) if l.strip()][a.start:a.start + a.n]
    else:
        boards = sorted(glob.glob(a.boards + "/board_*.kicad_pcb"))[a.start:a.start + a.n]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.demo_dir).mkdir(parents=True, exist_ok=True)
    done = set()
    if a.skip_done and Path(a.out).exists():
        done = {json.loads(l)["board"] for l in open(a.out) if l.strip()}
    with open(a.out, "a") as f:
        for b in boards:
            if Path(b).name in done:
                continue
            env, kw = build_env(b, a.ckpt)
            try:
                se = Searcher(env, budget=a.budget, shove=a.shove, trace=a.trace, use_oracle=not a.no_oracle, oracle_step=a.oracle_step); se.pin_trace = a.pin_trace; se.short_first = not a.no_short_first; se.verify_pins = not a.no_verify; se.board_time_cap = a.board_time_cap; r = se.run(b)
            finally:
                env.close()
            demo = r.pop("demo"); net_status = r.pop("net_status")
            json.dump(dict(board=b, actions=demo, clean_pass=r["clean_pass"], success=r["success"], net_status=net_status), open(Path(a.demo_dir) / (Path(b).stem + ".json"), "w"))
            print(json.dumps(r)); f.write(json.dumps(r) + "\n"); f.flush()
