"""Oracle-guided PNS executor (direct-first + re-plan).

Runs on a finished board through the training wrapper (KiCadRLWrapper) so every
emitted action is a native (action_type, pointer_idx, routing_mode) triple over
the same candidate pool the policy sees. The grid oracle is rebuilt from the env observation — outline / cutouts /
NPTH from ``board_static``, copper from pads + ``routing_geometry`` — and
re-synced after every successful action, so PNS-realized geometry (walkaround
detours, shove displacement) is what the next plan sees.

Fallback ladder per pin (recorded as ``level``):
  1 make_line -> direct target (pad / track end / via)   walkaround
  2 same target                                         shove
  3 waypoint = largest mres rung along the oracle path   walkaround
  4 waypoint                                            shove
  5 give up the pin
"""
from __future__ import annotations

import argparse, glob, json, math, os, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
CADET = HERE.parents[1]
sys.path.insert(0, str(CADET)); sys.path.insert(0, str(HERE))
from cadet.paths import use_pcbworld  # noqa: E402
REPO = use_pcbworld()   # the PCBWorld checkout

from grid_oracle import GridOracle  # noqa: E402
from pcb_world.core.action_schema import (  # noqa: E402
    ACT_NET_SELECT, ACT_START_ROUTE, ACT_NET_END, ACT_MAKE_LINE, ACT_MAKE_VIA, ACTION_NAMES,
)
from pcb_world.core.indexed_obs import is_indexed, arrays_to_dict  # noqa: E402

LADDER = (25.0, 5.0, 1.0, 0.2)
MODE_WALK, MODE_SHOVE = 2, 1
NC_NET = 10 ** 6   # owner id for no-net copper (NC pads)


def _key(x, y, l):
    return (round(float(x), 3), round(float(y), 3), int(l))


def _rule(bs: dict, net_key: str) -> dict:
    net = bs["nets"][net_key]
    c = net.get("constraints")
    if c:
        return dict(clearance=c["clearance"], width=c["track_width"], via_dia=c["via_diameter"], via_drill=c.get("via_drill", 0.4))
    b = bs["board_constraints"]
    return dict(clearance=b["clearance_mm"], width=b["track_width_mm"], via_dia=b["via_diameter_mm"], via_drill=b.get("via_drill_mm", 0.4))


# ---------------------------------------------------------------- oracle from obs
class _CachedOracle(GridOracle):
    """GridOracle variant for the executor.

    * ``_blocked_maps`` memoized until the copper changes (``epoch`` bumped by resync).
    * **Optimistic** margins: the generator's oracle is conservative (copper inflated by
      0.71·step, +0.5·step in the clearance test) so that a "yes" is realizable. Here PNS is
      the judge (a failed action just falls down the ladder), while PNS-realized walkaround
      copper hugs neighbours at *exact* clearance — the conservative oracle then declares
      pads boxed in that PNS can still leave. ``inflate``/``margin`` (in steps) default to 0.
    * the source cell is never blocked (the head is already standing there).
    """

    epoch = 0
    inflate = 0.0
    margin = 0.0
    src_cells: list = []

    def paint_capsule(self, layers, x0, y0, x1, y1, r, net, clearance, target=False):
        if not target:
            r = r + self.step * self.inflate
        def f(X, Y):
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            if L2 == 0:
                return (X - x0) ** 2 + (Y - y0) ** 2 <= r * r
            t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1)
            return (X - (x0 + t * dx)) ** 2 + (Y - (y0 + t * dy)) ** 2 <= r * r
        self._paint(layers, f, x0, y0, x1, y1, r, net, clearance, target)

    def paint_rect(self, layers, cx, cy, w, h, net, clearance, target=False):
        if not target:
            w, h = w + self.step * 2 * self.inflate, h + self.step * 2 * self.inflate
        def f(X, Y):
            return (np.abs(X - cx) <= w / 2) & (np.abs(Y - cy) <= h / 2)
        self._paint(layers, f, cx, cy, cx, cy, max(w, h) / 2, net, clearance, target)

    def _blocked_maps_raw(self, net, clearance, width, via_dia):
        s = self.step; m = s * self.margin
        other = (self.owner != 0) & (self.owner != net)
        blocked = np.zeros((2, self.ny, self.nx), bool)
        via_bad = np.zeros((self.ny, self.nx), bool)
        for l in range(2):
            for ck in np.unique(self.clear[l][other[l]]):
                mm = other[l] & (self.clear[l] == ck)
                d = ndimage.distance_transform_edt(~mm) * s
                req = max(clearance, float(ck))
                blocked[l] |= d < req + width / 2 + m
                via_bad |= d < req + via_dia / 2 + m
        edge_req = max(self.edge_clear, clearance)
        for l in range(2):
            blocked[l] |= self.static_dist < edge_req + width / 2 + m
        via_bad |= self.static_dist < edge_req + via_dia / 2 + m
        hole = self.hole_owner != 0
        if hole.any():
            dh = ndimage.distance_transform_edt(~hole) * s
            via_bad |= dh < via_dia * 0.35 + 0.3 + m
        return blocked, ~via_bad

    def _blocked_maps(self, net, clearance, width, via_dia):
        key = (self.epoch, int(net), round(clearance, 4), round(width, 4), round(via_dia, 4))
        cache = self.__dict__.setdefault("_maps", {})
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = self._blocked_maps_raw(net, clearance, width, via_dia)
        b, v = cache[key]
        b = b.copy(); v = v.copy()      # query mutates via_ok in place
        for l, y, x in self.src_cells:
            b[l, y, x] = False
        return b, v


class ObsOracle:
    """GridOracle whose static mask + copper come from the env observation."""

    def __init__(self, obs: dict, step: float = 0.2, via_cost: int = 1500, edge_clear: float = 0.3):
        bs = obs["board_static"]
        self.ox, self.oy = float(bs["bbox_x"]), float(bs["bbox_y"])
        bw, bh = float(bs["bbox_w"]), float(bs["bbox_h"])
        self.step = step
        self.orc = _CachedOracle(bw, bh, step, None, edge_clear=edge_clear, via_cost=via_cost)
        self._static_mask(bs)
        self.resync(obs)

    # -- coordinates
    def cell(self, x, y):
        return self.orc.cell(x - self.ox, y - self.oy)

    def to_mm(self, cy, cx):
        return cx * self.step + self.ox, cy * self.step + self.oy

    # -- static: outline loops (+arcs) / cutouts / NPTH
    def _static_mask(self, bs: dict):
        from matplotlib.path import Path as MPath
        o = self.orc
        segs = []
        for e in bs.get("boardlines", {}).values():
            p1, p2 = e["p1"]["xy"], e["p2"]["xy"]
            mid = e.get("mid")
            if mid is None:
                segs.append([tuple(p1), tuple(p2)])
            else:
                segs.append(self._arc_points(p1, mid["xy"], p2))
        loops = self._chain(segs)
        xs = (np.arange(o.nx) * self.step + self.ox); ys = (np.arange(o.ny) * self.step + self.oy)
        X, Y = np.meshgrid(xs, ys); pts = np.column_stack([X.ravel(), Y.ravel()])
        inside = np.zeros(o.ny * o.nx, bool)
        if loops:
            areas = [abs(self._area(lp)) for lp in loops]
            outer = loops[int(np.argmax(areas))]
            inside = MPath(outer).contains_points(pts)
            for lp, a in zip(loops, areas):
                if lp is outer or a < 1e-6:
                    continue
                inside &= ~MPath(lp).contains_points(pts)
        else:   # no outline in obs: rectangle bbox
            inside[:] = True
        inside = inside.reshape(o.ny, o.nx)
        # NPTH holes / slots (obstacles without net); keepout polygons excluded like the tokenizer
        for ob in bs.get("obstacles", {}).values():
            if ob.get("shape") == "polygon":
                continue
            cx, cy = ob["center"]["xy"]; w, h = float(ob["width"]), float(ob["height"])
            r = max(w, h) / 2
            ix0, ix1, iy0, iy1, XX, YY = o._window(cx - self.ox, cy - self.oy, cx - self.ox, cy - self.oy, r)
            if ob.get("shape") in ("circle", "oval", ""):
                m = (XX - (cx - self.ox)) ** 2 + (YY - (cy - self.oy)) ** 2 <= r * r
            else:
                m = (np.abs(XX - (cx - self.ox)) <= w / 2) & (np.abs(YY - (cy - self.oy)) <= h / 2)
            inside[iy0:iy1 + 1, ix0:ix1 + 1][m] = False
        # cells within half a step of the edge count as outside (as GridOracle does)
        o.static_dist = ndimage.distance_transform_edt(inside).astype(np.float32) * self.step
        self.inside = inside

    @staticmethod
    def _arc_points(p1, mid, p2, n=16):
        (x1, y1), (xm, ym), (x2, y2) = p1, mid, p2
        d = 2 * (x1 * (ym - y2) + xm * (y2 - y1) + x2 * (y1 - ym))
        if abs(d) < 1e-9:
            return [tuple(p1), tuple(p2)]
        ux = ((x1 * x1 + y1 * y1) * (ym - y2) + (xm * xm + ym * ym) * (y2 - y1) + (x2 * x2 + y2 * y2) * (y1 - ym)) / d
        uy = ((x1 * x1 + y1 * y1) * (x2 - xm) + (xm * xm + ym * ym) * (x1 - x2) + (x2 * x2 + y2 * y2) * (xm - x1)) / d
        r = math.hypot(x1 - ux, y1 - uy)
        a1, am, a2 = (math.atan2(y - uy, x - ux) for x, y in (p1, mid, p2))
        # sweep from a1 to a2 passing through am
        def norm(a): return (a + 2 * math.pi) % (2 * math.pi)
        sweep = norm(a2 - a1); through = norm(am - a1)
        if through > sweep:      # go the other way round
            sweep -= 2 * math.pi
        if abs(sweep) < 1e-9:    # full circle (p1 == p2)
            sweep = 2 * math.pi
        return [(ux + r * math.cos(a1 + sweep * t / n), uy + r * math.sin(a1 + sweep * t / n)) for t in range(n + 1)]

    @staticmethod
    def _chain(segs):
        key = lambda p: (round(p[0], 3), round(p[1], 3))
        used = [False] * len(segs); loops = []
        ends = {}
        for i, s in enumerate(segs):
            ends.setdefault(key(s[0]), []).append(i); ends.setdefault(key(s[-1]), []).append(i)
        for i in range(len(segs)):
            if used[i]:
                continue
            used[i] = True; loop = list(segs[i]); cur = key(loop[-1]); start = key(loop[0])
            while cur != start:
                nxt = next((j for j in ends.get(cur, []) if not used[j]), None)
                if nxt is None:
                    break
                used[nxt] = True; s = segs[nxt]
                if key(s[0]) != cur:
                    s = s[::-1]
                loop.extend(s[1:]); cur = key(loop[-1])
            loops.append(loop)
        return loops

    @staticmethod
    def _area(lp):
        return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(lp, lp[1:] + lp[:1]))

    # -- dynamic copper from obs
    def resync(self, obs: dict):
        o = self.orc; o.epoch += 1
        o.owner[:] = 0; o.clear[:] = 0; o.hole_owner[:] = 0; o.tgt_owner[:] = 0
        bs = obs["board_static"]; rg = obs.get("routing_geometry", {})
        self.rules = {}
        for net_key, net in bs["nets"].items():
            nid = int(net_key.split("_")[1]); rule = _rule(bs, net_key); self.rules[nid] = rule
            for pad in net["pads"].values():
                self._paint_pad(pad, nid, rule["clearance"])
            g = rg.get(net_key, {})
            for t in g.get("tracks", {}).values():
                (x0, y0), (x1, y1) = t["p1"]["xy"], t["p2"]["xy"]
                o.paint_capsule([t["layer"] - 1], x0 - self.ox, y0 - self.oy, x1 - self.ox, y1 - self.oy, t["width"] / 2, nid, rule["clearance"])
            for v in g.get("vias", {}).values():
                x, y = v["center"]["xy"]; r = v["via_width"] / 2
                o.paint_capsule([0, 1], x - self.ox, y - self.oy, x - self.ox, y - self.oy, r, nid, rule["clearance"])
                o.paint_hole(x - self.ox, y - self.oy, r * 0.6, nid)
        bc = bs["board_constraints"]
        for pad in bs.get("unconnected_pads", {}).values():
            self._paint_pad(pad, NC_NET, bc["clearance_mm"])

    def _paint_pad(self, pad, nid, clearance, target=False):
        o = self.orc; x, y = pad["center"]["xy"]; x -= self.ox; y -= self.oy
        w, h = float(pad["width"]), float(pad["height"]); pl = int(pad["layer"])
        layers = [0, 1] if pl == 0 else [pl - 1]
        if target:
            o.paint_capsule(layers, x, y, x, y, min(0.25, min(w, h) * 0.45) if pl == 0 else min(w, h) * 0.45, nid, clearance, target=True)
            return
        shape = pad.get("shape", "")
        if shape == "circle":
            o.paint_capsule(layers, x, y, x, y, w / 2, nid, clearance)
        elif shape == "oval":
            r = min(w, h) / 2; hl = (max(w, h) - min(w, h)) / 2
            if w >= h: o.paint_capsule(layers, x - hl, y, x + hl, y, r, nid, clearance)
            else: o.paint_capsule(layers, x, y - hl, x, y + hl, r, nid, clearance)
        else:
            o.paint_rect(layers, x, y, w, h, nid, clearance)
        if pl == 0:
            o.paint_hole(x, y, min(w, h) * 0.3, nid)

    # -- plan from the head to the net's copper outside the head's cluster
    def plan(self, obs: dict, nid: int, head_xy, head_layer: int, cluster: set):
        o = self.orc; o.tgt_owner[:] = 0
        bs = obs["board_static"]; net_key = f"net_{nid}"; g = obs.get("routing_geometry", {}).get(net_key, {})
        n_tgt = 0
        for pad in bs["nets"][net_key]["pads"].values():
            x, y = pad["center"]["xy"]; pl = int(pad["layer"])
            lays = (1, 2) if pl == 0 else (pl,)
            if any(_key(x, y, l) in cluster for l in lays):
                continue
            self._paint_pad(pad, nid, 0.0, target=True); n_tgt += 1
        for t in g.get("tracks", {}).values():
            (x0, y0), (x1, y1) = t["p1"]["xy"], t["p2"]["xy"]
            if _key(x0, y0, t["layer"]) in cluster or _key(x1, y1, t["layer"]) in cluster:
                continue
            o.paint_capsule([t["layer"] - 1], x0 - self.ox, y0 - self.oy, x1 - self.ox, y1 - self.oy, self.step * 0.6, nid, 0.0, target=True); n_tgt += 1
        for v in g.get("vias", {}).values():
            x, y = v["center"]["xy"]
            if any(_key(x, y, l) in cluster for l in (1, 2)):
                continue
            o.paint_capsule([0, 1], x - self.ox, y - self.oy, x - self.ox, y - self.oy, v["via_width"] * 0.45, nid, 0.0, target=True); n_tgt += 1
        if n_tgt == 0:
            return None, "no_target"
        r = self.rules[nid]
        cy, cx = self.cell(*head_xy)
        o.src_cells = [(head_layer - 1, cy, cx)]
        path = o.query(nid, r["clearance"], r["width"], r["via_dia"], [(head_layer - 1, cy, cx)], o.tgt_owner == nid)
        if path is None:
            blocked, _ = o._blocked_maps(nid, r["clearance"], r["width"], r["via_dia"])
            if blocked[head_layer - 1, cy, cx]:
                return None, "src_blocked"
            if not ((o.tgt_owner == nid) & ~blocked).any():
                return None, "target_blocked"
            return None, "no_path"
        verts = [(x + self.ox, y + self.oy, l) for x, y, l in o.path_to_polyline(path)]
        return verts, "ok"


# ---------------------------------------------------------------- executor
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


def _polyline_len(verts):
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0, _), (x1, y1, _) in zip(verts, verts[1:]))


class Executor:
    def __init__(self, env, oracle_step=0.1, via_cost=1500, max_actions=4000, log=print, trace=False):
        self.env, self.step, self.via_cost, self.max_actions, self.log, self.trace = env, oracle_step, via_cost, max_actions, log, trace

    def cand_index(self, x, y, layer=None):
        k2 = (round(float(x), 3), round(float(y), 3))
        best = None
        for i, (cx, cy, cl) in enumerate(self.env.cand_mm_list):
            if (round(cx, 3), round(cy, 3)) == k2 and (layer is None or cl == layer):
                if layer is not None:
                    return i
                best = i if best is None else best
        return best

    def act(self, at, ptr, mode=0):
        c = self.env.cand_mm_list
        tgt = c[ptr] if at in (ACT_START_ROUTE, ACT_MAKE_LINE, ACT_MAKE_VIA) and 0 <= ptr < len(c) else None
        obs, rew, term, trunc, info = self.env.step((int(at), int(ptr), int(mode)))
        self.n_act += 1; self.actions.append((int(at), int(ptr), int(mode), bool(info.get("action_success", True))))
        o = _obs(obs)
        if self.trace:
            rh = o["router_head"]
            extra = {k: v for k, v in info.items() if k in ("routed", "rejected", "via_placed", "reason", "reject_reason", "stuck", "error")}
            print(f"    [{self.n_act}] {ACTION_NAMES[at]} ptr={ptr} tgt={tgt} mode={mode} -> ok={info.get('action_success')} "
                  f"cls={info.get('action_class')} routing={rh.get('is_routing')} head={rh.get('current_xy')} L={rh.get('current_layer')} "
                  f"net={rh.get('current_net')} rats={sum(len(_ratsnest(o, n)) for n in self.env.sorted_net_codes)} {extra if extra else ''}")
        return o, term, trunc, info

    def cluster_keys(self, x, y, layer, obs=None, nid=None):
        """Connectivity cluster of (x, y, layer) as (x, y, human_layer) keys.

        Union of the engine query (same one the wrapper's candidate filter uses; empty on some
        states) and a geometric union-find over this net's pads / track endpoints / vias in
        ``obs`` (exact coordinate matching — the executor only ever targets pool points, so
        committed copper meets at candidate coordinates; thru pads and vias bridge layers).
        """
        keys = {_key(x, y, layer)}
        eng = getattr(self.env.env, "_engine", None)
        try:
            pts = eng.get_connected_points(float(x), float(y), int(layer)) if eng is not None else None
        except Exception:
            pts = None
        keys |= {_key(px, py, pl) for px, py, pl in (pts or [])}
        if obs is None or nid is None:
            return keys
        parent = {}
        def find(a):
            while parent.setdefault(a, a) != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb
        nl = max(int(obs["board_static"].get("copper_layers", 2)), 2)
        for pad in obs["board_static"]["nets"][f"net_{nid}"]["pads"].values():
            px, py = pad["center"]["xy"]; pl = int(pad["layer"])
            ks = [_key(px, py, l) for l in (range(1, nl + 1) if pl == 0 else (pl,))]
            for k in ks: union(ks[0], k)
        g = obs.get("routing_geometry", {}).get(f"net_{nid}", {})
        for t in g.get("tracks", {}).values():
            union(_key(*t["p1"]["xy"], t["layer"]), _key(*t["p2"]["xy"], t["layer"]))
        for v in g.get("vias", {}).values():
            vx, vy = v["center"]["xy"]; ks = [_key(vx, vy, l) for l in range(int(v.get("layer_start", 1)), int(v.get("layer_end", nl)) + 1)]
            for k in ks: union(ks[0], k)
        roots = {find(k) for k in list(keys) if k in parent}
        if roots:
            keys |= {k for k in parent if find(k) in roots}
        return keys

    def ensure_routing(self, obs, nid):
        """Open a route session at a ratsnest anchor of ``nid`` if none is active."""
        if obs["router_head"].get("is_routing"):
            return obs, True, False, False
        if not obs["_masks"]["action"][ACT_START_ROUTE]:
            return obs, False, False, False
        src = self._pick_source(obs, nid)
        if src is None:
            return obs, False, False, False
        obs, term, trunc, info = self.act(ACT_START_ROUTE, src)
        return obs, bool(info.get("action_success")) and bool(obs["router_head"].get("is_routing")), term, trunc

    def run(self, board: str):
        t0 = time.time(); env = self.env
        obs = _obs(env.reset()[0]); self.n_act = 0; self.actions = []
        self.lv = Counter(); self.reasons = Counter(); self.pin_fail = 0; self.oracle_fail = 0; self.oracle_len = 0.0; self.attempts = 0
        self._continue_from = None; self._bad_sources = set(); self._last_source = None; self._direct_failed_at = None; self._best_rem = 1e9; self._stall = 0; self._visited = set()
        orc = ObsOracle(obs, step=self.step, via_cost=self.via_cost)
        nets = list(env.sorted_net_codes); pins0 = sum(len(_ratsnest(obs, n)) for n in nets)
        term = trunc = False; info = {}
        for nid in nets:
            if term or trunc or self.n_act >= self.max_actions:
                break
            if not _ratsnest(obs, nid) or nid in (obs.get("closed_nets") or []):
                continue
            nv = obs["_masks"]["net_valid"]; idx = nets.index(nid)
            if idx >= len(nv) or not nv[idx]:
                continue
            obs, term, trunc, info = self.act(ACT_NET_SELECT, idx)
            pin_fails = 0; self._continue_from = None; self._bad_sources = set(); self._direct_failed_at = None
            while _ratsnest(obs, nid) and not (term or trunc) and self.n_act < self.max_actions:
                obs, term, trunc, level = self._route_pin(obs, nid, orc)
                self.lv[level] += 1
                if level < 5:
                    pin_fails = 0
                else:
                    pin_fails += 1; self.pin_fail += 1
                    if self._last_source is not None:
                        self._bad_sources.add(self._last_source)
                    self._continue_from = None
                    if pin_fails >= 4:
                        break
            if obs["_masks"]["action"][ACT_NET_END] and not (term or trunc):
                obs, term, trunc, info = self.act(ACT_NET_END, 0)
        drc_list = obs.get("drc_violations") or []
        drc = len(drc_list)
        drc_types = dict(Counter(str(v.get("type") or v.get("code") or v.get("kind") or "?") for v in drc_list if isinstance(v, dict)))
        unrouted = sum(len(_ratsnest(obs, n)) for n in nets)
        wl = _wirelength(obs)
        return dict(board=os.path.basename(board), nets=len(nets), pins=pins0 // 2, actions=self.n_act, attempts=self.attempts,
                    levels={str(k): v for k, v in sorted(self.lv.items())}, pin_fail=self.pin_fail, oracle_fail=self.oracle_fail, reasons=dict(self.reasons),
                    unrouted_pts=unrouted, drc=drc, terminated=bool(term), truncated=bool(trunc),
                    success=unrouted == 0, clean_pass=unrouted == 0 and drc == 0, drc_types=drc_types,
                    wirelength=round(wl, 2), oracle_len=round(self.oracle_len, 2), wl_ratio=round(wl / self.oracle_len, 3) if self.oracle_len else None,
                    wall_s=round(time.time() - t0, 1))

    def _pick_source(self, obs, nid):
        """Where to open the next session: the end of the stub we just committed (waypoint in progress),
        else the ratsnest anchor of this net with the smallest connected cluster (lone pin first)."""
        if self._continue_from is not None:
            i = self.cand_index(*self._continue_from)
            if i is not None:
                return i
            self._continue_from = None
        best = None
        for p in _ratsnest(obs, nid):
            x, y = p["xy"]; l = int(p.get("layer", 0))
            if (round(x, 3), round(y, 3)) in self._bad_sources:
                continue
            i = self.cand_index(x, y, l if l > 0 else None)
            if i is None:
                continue
            cl = self.env.cand_mm_list[i][2]
            size = len(self.cluster_keys(x, y, cl, obs, nid))
            if best is None or size < best[0]:
                best = (size, i)
        if best is not None:
            return best[1]
        for pad in obs["board_static"]["nets"][f"net_{nid}"]["pads"].values():
            i = self.cand_index(*pad["center"]["xy"])
            if i is not None:
                return i
        return None

    def _direct_target(self, obs, nid, verts, cluster):
        ex, ey, el = verts[-1]
        bs = obs["board_static"]; g = obs.get("routing_geometry", {}).get(f"net_{nid}", {})
        best = None
        for pad in bs["nets"][f"net_{nid}"]["pads"].values():
            x, y = pad["center"]["xy"]; d = math.hypot(x - ex, y - ey)
            if d <= 0.5 * min(pad["width"], pad["height"]) + 0.3 and (best is None or d < best[0]):
                best = (d, x, y, el + 1)
        if best is None:
            for v in g.get("vias", {}).values():
                x, y = v["center"]["xy"]; d = math.hypot(x - ex, y - ey)
                if d <= v["via_width"] and (best is None or d < best[0]):
                    best = (d, x, y, el + 1)
        if best is None:
            for t in g.get("tracks", {}).values():
                for p in (t["p1"], t["p2"]):
                    x, y = p["xy"]; d = math.hypot(x - ex, y - ey)
                    if _key(x, y, t["layer"]) in cluster:
                        continue
                    if best is None or d < best[0]:
                        best = (d, x, y, t["layer"])
        if best is None:
            return None
        return self.cand_index(best[1], best[2], best[3])

    def _plan_here(self, obs, nid, orc):
        rh = obs["router_head"]; head = tuple(rh["current_xy"]); hl = int(rh["current_layer"])
        cluster = self.cluster_keys(head[0], head[1], hl, obs, nid)
        orc.resync(obs)
        verts, why = orc.plan(obs, nid, head, hl, cluster)
        return head, hl, cluster, verts, why

    def _progressed(self, obs_before, obs_after, info, nid, head, hl):
        # every make_line/make_via commits AND closes the session (force_finish) in this env, so
        # progress == committed geometry (env class valid_effective) or a pin got connected.
        return bool(info.get("action_success", False)) and (
            info.get("action_class") == "valid_effective" or len(_ratsnest(obs_after, nid)) < len(_ratsnest(obs_before, nid)))

    def _route_pin(self, obs, nid, orc):
        """One pin: ladder of attempts, each = (ensure session) + plan + one action. Returns (obs, term, trunc, level)."""
        first_plan = True
        for level, mode, waypoint in ((1, MODE_WALK, False), (2, MODE_SHOVE, False), (3, MODE_WALK, True), (4, MODE_SHOVE, True)):
            slots = range(6) if waypoint else (0,)
            for slot in slots:
                obs, ok, term, trunc = self.ensure_routing(obs, nid)
                if term or trunc: return obs, term, trunc, 5
                if not ok:
                    self.log(f"  net {nid}: cannot open a route session"); return obs, False, False, 5
                head, hl, cluster, verts, why = self._plan_here(obs, nid, orc)
                self._last_source = (round(head[0], 3), round(head[1], 3))
                if verts is not None:
                    rem = _polyline_len(verts)
                    if self._continue_from is None:        # a fresh pin
                        self._best_rem, self._stall = rem, 0
                    elif rem < self._best_rem - 0.2:
                        self._best_rem, self._stall = rem, 0
                    else:
                        self._stall += 1
                        if self._stall >= 3:               # three waypoint steps without real progress
                            self.reasons["stall"] += 1; return obs, False, False, 5
                if verts is None:
                    # oracle says no corridor — PNS is the judge: still try a direct line to the
                    # nearest unconnected pad (walk, then shove) before giving the pin up
                    self.oracle_fail += 1; self.reasons[why] += 1; self.log(f"  net {nid}: plan {why}")
                    return self._blind_direct(obs, nid, head, hl, cluster)
                if first_plan and self._continue_from is None:
                    self.oracle_len += _polyline_len(verts)      # once per pin (not per waypoint step)
                first_plan = False
                if not waypoint:
                    cont_l = hl
                    if any(l != verts[0][2] for _, _, l in verts):
                        break          # target on the other layer: a via is needed first
                    df = self._direct_failed_at
                    # after a direct failure, retry direct only when the remaining oracle path is short
                    # (< 8 mm) or we have advanced well past the failed head (> 40% of that path)
                    if df is not None and _polyline_len(verts) >= 8.0 and math.hypot(df[0] - head[0], df[1] - head[1]) < 0.4 * df[2]:
                        break
                    if level == 2 and df is not None and df[3] == (round(head[0], 3), round(head[1], 3)) and df[4]:
                        break          # shove-direct already failed from this exact head
                    tgt = self._direct_target(obs, nid, verts, cluster); at = ACT_MAKE_LINE
                    if tgt is None:
                        break
                else:
                    opts = [o for o in self._waypoints(obs, nid, verts, head, hl, orc)
                            if (round(self.env.cand_mm_list[o[0]][0], 2), round(self.env.cand_mm_list[o[0]][1], 2)) not in self._visited]
                    if slot >= len(opts):
                        break
                    tgt, at, cont_l = opts[slot]
                self.attempts += 1
                tx, ty, tl = self.env.cand_mm_list[tgt]
                if self.trace:
                    print(f"      plan: head={tuple(round(v, 2) for v in head)} L{hl} remaining={_polyline_len(verts):.1f}mm verts={len(verts)} end={tuple(round(v, 2) for v in verts[-1])} -> level{level} {ACTION_NAMES[at]} ({tx:.1f},{ty:.1f})")
                o2, term, trunc, info = self.act(at, tgt, mode)
                if self._progressed(obs, o2, info, nid, head, hl):
                    connected = len(_ratsnest(o2, nid)) < len(_ratsnest(obs, nid))
                    self._visited.add((round(head[0], 2), round(head[1], 2)))
                    self._continue_from = None if connected else (tx, ty, cont_l)
                    if connected:
                        self._visited = set()
                    if connected:
                        self._direct_failed_at = None
                    return o2, term, trunc, level
                if not waypoint:
                    self._direct_failed_at = (head[0], head[1], _polyline_len(verts), (round(head[0], 3), round(head[1], 3)), level == 2)
                obs = o2
                if term or trunc: return obs, term, trunc, 5
        return obs, False, False, 5

    def _blind_direct(self, obs, nid, head, hl, cluster):
        """No oracle path: try make_line to the nearest same-layer unconnected pad, walk then shove."""
        bs = obs["board_static"]; best = None
        for pad in bs["nets"][f"net_{nid}"]["pads"].values():
            x, y = pad["center"]["xy"]; pl = int(pad["layer"])
            if pl not in (0, hl) or any(_key(x, y, l) in cluster for l in ((1, 2) if pl == 0 else (pl,))):
                continue
            d = math.hypot(x - head[0], y - head[1])
            if best is None or d < best[0]:
                best = (d, x, y)
        if best is None:
            return obs, False, False, 5
        for mode in (MODE_WALK, MODE_SHOVE):
            obs, ok, term, trunc = self.ensure_routing(obs, nid)
            if not ok or term or trunc:
                return obs, term, trunc, 5
            tgt = self.cand_index(best[1], best[2], hl)
            if tgt is None:
                return obs, False, False, 5
            self.attempts += 1
            o2, term, trunc, info = self.act(ACT_MAKE_LINE, tgt, mode)
            if self._progressed(obs, o2, info, nid, head, hl):
                self.reasons["no_path_but_pns_ok"] += 1
                return o2, term, trunc, 1 if mode == MODE_WALK else 2
            obs = o2
        return obs, False, False, 5

    def _waypoints(self, obs, nid, verts, head, hl, orc):
        """Directional candidates ranked by progress along the oracle polyline (farthest first).

        A candidate counts if it lies within ``tol`` of the polyline on the head's layer; the
        polyline's first via (layer change) caps the usable progress — the candidate nearest
        the via point is emitted as a make_via instead.
        """
        pts = [(x, y, l) for x, y, l in verts]
        # arc-length table on the head layer up to the first via
        seg, cum, via_at = [], [0.0], None
        for (x0, y0, l0), (x1, y1, l1) in zip(pts, pts[1:]):
            if l1 != l0:
                via_at = (x0, y0, cum[-1]); break
            seg.append(((x0, y0), (x1, y1))); cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
        if not seg:
            return []
        width = orc.rules[nid]["width"]; tol = max(0.3, width)
        scored = []
        for i, (cx, cy, cl) in enumerate(self.env.cand_mm_list):
            if cl != hl or (abs(cx - head[0]) < 1e-6 and abs(cy - head[1]) < 1e-6):
                continue
            best = None
            for k, ((x0, y0), (x1, y1)) in enumerate(seg):
                dx, dy = x1 - x0, y1 - y0; L2 = dx * dx + dy * dy
                t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((cx - x0) * dx + (cy - y0) * dy) / L2))
                px, py = x0 + t * dx, y0 + t * dy; d = math.hypot(cx - px, cy - py)
                prog = cum[k] + t * math.sqrt(L2)
                if d <= tol and (best is None or prog > best[0]):
                    best = (prog, d)
            if best is not None and best[0] > 0.05:
                scored.append((best[0], -best[1], i))
        scored.sort(reverse=True)
        out = []
        via_imminent = False
        if via_at is not None and obs["_masks"]["action"][ACT_MAKE_VIA]:
            vx, vy, vs = via_at
            via_imminent = vs < 1.0
            # layer after the via = the polyline's layer past the change (human index)
            next_l = next((l for (x, y, l) in pts if l != pts[0][2]), None)
            next_l = hl if next_l is None else int(next_l) + 1
            near = [(math.hypot(cx - vx, cy - vy), i) for i, (cx, cy, cl) in enumerate(self.env.cand_mm_list)
                    if cl == hl and not (abs(cx - head[0]) < 1e-6 and abs(cy - head[1]) < 1e-6)]
            near.sort()
            for d, i in near[:6]:
                if d <= 2.5:
                    out.append((i, ACT_MAKE_VIA, next_l))
        if not via_imminent:      # when the via is due right here, tiny same-layer moves just oscillate
            out.extend((i, ACT_MAKE_LINE, hl) for _, _, i in scored[:4])
        return out


# ---------------------------------------------------------------- main
def build_env(board, ckpt, max_steps=4096, obs_format="json"):
    from cadet.base_args import load_base_args
    from configs.loader.schema import RLEnvConfig
    from methods.rl_agent.wrappers.factory import make_decoder_env
    a = load_base_args(ckpt)
    kw = RLEnvConfig.from_checkpoint(a, max_steps).to_pool_kwargs()
    kw["keep_routing_fraction"] = None; kw["obs_format"] = obs_format
    return make_decoder_env(board, seed=0, policy_net_select=True, **kw), kw


CKPT = str(CADET / "cadet/configs/base_args/generation.json")   # base arguments (cadet/base_args.py); a trainer checkpoint also works
VAL = str(CADET / "var/datasets/routedemos/val")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default=VAL); ap.add_argument("--n", type=int, default=2); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ckpt", default=CKPT); ap.add_argument("--out", default=str(CADET / "var/results/search/executor.jsonl"))
    ap.add_argument("--quiet", action="store_true"); ap.add_argument("--trace", action="store_true"); ap.add_argument("--max-actions", type=int, default=4000); ap.add_argument("--step", type=float, default=0.1); ap.add_argument("--inflate", type=float, default=0.0); ap.add_argument("--margin", type=float, default=0.0)
    a = ap.parse_args()
    boards = sorted(glob.glob(a.boards + "/board_*.kicad_pcb"))[a.start:a.start + a.n]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    log = (lambda *x: None) if a.quiet else print
    with open(a.out, "a") as f:
        for b in boards:
            env, kw = build_env(b, a.ckpt); _CachedOracle.inflate = a.inflate; _CachedOracle.margin = a.margin
            try:
                ex = Executor(env, log=log, trace=a.trace, max_actions=a.max_actions, oracle_step=a.step); r = ex.run(b)
            finally:
                env.close()
            print(json.dumps(r)); f.write(json.dumps(r) + "\n"); f.flush()
