"""Grid routability oracle: 2-layer configuration-space grid + Dijkstra (numba).

Conservative by construction: a cell is blocked if the inflated obstacle
(clearance + width/2) covers its center; diagonal moves are refused when either
orthogonal neighbour is blocked (no corner cutting).  A "yes" is therefore
realizable; a "no" may be over-strict at grid resolution.
"""
from __future__ import annotations
import heapq, math
import numpy as np
from numba import njit
from scipy import ndimage

VIA_COST = 300     # in 1/10 cell units (orth=10, diag=14)


TURN_COST = 6      # per direction change (orth=10, diag=14) — removes staircase micro-steps
NDIR = 9           # 8 move directions + 8 = "no direction yet" (start / after via)
_DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64)
_DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64)


@njit(cache=True)
def _dijkstra(blocked, via_ok, src, target, parent, via_cost):
    """State = (cell, dir). blocked: (2,ny,nx) bool; via_ok: (ny,nx) bool; src: (k,3) int [l,y,x];
    target: (2,ny,nx) bool. parent: flat int64 array (2*ny*nx*NDIR), -1 init. Returns reached state or -1."""
    L, ny, nx = blocked.shape
    ncell = L * ny * nx
    dist = np.full(ncell * NDIR, np.iinfo(np.int64).max, np.int64)
    heap = [(np.int64(0), np.int64(0))]
    heap.pop()
    for i in range(src.shape[0]):
        l, y, x = src[i, 0], src[i, 1], src[i, 2]
        if blocked[l, y, x]:
            continue
        st = ((l * ny + y) * nx + x) * NDIR + 8
        dist[st] = 0
        parent[st] = st
        heapq.heappush(heap, (np.int64(0), np.int64(st)))
    while len(heap) > 0:
        d, st = heapq.heappop(heap)
        if d > dist[st]:
            continue
        idx = st // NDIR; dr = st % NDIR
        l = idx // (ny * nx); r = idx % (ny * nx); y = r // nx; x = r % nx
        if target[l, y, x]:
            return st
        for k in range(8):
            yy = y + _DY[k]; xx = x + _DX[k]
            if yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            if blocked[l, yy, xx]:
                continue
            if _DY[k] != 0 and _DX[k] != 0:
                if blocked[l, y, xx] or blocked[l, yy, x]:
                    continue
                nd = d + 14
            else:
                nd = d + 10
            if dr != 8 and dr != k:
                nd += TURN_COST
            j = ((l * ny + yy) * nx + xx) * NDIR + k
            if nd < dist[j]:
                dist[j] = nd; parent[j] = st
                heapq.heappush(heap, (np.int64(nd), np.int64(j)))
        if via_ok[y, x]:
            l2 = 1 - l
            if not blocked[l2, y, x]:
                j = ((l2 * ny + y) * nx + x) * NDIR + 8
                nd = d + via_cost
                if nd < dist[j]:
                    dist[j] = nd; parent[j] = st
                    heapq.heappush(heap, (np.int64(nd), np.int64(j)))
    return -1


class GridOracle:
    def __init__(self, bw: float, bh: float, step: float, geo, edge_clear: float = 0.3, via_cost: int = VIA_COST):
        self.bw, self.bh, self.step = bw, bh, step
        self.via_cost = int(via_cost)
        self.nx = int(math.ceil(bw / step)) + 1
        self.ny = int(math.ceil(bh / step)) + 1
        self.edge_clear = edge_clear
        xs = np.arange(self.nx) * step
        ys = np.arange(self.ny) * step
        inside = np.zeros((self.ny, self.nx), bool)
        m = step * 0.5   # cells within half a step of the edge count as "outside" so the EDT measures edge distance
        if geo is None:
            inside[:] = (ys[:, None] >= m) & (ys[:, None] <= bh - m) & (xs[None, :] >= m) & (xs[None, :] <= bw - m)
        else:
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    if not geo.outline.contains(x, y, m):
                        continue
                    ok = True
                    for c in geo.cutouts:
                        if c.blocks(x, y, 0.0): ok = False; break
                    if ok:
                        for h in geo.holes:
                            if h.blocks(x, y, 0.0): ok = False; break
                    inside[iy, ix] = ok
        # distance (mm) from each cell to the nearest non-inside cell (edge/cutout/NPTH)
        self.static_dist = ndimage.distance_transform_edt(inside).astype(np.float32) * step
        self.owner = np.zeros((2, self.ny, self.nx), np.int32)     # net id of copper (0 = none)
        self.clear = np.zeros((2, self.ny, self.nx), np.float32)   # clearance of that copper's net
        self.hole_owner = np.zeros((self.ny, self.nx), np.int32)   # drilled zones (THT pads, vias)
        self.tgt_owner = np.zeros((2, self.ny, self.nx), np.int32)  # copper a route may legally END on (real pad / track)

    # ---- painting -------------------------------------------------------
    def _window(self, x0, y0, x1, y1, r):
        s = self.step
        ix0 = max(0, int((min(x0, x1) - r) / s) - 1); ix1 = min(self.nx - 1, int((max(x0, x1) + r) / s) + 2)
        iy0 = max(0, int((min(y0, y1) - r) / s) - 1); iy1 = min(self.ny - 1, int((max(y0, y1) + r) / s) + 2)
        X = (np.arange(ix0, ix1 + 1) * s)[None, :]
        Y = (np.arange(iy0, iy1 + 1) * s)[:, None]
        return ix0, ix1, iy0, iy1, X, Y

    def _paint(self, layers, mask_fn, x0, y0, x1, y1, r, net, clearance, target=False):
        ix0, ix1, iy0, iy1, X, Y = self._window(x0, y0, x1, y1, r)
        m = mask_fn(X, Y)
        for l in layers:
            if target:
                self.tgt_owner[l, iy0:iy1 + 1, ix0:ix1 + 1][m] = net
                continue
            o = self.owner[l, iy0:iy1 + 1, ix0:ix1 + 1]
            c = self.clear[l, iy0:iy1 + 1, ix0:ix1 + 1]
            o[m] = net; c[m] = clearance

    def paint_capsule(self, layers, x0, y0, x1, y1, r, net, clearance, target=False):
        if not target:
            r = r + self.step * 0.71      # conservative: any cell the shape touches counts as copper
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
            w, h = w + self.step * 1.42, h + self.step * 1.42
        def f(X, Y):
            return (np.abs(X - cx) <= w / 2) & (np.abs(Y - cy) <= h / 2)
        self._paint(layers, f, cx, cy, cx, cy, max(w, h) / 2, net, clearance, target)

    def paint_target_cells(self, cells, net):
        """Mark exact path cells (layer,y,x) as legal landing points — endpoints land on the centerline."""
        for l, y, x in cells:
            self.tgt_owner[l, y, x] = net

    def paint_hole(self, cx, cy, r, net):
        r = r + self.step * 0.71
        ix0, ix1, iy0, iy1, X, Y = self._window(cx, cy, cx, cy, r)
        m = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
        self.hole_owner[iy0:iy1 + 1, ix0:ix1 + 1][m] = net

    def site_free(self, x, y, r, layers, net, any_hole=False):
        """True if no other-net copper lies within r of (x, y) on the given layers.
        any_hole=True (THT pin): also refuse drilled zones of the SAME net (hole-to-hole)."""
        ix0, ix1, iy0, iy1, X, Y = self._window(x, y, x, y, r)
        m = (X - x) ** 2 + (Y - y) ** 2 <= r * r
        for l in layers:
            o = self.owner[l, iy0:iy1 + 1, ix0:ix1 + 1]
            if np.any(m & (o != 0) & (o != net)):
                return False
        h = self.hole_owner[iy0:iy1 + 1, ix0:ix1 + 1]
        return not np.any(m & (h != 0) & ((h != net) | any_hole))

    def _shape_conflict(self, layers, mask_fn, x0, y0, x1, y1, r, net, any_copper):
        ix0, ix1, iy0, iy1, X, Y = self._window(x0, y0, x1, y1, r)
        m = mask_fn(X, Y)
        for l in layers:
            o = self.owner[l, iy0:iy1 + 1, ix0:ix1 + 1]
            if np.any(m & (o != 0) & ((o != net) | any_copper)):
                return True
        return False

    def pad_site_free(self, x, y, w, h, thru, layers, net, clearance, drill=0.0):
        """Exact-shape site check: pad body must not touch ANY copper; pad dilated by
        clearance must not touch other-net copper; THT hole must keep hole-to-hole distance."""
        rect = lambda dw: (lambda X, Y: (np.abs(X - x) <= w / 2 + dw) & (np.abs(Y - y) <= h / 2 + dw))
        if thru:
            rr = min(w, h) / 2; hl = (max(w, h) - min(w, h)) / 2
            ax0, ay0, ax1, ay1 = (x - hl, y, x + hl, y) if w >= h else (x, y - hl, x, y + hl)
            def caps(dw):
                def f(X, Y):
                    dx, dy = ax1 - ax0, ay1 - ay0; L2 = dx * dx + dy * dy
                    t = np.clip(((X - ax0) * dx + (Y - ay0) * dy) / L2, 0, 1) if L2 > 0 else 0
                    return (X - (ax0 + t * dx)) ** 2 + (Y - (ay0 + t * dy)) ** 2 <= (rr + dw) ** 2
                return f
            body, dil = caps(self.step * 0.5), caps(clearance + self.step * 0.5)
            ix0, ix1, iy0, iy1, X, Y = self._window(x, y, x, y, drill / 2 + 0.5)
            hz = (X - x) ** 2 + (Y - y) ** 2 <= (drill / 2 + 0.45) ** 2
            if np.any(hz & (self.hole_owner[iy0:iy1 + 1, ix0:ix1 + 1] != 0)):
                return False
        else:
            body, dil = rect(self.step * 0.5), rect(clearance + self.step * 0.5)
        r = max(w, h) / 2 + clearance + 1.0
        if self._shape_conflict(layers, body, x, y, x, y, r, net, True):
            return False
        return not self._shape_conflict(layers, dil, x, y, x, y, r, net, False)

    # ---- query ----------------------------------------------------------
    def _blocked_maps(self, net, clearance, width, via_dia):
        """Return blocked (2,ny,nx) for track width and via_ok (ny,nx)."""
        s = self.step
        other = (self.owner != 0) & (self.owner != net)
        r_track = np.full((2, self.ny, self.nx), np.inf, np.float32)   # dist to nearest other copper minus its clearance req
        blocked = np.zeros((2, self.ny, self.nx), bool)
        via_bad = np.zeros((self.ny, self.nx), bool)
        for l in range(2):
            for ck in np.unique(self.clear[l][other[l]]):
                m = other[l] & (self.clear[l] == ck)
                d = ndimage.distance_transform_edt(~m) * s
                req = max(clearance, float(ck))
                blocked[l] |= d < req + width / 2 + s * 0.5
                via_bad |= d < req + via_dia / 2 + s * 0.5
        edge_req = max(self.edge_clear, clearance)
        for l in range(2):
            blocked[l] |= self.static_dist < edge_req + width / 2 + s * 0.5
        via_bad |= self.static_dist < edge_req + via_dia / 2 + s * 0.5
        hole = self.hole_owner != 0
        if hole.any():                                # hole-to-hole (any net): via drill edge >= 0.3 from other hole edges
            dh = ndimage.distance_transform_edt(~hole) * s
            via_bad |= dh < via_dia * 0.35 + 0.3 + s * 0.5
        return blocked, ~via_bad

    def cell(self, x, y):
        return (int(round(y / self.step)), int(round(x / self.step)))

    def query(self, net, clearance, width, via_dia, src_cells, target_mask, via_block_r=0.0):
        """src_cells: list of (layer,y,x). target_mask: (2,ny,nx) bool of own copper.
        via_block_r: forbid vias within this radius (mm) of each source cell — the new
        pin's own pad is not painted yet, so a THT pin's hole must be excluded here
        (hole-to-hole). Returns list of (layer,y,x) cells from a source to the target, or None."""
        blocked, via_ok = self._blocked_maps(net, clearance, width, via_dia)
        if via_block_r > 0:
            rc = int(math.ceil(via_block_r / self.step))
            for l, y, x in src_cells:
                y0, y1 = max(0, y - rc), min(self.ny - 1, y + rc); x0, x1 = max(0, x - rc), min(self.nx - 1, x + rc)
                Y, X = np.ogrid[y0:y1 + 1, x0:x1 + 1]
                via_ok[y0:y1 + 1, x0:x1 + 1] &= ((Y - y) ** 2 + (X - x) ** 2) * self.step ** 2 > via_block_r ** 2
        src = np.array([c for c in src_cells if not blocked[c[0], c[1], c[2]]], np.int64).reshape(-1, 3)
        if src.shape[0] == 0:          # the pin's own cell already violates clearance (e.g. track wider than a thin pad next to other copper)
            return None
        for l, y, x in src:            # no via right at a pin or at the landing cell
            via_ok[y, x] = False
        via_ok &= ~(target_mask[0] | target_mask[1])
        tgt = target_mask & ~blocked   # must be able to stand on the target cell
        if not tgt.any():
            return None
        parent = np.full(2 * self.ny * self.nx * NDIR, -1, np.int64)
        end = _dijkstra(blocked, via_ok, src, tgt, parent, np.int64(self.via_cost))
        if end < 0:
            return None
        path = []
        st = end
        while True:
            idx = st // NDIR
            l = idx // (self.ny * self.nx); r = idx % (self.ny * self.nx)
            cell = (int(l), int(r // self.nx), int(r % self.nx))
            if not path or path[-1] != cell:
                path.append(cell)
            if parent[st] == st:
                break
            st = parent[st]
        path.reverse()
        return path

    def path_to_polyline(self, path):
        """Compress a cell path into vertices [(x_mm, y_mm, layer)], keeping
        direction changes and via points (layer change at the same cell)."""
        verts = [path[0]]
        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            d1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2]); d2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
            if d1 != d2:
                verts.append(b)
        if len(path) > 1:
            verts.append(path[-1])
        return [(x * self.step, y * self.step, l) for (l, y, x) in verts]
