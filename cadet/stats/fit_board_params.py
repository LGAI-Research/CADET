#!/usr/bin/env python3
"""Estimate the synthetic-board generator's target statistics from a real-board split and write them to one json.

Reads the boards of ``<split_json>[<difficulty>][<split>]`` (default: the generator-fit half of the D3 medium boards,
``cadet/configs/splits/d3_medium_half.json`` / ``fit``) and writes ``fit_params.json``: the ONLY input the generator's
front-end consumes, so which real boards shaped the synthetic distribution is recorded in one place (``source``).

Schema (all lengths mm, areas cm^2):
  source        {split_json, difficulty, split, n_boards, boards, excluded, filters}
  size          {dist: lognormal, median, sigma, clip: [lo, hi]}            side s = sqrt(W*H) of the Edge.Cuts bbox
  aspect        {square_p, sigma, max, observed_max}                         log(long/short) ~ |N(0, sigma)| for non-square boards; max = p99 clip
  net_density   {dist: lognormal, median, sigma, nets_clip: [2, max]}       nets (>= 2 pads) per cm^2
  locality      {dist: lognormal, median_frac_diag, sigma}                  same-net MST edge length / board diagonal
  pins_per_net  {pool: [...], min, max}                                      pads per net, empirical pool (drawn with replacement)
  pad_smd       [[w, h], ...]            pad_tht  [[w, h, drill], ...]       empirical pools
  netclass_sets [[{clearance, width, via_dia, via_drill, w}, ...], ...]     one entry per board = its class set (w = nets in class)
  layer_mix     [[f_tht, f_F_of_smd], ...]                                   one pair per board
  boards        [{id, W, H, area_cm2, n_nets, pins, f_tht, f_F, n_classes}]  per-board rows for reports

Lognormal parameters are the closed-form MLE with location 0 (median = exp(mean(ln x)), sigma = std(ln x)).
Pure stdlib: the s-expression reader below is local, so the script runs without the router engine.

    python cadet/stats/fit_board_params.py --split fit  --out /tmp/d3b_fit_half.json
    python cadet/stats/fit_board_params.py --split test --out /tmp/d3b_test_half.json
"""
from __future__ import annotations
import argparse, json, math, os, statistics, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLIT_JSON = REPO / "cadet/configs/splits/d3_medium_half.json"
PRO_NAME = "processed_v9_guide_v3.kicad_pro"
FILTERS = dict(area_cm2_min=1.0, pins_per_net=[2, 64], smd_min_side=0.2, smd_max_side=6.0, tht_drill_min=0.3, tht_max_side=6.0,
               class_min_clearance=0.05, class_min_width=0.05, mst_pads_cap=400)


# ---------------------------------------------------------------- minimal KiCad s-expression reader
def load_sexp(text: str):
    """Parse a KiCad s-expression into nested lists; atoms stay strings (quoted strings keep their quotes)."""
    tok = []; i = 0; n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n": i += 1
        elif c == "(" or c == ")": tok.append(c); i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            tok.append(text[i:j + 1]); i = j + 1
        else:
            j = i
            while j < n and text[j] not in " \t\r\n()": j += 1
            tok.append(text[i:j]); i = j
    stack = [[]]
    for t in tok:
        if t == "(": stack.append([])
        elif t == ")": node = stack.pop(); stack[-1].append(node)
        else: stack[-1].append(t)
    return stack[0][0]


def find_all(node, key):
    return [x for x in node if isinstance(x, list) and x and x[0] == key] if isinstance(node, list) else []


def find_first(node, key):
    return next((x for x in node if isinstance(x, list) and x and x[0] == key), None)


def unq(s): return s[1:-1] if isinstance(s, str) and len(s) >= 2 and s[0] == '"' else s
def num(s): return float(unq(s))


# ---------------------------------------------------------------- per-board extraction
def shape_extent(kind: str, p: dict, poly: list) -> list:
    """Points whose bbox is the Edge.Cuts shape's bbox. A circle's defining points (center + one rim point) and an arc's
    (start/mid/end) do not bound the shape — a round board read as a radius-long segment — so a circle contributes its four
    axis extremes and an arc its ends, its mid and every axis extreme its sweep passes."""
    pts = list(p.values()) + poly
    if kind == "circle" and "center" in p and "end" in p:
        (cx, cy), r = p["center"], math.dist(p["center"], p["end"])
        return [(cx - r, cy), (cx + r, cy), (cx, cy - r), (cx, cy + r)]
    if kind == "arc" and {"start", "mid", "end"} <= p.keys():
        (ax, ay), (bx, by), (qx, qy) = p["start"], p["mid"], p["end"]
        d = 2 * (ax * (by - qy) + bx * (qy - ay) + qx * (ay - by))
        if abs(d) < 1e-12: return pts                                    # collinear: a straight segment
        cx = ((ax * ax + ay * ay) * (by - qy) + (bx * bx + by * by) * (qy - ay) + (qx * qx + qy * qy) * (ay - by)) / d
        cy = ((ax * ax + ay * ay) * (qx - bx) + (bx * bx + by * by) * (ax - qx) + (qx * qx + qy * qy) * (bx - ax)) / d
        r = math.dist((cx, cy), (ax, ay)); tau = 2 * math.pi
        a0, am, a1 = (math.atan2(y - cy, x - cx) for x, y in (p["start"], p["mid"], p["end"]))
        ccw = (am - a0) % tau <= (a1 - a0) % tau                         # does the sweep start -> end in + angle pass mid?
        lo, span = (a0, (a1 - a0) % tau) if ccw else (a1, (a0 - a1) % tau)
        ext = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in (0, math.pi / 2, math.pi, 3 * math.pi / 2) if (t - lo) % tau <= span]
        x0, x1, y0, y1 = min(x for x, _ in pts), max(x for x, _ in pts), min(y for _, y in pts), max(y for _, y in pts)
        # keep only extremes that widen the defining points' bbox: an extreme sitting on an end point (rounded corner) would
        # otherwise add float noise to W/H
        return pts + [(x, y) for x, y in ext if x < x0 - 1e-6 or x > x1 + 1e-6 or y < y0 - 1e-6 or y > y1 + 1e-6]
    return pts


def extract(pcb_path: Path, pro_path: Path) -> dict | None:
    pcb = load_sexp(pcb_path.read_text())
    xs, ys = [], []

    def add_shape(kind, g, tf=lambda x, y: (x, y)):
        p = {k: tf(num(it[1]), num(it[2])) for k in ("start", "end", "mid", "center") for it in [find_first(g, k)] if it}
        poly = [tf(num(pt[1]), num(pt[2])) for ps in find_all(g, "pts") for pt in find_all(ps, "xy")]
        for x, y in shape_extent(kind, p, poly): xs.append(x); ys.append(y)

    for key in ("gr_line", "gr_arc", "gr_circle", "gr_rect", "gr_poly"):
        for g in find_all(pcb, key):
            lay = find_first(g, "layer")
            if lay and unq(lay[1]) != "Edge.Cuts": continue
            add_shape(key[3:], g)
    nets: dict[str, list] = {}; smd = []; tht = []; kinds = Counter()
    for fp in find_all(pcb, "footprint"):
        at = find_first(fp, "at"); fx, fy = num(at[1]), num(at[2]); ang = math.radians(num(at[3]) if len(at) > 3 else 0.0)
        # footprint-owned Edge.Cuts shapes are part of the outline too (KiCad rule; some boards draw the whole outline inside a
        # footprint) — local coordinates, rotated/translated like the pads
        for key in ("fp_line", "fp_arc", "fp_circle", "fp_rect", "fp_poly"):
            for g in find_all(fp, key):
                lay = find_first(g, "layer")
                if not lay or unq(lay[1]) != "Edge.Cuts": continue
                add_shape(key[3:], g, lambda lx, ly: (fx + lx * math.cos(ang) - ly * math.sin(ang), fy + lx * math.sin(ang) + ly * math.cos(ang)))
        for pad in find_all(fp, "pad"):
            if pad[2] == "np_thru_hole": continue
            net = find_first(pad, "net")
            if not net or str(net[1]) == "0": continue
            pat = find_first(pad, "at"); px, py = num(pat[1]), num(pat[2])
            x, y = fx + px * math.cos(ang) - py * math.sin(ang), fy + px * math.sin(ang) + py * math.cos(ang)
            sz = find_first(pad, "size"); w, h = num(sz[1]), num(sz[2]); dr = find_first(pad, "drill")
            layers = {unq(l) for l in (find_first(pad, "layers") or [])[1:]}
            cu = {l for l in layers if l.endswith(".Cu")}
            thru = pad[2] == "thru_hole" or "*.Cu" in cu or "F&B.Cu" in cu or {"F.Cu", "B.Cu"} <= cu
            kinds["T" if thru else ("F" if "F.Cu" in cu else "B")] += 1
            nets.setdefault(str(net[1]), []).append((x, y))
            drill = next((num(t) for t in (dr[1:] if dr else []) if isinstance(t, str) and t != "oval"), 0.0)   # (drill [oval] d [(offset ..)])
            (tht if thru else smd).append((w, h, drill))
    if not xs: return None
    W, H = max(xs) - min(xs), max(ys) - min(ys); diag = math.hypot(W, H)
    nets = {k: v for k, v in nets.items() if len(v) >= 2}
    edges = []
    for pts in nets.values():
        P = pts[:FILTERS["mst_pads_cap"]]; n = len(P); dist = [math.inf] * n; used = [False] * n; dist[0] = 0.0
        for _ in range(n):                                   # Prim's MST, O(n^2) on <= 400 pads
            i = min((j for j in range(n) if not used[j]), key=lambda j: dist[j]); used[i] = True
            if dist[i] > 0: edges.append(dist[i] / diag)
            for j in range(n):
                if not used[j]: dist[j] = min(dist[j], math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]))
    classes = []
    if pro_path.exists():
        p = json.load(open(pro_path)); pats = Counter(x["netclass"] for x in p["net_settings"].get("netclass_patterns", []))
        classes = [dict(name=c["name"], clearance=c.get("clearance"), width=c.get("track_width"), via=c.get("via_diameter"), drill=c.get("via_drill"), n_nets=pats.get(c["name"], 0))
                   for c in p["net_settings"]["classes"]]
    n_pads = sum(kinds.values())
    return dict(W=W, H=H, area=W * H, n_nets=len(nets), pins_per_net=[len(v) for v in nets.values()], smd=smd, tht=tht, mst_edges=edges, classes=classes,
                f_tht=kinds["T"] / max(1, n_pads), f_F=kinds["F"] / max(1, kinds["F"] + kinds["B"]))


def class_set(r: dict) -> list[dict]:
    """A board's usable netclass set (the generator's copy unit); [] if none survives the filters."""
    cs = [c for c in r["classes"] if c["n_nets"] > 0 and c["width"] and c["clearance"]]
    if not cs:
        cs = [dict(c, n_nets=1) for c in r["classes"] if c["name"] == "Default" and c["width"] and c["clearance"]]
    cs = [dict(clearance=float(c["clearance"]), width=float(c["width"]), via_dia=float(c["via"] or 0.6), via_drill=float(c["drill"] or 0.4), w=int(c["n_nets"])) for c in cs]
    return [c for c in cs if c["via_drill"] < c["via_dia"] and c["clearance"] >= FILTERS["class_min_clearance"] and c["width"] >= FILTERS["class_min_width"]]


def lognormal(x: list[float]) -> tuple[float, float]:
    lx = [math.log(v) for v in x if v > 0]
    return math.exp(statistics.fmean(lx)), statistics.pstdev(lx)


def aggregate(rows: dict[str, dict], meta: dict) -> dict:
    B = {b: r for b, r in rows.items() if r["area"] / 100.0 > FILTERS["area_cm2_min"] and r["n_nets"] >= 1}
    excluded = sorted(set(rows) - set(B))
    sides = [math.sqrt(r["area"]) for r in B.values()]
    la = [math.log(max(r["W"], r["H"]) / min(r["W"], r["H"])) for r in B.values()]
    nonsq = [v for v in la if v > 1e-6]
    dens = [r["n_nets"] / (r["area"] / 100.0) for r in B.values()]
    edges = [e for r in B.values() for e in r["mst_edges"]]
    lo, hi = FILTERS["pins_per_net"]
    pool = [f for r in B.values() for f in r["pins_per_net"] if lo <= f <= hi]
    smd = [list(t) for r in B.values() for w, h, _ in r["smd"] if FILTERS["smd_min_side"] <= min(w, h) and max(w, h) <= FILTERS["smd_max_side"] for t in [(round(w, 3), round(h, 3))]]
    tht = [[round(w, 3), round(h, 3), round(d, 3)] for r in B.values() for w, h, d in r["tht"] if FILTERS["tht_drill_min"] <= d < min(w, h) <= FILTERS["tht_max_side"]]
    csets = [cs for cs in (class_set(r) for r in B.values()) if cs]
    s_med, s_sig = lognormal(sides); d_med, d_sig = lognormal(dens); l_med, l_sig = lognormal(edges)
    return dict(
        source=dict(**meta, n_boards=len(B), boards=sorted(B), excluded=excluded, filters=FILTERS),
        size=dict(dist="lognormal", median=round(s_med, 3), sigma=round(s_sig, 4), clip=[math.floor(min(sides)), math.ceil(max(sides))]),
        aspect=dict(square_p=round(1 - len(nonsq) / len(la), 4), sigma=round(math.sqrt(statistics.fmean(v * v for v in nonsq)), 4),
                    max=round(math.exp(sorted(la)[int(0.99 * (len(la) - 1))]), 3), observed_max=round(math.exp(max(la)), 3)),   # clip at p99: one outlier board must not set the range
        net_density=dict(dist="lognormal", median=round(d_med, 4), sigma=round(d_sig, 4), nets_clip=[2, max(r["n_nets"] for r in B.values())]),
        locality=dict(dist="lognormal", median_frac_diag=round(l_med, 4), sigma=round(l_sig, 4)),
        pins_per_net=dict(pool=pool, min=lo, max=hi),
        pad_smd=smd, pad_tht=tht, netclass_sets=csets,
        layer_mix=[[round(r["f_tht"], 4), round(r["f_F"], 4)] for r in B.values()],
        boards=[dict(id=b, W=round(r["W"], 3), H=round(r["H"], 3), area_cm2=round(r["area"] / 100.0, 3), n_nets=r["n_nets"], pins=sum(r["pins_per_net"]),
                     f_tht=round(r["f_tht"], 4), f_F=round(r["f_F"], 4), n_classes=len(class_set(r))) for b, r in sorted(B.items())])


def build(split_json: Path, difficulty: str, split: str, root: Path | None = None) -> dict:
    d = json.load(open(split_json)); root = root or Path(os.path.expandvars(d["dataset_dirs"][split]))
    rows = {}; no_outline = []
    for b in d[difficulty][split]:
        r = extract(root / b / d["board_filename"], root / b / PRO_NAME)
        if r: rows[b] = r
        else: no_outline.append(b)
    out = aggregate(rows, dict(split_json=str(split_json.relative_to(REPO)) if split_json.is_relative_to(REPO) else str(split_json), difficulty=difficulty, split=split))
    out["source"]["excluded"] = sorted(out["source"]["excluded"] + no_outline)   # small-area filter + boards without an Edge.Cuts outline
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", type=Path, default=SPLIT_JSON); ap.add_argument("--difficulty", default="medium"); ap.add_argument("--split", default="fit")
    ap.add_argument("--out", type=Path, required=True, help="output json (never defaults to a file of this repository)"); ap.add_argument("--root", type=Path, default=None)
    a = ap.parse_args()
    out = build(a.split_json, a.difficulty, a.split, a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1) + "\n")
    s = out["source"]
    print(f"{a.out}: {s['n_boards']} boards ({s['difficulty']}/{s['split']}; excluded {s['excluded']}) | size {out['size']} | aspect {out['aspect']} | "
          f"net_density {out['net_density']} | locality {out['locality']} | pins_per_net pool {len(out['pins_per_net']['pool'])} | pads smd {len(out['pad_smd'])} tht {len(out['pad_tht'])} | "
          f"netclass sets {len(out['netclass_sets'])} | layer_mix {len(out['layer_mix'])}")


if __name__ == "__main__":
    main()
