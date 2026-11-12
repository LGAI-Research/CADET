#!/usr/bin/env python3
"""RouteDemos generator: autoregressive construction of boards with executable demonstrations.

The front end samples board size / net density / pins per net / pad tables / netclass sets / local placement from the
reference statistics; the routability gate (TryRoute) is the ENVIRONMENT ITSELF: a pin site is accepted iff the real router
(PCBWorld + KiCad PNS, walkaround / mark-obstacles [/ shove]) connects it with native policy actions while
the board keeps DRC == 0. The accepted action list IS the demonstration.

Per candidate pin: analytic site pre-check -> write the board (all pads so far + this one, routes so far)
-> fresh env, reset(preserve_routing) -> DRC must be 0 -> net_select -> executor2.Searcher.route_pin
(best-first over policy actions, engine checkpoint/restore backtracking) -> on success the routes are read
back from the obs and the pin's actions are kept; otherwise the site is rejected and re-sampled.

Later pads do not exist yet while earlier nets are routed, so the FINAL board is the authority: the assembled
demo is replayed from an empty board, a pin whose recorded actions diverge is re-searched from the replay
state (repair), and a pin that cannot be repaired has its pad PRUNED (net dropped below 2 pads) and the
board is re-verified — until one from-scratch pass reproduces every pin (100% solvable by construction).
Output = dataset board (pristine, unrouted — exactly what the acceptance replay ran on) + _routed/ copy carrying the
demo's routing + _demo/board_XXXXX.json.
"""
from __future__ import annotations
import argparse, gzip, json, math, os, random, sys, time, traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CADET = HERE.parents[1]
sys.path.insert(0, str(CADET)); sys.path.insert(0, str(CADET / "cadet/search"))
from cadet.paths import use_pcbworld  # noqa: E402
REPO = use_pcbworld()   # the PCBWorld checkout
sys.path.insert(0, str(REPO / "tools/datagen/synthetic_generator"))
import generate_synthetic_boards as gs   # noqa: E402
import outline_geometry as og            # noqa: E402
from executor2 import Searcher, build_env, _obs, _drc_real, _ratsnest, CKPT, MODE_WALK, MODE_SHOVE, SHOVE_PENALTY  # noqa: E402
import pcb_world.engine.router_client as router_client  # noqa: E402  (server reuse knob: fresh servers for the acceptance replay)


def _drc_strict(obs):
    """Every DRC violation except the open-ratsnest bookkeeping ('Missing connection'); dangling track ends COUNT
    (clean_pass counts them), unlike executor2._drc_real which also drops 'unconnected end'."""
    return sum(1 for v in (obs.get("drc_violations") or []) if isinstance(v, dict) and "missing connection" not in str(v.get("error_type", "")).lower())
from pcb_world.core.action_schema import ACT_NET_SELECT, ACT_START_ROUTE, ACT_NET_END, ACT_MAKE_LINE, ACT_MAKE_VIA  # noqa: E402
from replay_demo import replay as official_replay  # noqa: E402  (pure linear replay, no checkpoints — the acceptance test)

FIT_PARAMS = CADET / "cadet/configs/fit_params/d3b_fit_half.json"   # the ONLY front-end input (cadet/stats/fit_board_params.py)
MODE_MARK = 0
MODES = {"walk": (MODE_WALK, 0.0), "shove": (MODE_SHOVE, SHOVE_PENALTY), "mark": (MODE_MARK, 1.0)}
EDGE_CLEAR = 0.3
PRO_MINS = {"min_through_hole_diameter": 0.1, "min_via_diameter": 0.2, "min_via_annular_width": 0.05}
LAYER_NAME = {0: "F.Cu", 1: "B.Cu"}

# ---------------------------------------------------------------- targets: everything comes from fit_params.json
def load_fit(path=FIT_PARAMS):
    """Front-end tables + parametric targets from cadet/stats/fit_board_params.py output (one json = one provenance)."""
    f = json.load(open(path))
    T = dict(size_median=f["size"]["median"], size_sigma=f["size"]["sigma"], size_clip=tuple(f["size"]["clip"]),
             square_p=f["aspect"]["square_p"], aspect_sigma=f["aspect"]["sigma"], aspect_max=f["aspect"]["max"],
             net_density_median=f["net_density"]["median"], net_density_sigma=f["net_density"]["sigma"], nets_clip=tuple(f["net_density"]["nets_clip"]),
             local_median=f["locality"]["median_frac_diag"], local_sigma=f["locality"]["sigma"])
    return dict(T=T, pins_per_net=list(f["pins_per_net"]["pool"]), smd=[tuple(x) for x in f["pad_smd"]], tht=[tuple(x) for x in f["pad_tht"]],
                csets=f["netclass_sets"], mix=[tuple(x) for x in f["layer_mix"]],
                source=dict(path=str(path), n_boards=f["source"]["n_boards"], split=f"{f['source']['difficulty']}/{f['source']['split']}"))


def sample_board(rng, T):
    s = math.exp(math.log(T["size_median"]) + T["size_sigma"] * rng.gauss(0, 1)); s = min(max(s, T["size_clip"][0]), T["size_clip"][1])
    r = 1.0 if rng.random() < T["square_p"] else min(math.exp(abs(rng.gauss(0, T["aspect_sigma"]))), T["aspect_max"])
    r = min(r, (s / 12.0) ** 2)                      # short side >= 12mm
    w, h = s * math.sqrt(r), s / math.sqrt(r)
    if rng.random() < 0.5: w, h = h, w
    return w, h, s


# ---------------------------------------------------------------- analytic copper primitives (site pre-check)
# pad tuple: (x, y, layer_name, thru, L(0-based list), shape, w, h, drill)
def pad_prim(p):
    x, y, _ln, thru, L, _shape, w, h, _d = p
    if thru:
        r = min(w, h) / 2.0; hl = (max(w, h) - min(w, h)) / 2.0
        return ("cap", (x - hl, y, x + hl, y, r) if w >= h else (x, y - hl, x, y + hl, r), tuple(L))
    return ("rect", (x, y, w, h), tuple(L))


def _pt_seg(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0; L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / L2))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _seg_seg(a, b):
    x0, y0, x1, y1 = a; u0, v0, u1, v1 = b
    d1, d2 = _orient(x0, y0, x1, y1, u0, v0), _orient(x0, y0, x1, y1, u1, v1)
    d3, d4 = _orient(u0, v0, u1, v1, x0, y0), _orient(u0, v0, u1, v1, x1, y1)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0:
        return 0.0
    return min(_pt_seg(x0, y0, *b), _pt_seg(x1, y1, *b), _pt_seg(u0, v0, *a), _pt_seg(u1, v1, *a))


def _pt_rect(px, py, rx, ry, rw, rh):
    return math.hypot(max(abs(px - rx) - rw / 2.0, 0.0), max(abs(py - ry) - rh / 2.0, 0.0))


def _seg_rect(s, r, n=24):
    x0, y0, x1, y1 = s; rx, ry, rw, rh = r
    return min(_pt_rect(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, rx, ry, rw, rh) for i in range(n + 1))


def prim_dist(a, b):
    """Gap between two copper primitives (<= 0 = touching/overlapping)."""
    ka, ga, _ = a; kb, gb, _ = b
    if ka == "cap" and kb == "cap":
        return _seg_seg(ga[:4], gb[:4]) - ga[4] - gb[4]
    if ka == "cap" and kb == "rect":
        return _seg_rect(ga[:4], gb) - ga[4]
    if ka == "rect" and kb == "cap":
        return _seg_rect(gb[:4], ga) - gb[4]
    return math.hypot(max(abs(ga[0] - gb[0]) - (ga[2] + gb[2]) / 2.0, 0.0), max(abs(ga[1] - gb[1]) - (ga[3] + gb[3]) / 2.0, 0.0))


def route_prims(routes):
    out = []
    for s in routes["segments"]:
        out.append((("cap", (s["x1"], s["y1"], s["x2"], s["y2"], s["w"] / 2.0), (s["layer"],)), s["net"]))
    for v in routes["vias"]:
        out.append((("cap", (v["x"], v["y"], v["x"], v["y"], v["d"] / 2.0), (0, 1)), v["net"]))
    return out


def site_ok(new_pad, net_code, prims, clearance):
    """new pad body: no contact with ANY copper (same net incl.: overlapping own copper = born-connected pin),
    clearance to other-net copper. The engine DRC after loading is the final judge (holes, edges, netclass)."""
    pn = pad_prim(new_pad); layers = set(pn[2]); eps = 1e-6
    for prim, net in prims:
        if not (layers & set(prim[2])):
            continue
        d = prim_dist(pn, prim)
        if net == net_code:
            if d <= eps: return False
        elif d < clearance - eps:
            return False
    return True


# ---------------------------------------------------------------- board file
def render(nets, routes, geo, bw, bh, rng):
    """nets: list of dict(code, rule, pads); routes: dict(segments, vias) (may be empty). Legacy embedded net_class
    blocks -> the engine round-trip writes the .kicad_pro."""
    out = [gs._HEADER, "", '  (net 0 "")']
    for n in nets: out.append(f'  (net {n["code"]} "NET{n["code"]}")')
    out.append("")
    base = min((n["rule"] for n in nets), key=lambda r: r["clearance"])
    def nc_block(name, r, adds=()):
        out.append(f'  (net_class "{name}" ""'); out.append(f"    (clearance {r['clearance']})"); out.append(f"    (trace_width {r['width']})")
        out.append(f"    (via_dia {r['via_dia']:.4f})"); out.append(f"    (via_drill {r['via_drill']:.4f})"); out.append("    (uvia_dia 0.3)"); out.append("    (uvia_drill 0.1)")
        for a in adds: out.append(f'    (add_net "{a}")')
        out.append("  )")
    nc_block("Default", base)
    groups = {}
    for n in nets:
        key = tuple(round(n["rule"][k], 4) for k in ("clearance", "width", "via_dia", "via_drill")); groups.setdefault(key, (n["rule"], []))[1].append(f"NET{n['code']}")
    for gi, (rule, adds) in enumerate(groups.values()): nc_block(f"NC{gi+1}", rule, adds)
    out.append("")
    for n in nets:
        net_id = n["code"]
        for slot, (x, y, layer, thru, L, shape, w, h, drill) in enumerate(n["pads"]):
            fp = "SamplePad:PTH" if thru else ("SamplePad:FCu" if layer == "F.Cu" else "SamplePad:BCu")
            out.append(f'  (footprint "{fp}"'); out.append(f'    (layer "{"F.Cu" if thru else layer}")'); out.append(f"    (at {x:.6f} {y:.6f})"); out.append(f'    (uuid "{gs._u(rng)}")')
            out.append(f'    (property "Reference" "P{net_id}_{slot+1}"'); out.append("      (at 0 -1)"); out.append('      (layer "F.SilkS")'); out.append("      (effects (font (size 0.6 0.6) (thickness 0.1)))"); out.append("    )")
            out.append(f'    (property "Value" "Pad{slot+1}"'); out.append("      (at 0 1)"); out.append('      (layer "F.Fab")'); out.append("      (effects (font (size 0.6 0.6) (thickness 0.1)))"); out.append("    )")
            if thru:
                out.append(f'    (pad "1" thru_hole {shape}'); out.append("      (at 0 0)"); out.append(f"      (size {w} {h})"); out.append(f"      (drill {drill})"); out.append('      (layers "*.Cu" "*.Mask")')
            else:
                pl = '"F.Cu" "F.Paste" "F.Mask"' if layer == "F.Cu" else '"B.Cu" "B.Paste" "B.Mask"'
                out.append(f'    (pad "1" smd {shape}'); out.append("      (at 0 0)"); out.append(f"      (size {w} {h})"); out.append(f"      (layers {pl})")
            out.append(f'      (net {net_id} "NET{net_id}")'); out.append(f'      (uuid "{gs._u(rng)}")'); out.append("    )"); out.append("  )"); out.append("")
    drill_of = {n["code"]: n["rule"]["via_drill"] for n in nets}
    for s in routes.get("segments", []):
        out.append(f'  (segment (start {s["x1"]:.6f} {s["y1"]:.6f}) (end {s["x2"]:.6f} {s["y2"]:.6f}) (width {s["w"]:.6f}) (layer "{LAYER_NAME[s["layer"]]}") (net {s["net"]}) (uuid "{gs._u(rng)}"))')
    for v in routes.get("vias", []):
        out.append(f'  (via (at {v["x"]:.6f} {v["y"]:.6f}) (size {v["d"]:.6f}) (drill {drill_of[v["net"]]:.6f}) (layers "F.Cu" "B.Cu") (net {v["net"]}) (uuid "{gs._u(rng)}"))')
    geo.emit_hole_footprints(rng, out); geo.emit_edge_cuts(rng, out); out.append(")")
    return "\n".join(out)


PRO_TEMPLATE = json.loads((HERE / "pro_template.json").read_text())   # engine-written .kicad_pro of a generated board, classes/patterns stripped


def write_pro(pcb, nets):
    """Write the .kicad_pro directly: Default = the weakest rule, NC<k> per distinct rule with netclass_patterns NET<code> -> NC<k>,
    board minima PRO_MINS. Byte-equivalent in effect to the old engine round trip (load legacy net_class blocks -> engine save -> patch)."""
    import copy
    pj = copy.deepcopy(PRO_TEMPLATE); tmpl = pj["net_settings"]["classes"][0]
    def cls(name, r): c = dict(tmpl); c.update(name=name, clearance=r["clearance"], track_width=r["width"], via_diameter=round(r["via_dia"], 4), via_drill=round(r["via_drill"], 4)); return c
    base = min((n["rule"] for n in nets), key=lambda r: r["clearance"])
    groups = {}
    for n in nets:
        key = tuple(round(n["rule"][k], 4) for k in ("clearance", "width", "via_dia", "via_drill")); groups.setdefault(key, (n["rule"], []))[1].append(n["code"])
    classes = [cls("Default", base)]; patterns = []
    for gi, (rule, codes) in enumerate(groups.values()):
        classes.append(cls(f"NC{gi + 1}", rule)); patterns += [dict(netclass=f"NC{gi + 1}", pattern=f"NET{c}") for c in codes]
    pj["net_settings"]["classes"] = classes; pj["net_settings"]["netclass_patterns"] = patterns
    # the engine round trip derived these three board minima from the weakest (Default) legacy net_class clearance — reproduce that
    pj["board"]["design_settings"]["rules"].update(PRO_MINS, min_hole_to_hole=base["clearance"], min_hole_clearance=base["clearance"], min_copper_edge_clearance=base["clearance"])
    pj["meta"]["filename"] = Path(pcb).with_suffix(".kicad_pro").name
    Path(pcb).with_suffix(".kicad_pro").write_text(json.dumps(pj, indent=2))


def write_board(pcb, text, nets, cfg):
    """Default (``pro_mode=direct``): write the board text + a directly generated .kicad_pro — no engine round trip per attempt.
    ``pro_mode=engine`` keeps the old path (engine load -> save -> patch) for equivalence checks."""
    pcb = Path(pcb); pcb.write_text(text)
    if cfg.get("pro_mode", "direct") == "engine":
        from pcb_world.engine import KiCadEngine
        e0 = KiCadEngine(str(pcb), allow_default_rules=True); e0.save(str(pcb)); e0.close()
        patch_pro(pcb)
    else:
        write_pro(pcb, nets)


def patch_pro(pcb):
    pro = Path(pcb).with_suffix(".kicad_pro"); pj = json.loads(pro.read_text()); pj["board"]["design_settings"]["rules"].update(PRO_MINS); pro.write_text(json.dumps(pj, indent=2))


def extract_routes(obs):
    segs, vias = [], []
    for key, g in obs.get("routing_geometry", {}).items():
        code = int(key.split("_")[1])
        for t in g.get("tracks", {}).values():
            segs.append(dict(net=code, x1=t["p1"]["xy"][0], y1=t["p1"]["xy"][1], x2=t["p2"]["xy"][0], y2=t["p2"]["xy"][1], w=float(t["width"]), layer=int(t["layer"]) - 1))
        for v in g.get("vias", {}).values():
            vias.append(dict(net=code, x=v["center"]["xy"][0], y=v["center"]["xy"][1], d=float(v.get("via_width") or v.get("width") or 0.6)))
    return dict(segments=segs, vias=vias)


# ---------------------------------------------------------------- the PNS gate
def make_searcher(env, cfg, modes, budget):
    se = Searcher(env, budget=budget, shove=True, log=lambda *a: None, use_oracle=not cfg.get("no_oracle", False), oracle_step=cfg.get("oracle_step", 0.2))
    se.nl = 2; se.n_env_steps = 0; se._orc = None; se._field = None; se.term = False; se.verify_pins = True
    se.direct_modes = [MODES[m] for m in modes]
    se.strict_drc = True; se.skip_noncand = True
    return se


def pin_src(pad):
    x, y, _ln, _thru, L, *_ = pad
    return (x, y, 1 if 0 in L else 2)


def attempt(pcb, nets, routes, net_code, new_pad, geo, bw, bh, rng, cfg, tm):
    """Route the new pad of net ``net_code`` on a fresh env of the current board. Returns (actions, routes2, info) or (None, None, reason)."""
    t0 = time.time()
    write_board(pcb, render(nets, routes, geo, bw, bh, rng), nets, cfg)
    env, _ = build_env(str(pcb), cfg["ckpt"], preset=cfg["preset"], offboard_mask=True)
    try:
        obs = _obs(env.reset(options={"preserve_routing": True})[0]); tm["t_env"] += time.time() - t0; t1 = time.time()
        if _drc_real(obs) > 0:
            return None, None, "drc_after_pad"
        if not _ratsnest(obs, net_code):
            return None, None, "born_connected"
        se = make_searcher(env, cfg, cfg["gen_modes"], cfg["budget"])
        nets_env = list(env.sorted_net_codes)
        if net_code not in nets_env:
            return None, None, "net_missing"
        obs, term, trunc, info = se.step(ACT_NET_SELECT, nets_env.index(net_code))
        if not info.get("action_success", True):
            return None, None, "net_select_fail"
        src = pin_src(new_pad)
        if se.cand_index(*src) is None:
            return None, None, "no_start_cand"
        se.cap = True
        obs, acts, exp, ok = se.route_pin(obs, net_code, src)
        if not ok and cfg.get("uncapped_retry", True):
            se.cap = False; obs, acts, exp2, ok = se.route_pin(obs, net_code, src); exp += exp2
        if not ok:
            return None, None, "search_fail"
        if _drc_strict(obs) > 0 or _ratsnest(obs, net_code):
            return None, None, "post_check"
        return acts, extract_routes(obs), dict(expansions=exp, env_steps=se.n_env_steps)
    finally:
        tm["t_search"] += time.time() - t1 if "t1" in locals() else 0.0
        env.close()


# ---------------------------------------------------------------- one board (placement + incremental PNS gate)
def generate_one(seed, cfg, tables, work_dir, log):
    rng = random.Random(seed); T = tables["T"]; mix = tables["mix"]
    bw, bh, s = sample_board(rng, T)
    geo = og.sample_geo(rng, bw, bh, shape_census=False); bw, bh = geo.outline.w, geo.outline.h
    area_cm2 = bw * bh / 100.0
    dens = math.exp(math.log(T["net_density_median"]) + T["net_density_sigma"] * rng.gauss(0, 1)) * cfg.get("density_scale", 1.0)
    n_nets = int(min(max(round(dens * area_cm2), T["nets_clip"][0]), T["nets_clip"][1]))
    pins_per_net = [rng.choice(tables["pins_per_net"]) for _ in range(n_nets)]
    cset = rng.choice(tables["csets"]); wts = [c["w"] for c in cset]
    net_rules = [dict(rng.choices(cset, weights=wts)[0]) for _ in range(n_nets)]
    f_tht, f_F = rng.choice(mix)
    diag = math.hypot(bw, bh); grid = cfg["pad_grid"]
    max_c = max(c["clearance"] for c in cset)
    nets_out = []                  # dict(code, rule, pads, pins=[dict(src, actions, hops, ...)])
    routes = dict(segments=[], vias=[])
    prims = []                     # (prim, net_code) of every committed pad
    order = list(range(n_nets)); rng.shuffle(order)
    st = Counter(pins_target=sum(pins_per_net)); tm = Counter()
    pcb = work_dir / f"work_{seed}.kicad_pcb"

    def pad_geom(thru):
        if thru:
            w, h, d = rng.choice(tables["tht"]); shape = "circle" if abs(w - h) < 1e-6 else "oval"
        else:
            w, h = rng.choice(tables["smd"]); d = 0.0; shape = "rect"
        if rng.random() < 0.5: w, h = h, w
        return shape, w, h, d

    def sample_xy(anchor, r_env, c):
        for _ in range(60):
            st["tries"] += 1
            if anchor is None:
                x, y = rng.uniform(0, bw), rng.uniform(0, bh)
            else:
                d = math.exp(math.log(T["local_median"] * diag) + T["local_sigma"] * rng.gauss(0, 1)); a = rng.uniform(0, 2 * math.pi)
                x, y = anchor[0] + d * math.cos(a), anchor[1] + d * math.sin(a)
            x, y = round(round(x / grid) * grid, 4), round(round(y / grid) * grid, 4)
            if not (0 < x < bw and 0 < y < bh) or not geo.allows_pad(x, y, r_env, c):
                st["r_geo"] += 1; continue
            return x, y
        return None

    t_board = time.time()
    for ni in order:
        rule = net_rules[ni]; net_code = len(nets_out) + 1
        c = rule["clearance"]
        pads, pins, pending_first = [], [], None
        for k in range(pins_per_net[ni]):
            thru = rng.random() < f_tht
            layer = "F.Cu" if (thru or rng.random() < f_F) else "B.Cu"
            L = [0, 1] if thru else [0 if layer == "F.Cu" else 1]
            shape, pw, ph, drill = pad_geom(thru)
            r_env = math.hypot(pw, ph) / 2
            ok = False
            for _t in range(cfg["pin_tries"]):
                anchor = rng.choice(pads)[:2] if pads else None
                xy = sample_xy(anchor, r_env, c)
                if xy is None:
                    break
                x, y = xy
                new_pad = (x, y, layer, thru, L, shape, pw, ph, drill)
                own = [(pad_prim(p), net_code) for p in pads + ([pending_first] if pending_first is not None else [])]
                if not site_ok(new_pad, net_code, prims + own + route_prims(routes), max_c):
                    st["site_fail"] += 1; continue
                if k == 0:
                    pending_first = new_pad; ok = True; break
                if cfg.get("no_gate"):
                    # no-gate mode: the analytic site check is the only acceptance test (no routing, no demo);
                    # everything else (sampling, tables, render) is byte-identical to the gated run
                    if pending_first is not None:
                        pads.append(pending_first); prims.append((pad_prim(pending_first), net_code)); pending_first = None
                    pads.append(new_pad); prims.append((pad_prim(new_pad), net_code))
                    ok = True; break
                trial_pads = pads + ([pending_first] if pending_first is not None else []) + [new_pad]
                trial_nets = nets_out + [dict(code=net_code, rule=rule, pads=trial_pads)]
                acts, routes2, info = attempt(pcb, trial_nets, routes, net_code, new_pad, geo, bw, bh, rng, cfg, tm)
                st["attempts"] += 1
                if acts is None:
                    st["pins_rejected"] += 1; st[f"rej_{info}"] += 1; continue
                if pending_first is not None:
                    pads.append(pending_first); prims.append((pad_prim(pending_first), net_code)); pending_first = None
                pads.append(new_pad); prims.append((pad_prim(new_pad), net_code)); routes = routes2
                pins.append(dict(src=list(pin_src(new_pad)), actions=[list(a) for a in acts], hops=len(acts) // 2, **info))
                st["hops"] += len(acts) // 2; st["expansions"] += info["expansions"]
                ok = True; break
            if not ok:
                if k >= 2: st["nets_shrunk"] += 1
                break
        if len(pads) >= 2:
            nets_out.append(dict(code=net_code, rule=rule, pads=pads, pins=pins)); st["pins_placed"] += len(pads)
        else:
            st["nets_dropped"] += 1
        if cfg.get("board_time_cap") and time.time() - t_board > cfg["board_time_cap"]:
            st["timed_out"] += 1; break
    for f in work_dir.glob(f"work_{seed}.*"): f.unlink()
    if not nets_out:
        return None
    meta = dict(seed=seed, board_w=bw, board_h=bh, pad_grid=grid, f_tht=f_tht, f_F=f_F, gen="cadet", gen_modes=cfg["gen_modes"], gate=not cfg.get("no_gate", False), fit=tables["source"],
                target=dict(n_nets=n_nets, pins_per_net=pins_per_net, area_cm2=area_cm2), gen_stats=dict(st), timing=dict(gen_s=round(time.time() - t_board, 1), env_s=round(tm["t_env"], 1), search_s=round(tm["t_search"], 1)))
    log(f"seed={seed} board {bw:.1f}x{bh:.1f} nets {n_nets}->{len(nets_out)} pins {st['pins_target']}->{st['pins_placed']} rej={st['pins_rejected']} "
        f"site_fail={st['site_fail']} hops/pin={st['hops']/max(1, st['pins_placed'] - len(nets_out)):.2f} {time.time() - t_board:.0f}s")
    return nets_out, geo, bw, bh, meta


def assemble_demo(nets_out):
    actions, spans = [], []
    for i, n in enumerate(nets_out):
        actions.append([ACT_NET_SELECT, n["code"], 0, 0, 0])
        for p in n["pins"]:
            s = len(actions); actions.extend(p["actions"]); spans.append(dict(net=n["code"], start=s, end=len(actions), src=p["src"]))
        if i < len(nets_out) - 1:            # the last connection of the board terminates the episode: no net_end after it
            actions.append([ACT_NET_END, 0, 0, 0, 0])
    return actions, spans


# ---------------------------------------------------------------- verification replay with repair
def plain_step(env, at, ptr, mode):
    """The consumer's step: env.step without the searcher's via-toggle clearing (mirrors replay_demo exactly)."""
    raw, _r, term, trunc, info = env.step((int(at), int(ptr), int(mode)))
    return _obs(raw), term, trunc, info


def verify_and_repair(pcb, actions, spans, cfg, linear_until=0):
    """Replay the demo from an EMPTY board. Pins starting before ``linear_until`` are executed plainly (no checkpoint —
    they already passed the official linear replay); from there on every pin is snapshotted, and a pin whose recorded
    actions fail or do not connect is re-searched from the replay state (repair). Returns (actions, spans, stats);
    actions None on failure (stats['fail_at'] names the pin)."""
    env, _ = build_env(str(pcb), cfg["ckpt"], preset=cfg["preset"], offboard_mask=True)
    try:
        obs = _obs(env.reset()[0]); se = make_searcher(env, cfg, cfg["verify_modes"], cfg["verify_budget"]); nets_env = list(env.sorted_net_codes)
        out, out_spans = [], []; repaired = 0; term = trunc = False
        by_start = {sp["start"]: sp for sp in spans}
        i = 0
        while i < len(actions) and not (term or trunc):
            at, x, y, l, mode = actions[i]
            if i in by_start:
                sp = by_start[i]; nid = sp["net"]; rats0 = len(_ratsnest(obs, nid))
                linear = sp["start"] < linear_until
                snap = None if linear else se.snapshot(); good = True; local = []
                for a in actions[sp["start"]:sp["end"]]:
                    ptr = se.cand_index(a[1], a[2], a[3])
                    if ptr is None:
                        good = False; break
                    obs, term, trunc, info = plain_step(env, int(a[0]), ptr, int(a[4])); local.append(list(a))
                    if not info.get("action_success", True):
                        good = False; break
                    if term or trunc:
                        break
                if good and len(_ratsnest(obs, nid)) >= rats0 and not (term or trunc):
                    good = False
                if good and _drc_strict(obs) > 0:
                    good = False
                if not good:
                    if snap is None:      # a linearly-verified prefix pin failed anyway (engine nondeterminism): prune it
                        return None, None, dict(repaired=repaired, failed=1, fail_at=dict(net=nid, src=sp["src"], prefix=True))
                    obs = se.restore(snap); term = trunc = False; se.term = False
                    se.cap = True; obs, acts, _e, ok = se.route_pin(obs, nid, tuple(sp["src"]))
                    if not ok:
                        se.cap = False; obs, acts, _e, ok = se.route_pin(obs, nid, tuple(sp["src"]))
                    if not ok:
                        snap[0].release()
                        return None, None, dict(repaired=repaired, failed=1, fail_at=dict(net=nid, src=sp["src"]))
                    repaired += 1; local = [list(a) for a in acts]; term = se.term
                if snap is not None:
                    snap[0].release()
                s0 = len(out); out.extend(local); out_spans.append(dict(net=nid, start=s0, end=len(out), src=sp["src"])); i = sp["end"]; continue
            ptr = nets_env.index(int(x)) if at == ACT_NET_SELECT else 0
            obs, term, trunc, info = plain_step(env, int(at), ptr, int(mode)); out.append(list(actions[i])); i += 1
            if not info.get("action_success", True):
                return None, None, dict(repaired=repaired, failed=1, fail_at=dict(step=i - 1, at=int(at)))
        unrouted = sum(len(_ratsnest(obs, n)) for n in nets_env) // 2; drc = len(obs.get("drc_violations") or [])
        if unrouted == 0 and drc == 0:
            return out, out_spans, dict(repaired=repaired, failed=0, unrouted=0, drc=0, clean_pass=True, terminated=bool(term))
        return None, None, dict(repaired=repaired, failed=1, unrouted=unrouted, drc=drc, fail_at=dict(end_state=True))
    finally:
        env.close()


def official_replay_fresh(pcb, actions, cfg, work_dir):
    """The consumer's condition, literally: replay_demo.py in a FRESH python process (fresh engine server, no client-side
    history). Returns (ok, stats). Used with ``--accept-mode subprocess``; the default replays in-process on a fresh
    engine server."""
    demo = dict(board=str(pcb), actions=actions, clean_pass=True, env_knobs=dict(preset=cfg["preset"], offboard_mask=True))
    if cfg.get("accept_mode", "inprocess") == "inprocess":
        # same process, FRESH engine server (router_client.set_server_reuse(0) is in effect during finalize)
        try:
            ok, st, _ = official_replay(demo, cfg["ckpt"])
        except Exception as exc:
            return False, dict(error=repr(exc)[:300])
        return bool(ok) and bool(st.get("clean_pass")), st
    import subprocess
    demo_path = Path(work_dir) / f"accept_{Path(pcb).stem}_{os.getpid()}.json"
    json.dump(demo, open(demo_path, "w"))
    env = dict(os.environ, KICAD_ENGINE_REUSE="0", OMP_NUM_THREADS="1")
    try:
        r = subprocess.run([sys.executable, str(CADET / "cadet/search/replay_demo.py"), str(demo_path), "--ckpt", cfg["ckpt"]], capture_output=True, text=True, env=env, cwd=str(REPO), timeout=1800)
        line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("{")), None)
        if line is None:
            return False, dict(error="no result line", stderr=r.stderr[-500:])
        st = json.loads(line); return bool(st.get("ok")) and bool(st.get("clean_pass")), st
    except subprocess.TimeoutExpired:
        return False, dict(error="timeout")
    finally:
        demo_path.unlink(missing_ok=True)


def sync_pins(nets_out, actions, spans):
    """Write the (possibly repaired) span actions back into the pin records, so prune/assemble keep them."""
    idx = {(n["code"], round(p["src"][0], 6), round(p["src"][1], 6)): p for n in nets_out for p in n["pins"]}
    for sp in spans:
        p = idx.get((sp["net"], round(sp["src"][0], 6), round(sp["src"][1], 6)))
        if p is not None:
            p["actions"] = [list(a) for a in actions[sp["start"]:sp["end"]]]; p["hops"] = (sp["end"] - sp["start"]) // 2


def finalize_board(pcb, nets_out, geo, bw, bh, rng, cfg):
    """Verify/repair/prune until the OFFICIAL linear replay (replay_demo, fresh env, no checkpoints) reproduces every pin
    with clean_pass. Returns (actions, spans, stats, pruned) — actions None if the board could not be made to pass."""
    pruned = 0; linear_until = 0; iters = 0; linear_fail = 0; repaired_total = 0; fail_count = Counter(); vst = {}
    actions, spans = assemble_demo(nets_out); dirty = True
    while iters < cfg["max_verify_iters"] and nets_out:
        iters += 1
        if dirty:
            write_board(pcb, render(nets_out, dict(segments=[], vias=[]), geo, bw, bh, rng), nets_out, cfg); dirty = False
        actions2, spans2, vst = verify_and_repair(pcb, actions, spans, cfg, linear_until)
        if actions2 is None:
            fa = vst.get("fail_at") or {}
            if "net" not in fa:
                break
            pruned += prune(nets_out, fa) or 0; dirty = True; linear_until = 0; actions, spans = assemble_demo(nets_out); continue
        repaired_total += vst.get("repaired", 0); sync_pins(nets_out, actions2, spans2)
        ok_lin, lst = official_replay_fresh(pcb, actions2, cfg, Path(cfg["work_dir"]))
        if ok_lin and lst.get("clean_pass"):
            vst.update(linear_fail=linear_fail, repaired=repaired_total, iters=iters); return actions2, spans2, vst, pruned
        linear_fail += 1; k = lst.get("steps_ok", 0)
        sp = next((x for x in spans2 if x["start"] <= k < x["end"]), None)
        if sp is None:                       # divergence outside a pin (net_select / net_end): give up on this board
            vst["fail_at"] = dict(linear=True, step=k); break
        key = (sp["net"], round(sp["src"][0], 6), round(sp["src"][1], 6)); fail_count[key] += 1
        if fail_count[key] > 2:              # this pin keeps diverging under the plain replay: prune it
            pruned += prune(nets_out, dict(net=sp["net"], src=sp["src"])) or 0; dirty = True; linear_until = 0
            actions, spans = assemble_demo(nets_out); continue
        actions, spans = actions2, spans2; linear_until = sp["start"]   # prefix up to the diverging pin is linear-proven
    vst.update(linear_fail=linear_fail, repaired=repaired_total, iters=iters)
    return None, None, vst, pruned


def save_demo_routing(pcb, routed_pcb, actions, cfg):
    """Write the routing the accepted demo produces to ``routed_pcb`` (plain replay on the dataset file, engine save).
    The DATASET file stays the pristine unrouted board the acceptance replay ran on: an engine-saved routed copy is
    not replay-equivalent after the env's strip-at-reset (item UUID/order differences change PNS walkaround; measured
    3/4 divergent boards recovered by replaying the pristine file)."""
    env, _ = build_env(str(pcb), cfg["ckpt"], preset=cfg["preset"], offboard_mask=True)
    try:
        obs = _obs(env.reset()[0]); nets_env = list(env.sorted_net_codes)
        for at, x, y, l, mode in actions:
            ptr = nets_env.index(int(x)) if at == ACT_NET_SELECT else (0 if at == ACT_NET_END else next(i for i, (cx, cy, cl) in enumerate(env.cand_mm_list) if abs(cx - x) < 1e-3 and abs(cy - y) < 1e-3 and cl == int(l)))
            obs, term, trunc, info = plain_step(env, at, ptr, mode)
            if term or trunc: break
        routed_pcb = Path(routed_pcb); env.env._engine.save(str(routed_pcb), str(routed_pcb.with_suffix(".kicad_pro"))); patch_pro(routed_pcb)
    finally:
        env.close()


def prune(nets_out, fail_at):
    """Remove the pad at fail_at['src'] from net fail_at['net'] (and its pin record); drop the net below 2 pads;
    renumber net codes. Returns the number of pads removed (0 = nothing to prune)."""
    code, (sx, sy, _sl) = fail_at["net"], fail_at["src"]
    for n in nets_out:
        if n["code"] != code:
            continue
        before = len(n["pads"])
        n["pads"] = [p for p in n["pads"] if not (abs(p[0] - sx) < 1e-6 and abs(p[1] - sy) < 1e-6)]
        n["pins"] = [p for p in n["pins"] if not (abs(p["src"][0] - sx) < 1e-6 and abs(p["src"][1] - sy) < 1e-6)]
        removed = before - len(n["pads"])
        if len(n["pads"]) < 2:
            removed = before; nets_out.remove(n)
        for i, m in enumerate(nets_out): m["code"] = i + 1
        return removed
    return 0


def _finish_no_gate(i, pcb, out_dir, nets_out, geo, bw, bh, rng, meta, cfg, t0):
    """No-gate mode: write the pristine board, require a clean engine DRC on load (site_ok is analytic; the engine is
    the judge for holes/edges/netclass), save the witness meta. No demo, no _routed copy."""
    write_board(pcb, render(nets_out, dict(segments=[], vias=[]), geo, bw, bh, rng), nets_out, cfg)
    env, _ = build_env(str(pcb), cfg["ckpt"], preset=cfg["preset"], offboard_mask=True)
    try:
        drc = _drc_real(_obs(env.reset()[0]))
    finally:
        env.close()
    if drc > 0:
        pcb.unlink(missing_ok=True); pcb.with_suffix(".kicad_pro").unlink(missing_ok=True)
        return dict(idx=i, status="drc_fail", n_nets=len(nets_out), drc=drc, wall_s=round(time.time() - t0, 1))
    pins = sum(len(n["pads"]) for n in nets_out); st = meta["gen_stats"]
    meta.update(n_nets=len(nets_out), n_classes=len({tuple(round(n["rule"][k], 4) for k in ("clearance", "width", "via_dia", "via_drill")) for n in nets_out}),
                nets=[dict(code=n["code"], rule=n["rule"], pads=n["pads"], pins=[]) for n in nets_out], pins=pins, pruned=0, prune_rounds=0)
    with gzip.open(out_dir / "_witness" / f"board_{i:05d}.witness.json.gz", "wt") as fh: json.dump(meta, fh)
    return dict(idx=i, status="ok", bw=bw, bh=bh, n_nets=len(nets_out), pins_target=st["pins_target"], pins_placed=pins, pins_gen=st["pins_placed"], pins_rejected=0,
                site_fail=st.get("site_fail", 0), nets_dropped=st.get("nets_dropped", 0), nets_shrunk=st.get("nets_shrunk", 0), attempts=0,
                actions=0, actions_per_pin=0.0, within_budget=True, hops_hist={}, verify=dict(repaired=0), pruned=0, prune_rounds=0,
                timing=meta["timing"], wall_s=round(time.time() - t0, 1))


# ---------------------------------------------------------------- batch
_CFG = {}
def _init(cfg): _CFG.update(cfg)


def _one(i):
    cfg = _CFG; out_dir = Path(cfg["out_dir"]); pcb = out_dir / f"board_{i:05d}.kicad_pcb"; t0 = time.time()
    try:
        if pcb.exists() and (out_dir / ("_witness" if cfg.get("no_gate") else "_demo") / (f"board_{i:05d}.witness.json.gz" if cfg.get("no_gate") else f"board_{i:05d}.json")).exists():
            return dict(idx=i, status="exists")
        tables = load_fit(cfg["fit_params"])
        work = Path(cfg["work_dir"]); work.mkdir(parents=True, exist_ok=True)   # local scratch: per-attempt board files stay off the shared output directory
        # gate attempts may reuse parked engine servers (fast; their non-determinism only moves which sites get
        # accepted); the acceptance replay + repairs + final save run on FRESH servers = the consumer's condition
        router_client.set_server_reuse(2)
        res = generate_one(cfg["seed_base"] + i, cfg, tables, work, lambda m: None)
        if res is None: return dict(idx=i, status="empty")
        nets_out, geo, bw, bh, meta = res
        rng = random.Random(cfg["seed_base"] + i + 7)
        if cfg.get("no_gate"):
            return _finish_no_gate(i, pcb, out_dir, nets_out, geo, bw, bh, rng, meta, cfg, t0)
        t1 = time.time(); router_client.set_server_reuse(0)
        actions2, spans2, vst, pruned = finalize_board(pcb, nets_out, geo, bw, bh, rng, cfg)
        rounds = vst.get("iters", 0)
        if actions2 is None:
            return dict(idx=i, status="verify_fail", n_nets=len(nets_out), verify=vst, prune_rounds=rounds, pruned=pruned, wall_s=round(time.time() - t0, 1))
        unrouted = render(nets_out, dict(segments=[], vias=[]), geo, bw, bh, rng)
        # dataset file (pcb) = the pristine unrouted board the acceptance replay passed on; routed copy -> _routed/
        save_demo_routing(pcb, out_dir / "_routed" / pcb.name, actions2, cfg)
        pins = sum(len(n["pads"]) for n in nets_out); n_act = len(actions2)
        meta.update(n_nets=len(nets_out), n_classes=len({tuple(round(n["rule"][k], 4) for k in ("clearance", "width", "via_dia", "via_drill")) for n in nets_out}),
                    nets=[dict(code=n["code"], rule=n["rule"], pads=n["pads"], pins=n["pins"]) for n in nets_out], pins=pins, pruned=pruned, prune_rounds=rounds)
        meta["timing"]["verify_s"] = round(time.time() - t1, 1)
        demo = dict(board=str(pcb), actions=actions2, pin_spans=spans2, clean_pass=True, success=True, net_status={n["code"]: "ok" for n in nets_out},
                    env_knobs=dict(preset=cfg["preset"], offboard_mask=True), verify=vst, gen="cadet")
        json.dump(demo, open(out_dir / "_demo" / f"board_{i:05d}.json", "w"))
        with gzip.open(out_dir / "_witness" / f"board_{i:05d}.witness.json.gz", "wt") as fh: json.dump(meta, fh)
        hops = [(sp["end"] - sp["start"]) // 2 for sp in spans2]
        st = meta["gen_stats"]
        row = dict(idx=i, status="ok", bw=bw, bh=bh, n_nets=len(nets_out), pins_target=st["pins_target"], pins_placed=pins, pins_gen=st["pins_placed"], pins_rejected=st.get("pins_rejected", 0),
                   site_fail=st.get("site_fail", 0), nets_dropped=st.get("nets_dropped", 0), nets_shrunk=st.get("nets_shrunk", 0), attempts=st.get("attempts", 0),
                   **{k: v for k, v in st.items() if k.startswith("rej_")},
                   actions=n_act, actions_per_pin=round(n_act / max(1, pins), 2), within_budget=n_act <= cfg["budget_per_pin"] * pins,
                   hops_hist={str(k): v for k, v in sorted(Counter(hops).items())}, verify=vst, pruned=pruned, prune_rounds=rounds,
                   timing=meta["timing"], wall_s=round(time.time() - t0, 1))
        return row
    except Exception as exc:
        return dict(idx=i, status="error", error=repr(exc), tb=traceback.format_exc()[-1500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--n", type=int, default=16); ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--start-index", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-grid", type=float, default=0.1); ap.add_argument("--pin-tries", type=int, default=30)
    ap.add_argument("--preset", default="multi_resolution"); ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--gen-modes", default="walk", help="direct-hop router modes during generation (walk|shove|mark); mark never commits through obstacles "
                    "in the RL binding (dominated by walk), shove hops rarely survive later pads")
    ap.add_argument("--fit-params", type=Path, default=FIT_PARAMS, help="target statistics json (cadet/stats/fit_board_params.py)")
    ap.add_argument("--verify-modes", default="walk,shove,mark", help="router modes for the final from-scratch verify/repair pass")
    ap.add_argument("--budget", type=int, default=60, help="search expansions per pin attempt (generation)")
    ap.add_argument("--verify-budget", type=int, default=200, help="search expansions per repaired pin (verify)")
    ap.add_argument("--prune-rounds", type=int, default=4, help="(legacy name) unused; see --max-verify-iters"); ap.add_argument("--max-verify-iters", type=int, default=16, help="max verify/repair/prune iterations per board before giving up")
    ap.add_argument("--budget-per-pin", type=int, default=6, help="board-level action budget flag: actions <= N x pins")
    ap.add_argument("--no-oracle", action="store_true", help="Euclidean heuristic instead of the optimistic-grid distance field"); ap.add_argument("--oracle-step", type=float, default=0.2)
    ap.add_argument("--density-scale", type=float, default=1.0, help="multiply the target net density (over-sampling calibration)")
    ap.add_argument("--board-time-cap", type=float, default=0.0, help="stop placing nets after N s (partial board)")
    ap.add_argument("--no-gate", action="store_true", help="skip the PNS routing gate: every site_ok pad is accepted (no demos/_routed, no verify); "
                    "the only difference from the gated run — same sampling, tables, render")
    ap.add_argument("--work-dir", type=Path, default=None, help="local scratch for per-attempt board files (default /tmp/cadet_gen_<uid>/<out-dir name>)")
    ap.add_argument("--pro-mode", choices=["direct", "engine"], default="direct", help="how the per-attempt .kicad_pro is produced (engine = old round trip)")
    ap.add_argument("--accept-mode", choices=["inprocess", "subprocess"], default="inprocess", help="acceptance replay in this process (fresh engine server) or a fresh python subprocess (old)")
    ap.add_argument("--chunk-dir", type=Path, default=None, help="chunk queue: claim chunks of --chunk-size indices from [start-index, start-index+n) via mkdir; loop until none left")
    ap.add_argument("--chunk-size", type=int, default=200)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    for d in ("_routed", "_demo", "_witness"): (a.out_dir / d).mkdir(exist_ok=True)
    import uuid
    work_dir = a.work_dir or Path(f"/tmp/cadet_gen_{os.getuid()}") / a.out_dir.name
    cfg = dict(out_dir=str(a.out_dir), seed_base=a.seed_base, fit_params=str(a.fit_params), pad_grid=a.pad_grid, pin_tries=a.pin_tries, preset=a.preset, ckpt=a.ckpt, budget=a.budget,
               verify_budget=a.verify_budget, prune_rounds=a.prune_rounds, max_verify_iters=a.max_verify_iters, gen_modes=a.gen_modes.split(","), verify_modes=a.verify_modes.split(","),
               budget_per_pin=a.budget_per_pin, no_oracle=a.no_oracle, oracle_step=a.oracle_step, density_scale=a.density_scale, board_time_cap=a.board_time_cap, no_gate=a.no_gate,
               work_dir=str(work_dir), pro_mode=a.pro_mode, accept_mode=a.accept_mode)
    (a.out_dir / "_config.json").write_text(json.dumps(cfg, indent=1))
    import multiprocessing as mp
    t0 = time.time(); rows = []
    host = "w" + uuid.uuid4().hex[:8]   # anonymous worker id: names this process's row file and chunk claims
    tag = f"_{a.start_index}_{a.n}" if a.chunk_dir is None else f"_{host}_{os.getpid()}"; rows_path = a.out_dir / f"_gen_rows{tag}.jsonl"
    def run_indices(pool, rf, idxs, label):
        for k, row in enumerate(pool.imap_unordered(_one, idxs, chunksize=1), 1):
            rows.append(row); rf.write(json.dumps(row) + "\n"); rf.flush()
            if row["status"] == "error": print("[error]", row["idx"], row["error"], row["tb"], flush=True)
            elif row["status"] == "ok": print(f"[{label}{k}/{len(idxs)}] idx={row['idx']} nets={row['n_nets']} pins={row['pins_placed']}/{row['pins_target']} act/pin={row['actions_per_pin']} repaired={row['verify']['repaired']} pruned={row['pruned']} {row['timing']} {row['wall_s']}s", flush=True)
            else: print(f"[{label}{k}/{len(idxs)}] idx={row['idx']} {row['status']} {row.get('verify')} {row.get('wall_s')}s", flush=True)
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(cfg,), maxtasksperchild=20) as pool, open(rows_path, "a") as rf:
        if a.chunk_dir is None:
            run_indices(pool, rf, list(range(a.start_index, a.start_index + a.n)), "")
        else:
            # chunk queue: chunks of --chunk-size over [start, start+n); a chunk is claimed by an atomic mkdir (works on a shared filesystem), so any number
            # of machines can run this command concurrently and fast ones simply take more chunks. Chunks are STREAMED into one pool
            # (claimed lazily as workers free up), so the pool never idles on a chunk's slowest board (the 1500 s time-cap tail);
            # a chunk's .done marker is written when all of its indices have returned.
            a.chunk_dir.mkdir(parents=True, exist_ok=True); n_chunks = (a.n + a.chunk_size - 1) // a.chunk_size
            remaining = {}; taken = []; inflight = [0]; lim = 2 * a.workers   # backpressure: claim a chunk only when the pool is nearly drained,
            def stream():                                                        # else the first process would grab every chunk (imap feeds eagerly)
                for k in range(n_chunks):
                    while inflight[0] >= lim: time.sleep(1.0)
                    claim = a.chunk_dir / f"c{k:05d}.claimed"
                    try: os.mkdir(claim)
                    except FileExistsError: continue
                    (claim / "host").write_text(f"{host} {os.getpid()} {time.strftime('%F %T', time.gmtime())}\n")
                    s0 = a.start_index + k * a.chunk_size; n = min(a.chunk_size, a.start_index + a.n - s0)
                    remaining[k] = n; taken.append(k); print(f"[chunk {k}/{n_chunks}] idx {s0}+{n} claimed on {host}", flush=True)
                    for i in range(s0, s0 + n):
                        inflight[0] += 1; yield i
            done_k = 0
            for k_, row in enumerate(pool.imap_unordered(_one, stream(), chunksize=1), 1):
                rows.append(row); rf.write(json.dumps(row) + "\n"); rf.flush()
                if row["status"] == "error": print("[error]", row["idx"], row["error"], row["tb"], flush=True)
                elif row["status"] == "ok": print(f"[{k_}] idx={row['idx']} nets={row['n_nets']} pins={row['pins_placed']}/{row['pins_target']} act/pin={row['actions_per_pin']} repaired={row['verify']['repaired']} pruned={row['pruned']} {row['timing']} {row['wall_s']}s", flush=True)
                else: print(f"[{k_}] idx={row['idx']} {row['status']} {row.get('verify')} {row.get('wall_s')}s", flush=True)
                inflight[0] -= 1; k = (int(row["idx"]) - a.start_index) // a.chunk_size; remaining[k] -= 1
                if remaining[k] == 0:
                    (a.chunk_dir / f"c{k:05d}.done").touch(); done_k += 1; print(f"[chunk {k}] done ({done_k}/{len(taken)} of this host's chunks)", flush=True)
            print(f"[chunks] {host} took {len(taken)} chunks; queue exhausted", flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    def q(v): v = sorted(v); return [round(v[int(p * (len(v) - 1))], 2) for p in (0.1, 0.5, 0.9)] if v else []
    tot = lambda k: sum(r.get(k, 0) for r in ok)
    summ = dict(n=a.n, ok=len(ok), verify_fail=sum(r["status"] == "verify_fail" for r in rows), errors=sum(r["status"] == "error" for r in rows), wall_min=round((time.time() - t0) / 60, 1),
                pins_target=tot("pins_target"), pins_gen=tot("pins_gen"), pins_placed=tot("pins_placed"), pruned=tot("pruned"), pins_rejected=tot("pins_rejected"), site_fail=tot("site_fail"), nets_dropped=tot("nets_dropped"), nets_shrunk=tot("nets_shrunk"),
                rejections={k: tot(k) for k in sorted({k for r in ok for k in r if k.startswith("rej_")})},
                actions=tot("actions"), actions_per_pin=round(tot("actions") / max(1, tot("pins_placed")), 2), within_budget=sum(r["within_budget"] for r in ok),
                repaired_pins=sum(r["verify"]["repaired"] for r in ok), boards_pruned=sum(r["pruned"] > 0 for r in ok),
                n_nets_q=q([r["n_nets"] for r in ok]), size_q=q([math.sqrt(r["bw"] * r["bh"]) for r in ok]), wall_s_q=q([r["wall_s"] for r in ok]), wall_s_max=max([r["wall_s"] for r in ok] or [0]))
    (a.out_dir / f"_summary{tag}.json").write_text(json.dumps(summ, indent=1))
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
