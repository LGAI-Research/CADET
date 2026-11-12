"""Replay a demo (action list from executor2) through the training wrapper.

Actions are stored as (action_type, x, y, layer, mode) with coordinates, so pointer indices are
re-resolved against the live candidate pool at every step — the replay is therefore also the
check that the demo is deterministic / self-consistent (every step must be action_success and
the end state must match the recorded clean_pass). With --record, the per-step training tuples
(obs, masks, action triple) are yielded for BC.

    python replay_demo.py <demo.json> [--ckpt CKPT] [--record out.pkl]
"""
from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from cadet.paths import use_pcbworld  # noqa: E402
use_pcbworld()

from executor2 import build_env, _obs, _ratsnest, CKPT  # noqa: E402
from pcb_world.core.action_schema import ACT_NET_SELECT, ACT_START_ROUTE, ACT_NET_END, ACT_MAKE_LINE, ACT_MAKE_VIA, ACTION_NAMES  # noqa: E402


def cand_index(env, x, y, layer):
    k2 = (round(float(x), 3), round(float(y), 3))
    for i, (cx, cy, cl) in enumerate(env.cand_mm_list):
        if (round(cx, 3), round(cy, 3)) == k2 and cl == int(layer):
            return i
    return None


def replay(demo: dict, ckpt: str = CKPT, record: bool = False, skip_partial_nets: bool = True, verbose: bool = False):
    """Returns (ok, stats, records). ok = every step succeeded and final state matches."""
    knobs = demo.get("env_knobs") or {}          # demos carry the preset / off-board mask they were built with
    env, _ = build_env(demo["board"], ckpt, preset=knobs.get("preset"), offboard_mask=knobs.get("offboard_mask"))
    try:
        raw = env.reset()[0]; obs = _obs(raw)
        nets = list(env.sorted_net_codes); status = {int(k): v for k, v in demo.get("net_status", {}).items()}
        records = []; n_ok = 0; n_fail = 0; cur_net = None; skipping = False; nets_done = 0
        term = trunc = False        # an empty action list never enters the loop
        for at, x, y, l, mode in demo["actions"]:
            at = int(at)
            if at == ACT_NET_SELECT:
                cur_net = int(x); skipping = skip_partial_nets and status.get(cur_net) == "partial"
                ptr = nets.index(cur_net)
            elif at == ACT_NET_END:
                ptr = 0
            else:
                ptr = cand_index(env, x, y, l)
                if ptr is None and at == ACT_MAKE_VIA and obs["router_head"].get("is_routing"):
                    # the generator records make_via at the realized via centre on the post-via layer;
                    # the requested point is the nearest pool candidate on the head's layer within the via radius
                    hl = int(obs["router_head"]["current_layer"]); best = None
                    for i, (cx, cy, cl) in enumerate(env.cand_mm_list):
                        if cl != hl: continue
                        d = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
                        if d <= 0.8 and (best is None or d < best[0]): best = (d, i)
                    ptr = None if best is None else best[1]
                if ptr is None:
                    n_fail += 1
                    if verbose: print(f"  FAIL step {n_ok}: {ACTION_NAMES[at]} ({x},{y}) L{l} not in candidate pool; routing={obs['router_head'].get('is_routing')} masks={obs['_masks']['action']}")
                    break
            triple = (at, int(ptr), int(mode))
            if record and not skipping:
                records.append(dict(obs=raw, masks=raw.get("_masks"), action=triple, net=cur_net))
            raw, rew, term, trunc, info = env.step(triple); obs = _obs(raw)
            if info.get("action_success", True):
                n_ok += 1
                if at == ACT_NET_END and not skipping:
                    nets_done += 1
            else:
                n_fail += 1
                if verbose: print(f"  FAIL step {n_ok}: {ACTION_NAMES[at]} ({x},{y}) L{l} mode={mode} -> {info.get('action_class')} {info.get('rejected','')} routing={obs['router_head'].get('is_routing')}")
                break
            if term or trunc:
                break
        unrouted = sum(len(_ratsnest(obs, n)) for n in nets) // 2
        drc = len(obs.get("drc_violations") or [])
        if n_fail:   # keep the longest replayable prefix, cut back to the last completed net
            records = [r for r in records if r["net"] != cur_net]
        if term and not n_fail and cur_net is not None and not skipping:
            nets_done += 1          # the last net closes by episode termination, without a net_end
        nets_kept = nets_done
        stats = dict(steps_ok=n_ok, steps_fail=n_fail, unrouted_pins=unrouted, drc=drc,
                     clean_pass=unrouted == 0 and drc == 0, recorded=len(records), nets_kept=nets_kept,
                     nets_total=len(nets), prefix_frac=round(n_ok / max(1, len(demo["actions"])), 3))
        ok = n_fail == 0 and stats["clean_pass"] == bool(demo.get("clean_pass", False))
        return ok, stats, records
    finally:
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demo"); ap.add_argument("--ckpt", default=CKPT); ap.add_argument("--record", default=None); ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    demo = json.load(open(a.demo))
    ok, stats, records = replay(demo, a.ckpt, record=bool(a.record), verbose=a.v)
    print(json.dumps(dict(board=demo["board"], ok=ok, recorded_clean_pass=demo.get("clean_pass"), **stats)))
    if a.record:
        pickle.dump(records, open(a.record, "wb"))
