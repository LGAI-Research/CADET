"""Result table over scored cells: clean pass (CP), success, routability, final potential (fp), wirelength, vias,
DRC violations and serial time, as mean ± sample sd over samples.

    python cadet/eval/table.py --n-boards 141 --init <cell> \
        --row "FreeRouting @1=<cell>:@1" --row "CADET @1=<cell>:@1" --row "CADET @3=<cell>:@3" \
        [--time "CADET @3=episode_wall:<run>/episode_wall.jsonl"] [--time "FreeRouting @1=manifest:<seed0/manifest.json>,<seed1/manifest.json>,..."] [--time "LLM @1=cell:<cell>"]

A row is LABEL=CELL:GROUPS. GROUPS lists the samples; each sample is a group of rollout indices and the best
rollout of the group by (clean_pass, fp) represents it: "@1" = 0;1;2;3 (four single rollouts) and
"@3" = 0,1,2;3,4,5;6,7,8;9,10,11 (four best-of-three samples), or an explicit list in that syntax.
A missing rollout counts as an unrouted board (CP 0, fp = the board's initial potential, read from --init).
Quality columns average over the rollouts that produced a board. Time is the serial per-board sum over the
sample's rollouts.
"""
import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

Q = ["wirelength_mm", "via_count", "drv_errors_only_count", "drv_errors_and_promoted_count", "total_drv_count"]
COLS = [("CP", "cp", 3), ("success", "succ", 3), ("routability", "rout", 3), ("fp", "fp", 2), ("wirelength (mm)", "wirelength_mm", 0),
        ("vias", "via_count", 1), ("DRV err", "drv_errors_only_count", 2), ("DRV err+promoted", "drv_errors_and_promoted_count", 2),
        ("DRV total (incl. angle)", "total_drv_count", 2), ("time (s, serial)", "time", 0)]
SHORT = {"@1": "0;1;2;3", "@3": "0,1,2;3,4,5;6,7,8;9,10,11"}


def t(x): return str(x).lower() in ("true", "1", "1.0")


def f(x):
    try:
        v = float(x); return None if math.isnan(v) else v
    except (TypeError, ValueError): return None


def load(cell, n_boards, exact=True):
    d = defaultdict(dict)
    for r in csv.DictReader(open(Path(cell) / "per_rollout.csv")):
        d[r["board_id"]][int(r["rollout_idx"])] = r
    if len(d) != n_boards if exact else len(d) > n_boards:
        raise SystemExit(f"{cell}: {len(d)} boards, expected {n_boards}")
    return d


def rec(r, b, init):
    if r is None or f(r.get("final_potential")) is None:
        return {"cp": 0.0, "succ": 0.0, "rout": 0.0, "fp": init[b], "missing": 1}
    d = {"cp": float(t(r["clean_pass"])), "succ": float(t(r["success"])), "rout": f(r["routability"]) or 0.0,
         "fp": f(r["final_potential"]), "missing": 0}
    for q in Q: d[q] = f(r.get(q))
    return d


def sample_stats(boards, pick, secs=None):
    recs = [pick(b) for b in boards]
    out = {k: st.mean(x[k] for x in recs) for k in ("cp", "succ", "rout", "fp")}
    for q in Q:
        v = [x[q] for x in recs if x.get(q) is not None]
        out[q] = st.mean(v) if v else float("nan")
    out["missing"] = sum(x["missing"] for x in recs)
    out["time"] = secs
    return out


def times(spec):
    """(board_id, rollout) -> seconds."""
    kind, path = spec.split(":", 1)
    tb = {}
    if kind == "episode_wall":       # rollout_queue.py with --n-envs 1: consecutive completion-time differences
        prev = 0.0
        for x in (json.loads(l) for l in open(path)):
            tb[(x["board_id"], int(x["rollout"]))] = x.get("t_board_s", x["t_since_start_s"] - prev); prev = x["t_since_start_s"]
    elif kind == "manifest":         # rule-based runner: routing time per (board, seed), one manifest.json per seed
        for mp in path.split(","):
            m = json.load(open(mp))
            for x in m["results"]:
                tb[(x["board_id"], int(x.get("seed", m.get("seed", 0))))] = x["elapsed_routing_s"]
    elif kind == "cell":             # the run's own per-board time recorded in the cell (LLM runs)
        for x in csv.DictReader(open(Path(path) / "per_rollout.csv")):
            tb[(x["board_id"], int(x["rollout_idx"]))] = f(x.get("per_board_rollout_time")) or 0.0
    else:
        raise SystemExit(f"unknown time source {kind}")
    return tb


def fmt(v, d):
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in v):
        return "—"
    mu = st.mean(v); sd = st.stdev(v) if len(v) > 1 else None
    return f"{mu:,.{d}f}" + (f" ± {sd:,.{d}f}" if sd is not None else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boards", type=int, required=True)
    ap.add_argument("--init", required=True, help="cell whose rows give each board's initial potential")
    ap.add_argument("--row", action="append", required=True)
    ap.add_argument("--time", action="append", default=[])
    a = ap.parse_args()
    ref = load(a.init, a.n_boards)
    init = {b: f(next(iter(rs.values()))["initial_potential"]) for b, rs in ref.items()}
    boards = sorted(init)
    tsrc = {s.split("=", 1)[0]: times(s.split("=", 1)[1]) for s in a.time}
    print("| method | n | " + " | ".join(c[0] for c in COLS) + " | missing/sample |")
    print("|---|---:|" + "---:|" * (len(COLS) + 1))
    for spec in a.row:
        label, rest = spec.split("=", 1)
        cell, groups = rest.rsplit(":", 1)
        gs = [[int(i) for i in g.split(",")] for g in SHORT.get(groups, groups).split(";")]
        d, tb = load(cell, a.n_boards, exact=False), tsrc.get(label)      # absent boards count as missing
        ss = [sample_stats(boards, lambda b, g=g: max((rec(d[b].get(r), b, init) for r in g), key=lambda x: (x["cp"], x["fp"])),
                           sum(tb.get((b, r), 0.0) for b in boards for r in g) if tb else None) for g in gs]
        print(f"| {label} | {len(ss)} | " + " | ".join(fmt([s[k] for s in ss], dd) for _, k, dd in COLS)
              + f" | {st.mean(s['missing'] for s in ss):.1f} |")


if __name__ == "__main__":
    main()
