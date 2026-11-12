"""Paired board bootstrap (10,000 resamples, seed 0) of method A @1 / @3 minus method B @1 on CP and fp, using
per-board means over the four samples. A missing rollout counts as an unrouted board.

    python cadet/eval/bootstrap.py <cell A, 12 rollouts> <cell B, 4 rollouts>
"""
import csv, math, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path
def t(x): return str(x).lower() in ("true", "1", "1.0")
def f(x):
    try:
        v = float(x); return None if math.isnan(v) else v
    except (TypeError, ValueError): return None
def load(c):
    d = defaultdict(dict)
    for r in csv.DictReader(open(Path(c) / "per_rollout.csv")): d[r["board_id"]][int(r["rollout_idx"])] = r
    return d
cad, fr = load(sys.argv[1]), load(sys.argv[2])
init = {b: f(next(iter(v.values()))["initial_potential"]) for b, v in cad.items()}
boards = sorted(init)
def rec(r, b): return (0.0, init[b]) if r is None or f(r.get("final_potential")) is None else (float(t(r["clean_pass"])), f(r["final_potential"]))
G = {1: [[s] for s in range(4)], 3: [[3 * q, 3 * q + 1, 3 * q + 2] for q in range(4)]}
FR = {b: [rec(fr[b].get(s), b) for s in range(4)] for b in boards}
random.seed(0)
for k in (1, 3):
    C = {b: [max(rec(cad[b].get(r), b) for r in g) for g in G[k]] for b in boards}
    for j, name in ((0, "CP"), (1, "fp")):
        d = [st.mean(x[j] for x in C[b]) - st.mean(x[j] for x in FR[b]) for b in boards]
        bs = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(10000))
        print(f"A @{k} - B @1 {name}: {st.mean(d):+.3f} 95% CI [{bs[250]:+.3f}, {bs[9749]:+.3f}] (n={len(d)})")
