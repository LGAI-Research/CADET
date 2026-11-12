"""Fidelity statistics: one extractor (fit_board_params.extract) over four board sets -> 17 statistics, p10/p50/p90 and
two-sample KS distances. Sets: D3-B ref, D3-B test (real boards), RouteDemos train (a random sample of the pristine
dataset files) and D2 train.

    python cadet/stats/fidelity_extract.py --routedemos <RouteDemos dir>/train --d2 <D2 train dir> --out <dir>
    python cadet/stats/fidelity_tables.py <dir>          # KS-to-ref and KS-to-test tables
"""
import argparse, json, math, os, random, sys, statistics as S
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_board_params as F

def real_set(split):
    d = json.load(open(REPO / "cadet/configs/splits/d3_medium_half.json")); root = Path(os.path.expandvars(d["dataset_dirs"][split]))
    return [(str(root / b / d["board_filename"]), str(root / b / F.PRO_NAME)) for b in d["medium"][split]]
def synth_set(root, n=None, seed=0):
    fs = sorted(p for p in Path(root).glob("board_*.kicad_pcb"))
    if n: random.Random(seed).shuffle(fs); fs = fs[:n]
    return [(str(p), str(p.with_suffix(".kicad_pro"))) for p in fs]
def _ex(a):
    try: return F.extract(Path(a[0]), Path(a[1]))
    except Exception as e: return dict(error=f"{a[0]}: {e}")

def q(v, p): v = sorted(v); return v[int(p * (len(v) - 1))] if v else float("nan")
def q3(v): return [q(v, .1), q(v, .5), q(v, .9)]
def lognorm(x): lx = [math.log(v) for v in x if v > 0]; return (math.exp(S.fmean(lx)), S.pstdev(lx)) if lx else (float("nan"),) * 2
def ks(a, b):
    """Two-sample KS statistic, tie-safe: ECDFs compared at every distinct value of the union."""
    import bisect
    a, b = sorted(a), sorted(b); d = 0.0
    for v in sorted(set(a) | set(b)):
        d = max(d, abs(bisect.bisect_right(a, v) / len(a) - bisect.bisect_right(b, v) / len(b)))
    return d

def samples(rows):
    """Per-axis sample lists (same filters as fit_board_params.aggregate)."""
    B = [r for r in rows if r and "error" not in r and r["area"] / 100.0 > F.FILTERS["area_cm2_min"] and r["n_nets"] >= 1]
    lo, hi = F.FILTERS["pins_per_net"]
    smd = [(min(w, h), max(w, h)) for r in B for w, h, _ in r["smd"] if F.FILTERS["smd_min_side"] <= min(w, h) and max(w, h) <= F.FILTERS["smd_max_side"]]
    tht = [(w, d) for r in B for w, h, d in r["tht"] if F.FILTERS["tht_drill_min"] <= d < min(w, h) <= F.FILTERS["tht_max_side"]]
    cls = [(c["clearance"], c["width"], c["w"]) for r in B for c in F.class_set(r)]
    smdb = [r for r in B if r["f_tht"] < 0.999]
    return dict(n=len(B), n_raw=len(rows),
        size=[math.sqrt(r["area"]) for r in B], aspect=[math.log(max(r["W"], r["H"]) / min(r["W"], r["H"])) for r in B],
        nets_cm2=[r["n_nets"] / (r["area"] / 100) for r in B], pins_cm2=[sum(r["pins_per_net"]) / (r["area"] / 100) for r in B],
        locality=[e for r in B for e in r["mst_edges"]], pins_per_net=[f for r in B for f in r["pins_per_net"] if lo <= f <= hi],
        smd_short=[a for a, _ in smd], smd_long=[b for _, b in smd], tht_size=[a for a, _ in tht], tht_drill=[b for _, b in tht],
        clearance=[c for c, _, w in cls for _ in range(w)], width=[c for _, c, w in cls for _ in range(w)], n_classes=[len(F.class_set(r)) for r in B],
        f_tht=[r["f_tht"] for r in B], f_F=[r["f_F"] for r in smdb], nets=[r["n_nets"] for r in B], pins=[sum(r["pins_per_net"]) for r in B])

AXES = ["size", "aspect", "nets_cm2", "pins_cm2", "locality", "pins_per_net", "smd_short", "smd_long", "tht_size", "tht_drill", "clearance", "width", "n_classes", "f_tht", "f_F", "nets", "pins"]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routedemos", required=True, help="RouteDemos train dir (board_*.kicad_pcb)"); ap.add_argument("--d2", required=True, help="D2 train dir")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n", type=int, default=10000, help="RouteDemos sample size"); ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(); OUT = a.out; OUT.mkdir(parents=True, exist_ok=True); N_SAMPLE = a.n
    sets = {"fit": real_set("fit"), "test": real_set("test"), "routedemos": synth_set(a.routedemos, N_SAMPLE), "d2a": synth_set(a.d2)}
    rows = {}
    with Pool(a.workers) as pool:
        for k, lst in sets.items():
            rows[k] = pool.map(_ex, lst, chunksize=16); errs = [r for r in rows[k] if r and "error" in r]; none = sum(1 for r in rows[k] if r is None)
            print(f"{k}: {len(lst)} boards, extract ok {len(lst) - len(errs) - none}, no-outline {none}, errors {len(errs)}", errs[:2], flush=True)
    smp = {k: samples(v) for k, v in rows.items()}
    json.dump({k: {a: v for a, v in s.items()} for k, s in smp.items()}, open(OUT / "four_sets_samples.json", "w"))
    summ = {}
    for a in AXES:
        summ[a] = {k: dict(p=q3(smp[k][a]), n=len(smp[k][a]), lognorm=lognorm(smp[k][a]) if a not in ("aspect", "f_tht", "f_F") else None,
                           mean=S.fmean(smp[k][a]), ks_vs_test=None if k == "test" else ks(smp[k][a], smp["test"][a])) for k in smp}
    extra = {k: dict(n_boards=smp[k]["n"], square_p=sum(1 for v in smp[k]["aspect"] if v <= 1e-6) / len(smp[k]["aspect"]),
                     all_tht_boards=sum(1 for v in smp[k]["f_tht"] if v >= 0.999) / len(smp[k]["f_tht"]), all_front_smd=sum(1 for v in smp[k]["f_F"] if v >= 0.999) / max(1, len(smp[k]["f_F"])),
                     ppn2=Counter(smp[k]["pins_per_net"])[2] / len(smp[k]["pins_per_net"]), ppn3=Counter(smp[k]["pins_per_net"])[3] / len(smp[k]["pins_per_net"])) for k in smp}
    json.dump(dict(axes=summ, extra=extra), open(OUT / "four_sets_summary.json", "w"), indent=1)
    cols = ["test", "fit", "routedemos", "d2a"]
    print(f"\n{'axis':>13} | " + " | ".join(f"{c:^24}" for c in cols) + " || KS vs test: ref / RouteDemos / D2")
    for a in AXES:
        print(f"{a:>13} | " + " | ".join(f"{summ[a][c]['p'][0]:7.3f} {summ[a][c]['p'][1]:7.3f} {summ[a][c]['p'][2]:7.3f}" for c in cols)
              + f" || {summ[a]['fit']['ks_vs_test']:.3f} / {summ[a]['routedemos']['ks_vs_test']:.3f} / {summ[a]['d2a']['ks_vs_test']:.3f}")
    LBL = dict(size="size: side sqrt(W*H) mm", aspect="aspect: ln(long/short)", nets_cm2="net density nets/cm2", pins_cm2="pin density pins/cm2 (calibration axis)",
               locality="locality: MST edge / diag (pooled edges)", pins_per_net="pins per net (pooled nets)", smd_short="SMD pad short side mm (pooled pads)", smd_long="SMD pad long side mm",
               tht_size="THT pad size mm (pooled pads)", tht_drill="THT drill mm", clearance="netclass clearance mm (net-weighted)", width="netclass track width mm (net-weighted)",
               n_classes="netclasses per board", f_tht="THT share of pads per board", f_F="front share of SMD pads (boards with SMD)", nets="nets per board", pins="pads per board")
    L = ["# RouteDemos vs D3-B ref / test vs D2 — same extractor", "",
         "Extractor: `cadet/stats/fit_board_params.extract`, filters of `aggregate`. ",
         f"Sets: D3-B ref ({extra['fit']['n_boards']} boards), D3-B test ({extra['test']['n_boards']}), RouteDemos train random {N_SAMPLE} of 100k pristine files ({extra['routedemos']['n_boards']} after the area>1cm2 filter), D2 train ({extra['d2a']['n_boards']}).",
         "Reference column = **test half** (fit half shaped the generator; test half is held out). fit-vs-test = split noise floor. KS = two-sample Kolmogorov-Smirnov statistic (0..1, scale-free, no distribution assumed).", "",
         "## p10 / p50 / p90 per axis", "", "| axis | test (ref) | fit | RouteDemos | d2a |", "|---|---|---|---|---|"]
    fm = lambda p: " / ".join(f"{x:.3g}" for x in p)
    for a in AXES: L.append(f"| {LBL[a]} | {fm(summ[a]['test']['p'])} | {fm(summ[a]['fit']['p'])} | {fm(summ[a]['routedemos']['p'])} | {fm(summ[a]['d2a']['p'])} |")
    L += ["", "## KS distance to the test half (lower = closer; fit column = noise floor)", "", "| axis | fit | RouteDemos | d2a |", "|---|---|---|---|"]
    for a in AXES: L.append(f"| {LBL[a]} | {summ[a]['fit']['ks_vs_test']:.3f} | {summ[a]['routedemos']['ks_vs_test']:.3f} | {summ[a]['d2a']['ks_vs_test']:.3f} |")
    L += ["", "## Shares", "", "| | test | fit | RouteDemos | d2a |", "|---|---|---|---|---|"]
    for k, lbl in [("square_p", "square boards"), ("all_tht_boards", "all-THT boards"), ("all_front_smd", "all-front SMD (boards with SMD)"), ("ppn2", "2-pin nets"), ("ppn3", "3-pin nets")]:
        L.append(f"| {lbl} | " + " | ".join(f"{extra[c][k]:.3f}" for c in cols) + " |")
    L += ["", "## Notes", "",
          "- Pooled axes (locality, pins per net, pad dims, netclass) compare pooled samples, so KS there is dominated by the big synthetic sample; per-board axes compare boards.",
          "- Ties: netclass values and pins-per-net are discrete; the KS here is tie-safe (ECDFs at every distinct value).",
          "- fit-vs-test KS with 139-141 boards is itself noisy (no bootstrap CI computed here).",
          "- Raw per-axis samples: `four_sets_samples.json`; summary numbers: `four_sets_summary.json`."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n"); print("wrote", OUT / "REPORT.md")
