"""Fidelity tables from a fidelity_extract.py run: KS to D3-B ref (table A), KS to D3-B test (table B), shares and the
size x density correlation. Set keys: fit = D3-B ref, test = D3-B test, d2a = D2.

    python cadet/stats/fidelity_tables.py <dir with four_sets_samples.json + four_sets_summary.json>"""
import bisect, json, math, statistics as S, sys
from pathlib import Path
D = Path(sys.argv[1]); smp = json.load(open(D / "four_sets_samples.json")); summ = json.load(open(D / "four_sets_summary.json"))
def ks(a, b):
    a, b = sorted(a), sorted(b); return max(abs(bisect.bisect_right(a, v) / len(a) - bisect.bisect_right(b, v) / len(b)) for v in sorted(set(a) | set(b)))
AX = ["size", "aspect", "nets_cm2", "pins_cm2", "locality", "pins_per_net", "smd_short", "smd_long", "tht_size", "tht_drill", "clearance", "width", "n_classes", "f_tht", "f_F", "nets", "pins"]
print("n_boards", {k: summ["extra"][k]["n_boards"] for k in summ["extra"]})
print("A: KS to fit  | RouteDemos | test | d2a"); [print(f"{a:>13} | {ks(smp['routedemos'][a], smp['fit'][a]):.3f} | {ks(smp['test'][a], smp['fit'][a]):.3f} | {ks(smp['d2a'][a], smp['fit'][a]):.3f}") for a in AX]
print("B: KS to test | fit | RouteDemos | d2a"); [print(f"{a:>13} | {summ['axes'][a]['fit']['ks_vs_test']:.3f} | {summ['axes'][a]['routedemos']['ks_vs_test']:.3f} | {summ['axes'][a]['d2a']['ks_vs_test']:.3f}") for a in AX]
print("shares test/ref/RouteDemos/D2"); [print(f"{k:>14} | " + " | ".join(f"{summ['extra'][s][k]:.3f}" for s in ("test", "fit", "routedemos", "d2a"))) for k in ("square_p", "all_tht_boards", "all_front_smd", "ppn2", "ppn3")]
def corr(x, y): mx, my = S.fmean(x), S.fmean(y); return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
print("corr(log side, log pins/cm2) | corr(log side, log nets/cm2)")
for s in ("test", "fit", "routedemos", "d2a"):
    ls = [math.log(v) for v in smp[s]["size"]]
    print(f"{s:>5} | {corr(ls, [math.log(v) for v in smp[s]['pins_cm2']]):+.2f} | {corr(ls, [math.log(v) for v in smp[s]['nets_cm2']]):+.2f}")
