"""Report for a generated split: yield, actions/pin, hops, modes, timing, prune."""
import json, os, sys, glob, math
from collections import Counter
d = sys.argv[1]
rows = [json.loads(l) for f in glob.glob(d + '/_gen_rows_*.jsonl') for l in open(f) if l.strip()]
ok = [r for r in rows if r['status'] == 'ok']
print(f"boards {len(rows)}  ok {len(ok)}  verify_fail {sum(r['status']=='verify_fail' for r in rows)}  error {sum(r['status']=='error' for r in rows)}")
def q(v, ps=(0.5, 0.9, 1.0)):
    v = sorted(v); return [round(v[min(len(v) - 1, int(p * (len(v) - 1)))], 1) for p in ps] if v else []
pins = sum(r['pins_placed'] for r in ok); acts = sum(r['actions'] for r in ok)
print(f"pins placed {pins} (gen {sum(r['pins_gen'] for r in ok)}, pruned {sum(r['pruned'] for r in ok)} on {sum(r['pruned']>0 for r in ok)} boards)  target {sum(r['pins_target'] for r in ok)}")
print(f"actions/pin {acts/max(1,pins):.2f}  within 6x budget {sum(r['within_budget'] for r in ok)}/{len(ok)}  repaired pins {sum(r['verify']['repaired'] for r in ok)}")
hh = Counter()
for r in ok:
    for k, v in r['hops_hist'].items(): hh[int(k)] += v
n = sum(hh.values()); print('hops/pin hist', {k: f'{v/n:.1%}' for k, v in sorted(hh.items())})
rej = Counter()
for r in ok:
    for k, v in r.items():
        if k.startswith('rej_'): rej[k] += v
print('rejections', dict(rej), 'attempts', sum(r['attempts'] for r in ok), 'site_fail', sum(r['site_fail'] for r in ok))
print('wall_s p50/p90/max', q([r['wall_s'] for r in ok]), ' gen_s', q([r['timing']['gen_s'] for r in ok]), ' env_s', q([r['timing']['env_s'] for r in ok]), ' search_s', q([r['timing']['search_s'] for r in ok]), ' verify_s', q([r['timing']['verify_s'] for r in ok]))
modes = Counter(); mvia = Counter()
for r in ok:
    f = f"{d}/_demo/board_{r['idx']:05d}.json"
    if not os.path.exists(f):      # an accepted board beyond the kept N: select_first_n.py moved it to _extra/
        continue
    dm = json.load(open(f))
    for a in dm['actions']:
        if a[0] == 3: modes[{0: 'mark', 1: 'shove', 2: 'walk'}[a[4]]] += 1
        if a[0] == 4: mvia[{0: 'mark', 1: 'shove', 2: 'walk'}[a[4]]] += 1
print('make_line modes', dict(modes), ' make_via modes', dict(mvia))
# per-board table sorted by wall
print(f"{'idx':>4} {'size':>6} {'nets':>4} {'pins t/gen/fin':>15} {'act/pin':>7} {'rep':>3} {'prune':>5} {'gen_s':>6} {'ver_s':>6} {'wall':>6}")
for r in sorted(ok, key=lambda r: -r['wall_s'])[:12]:
    print(f"{r['idx']:>4} {math.sqrt(r['bw']*r['bh']):>6.1f} {r['n_nets']:>4} {r['pins_target']:>5}/{r['pins_gen']:>4}/{r['pins_placed']:>4} {r['actions_per_pin']:>7.2f} {r['verify']['repaired']:>3} {r['pruned']:>5} {r['timing']['gen_s']:>6.0f} {r['timing']['verify_s']:>6.0f} {r['wall_s']:>6.0f}")
