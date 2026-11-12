"""Difficulty axes: real boards (a fit_params json) vs a generated set (from _witness meta).
Axes: nets/cm2, pins/cm2, pins/net, board size, track width, clearance. Plus (generated only) actions/pin, hops."""
import json, gzip, glob, math, sys, statistics as S
def q(v, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = sorted(v); return [round(v[int(p * (len(v) - 1))], 2) for p in ps] if v else []
def show(name, rows):
    print(f"== {name} ({len(rows)} boards)")
    for k in ('size', 'nets_cm2', 'pins_cm2', 'pins_per_net', 'nets', 'pins'):
        print(f"  {k:>12} p10/25/50/75/90 = {q([r[k] for r in rows])}")
def real_rows(fit_json):
    """Difficulty axes of the real boards recorded in a fit_params json (cadet/stats/fit_board_params.py ``boards`` rows)."""
    rows = []
    for b in json.load(open(fit_json))['boards']:
        a = b['area_cm2']; rows.append(dict(size=math.sqrt(a * 100.0), nets=b['n_nets'], pins=b['pins'], nets_cm2=b['n_nets'] / a, pins_cm2=b['pins'] / a, pins_per_net=b['pins'] / max(1, b['n_nets'])))
    return rows
def gen(d):
    rows = []
    for f in glob.glob(d + '/_witness/*.witness.json.gz'):
        w = json.load(gzip.open(f, 'rt')); a = w['board_w'] * w['board_h'] / 100.0
        pins = sum(len(n.get('pads', n.get('pins', []))) + (0 if 'pads' in n else 1) for n in w['nets'])   # older witness format: pins = paths (pads-1)
        if 'pads' not in w['nets'][0]:
            pins = sum(len(n['pins']) + 1 for n in w['nets'])
        else:
            pins = sum(len(n['pads']) for n in w['nets'])
        nets = len(w['nets'])
        rows.append(dict(size=math.sqrt(w['board_w'] * w['board_h']), nets=nets, pins=pins, nets_cm2=nets / a, pins_cm2=pins / a, pins_per_net=pins / max(1, nets)))
    return rows
if __name__ == '__main__':
    # args: generated dirs and/or fit_params jsons (real halves)
    for d in sys.argv[1:]:
        show(d, real_rows(d) if d.endswith('.json') else gen(d))
