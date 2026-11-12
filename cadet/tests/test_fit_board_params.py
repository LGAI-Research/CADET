"""cadet/stats/fit_board_params.py — the generator's target statistics, estimated from the D3 medium fit half.

Guards the tracked artifact cadet/configs/fit_params/d3b_fit_half.json (schema, provenance = the fit half only,
value ranges the generator relies on) and the local s-expression reader. The rebuild test needs the PCB-bench data
root and is skipped without it.
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FIT = REPO / "cadet/configs/fit_params/d3b_fit_half.json"
SPLIT = REPO / "cadet/configs/splits/d3_medium_half.json"
sys.path.insert(0, str(REPO / "cadet/stats"))
import fit_board_params as fbp  # noqa: E402


@pytest.fixture(scope="module")
def fit():
    return json.load(open(FIT))


def test_sexp_reader_roundtrips_a_pad():
    node = fbp.load_sexp('(footprint "X:Y" (at 10 -2.5 90) (pad "1" smd rect (at 0.5 0) (size 1 0.6) (layers "F.Cu" "F.Mask") (net 3 "N\\"3")))')
    fp = node; at = fbp.find_first(fp, "at"); pad = fbp.find_first(fp, "pad")
    assert node[0] == "footprint" and fbp.num(at[3]) == 90.0 and pad[2] == "smd"
    assert {fbp.unq(l) for l in fbp.find_first(pad, "layers")[1:]} == {"F.Cu", "F.Mask"}
    assert fbp.find_first(pad, "net")[1] == "3" and fbp.find_all(fp, "pad") == [pad]


def _outline_wh(tmp_path, shapes: str) -> tuple[float, float]:
    pcb = tmp_path / "b.kicad_pcb"
    pcb.write_text(f'(kicad_pcb {shapes} (footprint "R" (at 0 0) (pad "1" smd rect (at 1 1) (size 1 1) (layers "F.Cu") (net 1 "A"))'
                   f' (pad "2" smd rect (at 3 1) (size 1 1) (layers "F.Cu") (net 1 "A"))))')
    r = fbp.extract(pcb, tmp_path / "missing.kicad_pro")
    return round(r["W"], 6), round(r["H"], 6)


def test_round_outlines_span_their_full_extent(tmp_path):
    # a circle's defining points are its center and one rim point — a round board used to read as a radius-long segment
    assert _outline_wh(tmp_path, '(gr_circle (center 100 50) (end 113 50) (layer "Edge.Cuts"))') == (26.0, 26.0)
    # footprint-owned circle: rotated/translated like the pads, still 2r wide
    assert _outline_wh(tmp_path, '(footprint "O" (at 50 50 90) (fp_circle (center 1 0) (end 3 0) (layer "Edge.Cuts")))') == (4.0, 4.0)
    # 3/4 arc (start 0 deg, mid 135 deg, end 270 deg around the origin, r 10) passes the 90 and 180 deg extremes, not in its points
    arc = '(gr_arc (start 10 0) (mid -7.0710678 7.0710678) (end 0 -10) (layer "Edge.Cuts"))'
    assert _outline_wh(tmp_path, arc) == (20.0, 20.0)
    # rounded-rectangle corner: the extreme sits on an end point, so W/H stay exactly the lines' span
    rr = ('(gr_line (start 0 1) (end 0 9) (layer "Edge.Cuts")) (gr_arc (start 0 1) (mid 0.292893 0.292893) (end 1 0) (layer "Edge.Cuts"))'
          ' (gr_line (start 1 0) (end 9 0) (layer "Edge.Cuts"))')
    assert _outline_wh(tmp_path, rr) == (9.0, 9.0)


def test_schema_and_provenance(fit):
    for k in ("source", "size", "aspect", "net_density", "locality", "pins_per_net", "pad_smd", "pad_tht", "netclass_sets", "layer_mix", "boards"):
        assert k in fit, k
    src = fit["source"]
    assert src["split_json"] == "cadet/configs/splits/d3_medium_half.json" and src["difficulty"] == "medium" and src["split"] == "fit"
    half = json.load(open(SPLIT))["medium"]
    assert set(src["boards"]).isdisjoint(half["test"]), "fit statistics must never see the held-out half"
    assert set(src["boards"]) | set(src["excluded"]) == set(half["fit"])
    assert src["n_boards"] == len(src["boards"]) == len(fit["boards"]) == len(fit["layer_mix"])


def test_parametric_fits_are_sane(fit):
    for key, med in (("size", "median"), ("net_density", "median"), ("locality", "median_frac_diag")):
        d = fit[key]; assert d["dist"] == "lognormal" and d[med] > 0 and 0 < d["sigma"] < 2 and math.isfinite(d["sigma"])
    assert 15 < fit["size"]["median"] < 80 and fit["size"]["clip"][0] < fit["size"]["median"] < fit["size"]["clip"][1]
    a = fit["aspect"]; assert 0 <= a["square_p"] < 0.5 and 0 < a["sigma"] < 1.5 and 1 < a["max"] <= a["observed_max"]
    assert fit["net_density"]["nets_clip"][0] == 2 and fit["net_density"]["nets_clip"][1] >= 10
    assert 0.02 < fit["locality"]["median_frac_diag"] < 0.5


def test_pools_respect_filters(fit):
    f = fit["source"]["filters"]; lo, hi = f["pins_per_net"]
    pool = fit["pins_per_net"]["pool"]; assert pool and all(lo <= v <= hi for v in pool) and fit["pins_per_net"]["min"] == lo
    assert fit["pad_smd"] and all(f["smd_min_side"] <= min(w, h) and max(w, h) <= f["smd_max_side"] for w, h in fit["pad_smd"])
    assert fit["pad_tht"] and all(f["tht_drill_min"] <= d < min(w, h) <= f["tht_max_side"] for w, h, d in fit["pad_tht"])
    assert fit["netclass_sets"] and all(cs for cs in fit["netclass_sets"])
    for cs in fit["netclass_sets"]:
        for c in cs:
            assert c["via_drill"] < c["via_dia"] and c["clearance"] >= f["class_min_clearance"] and c["width"] >= f["class_min_width"] and c["w"] >= 1
    assert all(0 <= t <= 1 and 0 <= F <= 1 for t, F in fit["layer_mix"])


def test_test_half_json_is_reports_only_and_disjoint(fit):
    t = json.load(open(FIT.parent / "d3b_test_half.json"))
    assert t["source"]["split"] == "test" and set(t["source"]["boards"]).isdisjoint(fit["source"]["boards"])
    half = json.load(open(SPLIT))["medium"]; assert set(t["source"]["boards"]) | set(t["source"]["excluded"]) == set(half["test"])

