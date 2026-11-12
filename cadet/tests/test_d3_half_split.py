"""cadet/configs/splits/d3_medium_half.json — the generator-fit / held-out halves of the D3 medium (d3b) real boards.

Guards the contract the synthetic-board pipeline relies on: the two halves are disjoint, 2-layer only, drawn from
d3.json's medium pool, balanced per wire-type stratum, and the derived board lists match the json. The rebuild test
needs the PCB-bench data root and is skipped without it.
"""
from __future__ import annotations
import json, os, statistics, sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPLIT = REPO / "cadet/configs/splits/d3_medium_half.json"
sys.path.insert(0, str(REPO))
from cadet.paths import pcbworld_root  # noqa: E402

D3 = pcbworld_root() / "configs/datasets/d3.json"
LISTS = REPO / "cadet/configs/splits"


@pytest.fixture(scope="module")
def split():
    return json.load(open(SPLIT))


def test_halves_disjoint_and_cover_2layer_medium(split):
    fit, test = set(split["medium"]["fit"]), set(split["medium"]["test"])
    assert fit.isdisjoint(test)
    pool = {str(b) for b in json.load(open(D3))["medium"]["train"]}
    assert (fit | test) <= pool
    excluded = set(split["_split_meta"]["excluded_non_2layer"])
    assert (fit | test | excluded) == pool
    feats = split["_board_features"]
    assert all(feats[b]["layers"] == 2 for b in fit | test)
    assert abs(len(fit) - len(test)) <= 1


def test_strata_balanced(split):
    feats = split["_board_features"]
    for name, st in split["_stratum_stats"].items():
        ids = [b for b, f in feats.items() if f["wire_type"] == name]
        assert st["n"] == len(ids) and st["fit"] + st["test"] == st["n"]
        assert abs(st["fit"] - st["test"]) <= 1
    assert sum(st["n"] for st in split["_stratum_stats"].values()) == len(split["medium"]["fit"]) + len(split["medium"]["test"])


def test_halves_have_matching_size_distribution(split):
    """Alternating on the sorted (pins, nets) order must leave both halves covering the same range."""
    feats = split["_board_features"]
    q = lambda ids, key, p: sorted(feats[b][key] for b in ids)[int(p * (len(ids) - 1))]
    for key in ("pins", "nets"):
        for p in (0.1, 0.5, 0.9):
            a, b = q(split["medium"]["fit"], key, p), q(split["medium"]["test"], key, p)
            assert abs(a - b) <= max(3, 0.1 * max(a, b)), (key, p, a, b)


def test_board_lists_derive_from_json(split):
    """Lists = the half minus the boards the engine cannot load (no Edge.Cuts outline)."""
    ds, fn = split["dataset_dirs"]["fit"], split["board_filename"]
    ex = split["_split_meta"]["eval_excluded"]
    for half in ("fit", "test"):
        lines = (LISTS / f"boards_d3b_{half}.txt").read_text().splitlines()
        assert lines == [f"{ds}/{b}/{fn}" for b in split["medium"][half] if b not in ex[half]]
        assert all(b in split["medium"][half] for b in ex[half])
    # every 2-layer medium board loads in the engine: nothing is excluded
    assert ex == {"fit": [], "test": []}, ex


@pytest.mark.skipif("PCBWORLD_DATA_ROOT" not in os.environ, reason="needs the PCB-bench data root")
def test_rebuild_is_deterministic(split):
    import sys
    sys.path.insert(0, str(REPO / "cadet/stats"))
    from make_d3_half_split import build
    d3 = json.load(open(D3)); root = Path(os.path.expandvars(d3["dataset_dirs"]["train"]))
    if not root.exists():
        pytest.skip(f"data root missing: {root}")
    rebuilt = build(d3, root, root / "pcb_characteristics_exacad_sorted.csv")
    assert rebuilt["medium"] == split["medium"] and rebuilt["_stratum_stats"] == split["_stratum_stats"]
