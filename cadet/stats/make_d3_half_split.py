#!/usr/bin/env python3
"""Split the D3 medium (d3b) real boards 50/50 into a generator-fit half and a held-out test half.

Why: the synthetic-board generator estimates its target statistics (size, net density, pins per net, netclass
sets, layer mix, ...) from real boards. Fitting on the evaluation boards leaks evaluation statistics into
the training distribution, so the fit source and the evaluation set must be disjoint.

Rule (deterministic, no seed): keep 2-layer boards only; stratify by wire type (number of
distinct track widths among the project's netclasses); inside each stratum sort by (pins, nets, board id) and
alternate fit/test, starting with a different side in consecutive strata so the two halves balance. Alternating on
the sorted order makes both halves cover the whole size/density range (a front/back split would not).

    python cadet/stats/make_d3_half_split.py            # rebuild cadet/configs/splits/d3_medium_half.json in place
    python cadet/stats/make_d3_half_split.py --out /tmp/x.json --lists-dir /tmp   # dry copy
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(REPO))
from cadet.paths import pcbworld_root  # noqa: E402
from fit_board_params import extract as _extract_board   # None when the board has no Edge.Cuts outline (engine cannot load it)
D3_JSON = pcbworld_root() / "configs/datasets/d3.json"   # PCBWorld's D3 difficulty split
OUT_JSON = REPO / "cadet/configs/splits/d3_medium_half.json"
LISTS_DIR = REPO / "cadet/configs/splits"
PRO_NAME = "processed_v9_guide_v3.kicad_pro"


def wire_type(n_widths: int) -> str:
    return "W1_uniform" if n_widths <= 1 else "W2_dual" if n_widths == 2 else "W3-4_multi" if n_widths <= 4 else "W5+_rich"


def board_features(root: Path, csv_path: Path, ids: list[str]) -> dict[str, dict]:
    """pins / nets / layers from the PCB-bench csv (sample = id without the NNNN_ prefix), wire type from the .kicad_pro."""
    by_sample = {r["sample"]: r for r in csv.DictReader(open(csv_path))}
    out = {}
    for bid in ids:
        r = by_sample[bid.split("_", 1)[1]]
        classes = json.load(open(root / bid / PRO_NAME))["net_settings"]["classes"]
        widths = {round(float(c["track_width"]), 6) for c in classes if c.get("track_width") is not None}
        out[bid] = dict(pins=int(r["pins"]), nets=int(r["nets"]), layers=int(r["layers"]), n_track_widths=len(widths), wire_type=wire_type(len(widths)))
    return out


def assign(features: dict[str, dict]) -> tuple[dict[str, list[str]], dict[str, dict]]:
    strata: dict[str, list[str]] = {}
    for bid, f in features.items():
        strata.setdefault(f["wire_type"], []).append(bid)
    halves = {"fit": [], "test": []}; stats = {}
    for k, name in enumerate(sorted(strata)):
        ids = sorted(strata[name], key=lambda b: (features[b]["pins"], features[b]["nets"], b))
        first = "fit" if k % 2 == 0 else "test"; other = "test" if first == "fit" else "fit"
        for i, b in enumerate(ids):
            halves[first if i % 2 == 0 else other].append(b)
        stats[name] = dict(n=len(ids), fit=sum(1 for i in range(len(ids)) if (i % 2 == 0) == (first == "fit")), test=0)
        stats[name]["test"] = stats[name]["n"] - stats[name]["fit"]
    return {k: sorted(v) for k, v in halves.items()}, stats


def build(d3: dict, root: Path, csv_path: Path) -> dict:
    ids = [str(b) for b in d3["medium"]["train"]]          # d3.json medium 'train' = the whole medium pool
    feats = board_features(root, csv_path, ids)
    kept = {b: f for b, f in feats.items() if f["layers"] == 2}
    excluded = sorted(b for b in ids if b not in kept)
    halves, stats = assign(kept)
    ds = d3["dataset_dirs"]["train"]
    # boards without an Edge.Cuts outline stay in the split (the partition is by design id) but cannot be loaded by the
    # engine, so the eval lists leave them out; the fit statistics extractor skips them by itself (fit_params source.excluded)
    no_outline = {h: [b for b in halves[h] if _extract_board(root / b / d3["board_filename"], root / b / PRO_NAME) is None] for h in halves}
    return dict(board_filename=d3["board_filename"], dataset_dirs=dict(fit=ds, test=ds), medium=halves,
                _split_meta=dict(purpose="generator-fit half vs held-out evaluation half (disjoint)", source="configs/datasets/d3.json medium (all 286)",
                                 keep="layers == 2 only", stratify_by=["wire_type"], wire_type_rule="distinct track_width among .kicad_pro net_settings.classes: 1/2/3-4/5+",
                                 assignment="per stratum sort by (pins, nets, id); alternate fit/test, first side alternates per stratum",
                                 excluded_non_2layer=excluded, n_fit=len(halves["fit"]), n_test=len(halves["test"]),
                                 eval_excluded=no_outline, eval_excluded_reason="no Edge.Cuts outline: the engine cannot load the board (kept in the split, dropped from the eval lists)"),
                _stratum_stats=stats, _board_features={b: kept[b] for b in sorted(kept)})


def eval_boards(split: dict, half: str) -> list[str]:
    """The half minus ``_split_meta.eval_excluded[half]`` (boards the engine cannot load): what the eval lists carry."""
    ex = set(split["_split_meta"].get("eval_excluded", {}).get(half, []))
    return [b for b in split["medium"][half] if b not in ex]


def write_lists(split: dict, lists_dir: Path) -> None:
    ds = split["dataset_dirs"]["fit"]; fn = split["board_filename"]
    for half in ("fit", "test"):
        (lists_dir / f"boards_d3b_{half}.txt").write_text("".join(f"{ds}/{b}/{fn}\n" for b in eval_boards(split, half)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_JSON); ap.add_argument("--lists-dir", type=Path, default=LISTS_DIR)
    ap.add_argument("--root", type=Path, default=None, help="exacad_sorted dir (default: from d3.json dataset_dirs via PCBWORLD_DATA_ROOT)")
    a = ap.parse_args()
    d3 = json.load(open(D3_JSON))
    root = a.root or Path(os.path.expandvars(d3["dataset_dirs"]["train"]))
    if "$" in str(root) or not root.exists():
        sys.exit(f"dataset root not found: {root} (set PCBWORLD_DATA_ROOT)")
    split = build(d3, root, root / "pcb_characteristics_exacad_sorted.csv")
    a.out.write_text(json.dumps(split, indent=1, ensure_ascii=False) + "\n"); a.lists_dir.mkdir(parents=True, exist_ok=True); write_lists(split, a.lists_dir)
    m = split["_split_meta"]
    print(f"{a.out}: fit {m['n_fit']} / test {m['n_test']} (excluded non-2L: {m['excluded_non_2layer']}; eval lists drop {m['eval_excluded']}); strata {split['_stratum_stats']}")


if __name__ == "__main__":
    main()
