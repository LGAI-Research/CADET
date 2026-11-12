#!/usr/bin/env python3
"""Build the split JSON of a flat synthetic dataset (``<root>/{train,val,test}/board_*.kicad_pcb``).

Output shape = what ``methods/_shared/board_loader.resolve_board_list`` reads: top-level
``dataset_dirs[split]`` (the directory holding ``<board_id>.kicad_pcb``) and
``data[difficulty][split]`` (the board ids). Ids are the boards ON DISK in sorted order,
skipping any board without a ``.kicad_pro`` twin (the trainer refuses pro-less boards), so
rebuilding the file from the same dataset yields the same list and the same draw order.
``--root`` is resolved, so ``dataset_dirs`` holds the real directory.

Usage:
    python make_split_json.py --root <dataset_root> --out <split.json> [--difficulty easy]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="dataset root holding train/ val/ test/")
    p.add_argument("--out", required=True, help="split json to write")
    p.add_argument("--difficulty", default="easy",
                   help="difficulty key ('easy' for the synthetic sets)")
    p.add_argument("--splits", default="train,val,test")
    a = p.parse_args()

    root = Path(a.root).resolve()
    splits = [s for s in a.splits.split(",") if s]
    data: dict = {"dataset_dirs": {}, a.difficulty: {}}
    for split in splits:
        d = root / split
        if not d.is_dir():
            raise SystemExit(f"missing split dir: {d}")
        ids, skipped = [], 0
        for pcb in sorted(d.glob("board_*.kicad_pcb")):
            if not (d / (pcb.stem + ".kicad_pro")).is_file():
                skipped += 1
                continue
            ids.append(pcb.stem)
        if not ids:
            raise SystemExit(f"no board_*.kicad_pcb (+.kicad_pro) in {d}")
        data["dataset_dirs"][split] = str(d)
        data[a.difficulty][split] = ids
        print(f"[split] {split}: {len(ids)} boards ({skipped} without .kicad_pro skipped) <- {d}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, out)
    print(f"[split] wrote {out}")


if __name__ == "__main__":
    main()
