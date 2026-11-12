#!/usr/bin/env python3
"""Expose a generated dataset split as the demonstration-run directory the BC index reads.

``cadet.bc.index.select_demos`` expects

    <run>/demos/<stem>.json          demonstration files
    <run>/shard_*.jsonl              executor rows: {board, clean_pass, success, ...}
    <run>/replay/shard_*.jsonl       replay rows:   {board, ok, ...}

while the generator (cadet/generation/run_generation.sh) writes

    <split>/_demo/board_XXXXX.json           demonstration files
    <split>/_gen_rows_<start>_<n>.jsonl      generator rows keyed by ``idx`` + ``status`` ('ok' | 'verify_fail'),
                                             with ``verify.clean_pass`` and no ``board`` key
    <dataset>/_replay/shard_*.jsonl          independent replay rows (replay_shard.py)

This script builds ``<out>`` from those without copying demonstrations: ``demos`` is a symlink, ``shard_0.jsonl`` holds
executor rows synthesised from the generator rows (clean_pass = status == 'ok' and verify.clean_pass), and
``replay/`` links every replay shard file. Standard library only.

    python cadet/bc/demos_to_bc_run.py --split-dir <dataset>/train --replay-dir <dataset>/_replay --out <run>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def read_gen_rows(split: Path) -> dict[int, dict]:
    """idx -> last generator row (shards are appended on resume; last wins)."""
    rows: dict[int, dict] = {}
    for path in sorted(split.glob("_gen_rows_*.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue            # partial last line of a killed shard
                if "idx" in row:
                    rows[int(row["idx"])] = row
    return rows


def exec_row(idx: int, row: dict) -> dict:
    ok = row.get("status") == "ok"
    verify = row.get("verify") or {}
    return {
        "board": f"board_{idx:05d}.kicad_pcb",
        "clean_pass": bool(ok and verify.get("clean_pass")),
        "success": bool(ok),
        "gen_status": row.get("status"),
        "actions": row.get("actions"),
    }


def link(target: Path, dest: Path) -> None:
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    os.symlink(str(target.resolve()), str(dest))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split-dir", required=True, help="<dataset>/train (holds _demo/ and _gen_rows_*.jsonl)")
    ap.add_argument("--replay-dir", action="append", default=[],
                    help="dir holding replay shard_*.jsonl; repeatable (rows from later dirs win on a repeated board)")
    ap.add_argument("--out", required=True, help="BC run directory to create")
    a = ap.parse_args()

    split = Path(a.split_dir).resolve()
    demo_dir = split / "_demo"
    if not demo_dir.is_dir():
        raise SystemExit(f"no _demo/ under {split}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay").mkdir(exist_ok=True)

    link(demo_dir, out / "demos")

    gen_rows = read_gen_rows(split)
    rows = [exec_row(i, r) for i, r in sorted(gen_rows.items())]
    with open(out / "shard_0.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # Demo files present on disk but without a generator row are invisible to
    # index.py (it iterates executor rows) — report them so a gap is noticed.
    demo_stems = {p[:-5] for p in os.listdir(demo_dir) if p.startswith("board_") and p.endswith(".json")}
    row_stems = {r["board"][:-len(".kicad_pcb")] for r in rows}
    n_replay_files = 0
    for rd in a.replay_dir:
        rd = Path(rd).resolve()
        tag = rd.name.strip("_")
        # "*shard_*": also accepts prefixed shard files
        for f in sorted(glob.glob(str(rd / "*shard_*.jsonl"))):
            link(Path(f), out / "replay" / f"shard_{tag}_{Path(f).name}")
            n_replay_files += 1

    meta = {
        "split_dir": str(split),
        "replay_dirs": [str(Path(d).resolve()) for d in a.replay_dir],
        "counts": {
            "gen_rows": len(rows),
            "gen_ok": sum(r["success"] for r in rows),
            "gen_clean_pass": sum(r["clean_pass"] for r in rows),
            "demo_files": len(demo_stems),
            "demo_without_row": len(demo_stems - row_stems),
            "ok_row_without_demo": sum(
                1 for r in rows if r["success"] and r["board"][:-len(".kicad_pcb")] not in demo_stems),
            "replay_shard_files": n_replay_files,
        },
    }
    with open(out / "adapter.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"run dir ready: {out}\n  next: python -m cadet.bc.index {out} --out {out}/all.manifest --require-replay-ok")


if __name__ == "__main__":
    main()
