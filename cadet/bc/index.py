"""Demonstration-run manifest: which demos to clone, and how they are split.

A demonstration run writes one JSON demo per board under ``<run>/demos/`` plus two
kinds of per-board stat rows:

* ``<run>/shard_*.jsonl`` — executor rows (``clean_pass`` / ``success`` /
  ``demo_actions``), written while the demos are produced;
* ``<run>/replay/shard_*.jsonl`` — replay-verification rows (``ok`` /
  ``prefix_frac`` / ``nets_kept``), written by the pass that follows.

The manifest is built from those rows, not from the demo files: reading 100k demo
JSONs to recover two booleans costs minutes, the rows cost seconds. A
board with no replay row is kept (the verification pass may still be running)
unless ``require_replay_ok`` is set.

CLI::

    python -m cadet.bc.index <run_dir> --out var/results/bc/<name>.manifest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["select_demos", "write_manifest", "read_manifest", "split_demos"]

#: Manifest header line: a JSON blob describing how the list was produced.
_HEADER = "# "


def _read_rows(paths: list[Path]) -> dict[str, dict]:
    """Board stem -> last row, over a set of jsonl shard files.

    Shards are appended to on resume, so a board can appear twice; the last row
    wins (it is the one that produced the demo currently on disk).
    """
    rows: dict[str, dict] = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a shard killed mid-write ends in a partial line
                board = row.get("board")
                if board:
                    rows[Path(str(board)).stem] = row
    return rows


def select_demos(
    run_dir: str | Path,
    *,
    require_clean_pass: bool = True,
    require_replay_ok: bool = False,
    limit: int | None = None,
) -> tuple[list[Path], dict]:
    """Demo files worth cloning, plus the counts behind the selection.

    ``require_clean_pass`` keeps only demos the executor finished with zero
    unrouted pins and zero DRC violations. ``require_replay_ok`` additionally
    demands a replay row that verified the demo end to end.
    """
    run = Path(run_dir)
    demo_dir = run / "demos"
    exec_rows = _read_rows(sorted(run.glob("shard_*.jsonl")))
    replay_rows = _read_rows(sorted((run / "replay").glob("shard_*.jsonl")))
    counts = {
        "exec_rows": len(exec_rows),
        "replay_rows": len(replay_rows),
        "no_demo_file": 0,
        "not_clean_pass": 0,
        "replay_failed": 0,
        "replay_missing": 0,
        "selected": 0,
    }
    demos: list[Path] = []
    for stem in sorted(exec_rows):
        row = exec_rows[stem]
        if require_clean_pass and not row.get("clean_pass"):
            counts["not_clean_pass"] += 1
            continue
        rep = replay_rows.get(stem)
        if rep is None:
            counts["replay_missing"] += 1
            if require_replay_ok:
                continue
        elif not rep.get("ok"):
            counts["replay_failed"] += 1
            continue
        path = demo_dir / f"{stem}.json"
        if not path.exists():
            counts["no_demo_file"] += 1
            continue
        demos.append(path)
        if limit is not None and len(demos) >= limit:
            break
    counts["selected"] = len(demos)
    return demos, counts


def write_manifest(path: str | Path, demos: list[Path], meta: dict) -> None:
    """One absolute demo path per line, behind a ``#``-commented JSON header."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(_HEADER + json.dumps(meta, sort_keys=True) + "\n")
        for demo in demos:
            fh.write(str(Path(demo).resolve()) + "\n")


def read_manifest(path: str | Path) -> list[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def split_demos(demos: list[str], n_val: int) -> tuple[list[str], list[str]]:
    """Hold out the LAST ``n_val`` entries (the list is sorted by board index).

    Taking a contiguous tail rather than a random subset keeps the split stable
    when the manifest is rebuilt after more demos land.
    """
    if n_val <= 0:
        return list(demos), []
    return list(demos[:-n_val]), list(demos[-n_val:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="run directory (holds demos/ and shard_*.jsonl)")
    ap.add_argument("--out", required=True, help="manifest path to write")
    ap.add_argument("--allow-unclean", action="store_true",
                    help="keep demos the executor did not finish clean")
    ap.add_argument("--require-replay-ok", action="store_true",
                    help="drop demos without a passing replay-verification row")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    demos, counts = select_demos(
        args.run_dir,
        require_clean_pass=not args.allow_unclean,
        require_replay_ok=args.require_replay_ok,
        limit=args.limit,
    )
    meta = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "require_clean_pass": not args.allow_unclean,
        "require_replay_ok": args.require_replay_ok,
        "counts": counts,
    }
    write_manifest(args.out, demos, meta)
    print(json.dumps(meta, indent=2))
    print(f"wrote {len(demos)} demo paths -> {args.out}")


if __name__ == "__main__":
    main()
