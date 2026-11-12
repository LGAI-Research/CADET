"""Replay-verify a slice of demo files and write one jsonl row per demo (no obs recording).

    python replay_shard.py --demo-dir <dir> --start S --n N --out rows.jsonl
Row: board, ok, prefix_frac, nets_kept, nets_total, steps_ok, steps_fail, clean_pass, recorded_clean_pass.
The BC loader uses these rows to pick demos (ok) or replayable prefixes (nets_kept) — see replay_demo.py.
"""
import argparse, glob, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from cadet.paths import use_pcbworld  # noqa: E402
use_pcbworld()
from replay_demo import replay, CKPT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--demo-dir"); ap.add_argument("--list", default=None, help="file with one demo path per line (avoids a per-worker scan of a 100k-entry dir)")
ap.add_argument("--start", type=int, default=0); ap.add_argument("--n", type=int, default=10**9)
ap.add_argument("--out", required=True); ap.add_argument("--ckpt", default=CKPT)
a = ap.parse_args()
all_files = [l.strip() for l in open(a.list) if l.strip()] if a.list else sorted(glob.glob(a.demo_dir + "/board_*.json"))
files = all_files[a.start:a.start + a.n]
with open(a.out, "a") as f:
    for p in files:
        t0 = time.time()
        try:
            demo = json.load(open(p))
            ok, st, _ = replay(demo, a.ckpt, record=False)
            row = dict(board=Path(demo["board"]).name, ok=ok, recorded_clean_pass=demo.get("clean_pass"), **st)
        except Exception as exc:  # noqa: BLE001
            row = dict(board=Path(p).stem + ".kicad_pcb", ok=False, error=repr(exc)[:200])
        row["wall_s"] = round(time.time() - t0, 1)
        f.write(json.dumps(row) + "\n"); f.flush()
