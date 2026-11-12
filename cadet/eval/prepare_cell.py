"""Turn a rollout output into an evaluation cell that PCBWorld's eval pipeline can score.

A cell is a directory with ``per_rollout.csv`` and the routed boards; scoring (``score_cell.sh``) runs the
post-hoc DRC and aggregation of ``eval.pipeline`` on it. The source run is never modified.

    python cadet/eval/prepare_cell.py rl <run dir of rollout_queue.py> <cell dir>
    python cadet/eval/prepare_cell.py rule_based <seed0/manifest.json> [<seed1/manifest.json> ...] <boards list> <cell dir>
    python cadet/eval/prepare_cell.py llm <output root of cadet/llm/run_llm.sh> <boards list> <cell dir>
    python cadet/eval/prepare_cell.py reference <boards list> <cell dir>

Rule-based runs (PCBWorld's FreeRouting / KRT runner writes one manifest.json per seed): rows take the board index of the boards list and
rollout_idx = seed, so the same board ids and sample indices line up with the RL cells.
LLM runs: one row per finished board (result board, steps, the run's own wall time). Reference: the original
human-routed boards of the list, scored like every method's output.
"""
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cadet.paths import use_pcbworld  # noqa: E402

use_pcbworld()
from eval import eval_utils as u  # noqa: E402
from eval.pipeline import flatten_rollout_artifacts  # noqa: E402

STATUS = {"ok": ("completed", dict(terminated=True)), "timeout": ("truncated", dict(truncated=True)),
          "no_output": ("crashed", dict(crashed=True)), "error": ("crashed", dict(crashed=True))}
UNROUTED, PRO = "processed_v9_guide_v3_unrouted.kicad_pcb", "processed_v9_guide_v3.kicad_pro"


def prep_rl(src: Path, dst: Path) -> None:
    (dst / "artifacts").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(src / "per_rollout.csv")))
    for r in rows:
        ap = r.get("artifact_path") or ""
        if ap:
            s = Path(ap)
            d = dst / "artifacts" / s.name
            shutil.copy2(s, d)
            shutil.copy2(s.with_suffix(".kicad_pro"), d.with_suffix(".kicad_pro"))
            r["artifact_path"] = str(d)
    with open(dst / "per_rollout.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def prep_rule_based(manifests: list[Path], boards_list: Path, dst: Path) -> None:
    (dst / "artifacts").mkdir(parents=True, exist_ok=True)
    paths = [Path(os.path.expandvars(l.strip())) for l in open(boards_list) if l.strip()]
    idx = {p.parent.name: (i, p.parent) for i, p in enumerate(paths)}
    rows = []
    results = []
    for manifest in manifests:
        m = json.loads(manifest.read_text())
        results += [dict(res, seed=res.get("seed", m.get("seed", 0))) for res in m["results"]]
    for res in results:
        b = res["board_id"]
        (i, bdir), r = idx[b], int(res["seed"])
        status, flags = STATUS.get(res["status"], ("crashed", dict(crashed=True)))
        ap = ""
        if res["status"] == "ok" and res.get("routed_pcb"):
            d = dst / "artifacts" / f"board_{i:05d}_rollout_{r}.kicad_pcb"
            shutil.copy2(res["routed_pcb"], d)
            shutil.copy2(bdir / PRO, d.with_suffix(".kicad_pro"))
            ap = str(d)
        row = {k: "" for k in u.PER_ROLLOUT_FIELDS}
        row.update({"board_index": i, "board_id": b, "board_path": str(bdir / UNROUTED), "rollout_idx": r, "seed": r,
                    "deterministic": False, "status": status,
                    "skip_reason": "" if res["status"] == "ok" else res["status"],
                    "terminated": flags.get("terminated", False), "truncated": flags.get("truncated", False),
                    "crashed": flags.get("crashed", False), "engine_crash": False,
                    "per_board_rollout_time": res["elapsed_routing_s"], "artifact_path": ap,
                    "error": res.get("error") or ""})
        rows.append(row)
    rows.sort(key=lambda x: (x["board_index"], x["rollout_idx"]))
    u.write_csv(dst / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)


def _board_dirs(boards_list: Path) -> list[Path]:
    return [Path(os.path.expandvars(l.strip())).parent for l in open(boards_list) if l.strip()]


def _row(i: int, bdir: Path, **kw) -> dict:
    row = {k: "" for k in u.PER_ROLLOUT_FIELDS}
    row.update({"board_index": i, "board_id": bdir.name, "board_path": str(bdir / UNROUTED), "rollout_idx": 0,
                "status": "completed", "crashed": False, "engine_crash": False, "error": ""})
    row.update(kw)
    return row


def prep_llm(src: Path, boards_list: Path, dst: Path) -> None:
    (dst / "artifacts").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, bdir in enumerate(_board_dirs(boards_list)):
        b4 = bdir.name[:4]
        cell = src / f"{b4}_v7_h4"
        if not (cell / "token_usage.json").is_file():
            continue                                   # not run (or not finished): table.py counts the board as missing
        r = next(csv.DictReader(open(cell / "per_rollout.csv")))
        early = str(r.get("early_stop", "")).lower() in ("true", "1", "1.0")
        done = str(r.get("terminated", "")).lower() in ("true", "1", "1.0") and not early
        s = cell / f"{b4}_episode_00_env_00.kicad_pcb"
        d = dst / "artifacts" / f"board_{i:05d}_rollout_0.kicad_pcb"
        shutil.copy2(s, d)
        shutil.copy2(s.with_suffix(".kicad_pro"), d.with_suffix(".kicad_pro"))
        rows.append(_row(i, bdir, seed=42, deterministic=False, early_stop=early,
                         early_stop_reason="no_geometry_progress" if early else "",
                         terminated=done, truncated=not done and not early, steps=r.get("steps", ""),
                         per_board_rollout_time=json.load(open(cell / "token_usage.json"))["wall_time_s"],
                         artifact_path=str(d)))
    u.write_csv(dst / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)


def prep_reference(boards_list: Path, dst: Path) -> None:
    (dst / "artifacts").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, bdir in enumerate(_board_dirs(boards_list)):
        src = bdir / PRO.replace(".kicad_pro", ".kicad_pcb")
        d = dst / "artifacts" / f"board_{i:05d}_rollout_0.kicad_pcb"
        shutil.copy2(src, d)
        shutil.copy2(bdir / PRO, d.with_suffix(".kicad_pro"))
        rows.append(_row(i, bdir, seed=0, deterministic=True, terminated=True, truncated=False, artifact_path=str(d)))
    u.write_csv(dst / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)


def main() -> None:
    kind, *rest = sys.argv[1:]
    dst = Path(rest[-1]).resolve()        # PCBWorld's flatten step resolves artifact paths against the cell directory
    if (dst / "PREPARED").exists():
        print(f"{dst}: already prepared")
        return
    if kind == "rl":
        prep_rl(Path(rest[0]).resolve(), dst)
    elif kind == "rule_based":
        prep_rule_based([Path(x) for x in rest[:-2]], Path(rest[-2]), dst)
    elif kind == "llm":
        prep_llm(Path(rest[0]).resolve(), Path(rest[1]), dst)
    elif kind == "reference":
        prep_reference(Path(rest[0]), dst)
    else:
        raise SystemExit(__doc__)
    flatten_rollout_artifacts(dst, ckpt_seed=0)
    (dst / "PREPARED").write_text(json.dumps({"src": str(rest[0]), "t": time.strftime("%FT%TZ", time.gmtime())}))
    print(f"{dst}: boards/={len(list((dst / 'boards').glob('*.kicad_pcb')))}")


if __name__ == "__main__":
    main()
