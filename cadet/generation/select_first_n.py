"""Keep exactly N accepted boards of a generated split: the first N ok boards in seed-index order stay, every other
board (later ok ones and all failures' leftovers) moves to <dir>/_extra/. Selection by index = no cherry-picking.

    python cadet/generation/select_first_n.py <dir> --n 128 [--dry]
"""
import argparse, json, re, shutil
from pathlib import Path

SIDE = [("_demo", "board_{i}.json"), ("_routed", "board_{i}.kicad_pcb"), ("_routed", "board_{i}.kicad_pro"), ("_witness", "board_{i}.witness.json.gz")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir", type=Path); ap.add_argument("--n", type=int, required=True); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(); d = a.dir
    # one readdir per directory (100k+ entries) instead of a per-file exists()
    import os
    have = {m.group(1) for n in os.listdir(d / "_witness") for m in [re.match(r"board_(\d+)\.witness\.json\.gz$", n)] if m}
    ok = sorted(int(m.group(1)) for n in os.listdir(d) for m in [re.match(r"board_(\d+)\.kicad_pcb$", n)] if m and m.group(1) in have)
    if len(ok) < a.n:
        raise SystemExit(f"{d}: only {len(ok)} accepted boards, need {a.n}")
    keep, extra = ok[:a.n], ok[a.n:]
    # boards whose final verification failed leave a board file without witness/demo (finalize wrote it): never part of the
    # dataset — sweep them into _extra/_unaccepted/ so consumers that glob board_*.kicad_pcb do not pick them up
    stray = sorted(int(m.group(1)) for n in os.listdir(d) for m in [re.match(r"board_(\d+)\.kicad_pcb$", n)] if m and m.group(1) not in have)
    moved = 0
    for i in stray:
        for ext in (".kicad_pcb", ".kicad_pro"):
            src = d / f"board_{i:05d}{ext}"
            if src.exists():
                dst = d / "_extra" / "_unaccepted" / src.name; moved += 1
                if not a.dry:
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(src), str(dst))
    for i in extra:
        tag = f"{i:05d}"
        for rel in [Path(f"board_{tag}.kicad_pcb"), Path(f"board_{tag}.kicad_pro")] + [Path(sd) / fn.format(i=tag) for sd, fn in SIDE]:
            src = d / rel
            if src.exists():
                dst = d / "_extra" / rel; moved += 1
                if not a.dry:
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(src), str(dst))
    (d / "_selection.json").write_text(json.dumps(dict(n=a.n, kept_idx=[keep[0], keep[-1]], n_accepted=len(ok), moved_to_extra=extra, unaccepted=stray, files_moved=moved), indent=1)) if not a.dry else None
    print(f"{d}: accepted {len(ok)} -> kept first {a.n} (idx {keep[0]}..{keep[-1]}), {len(extra)} surplus + {len(stray)} unaccepted boards / {moved} files -> _extra/{' (dry)' if a.dry else ''}")


if __name__ == "__main__":
    main()
