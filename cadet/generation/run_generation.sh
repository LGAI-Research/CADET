#!/usr/bin/env bash
# RouteDemos generation on one machine:
#   0 calibrate the pad density -> 1 generate val / test / train -> 1b keep the first N accepted boards (seed order)
#   -> 2 independent replay of the kept train demonstrations (TryRoute runs only) -> 3 report.
#
#   GATE=1 OUT=var/datasets/routedemos        bash cadet/generation/run_generation.sh   # with TryRoute (demonstrations)
#   GATE=0 OUT=var/datasets/routedemos_nogate bash cadet/generation/run_generation.sh   # same front end, no TryRoute
#
# Defaults are the paper's settings (100,000 train boards; calibration ended at density scale 1.6505 after four
# 3,000-board rounds; per-board time cap 1500 s). SCALE=1.6505 skips phase 0. A small run:
#   OUT=var/datasets/routedemos_small N_TRAIN=200 N_VAL=16 SCALE=1.6505 BOARD_TIME_CAP=300 bash cadet/generation/run_generation.sh
# Every phase is idempotent (done markers under $OUT/_dispatch); the generator draws work from a chunk queue, so
# rerunning the script resumes it.
set -u -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADET="$(cd "$HERE/../.." && pwd)"
GATE=${GATE:-1}; OUT=${OUT:-$CADET/var/datasets/routedemos}; OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
W=${W:-$(nproc)}; PY=${PY:-python}
N_TRAIN=${N_TRAIN:-100000}; N_VAL=${N_VAL:-128}; OVER=${OVER:-1.06}; CHUNK=${CHUNK:-200}
FIT=${FIT:-$CADET/cadet/configs/fit_params/d3b_fit_half.json}; PRESET=${PRESET:-multi_resolution}
CALIB_N=${CALIB_N:-3000}; CALIB_TOL=${CALIB_TOL:-0.02}; CALIB_MAX=${CALIB_MAX:-4}; CALIB_CHUNK=${CALIB_CHUNK:-50}
BASE_ARGS="--preset $PRESET --gen-modes walk --verify-modes walk,shove,mark --budget 60 --verify-budget 200 --max-verify-iters 16 --board-time-cap ${BOARD_TIME_CAP:-1500} --fit-params $FIT --workers $W"
[ "$GATE" = 0 ] && BASE_ARGS="$BASE_ARGS --no-gate"
export OMP_NUM_THREADS=1 KICAD_ENGINE_REUSE=0 PYTHONPATH="$CADET:${PCBWORLD_ROOT:-$CADET/third_party/PCBWorld}${PYTHONPATH:+:$PYTHONPATH}"
D=$OUT/_dispatch; mkdir -p "$D" "$OUT/train" "$OUT/val" "$OUT/test" "$OUT/_calib"
LOG=$OUT/_generation.log
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }
G=$HERE/generate.py; T_START=$(date +%s)
log "generation gate=$GATE fit=$FIT workers=$W chunk=$CHUNK out=$OUT"

# ---------------- phase 0: iterative density calibration (fresh seeds per iteration, p50 ratio)
if [ -n "${SCALE:-}" ] && [ ! -f "$OUT/_calib/density_scale.json" ]; then
  $PY -c "import json;json.dump(dict(density_scale=$SCALE,given=True),open('$OUT/_calib/density_scale.json','w'),indent=1)"
fi
if [ ! -f "$OUT/_calib/density_scale.json" ]; then
  SCALE=${SCALE0:-1.0}; iters="[]"; final=""
  for ((it=0; it<CALIB_MAX; it++)); do
    dir=$OUT/_calib/iter$it; mkdir -p "$dir"
    if [ ! -f "$dir/.gen.done" ]; then
      log "phase 0 iter $it: scale $SCALE, $CALIB_N boards"
      $PY "$G" --out-dir "$dir" --n "$CALIB_N" --seed-base $((3000000000 + it * 1000000)) --density-scale "$SCALE" \
        --chunk-dir "$dir/_chunks" --chunk-size "$CALIB_CHUNK" $BASE_ARGS > "$dir/gen.log" 2>&1 || { log "phase 0 iter $it FAILED (see $dir/gen.log)"; exit 1; }
      touch "$dir/.gen.done"
    fi
    st=$($PY "$HERE/calib_stats.py" "$FIT" "$dir" --json); ratio=$($PY -c "import json;print(json.loads('$st')['ratio'])")
    log "phase 0 iter $it: $st"
    iters=$($PY -c "import json;l=json.loads('$iters');l.append(dict(iter=$it,scale=$SCALE,**json.loads('$st')));print(json.dumps(l))")
    if $PY -c "import sys;sys.exit(0 if abs($ratio-1)<=$CALIB_TOL else 1)"; then final=$SCALE; break; fi
    SCALE=$($PY -c "print(round($SCALE*$ratio,4))")
  done
  [ -z "$final" ] && final=$SCALE
  $PY -c "import json;json.dump(dict(density_scale=$final,tol=$CALIB_TOL,n_per_iter=$CALIB_N,estimator='p50',iters=json.loads('$iters')),open('$OUT/_calib/density_scale.json','w'),indent=1)"
  log "phase 0 done: density_scale=$final"
fi
SCALE=$($PY -c "import json;print(json.load(open('$OUT/_calib/density_scale.json'))['density_scale'])")
GENARGS="$BASE_ARGS --density-scale $SCALE"; echo "$GENARGS" > "$D/genargs.txt"; log "density_scale=$SCALE"

# ---------------- phase 1: generation (val, test, then train chunk queues)
N_TRAIN_GEN=$($PY -c "import math;print(math.ceil($N_TRAIN*$OVER))"); N_VAL_GEN=$($PY -c "import math;print(math.ceil($N_VAL*$OVER)+4)")
if [ ! -f "$D/gen_all.done" ]; then
  log "phase 1 generation: train $N_TRAIN_GEN (keep $N_TRAIN) chunks of $CHUNK, val/test $N_VAL_GEN (keep $N_VAL)"
  $PY "$G" --out-dir "$OUT/val"   --n "$N_VAL_GEN"   --seed-base 1000000000 --chunk-dir "$D/chunks_val"   --chunk-size 20 $GENARGS > "$D/val.log" 2>&1 \
   && $PY "$G" --out-dir "$OUT/test"  --n "$N_VAL_GEN"   --seed-base 2000000000 --chunk-dir "$D/chunks_test"  --chunk-size 20 $GENARGS > "$D/test.log" 2>&1 \
   && $PY "$G" --out-dir "$OUT/train" --n "$N_TRAIN_GEN" --seed-base 0          --chunk-dir "$D/chunks_train" --chunk-size "$CHUNK" $GENARGS > "$D/gen.log" 2>&1 \
   || { log "phase 1 FAILED (see $D/*.log)"; exit 1; }
  touch "$D/gen_all.done"; log "phase 1 done ($(( ($(date +%s) - T_START) / 60 )) min)"
fi

# ---------------- phase 1b: exact counts (first N accepted in seed order; the rest -> _extra/)
if [ ! -f "$D/select.done" ]; then
  for sp in train val test; do n=$N_TRAIN; [ $sp != train ] && n=$N_VAL
    $PY "$HERE/select_first_n.py" "$OUT/$sp" --n $n 2>&1 | tee -a "$LOG" || { log "select FAILED for $sp"; exit 1; }
  done
  touch "$D/select.done"
fi

# ---------------- phase 2: independent replay check of the kept train demonstrations (fresh engine per worker)
if [ "$GATE" = 1 ] && [ ! -f "$D/replay_all.done" ]; then
  T2=$(date +%s); mkdir -p "$OUT/_replay"
  ls -U "$OUT/train/_demo" | grep '^board_.*\.json$' | sort | sed "s#^#$OUT/train/_demo/#" > "$D/_demos.txt"; N_DEMOS=$(wc -l < "$D/_demos.txt")
  per=$(( (N_DEMOS + W - 1) / W )); log "phase 2 replay: $N_DEMOS demos, $W workers, $per demos/worker"
  for ((w=0; w<W; w++)); do
    s0=$(( w * per )); (( s0 >= N_DEMOS )) && break
    tag=shard_$w; [ -f "$OUT/_replay/$tag.done" ] && continue
    ( $PY "$CADET/cadet/search/replay_shard.py" --list "$D/_demos.txt" --start $s0 --n $per --out "$OUT/_replay/$tag.jsonl" > "$OUT/_replay/$tag.log" 2>&1; touch "$OUT/_replay/$tag.done" ) &
  done
  wait
  touch "$D/replay_all.done"; log "phase 2 done ($(( ($(date +%s) - T2) / 60 )) min)"
fi

# ---------------- phase 3: report
{
  echo "# RouteDemos generation (gate=$GATE, fit=$(basename "$FIT"), density_scale=$SCALE) ($(date -u '+%F %T UTC'))"
  echo; echo "workers $W, chunk $CHUNK. args: \`$GENARGS\`. train/val/test hold exact counts; _extra/ = surplus accepted boards; a dataset file is the pristine unrouted board."
  echo; echo "## Calibration"; echo '```'; cat "$OUT/_calib/density_scale.json"; echo '```'
  echo; echo "## Selection"; echo '```'; for sp in train val test; do echo "$sp: $(cat "$OUT/$sp/_selection.json" 2>/dev/null | tr -d '\n' | cut -c1-200)"; done; echo '```'
  [ "$GATE" = 1 ] && { echo; echo "## Train set report"; echo '```'; $PY "$HERE/pilot_report.py" "$OUT/train" | head -12; echo '```'; }
  echo; echo "## Difficulty: D3-B ref / D3-B test / generated train"; echo '```'; $PY "$HERE/difficulty_report.py" "$FIT" "$CADET/cadet/configs/fit_params/d3b_test_half.json" "$OUT/train"; echo '```'
  echo; echo "## Realized density vs target (train)"; echo '```'; $PY "$HERE/calib_stats.py" "$FIT" "$OUT/train"; echo '```'
  if [ "$GATE" = 1 ]; then echo; echo "## Independent replay check"; echo '```'; $PY - "$OUT/_replay" <<'PY'
import json, glob, sys
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/shard_*.jsonl") for l in open(f) if l.strip()]
ok = sum(r.get("ok", False) for r in rows); cp = sum(bool(r.get("clean_pass")) for r in rows)
print(f"demos verified={len(rows)} fully replayable={ok} ({ok/max(1,len(rows)):.1%}) clean_pass={cp} errors={sum('error' in r for r in rows)}")
PY
  echo '```'; fi
  echo; echo "- total wall $(( ($(date +%s) - T_START) / 60 )) min"
} > "$OUT/RESULTS.md"
log "DONE -> $OUT/RESULTS.md"
