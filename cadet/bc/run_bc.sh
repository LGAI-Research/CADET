#!/usr/bin/env bash
# Behavior cloning on RouteDemos: adapter -> manifest -> training (the paper's settings: 4 GPUs x batch 128 = 512
# state-action pairs per step, 350,000 steps, checkpoints every 35,000; step 280,000 initializes fine-tuning).
# Training uses all train demonstrations; the validation NLL uses the val split's demonstrations (128 in the paper).
#   bash cadet/bc/run_bc.sh            (DATA=<RouteDemos dir> OUT=<checkpoint dir> GPUS=4 WORKERS=16 STEPS=350000)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADET="$(cd "$HERE/../.." && pwd)"
REPO="${PCBWORLD_ROOT:-$CADET/third_party/PCBWorld}"
PY="${PY:-python}"
DATA="${DATA:-$CADET/var/datasets/routedemos}"
OUT="${OUT:-$CADET/var/checkpoints/bc}"
BCRUN="${BCRUN:-$CADET/var/results/bc_run}"
BASE="${BASE:-$CADET/cadet/configs/base_args/bc.json}"      # architecture / observation settings; no weights are read
export OMP_NUM_THREADS=1 KICAD_ENGINE_REUSE=0 PYTHONPATH="$CADET:$REPO"
mkdir -p "$OUT" "$BCRUN"

[ -f "$DATA/_dispatch/replay_all.done" ] || { echo "ABORT: $DATA/_dispatch/replay_all.done missing (generation phase 2 not finished)"; exit 2; }
# 1. BC run directory: linked demonstrations + generator rows + replay rows (no copies)
"$PY" "$HERE/demos_to_bc_run.py" --split-dir "$DATA/train" --replay-dir "$DATA/_replay" --out "$BCRUN" | tee "$BCRUN/adapter.log"
# 2. manifest: clean-pass demonstrations whose independent replay row says ok
[ -d "$DATA/val/_demo" ] || { echo "ABORT: validation split $DATA/val (boards + _demo/) missing"; exit 3; }
if [ ! -f "$BCRUN/all.manifest" ]; then
  "$PY" -m cadet.bc.index "$BCRUN" --out "$BCRUN/all.manifest" --require-replay-ok | tee "$BCRUN/manifest_build.log"
  # validation = the demonstrations of the val split, appended as the manifest TAIL: --val-demos holds out the last
  # entries, so training uses every train demonstration and the NLL is measured on the val split
  ls "$DATA/val/_demo"/board_*.json | sort >> "$BCRUN/all.manifest"
fi
N_VAL_DEMOS=$(ls "$DATA/val/_demo"/board_*.json | wc -l)
# 3. train
cd "$CADET"
exec "$PY" -m cadet.bc.train \
  --manifest "$BCRUN/all.manifest" --base-ckpt "$BASE" --save-dir "$OUT" \
  --gpus "${GPUS:-4}" --num-workers "${WORKERS:-16}" --buffer-size "${BUFFER:-32768}" --batch-size "${BATCH:-128}" \
  --prefetch-factor "${PREFETCH:-4}" --attn flex --compile-regions efficient --max-steps 1024 \
  --total-steps "${STEPS:-350000}" --lr 4e-4 --warmup-steps 2000 --min-lr-frac 0.05 \
  --weight-decay 1e-4 --max-grad-norm 1.0 \
  --val-demos "${VAL_DEMOS:-$N_VAL_DEMOS}" --eval-every 2000 --eval-batches 16 \
  --rollout-every "${ROLLOUT_EVERY:-35000}" --rollout-boards 16 --rollout-envs 16 --rollout-boards-dir "$DATA/val" \
  --save-every "${SAVE_EVERY:-35000}" --log-every 50
