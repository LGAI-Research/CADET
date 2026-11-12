#!/usr/bin/env bash
# PPO cells of the paper: CADET fine-tuning and the from-scratch references.
#
#   bash cadet/rl/launch.sh <cell>     trainer (one GPU)
#   bash cadet/rl/launch.sh w<cell>    asynchronous validation watcher of <cell> (a second GPU)
#   DRY_RUN=1   bash ... <cell>        print the trainer command only
#   PREFLIGHT=1 bash ... <cell>        write <out>/checkpoints/config_resolved.yaml and exit (no engine)
#
# Cells:
#   ours        BC checkpoint -> PPO fine-tune   RouteDemos                200 iter
#   solvable    PPO from scratch                 RouteDemos                500 iter
#   unsolvable  PPO from scratch                 RouteDemos w/o TryRoute   500 iter
#   d2a_big     PPO from scratch                 D2                        500 iter  reduced observation
#   real_fit    PPO from scratch                 D3-B ref (real boards)    500 iter
# Checkpoint selection: fp_mean_of_means on val_d3b_fit (D3-B ref); D3-B test never selects.
# Env knobs: GPU=0  RESUME=<ckpt>  OUT_ROOT  BC_CKPT  ROUTEDEMOS  ROUTEDEMOS_NOGATE  PY  WANDB=1 (+ WANDB_PROJECT, WANDB_ENTITY)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADET="$(cd "$HERE/../.." && pwd)"
REPO="${PCBWORLD_ROOT:-$CADET/third_party/PCBWorld}"
[[ -d "$REPO/pcb_world" ]] || { echo "PCBWorld checkout not found at $REPO: run setup.sh or set PCBWORLD_ROOT" >&2; exit 2; }
PY="${PY:-python}"
: "${PCBWORLD_DATA_ROOT:?PCBWORLD_DATA_ROOT is unset (see README: Setup)}"
ARG="${1:?usage: launch.sh <cell> | w<cell>}"
CELL="${ARG#w}"
OUT_ROOT="${OUT_ROOT:-$CADET/var/checkpoints/rl}"
OUT_DIR="$OUT_ROOT/$CELL"
DRY_RUN="${DRY_RUN:-0}"; PREFLIGHT="${PREFLIGHT:-0}"
BC_CKPT="${BC_CKPT:-$CADET/checkpoints/cadet_bc.pt}"
ROUTEDEMOS="${ROUTEDEMOS:-$CADET/var/datasets/routedemos}"                       # cadet/generation/run_generation.sh GATE=1
ROUTEDEMOS_NOGATE="${ROUTEDEMOS_NOGATE:-$CADET/var/datasets/routedemos_nogate}"  # GATE=0

export KICAD_ENGINE_REUSE=0 OMP_NUM_THREADS=1 PYTHONPATH="$CADET:$REPO"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$ARG" == w* ]]; then
  mkdir -p "$OUT_DIR/checkpoints"
  CUDA_VISIBLE_DEVICES="${GPU:-0}" exec "$PY" -u "$HERE/val_watcher.py" --save-dir "$OUT_DIR/checkpoints" --n-envs 32 --device cuda --skip-sets val_d3bf
fi

# ------------------------------------------------------------------ cells
BIG=(--d-model 256 --n-heads 8 --n-layers 6 --d-ff 1024)
FULL_OBS=(--obstacle-obs --shape-obs --net-constraint-obs --outline-obs arc)
# D2 has no NPTH holes or no-net pads, no arc outlines, one netclass and only circle/roundrect
# pads, so a D2-trained policy never learns those tokens: its cell does not emit them.
REDUCED_OBS=(--no-obstacle-obs --no-shape-obs --no-net-constraint-obs --outline-obs poly16 --legacy-edge-encoding)
LISTS="$CADET/cadet/configs/splits"
FT=()
case "$CELL" in
  ours)       DATA=routedemos;        ITERS=200; OBS=("${FULL_OBS[@]}")
              FT=(--init-ckpt "$BC_CKPT" --critic-warmup-iters 10 --critic-warmup-lr 1e-4 --detach-critic
                  --critic-warmup-min-ev 0.9 --critic-warmup-max-iters 50
                  --kl-ref-coef 0.05 --kl-ref-mask-unseen-drc) ;;
  solvable)   DATA=routedemos;        ITERS=500; OBS=("${FULL_OBS[@]}") ;;
  unsolvable) DATA=routedemos_nogate; ITERS=500; OBS=("${FULL_OBS[@]}") ;;
  d2a_big)    DATA=d2a;               ITERS=500; OBS=("${REDUCED_OBS[@]}") ;;
  real_fit)   DATA=d3b_fit_real;      ITERS=500; OBS=("${FULL_OBS[@]}") ;;
  *) echo "unknown cell '$CELL' (ours|solvable|unsolvable|d2a_big|real_fit; w<cell> = watcher)" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------ boards
case "$DATA" in
  d2a) SPLIT="$REPO/experiments/kdd/configs/datasets/d2a.json" ;;
  d3b_fit_real)                                   # train = val = D3-B ref; the environment strips the routing at reset
    SPLIT="$OUT_ROOT/splits/$DATA.json"
    if [[ ! -f "$SPLIT" && "$DRY_RUN" != 1 ]]; then
      mkdir -p "$(dirname "$SPLIT")"
      "$PY" - "$LISTS/boards_d3b_fit.txt" "$SPLIT" <<'EOF'
import json, os, sys
fit = [l.strip().split("/exacad_sorted/")[1].split("/")[0] for l in open(sys.argv[1]) if l.strip()]
out = sys.argv[2]
root = "${PCBWORLD_DATA_ROOT}/pcbench/exacad_sorted"
data = {"board_filename": "processed_v9_guide_v3.kicad_pcb",
        "dataset_dirs": {"train": root, "val": root},
        "easy": {"train": fit, "val": fit}}
tmp = out + ".tmp"
open(tmp, "w").write(json.dumps(data, indent=1))
os.replace(tmp, out)
print(f"[split] train {len(fit)} / val {len(fit)} (both D3-B ref) -> {out}")
EOF
    fi ;;
  *) SPLIT="$OUT_ROOT/splits/$DATA.json"          # built once from the boards on disk
    ROOT="$ROUTEDEMOS"; [[ "$DATA" == routedemos_nogate ]] && ROOT="$ROUTEDEMOS_NOGATE"
    if [[ ! -f "$SPLIT" && "$DRY_RUN" != 1 ]]; then
      "$PY" "$HERE/make_split_json.py" --root "$ROOT" --out "$SPLIT"
    fi ;;
esac
EVAL=(
  --eval-split val --eval-board-limit 32 --eval-at-init --async-val
  --eval2-boards "$LISTS/boards_d3b_fit.txt" --eval2-prefix val_d3b_fit      # checkpoint selection set
  --eval3-boards "$LISTS/boards_d3a.txt" --eval3-prefix val_d3a
  --eval4-boards "$LISTS/boards_d3b_test.txt" --eval4-prefix val_d3b_test
  --eval5-boards "$LISTS/boards_d3bf.txt" --eval5-prefix val_d3bf            # an extra D3-B list kept for argument parity; the watcher skips it
  --eval-n-rollouts 5 --eval-every 10 --eval-diag-max-steps 1024 --eval-diag-masking-rule pcbworld_masking
)
FEATURES=(
  --directional-candidates multi_resolution --offboard-mask --connectivity-filter --simplify-outline
  --time-feature sin_remaining --time-feature-cap 10000 --drc-tokens --n-max-slots 512
  --policy-net-select --same-net-bias --disable-slot-emb --coord-encoding fourier --corner-mode 45
)
EXTRA=()
[[ -n "${RESUME:-}" ]] && EXTRA+=(--resume "$RESUME")
if [[ "${WANDB:-0}" == 1 && "$PREFLIGHT" != 1 ]]; then
  EXTRA+=(--wandb --wandb-project "${WANDB_PROJECT:?set WANDB_PROJECT}" --wandb-run-name "cadet-$CELL" --wandb-tags "$CELL,$DATA")
  [[ -n "${WANDB_ENTITY:-}" ]] && EXTRA+=(--wandb-entity "$WANDB_ENTITY")
fi

cmd=(
  "$PY" -u "$REPO/scripts/train.py" ppo
  --board "$REPO/tests/fixtures/simple_routing_board.kicad_pcb"
  --boards-order per_env_epoch --boards-json "$SPLIT" --boards-difficulty easy --boards-split train
  "${EVAL[@]}"
  --iterations "$ITERS" --max-steps 1024 --n-envs 64 --n-steps 1024
  --n-epochs 4 --batch-size 128 --lr 2e-5 --clip-eps 0.1 --entropy-coef 0.0
  --max-grad-norm 0.5 --warmup-iters 20 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5
  --no-truncation-bootstrap --reward-rule pcbworld_reward --masking-rule pcbworld_masking
  "${BIG[@]}" "${FEATURES[@]}" "${OBS[@]}" ${FT[@]+"${FT[@]}"}
  --device cuda --seed 42 --update-gpus 1 --mem-budget --bf16 --expect-env-diff seed,train_extras
  --log-dir "$OUT_DIR/tb" --save-dir "$OUT_DIR/checkpoints" --save-freq 10
  ${EXTRA[@]+"${EXTRA[@]}"}
)

echo "[cadet/rl] cell=$CELL data=$DATA iters=$ITERS out=$OUT_DIR resume=${RESUME:-none}"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%s ' "${GPU:-0}"; printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
mkdir -p "$OUT_DIR/tb" "$OUT_DIR/checkpoints"
cd "$REPO"
if [[ "$PREFLIGHT" == 1 ]]; then
  exec "${cmd[@]}" --dump-config-only
fi
CUDA_VISIBLE_DEVICES="${GPU:-0}" exec "${cmd[@]}"
