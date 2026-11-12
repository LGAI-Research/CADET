#!/usr/bin/env bash
# LLM baseline rollouts (one process per board, one rollout per board) with the settings of the paper's LLM rows:
# v7 prompt, history 4, s-expression state, 512 steps, early stop after 10 steps without geometry progress.
#   bash cadet/llm/apply_patch.sh                                   # once (see that script)
#   OPENAI_API_KEY=... API_MODEL=<model id> LIST=cadet/configs/splits/boards_d3b_test.txt OUT=var/runs/llm_d3b bash cadet/llm/run_llm.sh
#   ... BOARDS="0101 0103" bash cadet/llm/run_llm.sh                # listed 4-digit board ids only
# Output per board: OUT/<id>_v7_h4/{<id>_episode_00_env_00.{json,kicad_pcb,kicad_pro}, per_*.csv, token_usage.json, rollout.log}.
# Resumable: a board with token_usage.json + per_board.csv is skipped. This calls a paid API.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADET="$(cd "$HERE/../.." && pwd)"
REPO="${PCBWORLD_ROOT:-$CADET/third_party/PCBWorld}"
: "${OUT:?output root}" "${API_MODEL:?model id}" "${LIST:?boards list (cadet/configs/splits/boards_*.txt)}" "${OPENAI_API_KEY:?API key}"
: "${PCBWORLD_DATA_ROOT:?PCBWORLD_DATA_ROOT is unset (see README: Setup)}"
PY="${PY:-python}"
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"; LIST="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"
export PYTHONPATH="$CADET:$REPO" KICAD_ENGINE_REUSE=0 OMP_NUM_THREADS=1
cd "$REPO"

# Board files named <id>.kicad_pcb/.kicad_pro, so the board id is the 4-digit prefix of the board directory.
BOARD_DIR="$OUT/_boards"; mkdir -p "$BOARD_DIR"
"$PY" - "$LIST" "$BOARD_DIR" <<'PYEOF'
import os, sys
for line in open(sys.argv[1]):
    src = os.path.expandvars(line.strip())
    if not src:
        continue
    bid = os.path.basename(os.path.dirname(src))[:4]
    for ext in (".kicad_pcb", ".kicad_pro"):
        s, d = os.path.splitext(src)[0] + ext, os.path.join(sys.argv[2], bid + ext)
        if not os.path.lexists(d):
            os.symlink(s, d)
PYEOF
BOARDS=${BOARDS:-$(ls "$BOARD_DIR" | sed -n 's/\.kicad_pcb$//p' | sort)}

COMMON=(
  --mode api --api_provider openai --api_model "$API_MODEL"
  --api_reasoning_effort medium --max_new_tokens 4000
  --prompt_version v7 --state_format sexpr --outline_obs tess
  --masking_rule default_no_finish
  --reward_rule "$HERE/reward_wlnorm.yaml"
  --via_penalty 0.1 --max_steps 512 --early_stop_no_geometry_progress 10
  --rollout_episodes 1 --env_num 1 --seed 42 --history_length 4 --boards_order single
)

run() {      # Ray is started without its dashboard; the rollout reuses the initialised Ray
  "$PY" -u -c 'import sys, ray
ray.init(include_dashboard=False)
from methods.llm_agent.rollout import pcbworld as R
sys.argv = ["pcbworld"] + sys.argv[1:]
R.main()' "$@"
}

for b in $BOARDS; do
  cell="$OUT/${b}_v7_h4"
  if [ -f "$cell/token_usage.json" ] && [ -f "$cell/per_board.csv" ]; then
    echo "== $b done: skip"; continue
  fi
  [ -f "$BOARD_DIR/$b.kicad_pcb" ] || { echo "missing board $BOARD_DIR/$b.kicad_pcb" >&2; exit 1; }
  mkdir -p "$cell"
  echo "== $b start $(date -u +%T)"
  run "${COMMON[@]}" --board_path "$BOARD_DIR/$b.kicad_pcb" --dump_dir "$cell" > "$cell/rollout.log" 2>&1
  echo "== $b end $(date -u +%T)"
done
echo "== RUN_DONE model=$API_MODEL"
