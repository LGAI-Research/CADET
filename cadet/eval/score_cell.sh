#!/usr/bin/env bash
# Score a prepared cell: post-hoc DRC + aggregation of PCBWorld's eval pipeline (no rollout).
#   bash cadet/eval/score_cell.sh <cell dir> [<cell dir> ...]
set -euo pipefail
CADET="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${PCBWORLD_ROOT:-$CADET/third_party/PCBWorld}"
PY="${PY:-python}"
export KICAD_ENGINE_REUSE=0 OMP_NUM_THREADS=1 PYTHONPATH="$CADET:$REPO"
cells=(); for c in "$@"; do cells+=("$(cd "$c" && pwd)"); done     # absolute before leaving the caller's directory
cd "$REPO"
for cell in "${cells[@]}"; do
  [ -f "$cell/SCORED" ] && { echo "== already scored: $cell"; continue; }
  "$PY" -m eval.pipeline --skip-rollout --rollout-mode parallel --output-dir "$cell" --n-envs "${N_ENVS:-64}" \
    --reward-config pcbworld_reward --check-angle 45 > "$cell/score.log" 2>&1 && date -u +%FT%TZ > "$cell/SCORED"
  echo "== scored $cell"
done
