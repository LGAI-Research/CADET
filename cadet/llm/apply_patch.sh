#!/usr/bin/env bash
# Set up the LLM agent used by cadet/llm/run_llm.sh (run once before it).
#   bash cadet/llm/apply_patch.sh            apply          bash cadet/llm/apply_patch.sh -R     revert
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PCBWORLD_ROOT:-$HERE/../../third_party/PCBWorld}"
git -C "$REPO" apply --check "$@" "$HERE/pcbworld_llm.patch"
git -C "$REPO" apply "$@" "$HERE/pcbworld_llm.patch"
if [ "${1:-}" = "-R" ]; then echo "patch reverted in $REPO"; else echo "patch applied to $REPO"; fi
