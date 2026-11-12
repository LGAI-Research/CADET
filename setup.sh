#!/usr/bin/env bash
# Install the PCBWorld benchmark that CADET runs on, then check that CADET imports against it.
#   bash setup.sh                                  # clone v1.0.1 into third_party/PCBWorld and build it
#   PCBWORLD_ROOT=<existing checkout> bash setup.sh   # use a checkout you already built
#   PCBWORLD_ENV=<name> bash setup.sh              # conda env name (default: cadet)
set -euo pipefail
CADET="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PCBWORLD_ROOT:-$CADET/third_party/PCBWorld}"
TAG="${PCBWORLD_TAG:-v1.0.1}"
URL="${PCBWORLD_GIT_URL:-https://github.com/LGAI-Research/PCBWorld.git}"
export PCBWORLD_ENV="${PCBWORLD_ENV:-cadet}"     # conda env of this installation; an existing `pcbworld` env is left alone
if [ ! -d "$ROOT/pcb_world" ]; then
  git clone --recursive --branch "$TAG" "$URL" "$ROOT"
  (cd "$ROOT" && bash tools/setup/setup_all.sh)      # conda env `$PCBWORLD_ENV`, engine build, import smoke
fi
echo "PCBWorld: $ROOT ($(git -C "$ROOT" describe --tags --always 2>/dev/null || echo unknown))"
echo "next: conda activate $PCBWORLD_ENV && source $CADET/env.sh && python -m pytest $CADET/cadet/tests -q"
