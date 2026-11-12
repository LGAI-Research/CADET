# source ./env.sh  — paths for every command in the README.
#   PCBWORLD_ROOT        PCBWorld checkout (default: third_party/PCBWorld)
#   PCBWORLD_DATA_ROOT   dataset root in PCBWorld's layout (pcbench/exacad_sorted, synthetic/...)
CADET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CADET_ROOT
export PCBWORLD_ROOT="${PCBWORLD_ROOT:-$CADET_ROOT/third_party/PCBWorld}"
export PCBWORLD_DATA_ROOT="${PCBWORLD_DATA_ROOT:-$PCBWORLD_ROOT/var/datasets}"
export PYTHONPATH="$CADET_ROOT:$PCBWORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export KICAD_ENGINE_REUSE=0 OMP_NUM_THREADS=1
[ -d "$PCBWORLD_ROOT/pcb_world" ] || echo "env.sh: no PCBWorld checkout at $PCBWORLD_ROOT (see README: Setup)" >&2
