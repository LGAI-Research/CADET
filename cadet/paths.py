"""Locations: this repository and the PCBWorld checkout it runs on.

CADET does not vendor PCBWorld. ``setup.sh`` clones it to ``third_party/PCBWorld``; set
``PCBWORLD_ROOT`` to use another checkout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CADET_ROOT = Path(__file__).resolve().parents[1]


def pcbworld_root() -> Path:
    root = Path(os.environ.get("PCBWORLD_ROOT") or CADET_ROOT / "third_party" / "PCBWorld")
    if not (root / "pcb_world").is_dir():
        raise SystemExit(f"PCBWorld checkout not found at {root}: run setup.sh or set PCBWORLD_ROOT")
    return root


def use_pcbworld() -> Path:
    """Put the PCBWorld checkout on ``sys.path`` (after this repository) and return its root."""
    root = pcbworld_root()
    if str(root) not in sys.path:
        sys.path.insert(1 if sys.path and sys.path[0] == str(CADET_ROOT) else 0, str(root))
    return root
