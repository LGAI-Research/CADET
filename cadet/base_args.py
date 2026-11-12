"""Base arguments: the environment / observation / architecture settings that the generator, the
replay checker and the BC trainer rebuild their environments and policy from.

A base file is either a JSON dump of a trainer's argument dict (``cadet/configs/base_args/*.json``)
or a trainer checkpoint, whose ``args`` entry is read. Weights are never loaded from it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_base_args(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    import torch

    return dict(torch.load(path, map_location="cpu", weights_only=False)["args"])
