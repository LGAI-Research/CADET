"""Replay a recorded demo through the RL wrapper to recover teacher-forced samples.

Demos store every action as ``(action_type, x_mm, y_mm, layer, mode)`` —
coordinates, not pointer indices. The candidate pool is rebuilt after every
commit, so the pointer index the policy has to emit exists only against the LIVE
pool and is re-resolved here at each step. Replaying is therefore also the check
that a demo is self-consistent: a step whose point is not in the pool, or that the
router rejects, ends the stream.

Two filters keep the cloned distribution honest:

* nets the demo builder marked ``partial`` are executed (later nets see their
  copper) but never recorded;
* a net whose replay diverges is dropped whole — its prefix is a half-routed net
  the policy should not be asked to imitate — and replay stops there.

The demo's ``env_knobs`` (candidate preset, off-board mask) rebuild the same
action space the demo was searched in; everything else comes from the checkpoint
whose env configuration the generation run used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from pcb_world.core.action_schema import (
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
)

__all__ = [
    "DEFAULT_MAX_STEPS",
    "env_kwargs_for_demo",
    "load_demo",
    "replay_demo",
]

#: Step budget for a replay env. Demos run to ~3k actions; the env must never
#: truncate one, so this sits well above the longest recorded demo.
DEFAULT_MAX_STEPS = 8192

#: A make_via is recorded at the realized via centre on the POST-via layer. The
#: matching pool candidate sits on the head's layer within a via radius of it.
_VIA_SNAP_MM = 0.8


def env_kwargs_for_demo(
    ckpt_args: dict[str, Any],
    env_knobs: dict[str, Any] | None = None,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """``make_decoder_env`` kwargs reproducing the env a demo was built in."""
    from configs.loader.schema import RLEnvConfig

    kwargs = RLEnvConfig.from_checkpoint(ckpt_args, max_steps).to_pool_kwargs()
    kwargs["keep_routing_fraction"] = None
    kwargs["obs_format"] = "json"
    knobs = env_knobs or {}
    if knobs.get("preset"):
        kwargs["directional_candidates"] = str(knobs["preset"])
    if knobs.get("offboard_mask") is not None:
        kwargs["offboard_mask"] = bool(knobs["offboard_mask"])
    return kwargs


def load_demo(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _cand_index(cand_mm: list[tuple[float, float, int]], x: float, y: float,
                layer: int) -> int | None:
    key = (round(float(x), 3), round(float(y), 3), int(layer))
    for i, (cx, cy, cl) in enumerate(cand_mm):
        if (round(cx, 3), round(cy, 3), int(cl)) == key:
            return i
    return None


def _via_snap_index(cand_mm: list[tuple[float, float, int]], x: float, y: float,
                    head_layer: int) -> int | None:
    best: tuple[float, int] | None = None
    for i, (cx, cy, cl) in enumerate(cand_mm):
        if int(cl) != head_layer:
            continue
        dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
        if dist <= _VIA_SNAP_MM and (best is None or dist < best[0]):
            best = (dist, i)
    return None if best is None else best[1]


def replay_demo(
    demo: dict,
    env_kwargs: dict[str, Any],
    *,
    skip_partial_nets: bool = True,
    seed: int = 0,
) -> Iterator[tuple[dict, tuple[int, int, int]]]:
    """Yield ``(obs, (action_type, pointer_idx, routing_mode))`` per recorded step.

    A net's samples are buffered until the net closes, so a net that fails
    mid-replay is emitted for exactly zero steps. The env (and its engine
    process) is owned by the generator and closed when it finishes or is dropped.
    """
    from methods.rl_agent.wrappers.factory import make_decoder_env

    env = make_decoder_env(
        demo["board"], seed=seed, policy_net_select=True, **env_kwargs,
    )
    try:
        obs = env.reset()[0]
        nets = list(env.sorted_net_codes)
        status = {int(k): v for k, v in (demo.get("net_status") or {}).items()}
        pending: list[tuple[dict, tuple[int, int, int]]] = []
        cur_net: int | None = None
        skipping = False

        for action_type, x, y, layer, mode in demo["actions"]:
            action_type = int(action_type)
            if action_type == ACT_NET_SELECT:
                cur_net = int(x)
                skipping = skip_partial_nets and status.get(cur_net) == "partial"
                if cur_net not in nets:
                    return                       # net vanished — demo is stale
                pointer = nets.index(cur_net)
            elif action_type == ACT_NET_END:
                pointer = 0                      # unused slot (SLOT_USAGE)
            else:
                cand_mm = env.cand_mm_list
                idx = _cand_index(cand_mm, x, y, layer)
                if idx is None and action_type == ACT_MAKE_VIA and \
                        obs["router_head"].get("is_routing"):
                    idx = _via_snap_index(
                        cand_mm, float(x), float(y),
                        int(obs["router_head"]["current_layer"]),
                    )
                if idx is None:
                    return                       # unresolvable point: stop here
                pointer = idx

            action = (action_type, int(pointer), int(mode))
            if not skipping:
                pending.append((obs, action))
            obs, _reward, terminated, truncated, info = env.step(action)
            if not info.get("action_success", True):
                return                           # divergence: drop the open net
            if action_type == ACT_NET_END:
                yield from pending
                pending = []
            if terminated:
                break
            if truncated:
                return                           # step budget hit mid-net
        # The board's last connection terminates the episode without a net_end.
        yield from pending
    finally:
        env.close()
