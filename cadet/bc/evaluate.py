"""Held-out evaluation for a BC run.

Two questions, two measurements:

* *does the policy reproduce the teacher's action?* — the same teacher-forced
  metrics as training, over demos no training worker ever replays
  (:func:`action_eval`);
* *does it route a board?* — an ordinary rollout on held-out boards through the
  shared eval path (:func:`rollout_eval`), so completion / DRC / final-potential
  are the same numbers the RL trainer and the eval pipeline report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cadet.bc.data import batch_to_device
from cadet.bc.objective import bc_loss, component_metrics

__all__ = ["action_eval", "rollout_eval"]


@torch.no_grad()
def action_eval(policy: Any, loader: Any, device: Any, *,
                max_batches: int = 32) -> dict[str, float]:
    """Mean teacher-forced loss + per-component metrics over held-out batches."""
    was_training = policy.training
    policy.eval()
    totals: dict[str, float] = {}
    n = 0
    for batch_np in loader:
        batch = batch_to_device(batch_np, device)
        _loss, stats = bc_loss(policy, batch)
        row = {**stats, **component_metrics(policy, batch)}
        for key, value in row.items():
            if value == value:                      # skip NaN (head unused here)
                totals[key] = totals.get(key, 0.0) + value
        n += 1
        if n >= max_batches:
            break
    if was_training:
        policy.train()
    return {k: v / max(n, 1) for k, v in totals.items()} | {"batches": float(n)}


def rollout_eval(
    policy: Any,
    device: Any,
    boards_dir: str | Path,
    env_kwargs: dict[str, Any],
    *,
    n_boards: int = 32,
    n_rollouts: int = 1,
    n_envs: int = 32,
    max_steps: int = 1024,
    deterministic: bool = True,
    base_seed: int = 1000,
    early_stop_no_geometry_progress: int = 0,
) -> dict[str, float]:
    """Route held-out boards with the current weights -> flat eval scalars.

    Runs through :func:`eval.rollout.rl.eval_transformer` (the path training
    validation and the standalone evaluator both use), so ``completion_rate`` /
    ``drc_violations`` / ``final_potential`` mean exactly what they mean
    everywhere else. ``max_steps`` overrides the env budget: a BC demo on this
    board set runs to several hundred actions, well past a short rollout budget.
    """
    from eval.evaluator import Evaluator
    from eval.rollout.rl import eval_transformer
    from methods._shared.board_loader import load_boards_from_dir_or_list
    from methods.rl_agent.policy.agent import KiCadRLAgent

    boards = load_boards_from_dir_or_list(boards_dir=Path(boards_dir))[:n_boards]
    kwargs = dict(env_kwargs)
    kwargs["max_steps"] = int(max_steps)
    agent = KiCadRLAgent(policy, device=device, deterministic=deterministic)

    was_training = policy.training
    policy.eval()
    try:
        summary = Evaluator(
            lambda bs: eval_transformer(
                agent, torch.device(device), bs, env_kwargs=kwargs,
                n_rollouts=n_rollouts, n_envs=n_envs, base_seed=base_seed,
                max_steps=int(max_steps), deterministic=deterministic,
                step_drc=True, final_drc=False, save_artifacts=False,
                early_stop_no_geometry_progress=early_stop_no_geometry_progress,
            ),
            boards,
        ).run()
    finally:
        if was_training:
            policy.train()
    return {k: float(v) for k, v in summary.overall.items()
            if isinstance(v, (int, float))}
