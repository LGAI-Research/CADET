"""Teacher-forced behavior-cloning objective over the decoder policy.

The BC loss is the negative log-likelihood of the recorded action under exactly
the factorisation the PPO update scores:
:meth:`~methods.rl_agent.models.v1.net.KiCadRLModel.evaluate_actions_and_value`
returns the joint log-prob of ``(action_type, pointer_idx, routing_mode)`` under
the same masks and the same ``SLOT_USAGE`` weighting, so a BC-initialised policy
and a PPO update agree on what an action's probability means. The demos come from
a policy-driven-net-select configuration, so ``allow_net_select_lp`` is on
(supplied by ``mask_arrays_to_tensors`` together with the net-valid mask) and
``net_select`` contributes to the loss instead of being treated as an external
choice.

Per-component numbers — NLL and top-1 accuracy for each of the three heads — come
from :meth:`factored_action_logits`, which returns the same autoregressive factors
for every action type from one state pass. The loss therefore stays on the PPO
code path and the diagnostics do not duplicate it.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from pcb_world.core.action_schema import ACTION_NAMES
from methods.rl_agent.models.v1.spec import (
    ACT_FINISH,
    ACT_NET_SELECT,
    NUM_ACTION_TYPES,
    SLOT_USAGE,
)

__all__ = ["bc_loss", "component_metrics", "per_type_recall"]


def per_type_recall(pred: torch.Tensor, target: torch.Tensor,
                    n_types: int = NUM_ACTION_TYPES) -> dict[str, float]:
    """Recall of each action type, plus the macro (unweighted) average.

    The aggregate ``type_acc`` is a micro average, so it is dominated by the
    majority types and **cannot see a minority class failing outright**:
    ``make_via`` is ~4.4% of the demo actions, so a policy that never predicts a
    via still scores ~0.98. Measured in one run at step 15000: type_acc 0.983
    while make_via recall was 0.083 — and the rollout placed 0.02 vias/board.
    Types absent from the batch are omitted rather than reported as zero.
    """
    out: dict[str, float] = {}
    recalls = []
    for t in range(n_types):
        sel = target == t
        n = int(sel.sum())
        if not n:
            continue
        r = float((pred[sel] == t).float().mean())
        out[f"type_recall_{ACTION_NAMES[t]}"] = r
        out[f"type_n_{ACTION_NAMES[t]}"] = float(n)
        recalls.append(r)
    if recalls:
        out["type_acc_macro"] = sum(recalls) / len(recalls)
    return out


def bc_loss(policy: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean teacher-forced NLL of the recorded actions, plus scalar diagnostics."""
    log_probs, entropy, _values = policy.evaluate_actions_and_value(
        None,
        batch["actions"],
        action_masks=batch["action_masks"],
        pointer_masks=batch["pointer_masks"],
        offlayer_masks=batch.get("offlayer_masks"),
        mode_mask=batch["mode_mask"],
        net_valid_mask=batch.get("net_valid_mask"),
        allow_net_select_lp=batch.get("allow_net_select_lp", True),
        walked=batch["walked"],
    )
    loss = -log_probs.mean()
    stats = {
        "nll": float(loss.detach()),
        "entropy": float(entropy.detach().mean()),
        "perplexity": float(torch.exp(loss.detach())),
    }
    return loss, stats


@torch.no_grad()
def component_metrics(policy: Any, batch: dict[str, Any]) -> dict[str, float]:
    """Per-head NLL and top-1 accuracy of the recorded action (diagnostics only).

    Each head is scored where the action actually uses it (``SLOT_USAGE``), so a
    ``net_end`` never counts towards pointer accuracy and only ``make_line`` /
    ``make_via`` / ``finish`` count towards the mode head. ``net_select`` counts
    towards the pointer head — it is a real decision here, not an external one.
    """
    out = policy.factored_action_logits(
        None,
        action_masks=batch["action_masks"],
        pointer_masks=batch["pointer_masks"],
        offlayer_masks=batch.get("offlayer_masks"),
        mode_mask=batch["mode_mask"],
        net_valid_mask=batch.get("net_valid_mask"),
        walked=batch["walked"],
    )
    actions = batch["actions"]
    action_type, pointer, mode = actions[:, 0], actions[:, 1], actions[:, 2]
    rows = torch.arange(actions.size(0), device=actions.device)
    usage = SLOT_USAGE.to(actions.device)
    needs_ptr = usage[action_type, 0].bool()
    needs_mode = usage[action_type, 1].bool()
    is_finish = action_type == ACT_FINISH

    metrics: dict[str, float] = {}

    at_logp = F.log_softmax(out["at_logits"].float(), dim=-1)
    pred_type = out["at_logits"].argmax(-1)
    metrics["type_nll"] = float(-at_logp[rows, action_type].mean())
    metrics["type_acc"] = float((pred_type == action_type).float().mean())
    metrics.update(per_type_recall(pred_type, action_type))

    # Pointer: the factor for the RECORDED type; its index space is the combined
    # net/cand one the sampler uses, so argmax is directly comparable.
    ptr_logits = out["ptr_logits"][rows, action_type].float()
    ptr_logp = F.log_softmax(ptr_logits, dim=-1)
    safe_ptr = pointer.clamp(min=0, max=max(ptr_logits.size(-1) - 1, 0))
    metrics.update(_masked(
        "pointer", -ptr_logp[rows, safe_ptr], ptr_logits.argmax(-1) == pointer,
        needs_ptr,
    ))
    metrics.update(_masked(
        "net_select", -ptr_logp[rows, safe_ptr],
        ptr_logits.argmax(-1) == pointer, action_type == ACT_NET_SELECT,
    ))

    # Mode: finish reads the PRE-pointer hidden, make_line/make_via the
    # POST-pointer one — mirror of evaluate_actions_and_value.
    n_cands = out["mode_pt_logits"].size(2)
    cand_ptr = pointer.clamp(min=0, max=max(n_cands - 1, 0))
    mode_pt = out["mode_pt_logits"][rows, action_type, cand_ptr].float()
    mode_at = out["mode_at_logits"][rows, action_type].float()
    mode_logits = torch.where(is_finish.unsqueeze(-1), mode_at, mode_pt)
    mode_logp = F.log_softmax(mode_logits, dim=-1)
    metrics.update(_masked(
        "mode", -mode_logp[rows, mode.clamp(min=0)],
        mode_logits.argmax(-1) == mode, needs_mode,
    ))
    metrics["joint_acc"] = float(_joint_acc(
        out["at_logits"], ptr_logits, mode_logits,
        action_type, pointer, mode, needs_ptr, needs_mode,
    ))
    return metrics


def _masked(name: str, nll: torch.Tensor, correct: torch.Tensor,
            mask: torch.Tensor) -> dict[str, float]:
    """Mean NLL / accuracy over the rows the head is actually used on."""
    n = int(mask.sum())
    if n == 0:
        return {f"{name}_nll": float("nan"), f"{name}_acc": float("nan"),
                f"{name}_n": 0.0}
    return {
        f"{name}_nll": float((nll * mask).sum() / n),
        f"{name}_acc": float((correct & mask).sum() / n),
        f"{name}_n": float(n),
    }


def _joint_acc(at_logits, ptr_logits, mode_logits, action_type, pointer, mode,
               needs_ptr, needs_mode) -> torch.Tensor:
    """Fraction of rows where every head the action uses picks the recorded value."""
    ok = at_logits.argmax(-1) == action_type
    ok = ok & (~needs_ptr | (ptr_logits.argmax(-1) == pointer))
    ok = ok & (~needs_mode | (mode_logits.argmax(-1) == mode))
    return ok.float().mean()
