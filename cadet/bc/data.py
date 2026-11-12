"""BC input pipeline: demo replay in worker processes, walked batches out.

Every training sample needs a whole-board observation, so ~50M of them cannot be
materialised — demos are replayed on the fly instead. Two measurements shape the
split of work between the workers and the training process:

* replay costs ~10 ms/step and the tokenizer's CPU walk another ~1.1 ms/step, so
  both belong in the workers — a walk done in the training process alone would
  cap the whole run at ~1k steps/s;
* a walked batch is a handful of numpy columns: 3x the bytes of the obs dicts it
  replaces, but ~35x cheaper to unpickle (0.9 ms vs 33 ms per 128 samples), and
  the unpickling is what the training process pays for.

So a worker replays demos into a local shuffle buffer and emits ready-made
batches — the walk, the five act-time mask arrays, and the action triples — and
the training process only tensorises them.

Batch decorrelation is the shuffle buffer's job: consecutive steps of one board
are near-duplicates, so each worker holds ``buffer_size`` samples and draws every
batch from it at random. A demo averages ~536 actions, so a 1024-sample buffer
spans a couple of boards; the diversity BETWEEN batches comes from the worker
pool, which the loader round-robins, and from sharding the demo list across
``(rank, worker id)`` so no two workers replay the same board.
"""
from __future__ import annotations

import random
import sys
from typing import Any, Iterator

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from cadet.bc.replay import load_demo, replay_demo

__all__ = [
    "DemoBatchStream", "collate_batch", "batch_to_device", "make_loader",
    "tokenizer_kwargs_from_policy_config",
]


def tokenizer_kwargs_from_policy_config(policy_cfg: Any) -> dict[str, Any]:
    """The :class:`BatchedStateTokenizer` subset of an ``RLPolicyConfig``.

    Derived from the constructor signature rather than hand-listed: the walk a
    worker produces has to match the model that consumes it, so a tokenizer knob
    added to the policy config must reach the workers without a second edit.
    """
    import inspect
    from dataclasses import asdict

    from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer

    names = set(inspect.signature(BatchedStateTokenizer.__init__).parameters)
    return {k: v for k, v in asdict(policy_cfg).items() if k in names}


def collate_batch(
    obs_chunk: list[dict],
    actions: list[tuple[int, int, int]],
    tokenizer: Any,
    *,
    policy_net_select: bool = True,
) -> dict[str, Any]:
    """Walk + mask-gather one chunk of samples into a numpy-only batch payload."""
    from methods.rl_agent.policy.agent import gather_mask_arrays

    masks, ptr_masks, mode_masks, nvm, off_masks = gather_mask_arrays(
        None, list(range(len(obs_chunk))),
        policy_net_select=policy_net_select, obs_list=obs_chunk,
    )
    return {
        "walk": tokenizer.walk_timed(obs_chunk),
        "action_masks": masks,
        "pointer_masks": ptr_masks,
        "mode_masks": mode_masks,
        "net_valid_masks": nvm,
        "offlayer_masks": off_masks,
        "actions": np.asarray(actions, dtype=np.int64),
    }


def batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    """Tensorise a batch payload into ``evaluate_actions_and_value`` kwargs.

    ``mode_none_if_all_true=True`` is the training convention (the PPO collector
    and its update path both use it): an all-permissive mode mask becomes None
    rather than an all-True tensor, which is a semantic difference in the
    log-prob composition, not a style one.
    """
    import torch

    from methods.rl_agent.policy.agent import mask_arrays_to_tensors

    mask_t, ptr_t, mode_t, extra = mask_arrays_to_tensors(
        batch["action_masks"], batch["pointer_masks"], batch["mode_masks"],
        batch["net_valid_masks"], device,
        mode_none_if_all_true=True, off_masks=batch["offlayer_masks"],
    )
    return {
        "actions": torch.as_tensor(batch["actions"], device=device),
        "walked": batch["walk"],
        "action_masks": mask_t,
        "pointer_masks": ptr_t,
        "mode_mask": mode_t,
        **extra,
    }


class DemoBatchStream(IterableDataset):
    """Replays demos and yields walk-collated training batches.

    Args:
        demos: demo JSON paths (a manifest slice).
        env_kwargs: ``make_decoder_env`` kwargs, from
            :func:`~cadet.bc.replay.env_kwargs_for_demo`.
        tokenizer_kwargs: :class:`BatchedStateTokenizer` constructor kwargs —
            the walk must match the model that consumes it.
        batch_size: samples per emitted batch.
        buffer_size: shuffle-buffer depth, in samples.
        epochs: passes over ``demos``; None = never stop (the trainer decides).
        rank / world_size: DDP sharding, composed with the worker sharding.
    """

    def __init__(
        self,
        demos: list[str],
        env_kwargs: dict[str, Any],
        tokenizer_kwargs: dict[str, Any],
        *,
        batch_size: int = 128,
        buffer_size: int = 1024,
        epochs: int | None = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        policy_net_select: bool = True,
    ) -> None:
        self.demos = list(demos)
        self.env_kwargs = dict(env_kwargs)
        self.tokenizer_kwargs = dict(tokenizer_kwargs)
        self.batch_size = int(batch_size)
        self.buffer_size = max(int(buffer_size), int(batch_size))
        self.epochs = epochs
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.policy_net_select = bool(policy_net_select)

    def _shard(self) -> tuple[int, int]:
        info = get_worker_info()
        n_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        return self.rank * n_workers + worker_id, self.world_size * n_workers

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer

        shard, n_shards = self._shard()
        tokenizer = BatchedStateTokenizer(**self.tokenizer_kwargs).eval()
        rng = random.Random(self.seed * 1_000_003 + shard)
        obs_buf: list[dict] = []
        act_buf: list[tuple[int, int, int]] = []
        n_failed = 0
        epoch = 0

        while self.epochs is None or epoch < self.epochs:
            order = list(self.demos)
            random.Random(self.seed + epoch).shuffle(order)
            for path in order[shard::n_shards]:
                try:
                    for obs, action in replay_demo(load_demo(path), self.env_kwargs):
                        obs_buf.append(obs)
                        act_buf.append(action)
                        if len(obs_buf) >= self.buffer_size:
                            yield self._emit(obs_buf, act_buf, rng, tokenizer)
                except Exception as exc:                        # noqa: BLE001
                    # A demo that cannot be replayed (engine hiccup, stale board)
                    # must not end the shard. Report the first few per worker;
                    # the rate is a training metric, not a per-board log.
                    n_failed += 1
                    if n_failed <= 3:
                        print(f"[bc-data] shard {shard}: replay failed for "
                              f"{path}: {exc!r}", file=sys.stderr, flush=True)
            epoch += 1

        while len(obs_buf) >= self.batch_size:
            yield self._emit(obs_buf, act_buf, rng, tokenizer)

    def _emit(self, obs_buf: list, act_buf: list, rng: random.Random,
              tokenizer: Any) -> dict[str, Any]:
        """Pop ``batch_size`` random samples from the buffer and collate them."""
        picks = rng.sample(range(len(obs_buf)), self.batch_size)
        obs_chunk, act_chunk = [], []
        for i in sorted(picks, reverse=True):
            obs_chunk.append(obs_buf[i])
            act_chunk.append(act_buf[i])
            obs_buf[i] = obs_buf[-1]
            act_buf[i] = act_buf[-1]
            obs_buf.pop()
            act_buf.pop()
        return collate_batch(
            obs_chunk, act_chunk, tokenizer,
            policy_net_select=self.policy_net_select,
        )


def _passthrough(batch: Any) -> Any:
    """DataLoader collate for an already-collated payload (module-level: picklable)."""
    return batch


def make_loader(stream: DemoBatchStream, num_workers: int, *,
                prefetch_factor: int = 2, persistent: bool = True,
                in_order: bool = True):
    """DataLoader over :class:`DemoBatchStream` — batching already happened.

    ``multiprocessing_context="spawn"``: workers build KiCad engine processes,
    and the training process has CUDA initialised by the time iteration starts,
    which fork cannot inherit safely.

    ``in_order=False`` lets the loader hand out whichever worker's batch is ready
    instead of cycling through the workers in a fixed order. Replay time per
    board is heavy-tailed (p50 3 s, p99 50 s), so the fixed
    order stalls every rank on its slowest worker while the others sit on full
    prefetch queues — measured as 30-50% data wait with the CPUs mostly idle.
    Batches are random draws from per-worker shuffle buffers either way, so the
    sample distribution does not change; keep the default for a loader whose
    batch sequence must be reproducible (the fixed val set).
    """
    from torch.utils.data import DataLoader

    if num_workers <= 0:
        return DataLoader(stream, batch_size=None, num_workers=0,
                          collate_fn=_passthrough)
    return DataLoader(
        stream, batch_size=None, num_workers=num_workers,
        collate_fn=_passthrough, persistent_workers=persistent,
        prefetch_factor=prefetch_factor, multiprocessing_context="spawn",
        in_order=in_order,
    )
