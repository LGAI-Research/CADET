"""Behavior-cloning trainer for the decoder-only routing policy.

Teacher-forced imitation of recorded demonstrations: the replay stream
(:mod:`.data`) supplies ``(state, action)`` batches and the policy is trained to
maximise the log-probability of the recorded action under the same head
factorisation PPO scores (:mod:`.objective`). What it writes is an ordinary
training checkpoint — same ``args`` / ``policy_state_dict`` / ``obs_probe``
payload — so PPO can resume from it and the eval pipeline scores it with no
special case.

Configuration comes from a **base checkpoint** (``--base-ckpt``): the demos were
searched in the env that checkpoint defines, so its args pin the policy
architecture, the observation semantics and the env knobs. The demos' own
``env_knobs`` (candidate preset, off-board mask) override the env side — a
generation run may use a finer candidate ladder than the checkpoint was trained on.

Multi-GPU follows PCBWorld's PPO trainer: manual ``torch.distributed``
collectives through :class:`~methods.rl_agent.training.ddp.DDPCtx`, not the DDP
module wrapper. Each rank owns a disjoint shard of the demo list and its own
replay pool; gradients are SUM-allreduced once per step and every rank scales its
loss by ``1 / world_size``, so N ranks x B == one rank x N·B.

Run::

    python -m cadet.bc.train --demos-run <run dir> --base-ckpt cadet/configs/base_args/bc.json \\
        --save-dir var/checkpoints/bc/<run> --total-steps 40000 --gpus 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from cadet.base_args import load_base_args
from cadet.bc import data as bcdata
from cadet.bc.evaluate import action_eval, rollout_eval
from cadet.bc.index import read_manifest, select_demos, split_demos, write_manifest
from cadet.bc.objective import bc_loss
from cadet.bc.replay import env_kwargs_for_demo, load_demo

__all__ = ["main", "build_argparser"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Behavior cloning of recorded routing demonstrations.",
    )
    src = ap.add_argument_group("data")
    src.add_argument("--demos-run", default=None,
                     help="demonstration run directory; a manifest is built (and cached) from it")
    src.add_argument("--manifest", default=None,
                     help="pre-built manifest (see cadet.bc.index)")
    src.add_argument("--val-demos", type=int, default=256,
                     help="demos held out of training for the action-level eval")
    src.add_argument("--limit-demos", type=int, default=None,
                     help="cap the manifest (debugging / short runs)")
    src.add_argument("--max-steps", type=int, default=4096,
                     help="env step budget — shared by demo replay and rollout eval so "
                          "the policy sees one time-feature scale")

    cfg = ap.add_argument_group("model")
    cfg.add_argument("--base-ckpt", required=True,
                     help="checkpoint whose args define the policy + env configuration")
    cfg.add_argument("--init-ckpt", default=None,
                     help="optional weight initialisation (default: fresh init)")
    cfg.add_argument("--no-bf16", action="store_true")
    cfg.add_argument("--attn", default="sdpa", choices=["sdpa", "flex"],
                     help="state-pass attention kernel (see KiCadRLModel.configure_speed)")
    cfg.add_argument("--compile-regions", default="",
                     help="torch.compile regions, e.g. 'efficient' — same knob as the PPO trainer")

    opt = ap.add_argument_group("optimisation")
    opt.add_argument("--total-steps", type=int, default=40000)
    opt.add_argument("--batch-size", type=int, default=128, help="per rank")
    opt.add_argument("--lr", type=float, default=3e-4)
    opt.add_argument("--weight-decay", type=float, default=1e-4)
    opt.add_argument("--warmup-steps", type=int, default=1000)
    opt.add_argument("--min-lr-frac", type=float, default=0.05)
    opt.add_argument("--max-grad-norm", type=float, default=1.0)
    opt.add_argument("--seed", type=int, default=0)

    run = ap.add_argument_group("runtime")
    run.add_argument("--gpus", type=int, default=1)
    run.add_argument("--num-workers", type=int, default=32, help="replay workers per rank")
    run.add_argument("--buffer-size", type=int, default=1024, help="shuffle buffer per worker")
    run.add_argument("--prefetch-factor", type=int, default=2,
                     help="batches each replay worker keeps ready ahead of the trainer")
    run.add_argument("--save-dir", required=True)
    run.add_argument("--log-dir", default=None, help="TensorBoard dir (default <save-dir>/tb)")
    run.add_argument("--save-every", type=int, default=2000)
    run.add_argument("--log-every", type=int, default=25)
    run.add_argument("--no-resume", action="store_true",
                     help="ignore an existing policy_last.pt in --save-dir")

    ev = ap.add_argument_group("evaluation")
    ev.add_argument("--eval-every", type=int, default=1000)
    ev.add_argument("--eval-batches", type=int, default=16)
    ev.add_argument("--rollout-every", type=int, default=5000, help="0 disables")
    ev.add_argument("--rollout-boards-dir", default=None,
                    help="held-out board directory (default: <demo board dir>/../val)")
    ev.add_argument("--rollout-boards", type=int, default=32)
    ev.add_argument("--rollout-envs", type=int, default=32)
    ev.add_argument("--rollout-early-stop", type=int, default=0,
                    help="abandon a rollout after N steps with no geometry change. "
                         "Off by default: an early-stopped rollout is not DRC-scored, "
                         "so final_potential / clean_pass / drc_violations come back NaN")

    wb = ap.add_argument_group("logging sinks")
    wb.add_argument("--wandb", action="store_true")
    wb.add_argument("--wandb-project", default=None)
    wb.add_argument("--wandb-entity", default=None)
    wb.add_argument("--wandb-group", default=None)
    wb.add_argument("--wandb-run-name", default=None)
    wb.add_argument("--wandb-tags", default=None)
    return ap


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def ensure_manifest(args: Namespace) -> str:
    """Path to the manifest, building it from ``--demos-run`` on first use.

    Called once in the launcher, never per rank: the ranks read the file, so a
    100k-file selection is not repeated (nor raced on) once per rank.
    """
    if args.manifest:
        return args.manifest
    if not args.demos_run:
        raise SystemExit("provide --manifest or --demos-run")
    cached = Path(args.save_dir) / "demos.manifest"
    if not cached.exists():
        selected, counts = select_demos(args.demos_run)
        write_manifest(cached, selected,
                       {"run_dir": str(args.demos_run), "counts": counts})
        print(f"  manifest: {len(selected)} demos <- {args.demos_run} {counts}")
    return str(cached)


def resolve_demos(args: Namespace) -> tuple[list[str], list[str]]:
    """(train, val) demo path lists from the manifest."""
    demos = read_manifest(args.manifest)
    if args.limit_demos:
        demos = demos[: args.limit_demos]
    return split_demos(demos, args.val_demos)


def lr_lambda(step: int, *, warmup: int, total: int, min_frac: float) -> float:
    """Linear warmup then cosine decay to ``min_frac`` of the peak LR."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))


def resolve_rollout_target(args: Namespace, first_board: str,
                           *, announce: bool = True) -> None:
    """Fill in ``--rollout-boards-dir``, or turn the rollout OFF loudly.

    A rollout cadence that silently evaluates nothing is the worst outcome: the
    run looks configured, the log stays quiet, and the gap only shows up when
    someone goes looking for curves that were never written. So when the flag is
    omitted the board set is derived from the demo dataset's sibling ``val/``
    split, and when that does not exist ``--rollout-every`` is zeroed IN THE RUN
    ARGS (which ride into every checkpoint) behind one loud line — the recorded
    configuration then matches what actually happened.

    Mutates ``args``; ``announce`` keeps the message to rank 0.
    """
    if not args.rollout_every:
        return
    if not args.rollout_boards_dir:
        # Demos are built from <dataset>/train/board_XXXXX.kicad_pcb, so the
        # sibling val/ split is the natural held-out board set.
        derived = Path(first_board).parent.parent / "val"
        if derived.is_dir():
            args.rollout_boards_dir = str(derived)
            if announce:
                print(f"  rollout boards (derived from the demo dataset): {derived}")
        else:
            args.rollout_every = 0
            if announce:
                print("  WARNING: --rollout-every set but --rollout-boards-dir is "
                      f"empty and no val/ split exists at {derived} — rollout "
                      "evaluation DISABLED (recorded as rollout_every=0).",
                      flush=True)
            return
    if not Path(args.rollout_boards_dir).is_dir():
        args.rollout_every = 0
        if announce:
            print(f"  WARNING: --rollout-boards-dir {args.rollout_boards_dir} is "
                  "not a directory — rollout evaluation DISABLED (recorded as "
                  "rollout_every=0).", flush=True)


def build_policy(base_args: dict, device: Any, init_ckpt: str | None,
                 *, bf16: bool, attn: str = "sdpa",
                 compile_regions: str = "") -> tuple[Any, Any]:
    from configs.loader.schema import RLPolicyConfig

    cfg = RLPolicyConfig.from_namespace(Namespace(**base_args), use_critic=True)
    policy = cfg.build(device)
    if init_ckpt:
        state = torch.load(init_ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(state["policy_state_dict"])
        print(f"  weights initialised from {init_ckpt}")
    regions = tuple(r for r in compile_regions.split(",") if r)
    if bf16 or regions or attn != "sdpa":
        policy.configure_speed(bf16=bf16, compile_regions=regions, attn=attn)
    return policy, cfg


def broadcast_params(policy: Any, ctx: Any) -> None:
    """Put every rank on rank 0's weights.

    ``DDPCtx.sync_grads`` averages GRADIENTS and nothing else. Ranks that built
    (or resumed) different weights would then keep applying one shared update at
    different points forever — the offset never closes, and rank 0's
    checkpoint is trained partly on gradients evaluated somewhere it has never
    been. The PPO path does the same thing through ``DDPUpdateGroup``'s initial
    state broadcast.
    """
    import torch.distributed as dist

    for tensor in policy.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=0, group=ctx.pg)


def checkpoint_payload(policy, optimizer, scheduler, step: int, saved_args: dict,
                       metrics: dict) -> dict:
    """The ordinary training-checkpoint payload, plus the BC scheduler state.

    Carries the obs-probe stamp so a loader can detect observation-semantics
    drift exactly as it does for a PPO checkpoint.
    """
    from methods._shared.config_dump import collect_provenance
    from methods.rl_agent.models.v1.obs_probe import build_probe_obs, probe_digest

    probe = build_probe_obs()
    return {
        "iteration": step,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "args": saved_args,
        "provenance": collect_provenance(),
        "obs_probe": {"obs": probe, "digest": probe_digest(policy.tokenizer, probe)},
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_rank(rank: int, world_size: int, args: Namespace,
               rdv_file: str | None = None) -> None:
    from methods._shared.logger import build_logger
    from methods.rl_agent.training.ddp import DDPCtx

    is_main = rank == 0
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # Same seed on every rank ON PURPOSE: sync_grads averages gradients, it does
    # not synchronise parameters, so ranks must construct identical weights.
    # (broadcast_params below enforces it regardless; per-rank randomness lives
    # in the data stream, which seeds from the rank.)
    torch.manual_seed(args.seed)

    ctx = None
    if world_size > 1:
        import torch.distributed as dist

        from methods.rl_agent.training.ddp import DDP_TIMEOUT

        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            init_method=f"file://{rdv_file}", rank=rank, world_size=world_size,
            timeout=DDP_TIMEOUT,
        )
        ctx = DDPCtx(rank, world_size, device)

    base_args = load_base_args(args.base_ckpt)
    train_demos, val_demos = resolve_demos(args)
    if not train_demos:
        raise SystemExit("no demos selected — check --demos-run / --manifest")

    first_demo = load_demo(train_demos[0])
    knobs = first_demo.get("env_knobs") or {}
    resolve_rollout_target(args, first_demo["board"], announce=is_main)
    env_kwargs = env_kwargs_for_demo(base_args, knobs, max_steps=args.max_steps)
    policy, policy_cfg = build_policy(base_args, device, args.init_ckpt,
                                      bf16=not args.no_bf16, attn=args.attn,
                                      compile_regions=args.compile_regions)
    tokenizer_kwargs = bcdata.tokenizer_kwargs_from_policy_config(policy_cfg)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, eps=1e-5,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: lr_lambda(s, warmup=args.warmup_steps, total=args.total_steps,
                            min_frac=args.min_lr_frac),
    )

    save_dir = Path(args.save_dir)
    start_step = 0
    last_path = save_dir / "policy_last.pt"
    if last_path.exists() and not args.no_resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        policy.load_state_dict(state["policy_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_step = int(state["iteration"])
        if is_main:
            print(f"  resumed from {last_path} at step {start_step} "
                  "(the replay stream restarts — BC data is unordered)")

    if ctx is not None:
        broadcast_params(policy, ctx)

    saved_args = dict(base_args)
    saved_args.update({
        "directional_candidates": env_kwargs["directional_candidates"],
        "offboard_mask": env_kwargs["offboard_mask"],
        "max_steps": args.max_steps,
        "bc": {k: v for k, v in vars(args).items()},
    })

    stream = bcdata.DemoBatchStream(
        train_demos, env_kwargs, tokenizer_kwargs,
        batch_size=args.batch_size, buffer_size=args.buffer_size,
        epochs=None, seed=args.seed, rank=rank, world_size=world_size,
    )
    loader = bcdata.make_loader(stream, args.num_workers,
                                prefetch_factor=args.prefetch_factor, in_order=False)

    logger = None
    if is_main:
        log_dir = args.log_dir or str(save_dir / "tb")
        logger = build_logger(log_dir, args=args)
        print(f"  bc: {len(train_demos)} train demos, {len(val_demos)} val demos")
        print(f"  bc: batch {args.batch_size} x {world_size} rank(s), "
              f"{args.num_workers} replay workers/rank, {args.total_steps} steps")
        print(f"  env: preset={env_kwargs['directional_candidates']} "
              f"offboard_mask={env_kwargs['offboard_mask']} "
              f"max_steps={env_kwargs['max_steps']}")

    val_batches = (
        materialise_val_batches(val_demos, env_kwargs, tokenizer_kwargs, args)
        if is_main else []
    )

    policy.train()
    step = start_step
    t_start = time.perf_counter()
    t_window = t_start
    wait_s = 0.0
    seen = 0
    acc: dict[str, float] = {}
    it = iter(loader)

    while step < args.total_steps:
        t0 = time.perf_counter()
        batch_np = next(it)
        wait_s += time.perf_counter() - t0

        batch = bcdata.batch_to_device(batch_np, device)
        loss, stats = bc_loss(policy, batch)
        (loss / world_size).backward()
        if ctx is not None:
            ctx.sync_grads(policy)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), args.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        seen += args.batch_size * world_size
        for key, value in stats.items():
            acc[key] = acc.get(key, 0.0) + value
        acc["grad_norm"] = acc.get("grad_norm", 0.0) + float(grad_norm)

        if step % args.log_every == 0:
            now = time.perf_counter()
            row = {k: v / args.log_every for k, v in acc.items()}
            if ctx is not None:
                row = {k: v / world_size
                       for k, v in ctx.allreduce_sums(row).items()}
            acc.clear()
            if is_main:
                elapsed = now - t_window
                row["lr"] = scheduler.get_last_lr()[0]
                row["steps_per_s"] = args.log_every / elapsed
                row["samples_per_s"] = args.log_every * args.batch_size * world_size / elapsed
                row["data_wait_frac"] = wait_s / elapsed
                for key, value in row.items():
                    logger.add_scalar(f"bc/{key}", value, step)
                print(f"  step {step:>7} nll {row['nll']:.4f} "
                      f"ent {row['entropy']:.3f} gn {row['grad_norm']:.2f} "
                      f"lr {row['lr']:.2e} {row['samples_per_s']:.0f} samples/s "
                      f"(data wait {row['data_wait_frac']*100:.0f}%)", flush=True)
            wait_s = 0.0
            t_window = now

        # The two cadences are independent: gating the rollout inside an
        # eval-cadence branch would silently reduce it to the steps where BOTH
        # divide (lcm), e.g. --eval-every 2000 --rollout-every 15000 fires the
        # rollout at 30000, not 15000. Every rank evaluates the same condition,
        # so the barriers inside run_eval stay matched.
        last = step == args.total_steps
        due_val = step % args.eval_every == 0 or last
        due_rollout = bool(args.rollout_every) and (
            step % args.rollout_every == 0 or last)
        if due_val or due_rollout:
            metrics = run_eval(policy, device, val_batches, args, ctx, is_main,
                               logger, step, env_kwargs,
                               do_val=due_val, do_rollout=due_rollout)
        else:
            metrics = {}

        if is_main and (step % args.save_every == 0 or step == args.total_steps):
            save_checkpoint(
                save_dir / "policy_last.pt",
                checkpoint_payload(policy, optimizer, scheduler, step,
                                   saved_args, metrics),
            )
            save_checkpoint(
                save_dir / f"policy_step_{step}.pt",
                checkpoint_payload(policy, optimizer, scheduler, step,
                                   saved_args, metrics),
            )
            print(f"  saved {save_dir}/policy_step_{step}.pt", flush=True)

    if is_main:
        save_checkpoint(
            save_dir / "policy_final.pt",
            checkpoint_payload(policy, optimizer, scheduler, step, saved_args, {}),
        )
        print(f"  done: {step} steps, {seen} samples, "
              f"{(time.perf_counter() - t_start) / 60:.1f} min")
        if logger is not None:
            logger.close()
    if ctx is not None:
        import torch.distributed as dist

        dist.destroy_process_group()


def materialise_val_batches(val_demos: list[str], env_kwargs: dict,
                            tokenizer_kwargs: dict, args: Namespace) -> list[dict]:
    """Replay the held-out demos ONCE into a fixed set of batches.

    A fixed set (a few tens of MB of numpy) makes the validation curve comparable
    across steps and removes a worker pool restart from every eval; the
    alternative — a fresh draw each time — moves the metric for reasons that have
    nothing to do with the weights.
    """
    if not val_demos or args.eval_batches <= 0:
        return []
    stream = bcdata.DemoBatchStream(
        val_demos, env_kwargs, tokenizer_kwargs,
        batch_size=args.batch_size,
        buffer_size=max(args.batch_size, args.buffer_size // 4),
        epochs=1, seed=args.seed + 7919,
    )
    loader = bcdata.make_loader(stream, min(args.num_workers, 8), persistent=False)
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= args.eval_batches:
            break
    print(f"  val set: {len(batches)} fixed batches "
          f"({len(batches) * args.batch_size} samples) from {len(val_demos)} demos",
          flush=True)
    return batches


def run_eval(policy, device, val_batches, args, ctx, is_main: bool, logger,
             step: int, env_kwargs: dict, *, do_val: bool = True,
             do_rollout: bool = False) -> dict[str, float]:
    """Rank-0 evaluation, with the other ranks parked on a barrier."""
    import torch.distributed as dist

    if ctx is not None:
        dist.barrier()
    metrics: dict[str, float] = {}
    if is_main and do_val and val_batches:
        metrics = action_eval(policy, val_batches, device,
                              max_batches=len(val_batches))
        for key, value in metrics.items():
            logger.add_scalar(f"val/{key}", value, step)
        print("  val " + " ".join(
            f"{k}={v:.4f}" for k, v in sorted(metrics.items())), flush=True)

    if is_main and do_rollout and args.rollout_boards_dir:
        t0 = time.perf_counter()
        roll = rollout_eval(
            policy, device, args.rollout_boards_dir, env_kwargs,
            n_boards=args.rollout_boards, n_envs=args.rollout_envs,
            max_steps=args.max_steps, deterministic=True,
            early_stop_no_geometry_progress=args.rollout_early_stop,
        )
        for key, value in roll.items():
            logger.add_scalar(f"rollout/{key}", value, step)
        # Console line only — every rollout scalar reaches TensorBoard above.
        keep = ("success_rate", "clean_pass", "ratsnest_reduction",
                "final_potential", "drc_violations", "steps")
        print(f"  rollout ({time.perf_counter() - t0:.0f}s) " + " ".join(
            f"{k}={roll[k]:.4f}" for k in keep if k in roll), flush=True)
        metrics = {**metrics, **{f"rollout_{k}": v for k, v in roll.items()}}
    if ctx is not None:
        dist.barrier()
    return metrics


def main() -> None:
    args = build_argparser().parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.save_dir) / "bc_args.json", "w") as fh:
        json.dump(vars(args), fh, indent=2, sort_keys=True)
    args.manifest = ensure_manifest(args)
    if args.gpus <= 1:
        train_rank(0, 1, args)
        return
    import signal

    import torch.multiprocessing as mp

    fd, rdv = tempfile.mkstemp(prefix="cadet_bc_rdv_")
    os.close(fd)
    os.unlink(rdv)
    spawn_ctx = mp.spawn(train_rank, args=(args.gpus, args, rdv),
                         nprocs=args.gpus, join=False)

    def _terminate(signum, _frame):
        # Without this, a SIGTERM to the launcher orphans the ranks: they keep
        # their GPU memory and their replay workers, and have to be hunted down
        # by PID. (Their engine children still need a sweep — the engine server
        # exits on the client's explicit kill(), not on the client's death.)
        for proc in spawn_ctx.processes:
            if proc.is_alive():
                proc.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    try:
        while not spawn_ctx.join(timeout=10):
            pass
    finally:
        if os.path.exists(rdv):
            os.unlink(rdv)


if __name__ == "__main__":
    main()
