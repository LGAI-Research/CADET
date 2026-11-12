"""Queue-scheduled (continuous) evaluation rollout for one checkpoint.

Same settings as PCBWorld's validation rollout during training (environment arguments from the checkpoint's training args with eval_diag_max_steps / eval_diag_masking_rule,
step_drc=True, final_drc=False, stochastic policy, seed = eval_base_seed + board.index*100 + r,
same per-rollout rows and aggregate_inline), but scheduled as one job queue instead of waves:
jobs = board x rollout sorted by pad count DESCENDING (largest first); n_envs slots; when a slot's
episode ends its row is finalized and the slot is reloaded with the next job immediately, so no
slot waits for a wave to drain. Episode length is capped by the env itself (env.py truncates at
max_steps), not by the rollout loop.

Run with PYTHONPATH=<PCBWorld checkout> (source ./env.sh).
    python cadet/eval/rollout_queue.py --ckpt checkpoints/cadet.pt --boards-list cadet/configs/splits/boards_d3b_test.txt \
        --prefix val_d3b_test --n-rollouts 12 --n-envs 1 --bf16 --save-boards --out var/runs/cadet_d3b_test
Outputs in --out: per_rollout.csv, summary.json (scalars as <prefix>/*, timing), artifacts/ if --save-boards.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from configs.loader.schema import DEFAULTS
from eval import eval_utils as u
from eval.aggregation import aggregate_inline
from eval.metrics import EvalResult, EvalSummary, _jsonable
from eval.rollout.rl import resolve_eval_env_kwargs
from methods._shared.board_loader import count_pads, load_boards_from_dir_or_list
from methods.rl_agent.models.loader import env_kwargs_from_training_args
from methods.rl_agent.policy.agent import KiCadRLAgent
from methods.rl_agent.rollout.primitive import iter_rollout
from methods.rl_agent.rollout.transformer import (
    _board_net_count_from_obs,
    _finalize_slot,
    _incompatible_row,
    max_net_slots_for_policy,
)
from methods.rl_agent.training.utils import auto_device
from methods.rl_agent.wrappers.factory import make_decoder_env_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--boards-list", required=True)
    ap.add_argument("--prefix", required=True, help="scalar prefix, e.g. val_d3b_test")
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="first N boards of the list (smoke)")
    ap.add_argument("--deterministic", action="store_true", help="smoke equivalence check only")
    ap.add_argument("--save-boards", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true", help="autocast bf16 (training used bf16; default fp32 = async val)")
    ap.add_argument("--compile", action="store_true", help="torch.compile regions stack,decode,heads (= 'efficient')")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    dev = auto_device(a.device)
    agent = KiCadRLAgent.from_checkpoint(a.ckpt, dev)
    if a.bf16 or a.compile:
        agent.model.configure_speed(
            bf16=a.bf16, compile_regions=("stack", "decode", "heads") if a.compile else (),
            compile_mode="default", attn="sdpa")
    targs = SimpleNamespace(**torch.load(a.ckpt, map_location="cpu", weights_only=False)["args"])

    # == build_evaluators' diagnostic env (eval2..eval5)
    env_kwargs = env_kwargs_from_training_args(targs)
    max_steps = int(targs.eval_diag_max_steps) if targs.eval_diag_max_steps is not None else int(targs.max_steps)
    env_kwargs["max_steps"] = max_steps
    if getattr(targs, "eval_diag_masking_rule", None) is not None:
        env_kwargs["masking_rule"] = str(targs.eval_diag_masking_rule)
    env_kwargs = resolve_eval_env_kwargs(env_kwargs, agent, step_drc=True)
    base_seed = int(targs.eval_base_seed)

    boards = load_boards_from_dir_or_list(boards_list=Path(a.boards_list))
    if a.limit:
        boards = boards[: a.limit]
    pads = {b.index: count_pads(b.path) for b in boards}
    ordered = sorted(boards, key=lambda b: -pads[b.index])      # largest first (stable)
    jobs = [(b, r) for b in ordered for r in range(a.n_rollouts)]
    n_envs = min(a.n_envs, len(jobs))
    artifacts_dir = out / "artifacts" if a.save_boards else None
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    t_pool = time.time()
    pool = make_decoder_env_pool(boards[0].path, n_envs, seed=base_seed, **env_kwargs)
    pool_spawn_s = time.time() - t_pool
    max_net_slots = max_net_slots_for_policy(agent)
    agent.model.eval()

    rows: list[dict] = []
    ep_wall: list[dict] = []
    next_job = 0
    slot_job: dict[int, tuple] = {}
    slot_seed: dict[int, int] = {}
    slot_t0: dict[int, float] = {}
    rewards = {s: 0.0 for s in range(n_envs)}
    lengths = {s: 0 for s in range(n_envs)}
    last_info: dict[int, dict] = {s: {} for s in range(n_envs)}
    obs_by_slot: dict[int, dict] = {}
    done = {s: True for s in range(n_envs)}
    # main-loop blocking time: while one slot reloads/finalizes, every other slot waits
    stall = {"load_n": 0, "finalize_s": 0.0}

    # Async load: a finished slot gets its reload+reset commands SENT (not awaited) and sits out
    # of the step loop (done=True) until both replies arrive; the other slots keep stepping.
    # Bookkeeping mirrors SubprocDecoderVecEnv._reload_indices (_last_boards, _rng_reload_seq).
    loading: dict[int, dict] = {}
    stall["async_load_latency_s"] = 0.0
    stall["blocked_on_load_s"] = 0.0   # time with NO live slot, waiting only on loads

    def dispatch_load(slot: int) -> None:
        nonlocal next_job
        if next_job >= len(jobs):
            done[slot] = True
            return
        board, r = jobs[next_job]
        next_job += 1
        seed = base_seed + board.index * 100 + r
        pool._last_boards[slot] = board.path
        if pool._advance_rng_on_reload:
            pool._rng_reload_seq[slot] += 1
        pool.remotes[slot].send(("reload_board", (board.path, pool._rng_reload_seq[slot])))
        pool.remotes[slot].send(("reset", seed))
        loading[slot] = {"job": (board, r), "seed": seed, "acks": 0, "t": time.time(), "obs": None}
        done[slot] = True

    def poll_loads(block: bool) -> None:
        for slot in list(loading):
            ld = loading[slot]
            rem = pool.remotes[slot]
            while ld["acks"] < 2 and (block or rem.poll(0)):
                res = pool._check(rem.recv())
                ld["acks"] += 1
                if ld["acks"] == 2:
                    ld["obs"] = res[0]
            if ld["acks"] < 2:
                continue
            del loading[slot]
            board, r = ld["job"]
            stall["async_load_latency_s"] += time.time() - ld["t"]
            stall["load_n"] += 1
            obs = ld["obs"]
            if max_net_slots is not None:
                nc = _board_net_count_from_obs(obs)
                if nc > max_net_slots:
                    rows.append(_incompatible_row(
                        board=board, rollout_idx=r, seed=ld["seed"], deterministic=a.deterministic,
                        board_net_count=nc, max_net_slots=max_net_slots))
                    dispatch_load(slot)
                    continue
            slot_job[slot], slot_seed[slot] = (board, r), ld["seed"]
            obs_by_slot[slot] = obs
            rewards[slot], lengths[slot], last_info[slot] = 0.0, 0, {}
            done[slot] = False
            slot_t0[slot] = time.time()

    def finalize(slot: int, info: dict, terminated: bool, truncated: bool) -> None:
        board, r = slot_job[slot]
        wall = time.time() - slot_t0[slot]
        t_f = time.time()
        rows.append(_finalize_slot(
            pool=pool, slot=slot, board=board, rollout_idx=r, seed=slot_seed[slot],
            deterministic=a.deterministic, info=info, reward=rewards[slot], steps=lengths[slot],
            terminated=terminated, truncated=truncated,
            save_artifacts=a.save_boards, artifacts_dir=artifacts_dir,
            final_drc=False, reward_config=DEFAULTS.reward_config,
            check_angle=DEFAULTS.check_angle, per_board_rollout_time=float("nan"),
        ))
        stall["finalize_s"] += time.time() - t_f
        e = {"board_index": board.index, "board_id": board.board_id, "rollout": r,
             "pads": pads[board.index], "steps": lengths[slot], "episode_wall_s": round(wall, 3),
             "t_since_start_s": round(time.time() - t_roll, 1)}
        ep_wall.append(e)
        with open(out / "episode_wall.jsonl", "a") as f:     # incremental: survives a kill
            f.write(json.dumps(e) + "\n")
        if len(rows) % 20 == 0:
            u.write_csv(out / "per_rollout.partial.csv", rows, u.PER_ROLLOUT_FIELDS)

    (out / "episode_wall.jsonl").write_text("")
    t_roll = time.time()
    try:
        for s in range(n_envs):             # initial fill: all sends first, then one parallel wait
            dispatch_load(s)
        slots = list(range(n_envs))
        while True:
            if not any(not done[s] for s in slots):
                if not loading:
                    break
                t_b = time.time()
                poll_loads(block=True)       # nothing live: only loads remain in flight
                stall["blocked_on_load_s"] += time.time() - t_b
                continue
            # iter_rollout returns when no slot is live (e.g. all remaining slots loading) -> outer loop
            for batch in iter_rollout(pool, agent, dev, obs_by_slot, slots, done,
                                      want_value=False, deterministic=a.deterministic,
                                      max_steps=10**9, mem_budget=None):
                for k, slot in enumerate(batch.live):
                    rewards[slot] += float(batch.rewards[k])
                    lengths[slot] += 1
                    info = batch.infos[k]
                    last_info[slot] = info
                    term, trunc = bool(batch.terminateds[k]), bool(batch.truncateds[k])
                    if term or trunc:
                        finalize(slot, info, term, trunc)
                        dispatch_load(slot)      # async: slot rejoins once its reset reply arrives
                poll_loads(block=False)          # activated slots are live from the next iteration
                if len(rows) // 50 != getattr(main, "_last", -1):
                    main._last = len(rows) // 50
                    print(f"[queue] done={len(rows)}/{len(jobs)} t={time.time() - t_roll:.0f}s "
                          f"loads={stall['load_n']} async_latency={stall['async_load_latency_s']:.0f}s "
                          f"blocked={stall['blocked_on_load_s']:.0f}s finalize={stall['finalize_s']:.0f}s",
                          flush=True)
        rollout_wall_s = time.time() - t_roll
    finally:
        try:
            pool.close()
        except Exception:  # noqa: BLE001
            pass

    per_board, overall = aggregate_inline(rows, boards, selection_mode=None)
    summary = EvalSummary.from_eval_result(EvalResult(per_rollout=rows, per_board=per_board, overall=overall))
    scalars = summary.to_logger_dict(prefix=a.prefix, include_per_board=False)
    u.write_csv(out / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    res = {
        "ckpt": a.ckpt, "boards_list": a.boards_list, "n_boards": len(boards), "n_rollouts": a.n_rollouts,
        "n_envs": n_envs, "deterministic": a.deterministic, "save_boards": a.save_boards,
        "max_steps": max_steps, "base_seed": base_seed, "order": "pads_desc", "load_mode": "async", "bf16": a.bf16, "compile": a.compile,
        "timing": {"pool_spawn_s": round(pool_spawn_s, 2), "rollout_wall_s": round(rollout_wall_s, 2),
                   "total_wall_s": round(time.time() - t_start, 2),
                   **{f"stall_{k}": round(v, 2) for k, v in stall.items()}},
        "scalars": scalars,
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(res), indent=1))
    keys = [f"{a.prefix}/{k}" for k in ("fp_mean_of_means", "success_rate", "clean_pass_rate", "routability")]
    print(json.dumps({"timing": res["timing"], **{k: scalars.get(k) for k in keys if k in scalars}}, indent=1))


if __name__ == "__main__":
    main()
