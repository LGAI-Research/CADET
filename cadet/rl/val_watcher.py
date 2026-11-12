#!/usr/bin/env python3
"""Asynchronous validation watcher with some diagnostic eval sets switched off (PCBWorld's stock watcher otherwise).

The stock watcher (methods/rl_agent/training/async_val.py watcher_main) builds its evaluators once from the first
checkpoint's args via training.loop.build_evaluators; this wrapper clears eval<k>_boards for the skipped prefixes before
that call. Nothing else changes (boards, steps, rollouts, seeds, result format).
    python cadet/rl/val_watcher.py --save-dir <ckpt dir> --n-envs 32 --device cuda [--skip-sets val_d3bf]
"""
import argparse

import methods.rl_agent.training.loop as loop
from methods.rl_agent.training.async_val import watcher_main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--poll-interval", type=float, default=10.0)
    p.add_argument("--skip-sets", default="val_d3bf", help="comma-separated eval prefixes to leave out")
    a = p.parse_args()
    skip = {s for s in a.skip_sets.split(",") if s}
    stock = loop.build_evaluators

    def build_evaluators(args, *rest, **kw):
        for k in (2, 3, 4, 5):
            if getattr(args, f"eval{k}_prefix", None) in skip and getattr(args, f"eval{k}_boards", None):
                print(f"[watcher] eval set {getattr(args, f'eval{k}_prefix')} switched off (--skip-sets)", flush=True)
                setattr(args, f"eval{k}_boards", None)
        if "val_greedy" in skip:
            args.no_eval_greedy = True
        return stock(args, *rest, **kw)

    loop.build_evaluators = build_evaluators   # watcher_main imports it from the module at bootstrap
    watcher_main(a.save_dir, device=a.device, n_envs=a.n_envs, poll_interval=a.poll_interval)


if __name__ == "__main__":
    main()
