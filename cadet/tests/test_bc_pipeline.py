"""BC pipeline guards that need neither the C++ engine nor a GPU.

Covers the three places a silent mistake would corrupt a BC run without failing
loudly: the demo-selection filters, the LR schedule, and — most important — the
shuffle buffer's obs/action pairing (a batch whose actions no longer belong to
their observations trains perfectly happily on nonsense).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from cadet.bc.data import DemoBatchStream
from cadet.bc.index import select_demos, split_demos
from cadet.bc.train import lr_lambda, resolve_rollout_target


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------
def _run_dir(tmp_path, exec_rows, replay_rows=()):
    (tmp_path / "demos").mkdir(parents=True)
    (tmp_path / "replay").mkdir(parents=True)
    for row in exec_rows:
        stem = row["board"].replace(".kicad_pcb", "")
        if row.pop("_no_file", False):
            continue
        (tmp_path / "demos" / f"{stem}.json").write_text("{}")
    with open(tmp_path / "shard_00.jsonl", "w") as fh:
        for row in exec_rows:
            fh.write(json.dumps(row) + "\n")
    if replay_rows:
        with open(tmp_path / "replay" / "shard_00.jsonl", "w") as fh:
            for row in replay_rows:
                fh.write(json.dumps(row) + "\n")
    return tmp_path


def test_select_demos_filters(tmp_path):
    run = _run_dir(tmp_path, [
        {"board": "board_00000.kicad_pcb", "clean_pass": True},
        {"board": "board_00001.kicad_pcb", "clean_pass": False},
        {"board": "board_00002.kicad_pcb", "clean_pass": True, "_no_file": True},
        {"board": "board_00003.kicad_pcb", "clean_pass": True},
    ], replay_rows=[
        {"board": "board_00000.kicad_pcb", "ok": True},
        {"board": "board_00003.kicad_pcb", "ok": False},
    ])
    demos, counts = select_demos(run)
    assert [p.stem for p in demos] == ["board_00000"]
    assert counts["not_clean_pass"] == 1
    assert counts["replay_failed"] == 1
    assert counts["no_demo_file"] == 1

    # A demo the verification pass has not reached yet is kept by default and
    # dropped under require_replay_ok.
    run2 = _run_dir(tmp_path / "b", [{"board": "board_00000.kicad_pcb", "clean_pass": True}])
    assert len(select_demos(run2)[0]) == 1
    assert len(select_demos(run2, require_replay_ok=True)[0]) == 0


def test_select_demos_last_row_wins(tmp_path):
    """Shards are appended to on resume — the surviving row is the last one."""
    run = _run_dir(tmp_path, [
        {"board": "board_00000.kicad_pcb", "clean_pass": False},
        {"board": "board_00000.kicad_pcb", "clean_pass": True},
    ])
    assert [p.stem for p in select_demos(run)[0]] == ["board_00000"]


def test_split_demos_holds_out_the_tail():
    demos = [f"d{i}" for i in range(10)]
    train, val = split_demos(demos, 3)
    assert train == demos[:7] and val == demos[7:]
    assert split_demos(demos, 0) == (demos, [])


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
def test_lr_lambda_warmup_then_cosine():
    kw = dict(warmup=100, total=1000, min_frac=0.05)
    assert lr_lambda(0, **kw) == pytest.approx(0.01)
    assert lr_lambda(99, **kw) == pytest.approx(1.0)
    assert lr_lambda(100, **kw) == pytest.approx(1.0)
    assert lr_lambda(550, **kw) == pytest.approx(0.525, abs=1e-3)   # cosine midpoint
    assert lr_lambda(1000, **kw) == pytest.approx(0.05)
    assert lr_lambda(5000, **kw) == pytest.approx(0.05)             # never below the floor


# ---------------------------------------------------------------------------
# shuffle buffer
# ---------------------------------------------------------------------------
class _StubTokenizer:
    """Stands in for BatchedStateTokenizer: the walk is just the obs it saw."""

    def walk_timed(self, obs_list):
        return {"tag": [o["tag"] for o in obs_list]}


def _fake_obs(tag: int) -> dict:
    return {
        "tag": tag,
        "_masks": {
            "action": np.ones(7, dtype=bool),
            "pointer": np.zeros(0, dtype=np.int64),
            "mode": np.ones(3, dtype=bool),
            "net_valid": np.ones(4, dtype=bool),
            "offlayer": np.zeros(0, dtype=np.int64),
            "offboard": np.zeros(0, dtype=np.int64),
        },
    }


def test_emit_keeps_obs_and_action_paired():
    """Popping from the buffer must not decouple an obs from its action.

    ``_emit`` swap-removes the sampled positions; the action recorded for a
    sample is tag+1 here, so any mis-pairing shows up as a broken invariant.
    """
    import random

    stream = DemoBatchStream([], {}, {}, batch_size=8, buffer_size=32)
    obs_buf = [_fake_obs(i) for i in range(32)]
    act_buf = [(0, i + 1, 0) for i in range(32)]
    batch = stream._emit(obs_buf, act_buf, random.Random(0), _StubTokenizer())

    assert len(obs_buf) == len(act_buf) == 24            # exactly batch_size popped
    tags = batch["walk"]["tag"]
    assert len(tags) == 8 and len(set(tags)) == 8        # distinct samples
    assert [a[1] for a in batch["actions"]] == [t + 1 for t in tags]
    # what stayed behind is still paired, and disjoint from what left
    assert [o["tag"] + 1 for o in obs_buf] == [a[1] for a in act_buf]
    assert not set(tags) & {o["tag"] for o in obs_buf}
    assert batch["actions"].shape == (8, 3)
    assert batch["action_masks"].shape == (8, 7)


# ---------------------------------------------------------------------------
# rollout target
# ---------------------------------------------------------------------------
def _rollout_args(**kw):
    from argparse import Namespace

    return Namespace(**{"rollout_every": 5000, "rollout_boards_dir": None, **kw})


def _dataset(tmp_path, with_val: bool):
    """A demo dataset layout: <root>/train/board_00000.kicad_pcb (+ optional val/)."""
    (tmp_path / "train").mkdir()
    board = tmp_path / "train" / "board_00000.kicad_pcb"
    board.write_text("")
    if with_val:
        (tmp_path / "val").mkdir()
    return str(board)


def test_rollout_dir_derived_from_the_demo_dataset(tmp_path, capsys):
    board = _dataset(tmp_path, with_val=True)
    args = _rollout_args()
    resolve_rollout_target(args, board)
    assert args.rollout_boards_dir == str(tmp_path / "val")
    assert args.rollout_every == 5000
    assert "rollout boards" in capsys.readouterr().out


def test_rollout_without_a_board_set_is_disabled_loudly(tmp_path, capsys):
    """A rollout cadence that evaluates nothing must not pass silently.

    The run args ride into every checkpoint, so the cadence is zeroed rather
    than left claiming a rollout that never happens.
    """
    board = _dataset(tmp_path, with_val=False)
    args = _rollout_args()
    resolve_rollout_target(args, board)
    assert args.rollout_every == 0
    assert "WARNING" in capsys.readouterr().out


def test_rollout_dir_that_does_not_exist_is_disabled_loudly(tmp_path, capsys):
    board = _dataset(tmp_path, with_val=True)
    args = _rollout_args(rollout_boards_dir=str(tmp_path / "nope"))
    resolve_rollout_target(args, board)
    assert args.rollout_every == 0
    assert "WARNING" in capsys.readouterr().out


def test_explicit_rollout_dir_is_kept(tmp_path):
    board = _dataset(tmp_path, with_val=True)
    explicit = tmp_path / "elsewhere"
    explicit.mkdir()
    args = _rollout_args(rollout_boards_dir=str(explicit))
    resolve_rollout_target(args, board)
    assert args.rollout_boards_dir == str(explicit)
    assert args.rollout_every == 5000


def test_rollout_disabled_needs_no_board_set(tmp_path):
    """--rollout-every 0 is a deliberate choice, not a misconfiguration."""
    board = _dataset(tmp_path, with_val=False)
    args = _rollout_args(rollout_every=0)
    resolve_rollout_target(args, board)
    assert args.rollout_every == 0 and args.rollout_boards_dir is None


# ---------------------------------------------------------------------------
# per-type recall
# ---------------------------------------------------------------------------
def test_per_type_recall_exposes_a_failing_minority_class():
    """The aggregate type accuracy cannot see a rare class failing outright.

    In miniature: 96 make_line + 4 make_via, with
    every via mispredicted as a line. Micro accuracy reads 0.96 ("fine"), while
    the macro average and the per-type recall show the via head is dead.
    """
    import torch

    from cadet.bc.objective import per_type_recall
    from methods.rl_agent.models.v1.spec import ACT_MAKE_LINE, ACT_MAKE_VIA

    target = torch.tensor([ACT_MAKE_LINE] * 96 + [ACT_MAKE_VIA] * 4)
    pred = torch.full_like(target, ACT_MAKE_LINE)          # never predicts a via
    out = per_type_recall(pred, target)

    assert float((pred == target).float().mean()) == pytest.approx(0.96)
    assert out["type_recall_make_line"] == pytest.approx(1.0)
    assert out["type_recall_make_via"] == pytest.approx(0.0)
    assert out["type_acc_macro"] == pytest.approx(0.5)
    assert out["type_n_make_via"] == 4.0
    # types absent from the batch are omitted, not reported as a zero recall
    assert "type_recall_net_select" not in out
