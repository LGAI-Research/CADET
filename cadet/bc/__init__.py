"""Behavior cloning for the decoder-only routing policy.

Trains :class:`~methods.rl_agent.models.v1.net.KiCadRLModel` by teacher-forced
imitation of recorded demonstrations instead of by reward, and writes a
checkpoint in the ordinary training format so PPO/GRPO can fine-tune from it and
the eval pipeline can score it unchanged.

Modules:
  * :mod:`.index`     — pick the demos of a generation run and split them.
  * :mod:`.replay`    — replay a demo through the RL wrapper -> (obs, action).
  * :mod:`.data`      — worker-side replay + walk collation -> training batches.
  * :mod:`.objective` — the teacher-forced NLL and its per-component diagnostics.
  * :mod:`.evaluate`  — held-out action metrics + a rollout eval on real boards.
  * :mod:`.train`     — the trainer (``python -m cadet.bc.train``).
"""
