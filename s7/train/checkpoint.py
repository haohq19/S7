"""Thin wrappers around flax.training.checkpoints.

Uses ``keep=10_000`` (effectively unbounded) to sidestep the built-in
``_remove_invalid_ckpts`` GC path in flax, which calls ``io.rmtree`` on the
old checkpoint's orbax temp directory. Under pmap on Lustre that rmtree
occasionally races with the save it's trying to retire and fails with
``FailedPreconditionError: ... ocdbt.process_0/d; Directory not empty``.

We never actually end up with 10 000 checkpoints because the trainer only
calls ``save_checkpoint`` when the validation metric improves.
"""

from __future__ import annotations

import os

from flax.training import checkpoints


def save_checkpoint(state, ckpt_dir: str, step: int):
    os.makedirs(ckpt_dir, exist_ok=True)
    return checkpoints.save_checkpoint(
        ckpt_dir=os.path.abspath(ckpt_dir),
        target=state,
        step=int(step),
        overwrite=True,
        keep=10_000,
    )


def load_checkpoint(state, ckpt_dir: str):
    return checkpoints.restore_checkpoint(
        ckpt_dir=os.path.abspath(ckpt_dir),
        target=state,
    )
