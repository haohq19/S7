"""Thin wrappers around flax.training.checkpoints (single-device only)."""

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
        keep=1,
    )


def load_checkpoint(state, ckpt_dir: str):
    return checkpoints.restore_checkpoint(
        ckpt_dir=os.path.abspath(ckpt_dir),
        target=state,
    )
