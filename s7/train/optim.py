"""Optimizer construction with separate parameter groups for SSM vs. projection vs. regular weights.

The SSM core (``Lambda_re``, ``Lambda_im``, ``B``) is trained at a small base
learning rate with no weight decay; the input-dependent Δ projection (and any
B/C projections, if present) and the rest of the network use a larger LR
(``ssm_lr × lr_factor``) and standard weight decay.
"""

from __future__ import annotations

from functools import partial

import optax


# Parameter-name leaves that belong to the SSM core (low LR, no WD).
_SSM_LEAVES = frozenset({"B", "Lambda_re", "Lambda_im"})

# Parameter-name leaves that belong to the input-dependent Δ projection (and
# any optional B/C projection variants we may add later).
_PROJ_LEAVES = frozenset({
    "step_proj_kernel", "step_proj_bias",
    # Legacy names retained for old checkpoints that exposed full B/C projection.
    "x_proj", "B_proj", "C_proj",
})


def _label_leaf(name: str) -> str:
    if name in _SSM_LEAVES:
        return "ssm"
    if name in _PROJ_LEAVES:
        return "proj"
    return "regular"


def _label_tree(tree):
    """Recursively label every leaf of a Flax param tree as ssm/proj/regular."""
    return {
        k: (_label_tree(v) if hasattr(v, "keys") else _label_leaf(k))
        for k, v in tree.items()
    }


def build_learning_rate_fn(*, lr: float, total_steps: int, warmup_steps: int, schedule: str):
    """Cosine or constant LR with linear warm-up."""
    if schedule == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=lr,
            warmup_steps=warmup_steps, decay_steps=total_steps,
        )
    if schedule == "constant":
        return optax.join_schedules(
            [
                optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps),
                optax.constant_schedule(lr),
            ],
            [warmup_steps],
        )
    raise ValueError(f"unknown schedule: {schedule}")


def build_optimizer(opt_config):
    """Return a multi-group AdamW optimizer wired up from an OmegaConf config.

    Expected ``opt_config`` keys:
        ssm_lr, lr_factor, total_steps, warmup_steps, schedule,
        ssm_weight_decay, weight_decay, proj_weight_decay,
        accumulation_steps  (optional)
    """
    lr_fn = partial(
        build_learning_rate_fn,
        total_steps=opt_config.total_steps,
        warmup_steps=opt_config.warmup_steps,
        schedule=opt_config.schedule,
    )

    base_lr = opt_config.ssm_lr
    other_lr = base_lr * opt_config.lr_factor

    tx = optax.multi_transform(
        {
            "ssm": optax.inject_hyperparams(
                partial(optax.adamw, b1=0.9, b2=0.999, weight_decay=opt_config.ssm_weight_decay),
            )(learning_rate=lr_fn(lr=base_lr)),
            "regular": optax.adamw(
                learning_rate=lr_fn(lr=other_lr),
                b1=0.9, b2=0.999, weight_decay=opt_config.weight_decay,
            ),
            "proj": optax.adamw(
                learning_rate=lr_fn(lr=other_lr),
                b1=0.9, b2=0.999, weight_decay=opt_config.proj_weight_decay,
            ),
        },
        _label_tree,
    )

    if opt_config.get("accumulation_steps", 0):
        tx = optax.MultiSteps(tx, every_k_schedule=opt_config.accumulation_steps)
    return tx
