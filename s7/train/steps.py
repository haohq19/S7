"""Single-step train/eval functions for S7 classification and regression tasks."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array


def _compute_loss(logits, targets, loss_type: str):
    if loss_type == "cross_entropy":
        return optax.softmax_cross_entropy(logits, targets).mean()
    if loss_type == "one_hot_cross_entropy":
        one_hot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
        return optax.softmax_cross_entropy(logits, one_hot).mean()
    if loss_type == "mse":
        return optax.l2_loss(logits, targets).mean()
    raise ValueError(f"unknown loss_type: {loss_type}")


def _compute_accuracy(logits, targets, loss_type: str):
    if loss_type == "mse":
        return jnp.full((), jnp.nan)
    preds = jnp.argmax(logits, axis=-1)
    if loss_type == "one_hot_cross_entropy":
        target_class = targets
    else:
        target_class = jnp.argmax(targets, axis=-1)
    return (preds == target_class).mean()


def train_step(
    state,
    batch: Tuple[Array, Array, Array, Array],
    dropout_key: Array,
    *,
    distributed: bool = False,
    loss_type: str = "cross_entropy",
) -> Tuple[Any, Dict[str, Array]]:
    """One training step: forward, loss, gradient, optimizer update."""
    inputs, targets, dt, lengths = batch

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, **state.model_state},
            inputs, dt, lengths, True,
            rngs={"dropout": dropout_key},
            mutable=["batch_stats"],
        )
        return _compute_loss(logits, targets, loss_type), (logits, updates)

    (loss, (logits, batch_updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    accuracy = _compute_accuracy(logits, targets, loss_type)

    if distributed:
        grads = jax.lax.pmean(grads, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        accuracy = jax.lax.pmean(accuracy, axis_name="data")

    state = state.apply_gradients(grads=grads).replace(model_state=batch_updates)
    return state, {"loss": loss, "accuracy": accuracy}


def eval_step(
    state,
    batch: Tuple[Array, Array, Array, Array],
    *,
    distributed: bool = False,
    loss_type: str = "cross_entropy",
) -> Tuple[Any, Dict[str, Array]]:
    """One evaluation step: forward, loss, accuracy."""
    inputs, targets, dt, lengths = batch
    logits = state.apply_fn(
        {"params": state.params, **state.model_state},
        inputs, dt, lengths, False,
    )
    loss = _compute_loss(logits, targets, loss_type)
    accuracy = _compute_accuracy(logits, targets, loss_type)
    if distributed:
        loss = jax.lax.pmean(loss, axis_name="data")
        accuracy = jax.lax.pmean(accuracy, axis_name="data")
    return state, {"loss": loss, "accuracy": accuracy}
