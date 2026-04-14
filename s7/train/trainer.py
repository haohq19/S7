"""S7 training loop with per-epoch metric logging and best-checkpoint tracking."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training import train_state as ts
from jaxtyping import Array

from s7.train.checkpoint import load_checkpoint, save_checkpoint


class TrainState(ts.TrainState):
    key: Array
    model_state: Dict


@partial(jax.jit, static_argnums=(1,))
def _reshape_batch(batch, num_devices):
    def _reshape(x):
        rest, ragged = divmod(x.shape[0], num_devices)
        if ragged:
            raise ValueError(f"batch dim {x.shape[0]} not divisible by {num_devices}")
        return x.reshape((num_devices, rest) + x.shape[1:])
    return jax.tree_util.tree_map(_reshape, batch)


def _opt_state_lr(opt_state):
    """Pull the current SSM-group learning rate out of an optax MultiSteps state."""
    from optax import MultiStepsState
    if isinstance(opt_state, MultiStepsState):
        opt_state = opt_state.inner_opt_state
    try:
        return opt_state.inner_states["ssm"].inner_state.hyperparams["learning_rate"]
    except (AttributeError, KeyError, TypeError):
        return None


class Trainer:
    """Owns a TrainState + train/eval step fns + logging + checkpointing."""

    def __init__(
        self,
        *,
        train_state: TrainState,
        train_step_fn: Callable,
        eval_step_fn: Callable,
        world_size: int,
        log_dir: str,
        num_epochs: int,
        log_interval: int = 100,
        wandb_run=None,
        summary_metric: str = "Performance/Validation accuracy",
    ):
        self.state = train_state
        self.train_step = train_step_fn
        self.eval_step = eval_step_fn
        self.world_size = world_size
        self.log_dir = log_dir
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.wandb_run = wandb_run
        self.summary_metric = summary_metric
        self.epoch_idx = 0
        self.best_eval_metrics: Dict[str, Any] = {}

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "checkpoints"), exist_ok=True)

        n_params = sum(
            arr.size for arr in jax.tree_util.tree_leaves(self.state.params)
            if isinstance(arr, jnp.ndarray)
        ) // max(world_size, 1)
        print(f"[*] {n_params:,} parameters")

    # ------------------------------------------------------------------
    def fit(self, train_loader: Iterator, val_loader: Iterator,
            dropout_key: Array, test_loader: Optional[Iterator] = None) -> Dict[str, Any]:
        for epoch in range(1, self.num_epochs + 1):
            self.epoch_idx = epoch
            train_metrics = self._train_one_epoch(train_loader, dropout_key)
            self._log_epoch("Train", train_metrics)

            val_metrics = self._evaluate(val_loader, "Performance/Validation")
            self._log_epoch("Val  ", val_metrics)

            if self._is_better(val_metrics):
                self.best_eval_metrics = val_metrics
                self._save_state()
                self._dump_metrics("best_eval", val_metrics)

            self._wandb_log(epoch, train_metrics, val_metrics)

        if test_loader is not None:
            self._restore_state()
            test_metrics = self._evaluate(test_loader, "Performance/Test")
            self._dump_metrics("test", test_metrics)
            self.best_eval_metrics.update(test_metrics)
            print("─" * 80)
            print(f"| End of Training |  test/acc = {test_metrics.get('Performance/Test accuracy', float('nan')):.4f}"
                  f"  test/loss = {test_metrics.get('Performance/Test loss', float('nan')):.4f}")
            print("─" * 80)
            if self.wandb_run is not None:
                self.wandb_run.log(test_metrics)

        return self.best_eval_metrics

    # ------------------------------------------------------------------
    def _train_one_epoch(self, loader: Iterator, dropout_key: Array) -> Dict[str, Any]:
        sums: Dict[str, float] = defaultdict(float)
        running: Dict[str, float] = defaultdict(float)
        n = 0
        epoch_start = time.time()
        window_start = epoch_start

        for i, batch in enumerate(loader):
            _, _, _, lengths = batch
            if jnp.any(lengths == 0):
                continue

            if self.world_size > 1:
                step_key, dropout_key = jax.vmap(jax.random.split, in_axes=0, out_axes=1)(dropout_key)
                step_key = jax.vmap(jax.random.fold_in)(step_key, jnp.arange(self.world_size))
                batch = _reshape_batch(batch, self.world_size)
            else:
                step_key, dropout_key = jax.random.split(dropout_key)

            self.state, m = self.train_step(self.state, batch, step_key)

            if jnp.isnan(m["loss"]).any():
                print("EXITING TRAINING DUE TO NaN LOSS")
                sys.exit(1)

            for k, v in m.items():
                sums[k] += float(jnp.mean(v))
                running[k] += float(jnp.mean(v))
            n += 1

            if (i + 1) % self.log_interval == 0:
                dt = time.time() - window_start
                window_start = time.time()
                msg = " | ".join(f"{k} {v / self.log_interval:.4f}" for k, v in running.items())
                print(f"| epoch {self.epoch_idx} | step {i+1} | {dt*1000/self.log_interval:.1f} ms/it | {msg}")
                running.clear()

        out = {f"Performance/Training {k}": v / max(n, 1) for k, v in sums.items()}
        out["epoch_time"] = time.time() - epoch_start
        return out

    # ------------------------------------------------------------------
    def _evaluate(self, loader: Iterator, prefix: str) -> Dict[str, Any]:
        sums: Dict[str, float] = defaultdict(float)
        n = 0
        skipped = 0
        for batch in loader:
            if self.world_size > 1:
                # Truncate the last ragged batch to a multiple of num_devices.
                # Loaders use drop_last=False for eval so every sample is seen
                # under single-device, but pmap can't ingest a ragged leading
                # axis. Skipped samples are at most (num_devices - 1) per pass
                # and are accumulated into ``skipped`` for the final log line.
                bsz = batch[0].shape[0]
                rest = (bsz // self.world_size) * self.world_size
                if rest == 0:
                    skipped += bsz
                    continue
                if rest != bsz:
                    skipped += bsz - rest
                    batch = tuple(x[:rest] for x in batch)
                batch = _reshape_batch(batch, self.world_size)
            self.state, m = self.eval_step(self.state, batch)
            for k, v in m.items():
                sums[k] += float(jnp.mean(v))
            n += 1
        if skipped and self.world_size > 1:
            print(f"[{prefix}] dropped {skipped} ragged tail samples for pmap")
        return {f"{prefix} {k}": v / max(n, 1) for k, v in sums.items()}

    # ------------------------------------------------------------------
    def _is_better(self, new_metrics: Dict[str, Any]) -> bool:
        if not self.best_eval_metrics:
            return True
        is_max = "perplexity" not in self.summary_metric
        for key in (self.summary_metric, "Performance/Validation accuracy", "Performance/Validation loss"):
            if key in new_metrics and key in self.best_eval_metrics:
                a, b = new_metrics[key], self.best_eval_metrics[key]
                return a > b if is_max else a < b
        return False

    # ------------------------------------------------------------------
    def _log_epoch(self, label: str, metrics: Dict[str, Any]):
        kv = " | ".join(
            f"{k.split('/')[-1]} {v:.4f}"
            for k, v in metrics.items() if isinstance(v, (int, float))
        )
        print(f"| epoch {self.epoch_idx:3d} | {label} | {kv}")

    def _wandb_log(self, epoch, train_metrics, val_metrics):
        if self.wandb_run is None:
            return
        merged = {"Performance/epoch": epoch, **train_metrics, **val_metrics}
        lr = _opt_state_lr(
            jax_utils.unreplicate(self.state.opt_state) if self.world_size > 1 else self.state.opt_state
        )
        if lr is not None:
            merged["learning_rate"] = float(jnp.mean(lr))
        self.wandb_run.log(merged)

    def _dump_metrics(self, name: str, metrics: Dict[str, Any]):
        with open(os.path.join(self.log_dir, "metrics", f"{name}.json"), "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    def _save_state(self):
        state = jax_utils.unreplicate(self.state) if self.world_size > 1 else self.state
        save_checkpoint(state, os.path.join(self.log_dir, "checkpoints"), int(state.step))

    def _restore_state(self):
        state = jax_utils.unreplicate(self.state) if self.world_size > 1 else self.state
        restored = load_checkpoint(state, os.path.join(self.log_dir, "checkpoints"))
        self.state = jax_utils.replicate(restored) if self.world_size > 1 else restored
