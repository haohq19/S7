"""S7 training entry point.

Usage:

    python scripts/train.py task=dvs-gesture

This is the rewritten counterpart of the legacy ``run_training.py`` and reads
the same Hydra config tree under ``configs/``.
"""

from __future__ import annotations

import os

# Give XLA's BFC allocator a larger chunk of each GPU up-front. The default of
# 0.75 is not enough for DVS128 at L=524288, B=8 — init allocates ~22 GB.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

from functools import partial

import hydra
import jax
import jax.numpy as jnp
from flax import jax_utils
from omegaconf import DictConfig, OmegaConf, open_dict

from s7.data import get_builder
from s7.model import BatchClassificationModel, RetrievalModel
from s7.ssm import build_s7
from s7.train.optim import build_optimizer
from s7.train.steps import eval_step, train_step
from s7.train.trainer import Trainer, TrainState


def _seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def _build_dataset(cfg: DictConfig, world_size: int):
    builder = get_builder(cfg.task.name)
    return builder(
        cache_dir=cfg.data_dir,
        seed=cfg.seed,
        world_size=world_size,
        **cfg.training,
        **cfg.model.ssm,
    )


def _build_model(cfg: DictConfig, info):
    ssm_init_fn = build_s7(**cfg.model.ssm_init)
    if cfg.task.name == "retrieval-classification":
        model_cls = RetrievalModel
    else:
        model_cls = BatchClassificationModel
    return model_cls(
        ssm=ssm_init_fn,
        num_classes=info.n_classes,
        num_embeddings=info.num_embeddings,
        **cfg.model.ssm,
    )


def _init_train_state(rng, model, sample_batch, opt_config) -> TrainState:
    inputs, _, dt, lengths = sample_batch
    init_key, dropout_key = jax.random.split(rng)
    variables = model.init({"params": init_key, "dropout": dropout_key}, inputs, dt, lengths, True)
    params = variables.pop("params")
    model_state = variables

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"[*] {n_params:,} model parameters")

    tx = build_optimizer(opt_config)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        key=dropout_key,
        model_state=model_state,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    with open(os.path.join(cfg.logging.log_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    _seed_everything(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)
    init_key, dropout_key = jax.random.split(key)

    world_size = jax.local_device_count()
    if world_size > 1:
        dropout_key = jax.random.split(dropout_key, world_size)

    print("[*] Loading dataset")
    train_loader, val_loader, test_loader, info = _build_dataset(cfg, world_size)

    with open_dict(cfg):
        steps_per_epoch = max(1, len(train_loader) // cfg.optimizer.accumulation_steps)
        cfg.optimizer.total_steps = cfg.training.num_epochs * steps_per_epoch
        cfg.optimizer.warmup_steps = cfg.optimizer.warmup_epochs * steps_per_epoch
        cfg.optimizer.ssm_lr = (
            cfg.optimizer.ssm_base_lr
            * cfg.training.per_device_batch_size
            * world_size
            * cfg.optimizer.accumulation_steps
        )

    print("[*] Building model")
    model = _build_model(cfg, info)

    sample = next(iter(train_loader))
    bsz = cfg.training.per_device_batch_size
    sample = tuple(x[:bsz] for x in sample)

    print("[*] Initializing parameters")
    state = _init_train_state(init_key, model, sample, cfg.optimizer)

    if world_size > 1:
        state = jax_utils.replicate(state)
        train_step_fn = jax.pmap(
            partial(train_step, distributed=True, loss_type=cfg.training.loss_type),
            axis_name="data",
        )
        eval_step_fn = jax.pmap(
            partial(eval_step, distributed=True, loss_type=cfg.training.loss_type),
            axis_name="data",
        )
    else:
        train_step_fn = jax.jit(partial(train_step, loss_type=cfg.training.loss_type))
        eval_step_fn = jax.jit(partial(eval_step, loss_type=cfg.training.loss_type))

    wandb_run = None
    if cfg.logging.wandb:
        import wandb
        wandb_run = wandb.init(
            dir=cfg.logging.log_dir,
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    trainer = Trainer(
        train_state=state,
        train_step_fn=train_step_fn,
        eval_step_fn=eval_step_fn,
        world_size=world_size,
        log_dir=cfg.logging.log_dir,
        num_epochs=cfg.training.num_epochs,
        log_interval=cfg.logging.interval,
        wandb_run=wandb_run,
        summary_metric=cfg.logging.summary_metric,
    )

    print("[*] Training")
    trainer.fit(train_loader, val_loader, dropout_key, test_loader)


if __name__ == "__main__":
    main()
