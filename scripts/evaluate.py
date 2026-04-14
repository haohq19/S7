"""Evaluate an S7 checkpoint on its task's test set."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

from functools import partial

import hydra
import jax
from omegaconf import DictConfig, OmegaConf

from s7.data import get_builder
from s7.model import BatchClassificationModel, RetrievalModel
from s7.ssm import build_s7
from s7.train.checkpoint import load_checkpoint
from s7.train.optim import build_optimizer
from s7.train.steps import eval_step
from s7.train.trainer import TrainState


def _build(cfg: DictConfig):
    builder = get_builder(cfg.task.name)
    train_loader, _val, test_loader, info = builder(
        cache_dir=cfg.data_dir, seed=cfg.seed, world_size=1,
        **cfg.training, **cfg.model.ssm,
    )
    ssm_init = build_s7(**cfg.model.ssm_init)
    model_cls = RetrievalModel if cfg.task.name == "retrieval-classification" else BatchClassificationModel
    model = model_cls(
        ssm=ssm_init, num_classes=info.n_classes, num_embeddings=info.num_embeddings,
        **cfg.model.ssm,
    )
    return train_loader, test_loader, model, info


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    if not cfg.checkpoint:
        raise SystemExit("checkpoint=<path> required")
    print(OmegaConf.to_yaml(cfg))

    key = jax.random.PRNGKey(cfg.seed)
    train_loader, test_loader, model, info = _build(cfg)

    sample = next(iter(train_loader))
    bsz = cfg.training.per_device_batch_size
    sample = tuple(x[:bsz] for x in sample)
    inputs, _, dt, lengths = sample

    init_key, _ = jax.random.split(key)
    variables = model.init({"params": init_key, "dropout": init_key}, inputs, dt, lengths, True)
    params = variables.pop("params")
    model_state = variables

    cfg.optimizer.total_steps = max(1, cfg.training.num_epochs)
    cfg.optimizer.warmup_steps = 1
    cfg.optimizer.ssm_lr = 1e-4
    tx = build_optimizer(cfg.optimizer)

    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, key=init_key, model_state=model_state,
    )
    state = load_checkpoint(state, cfg.checkpoint)

    eval_fn = jax.jit(partial(eval_step, loss_type=cfg.training.loss_type))
    sums = {}
    n = 0
    for batch in test_loader:
        state, m = eval_fn(state, batch)
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    avg = {k: v / max(n, 1) for k, v in sums.items()}
    print("test:", avg)


if __name__ == "__main__":
    main()
