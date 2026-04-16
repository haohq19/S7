"""Frozen-reference regression test for the full ClassificationModel.

Compares against hardcoded output values (generated once from the
parity-verified code). Replaces the live-comparison test that imported
the legacy event_ssm.seq_model module (now deleted).
"""

from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from s7.model import ClassificationModel
from s7.ssm import build_s7


def main():
    kw_init = dict(
        C_init="lecun_normal", dt_min=1e-3, dt_max=0.1,
        conj_sym=False, clip_eigs=True,
        log_a=False, stablessm_a=False,
        input_dependent=True, bidirectional=False,
    )
    kw = dict(
        discretization="async", d_model=16, d_ssm=16, ssm_block_size=16,
        num_stages=2, num_layers_per_stage=2,
        dropout=0.0, classification_mode="timepool",
        prenorm=True, batchnorm=False,
        pooling_stride=4, pooling_mode="timepool",
        state_expansion_factor=1, encoder_type="embed",
    )

    model = ClassificationModel(
        ssm=build_s7(**kw_init), num_classes=5, num_embeddings=64, **kw,
    )

    rng = jax.random.PRNGKey(1)
    ik, dk = jax.random.split(rng)
    x = jax.random.randint(dk, (128,), 0, 64)
    dt = jnp.ones((128,)) * 0.1
    length = jnp.int32(128)

    vars_ = model.init({"params": ik, "dropout": dk}, x, dt, length, True)
    y = model.apply(vars_, x, dt, length, False)

    # Frozen reference (generated 2026-04-16 from the parity-verified code)
    REF = jnp.array([-0.417980, 0.531979, -0.702640, 0.701134, -0.635242])

    diff = float(jnp.max(jnp.abs(y - REF)))
    print(f"max |Δy| = {diff:.3e}")
    if diff > 1e-4:
        raise SystemExit(f"MODEL REGRESSION FAILED: max diff = {diff:.3e}")
    print("MODEL REGRESSION TEST OK")


if __name__ == "__main__":
    main()
