"""Full-stack parity: new s7.ClassificationModel vs legacy event_ssm ClassificationModel.

Stacks multiple SequenceLayers + pooling + classification head, to catch any
subtle ordering / broadcasting regressions the layer-level parity test might
miss. CPU-only so it can run alongside a GPU training job.
"""

from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from event_ssm.seq_model import ClassificationModel as LegacyCM
from event_ssm.ssm import init_S5SSM as legacy_build_ssm

from s7.model import ClassificationModel as NewCM
from s7.ssm import build_s7


def main():
    d_model = 16
    d_ssm = 16
    L = 128
    num_classes = 5
    num_embeddings = 64

    ssm_init_kwargs = dict(
        C_init="lecun_normal",
        dt_min=1e-3, dt_max=1e-1,
        conj_sym=False, clip_eigs=True,
        log_a=False, stablessm_a=False,
        input_dependent=True, bidirectional=False,
    )
    ssm_kwargs = dict(
        discretization="async",
        d_model=d_model, d_ssm=d_ssm, ssm_block_size=d_ssm,
        num_stages=2, num_layers_per_stage=2,
        dropout=0.0,
        classification_mode="timepool",
        prenorm=True, batchnorm=False,
        pooling_stride=4, pooling_mode="timepool",
        state_expansion_factor=1,
        encoder_type="embed",
    )

    new_model = NewCM(ssm=build_s7(**ssm_init_kwargs),
                     num_classes=num_classes, num_embeddings=num_embeddings, **ssm_kwargs)
    old_model = LegacyCM(ssm=legacy_build_ssm(**ssm_init_kwargs),
                        num_classes=num_classes, num_embeddings=num_embeddings, **ssm_kwargs)

    rng = jax.random.PRNGKey(1)
    init_key, data_key, dropout_key = jax.random.split(rng, 3)
    x = jax.random.randint(data_key, (L,), 0, num_embeddings)
    dt = jnp.ones((L,)) * 0.1
    length = jnp.int32(L)

    vars_new = new_model.init({"params": init_key, "dropout": dropout_key}, x, dt, length, True)
    vars_old = old_model.init({"params": init_key, "dropout": dropout_key}, x, dt, length, True)

    # Forward in eval mode to avoid dropout noise affecting the comparison.
    y_new = new_model.apply(vars_new, x, dt, length, False)
    y_old = old_model.apply(vars_old, x, dt, length, False)

    diff = float(jnp.max(jnp.abs(y_new - y_old)))
    print(f"new shape: {y_new.shape}  old shape: {y_old.shape}")
    print(f"max |Δy| = {diff:.3e}")
    if diff > 1e-5:
        raise SystemExit(f"MODEL PARITY FAILED: max diff = {diff:.3e}")
    print("MODEL PARITY OK")


if __name__ == "__main__":
    main()
