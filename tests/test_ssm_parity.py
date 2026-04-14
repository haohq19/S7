"""Numerical parity test: new s7.S7Layer vs legacy event_ssm.S7.

Both should produce identical outputs given identical PRNG keys and inputs.
Runs on CPU so it can be executed concurrently with a GPU training job.
"""

from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

# Legacy module
from event_ssm.ssm import S7 as LegacyS7
# Rewrite
from s7.ssm import S7 as S7Layer


def _run_layer(LayerCls, *, key, common, u, dt):
    layer = LayerCls(**common)
    variables = layer.init(key, u, dt)
    return layer.apply(variables, u, dt), variables


def main():
    L, H_in = 64, 16
    P = 32
    block_size = 16

    common = dict(
        H_in=H_in, H_out=H_in, P=P, block_size=block_size,
        C_init="lecun_normal", discretization="async",
        dt_min=1e-3, dt_max=1e-1,
        log_a=False, stablessm_a=False, bidirectional=False,
        conj_sym=False, clip_eigs=True, input_dependent=True,
    )

    rng = jax.random.PRNGKey(0)
    init_key, data_key = jax.random.split(rng)
    u = jax.random.normal(data_key, (L, H_in))
    dt = jnp.ones((L,))

    y_new, vars_new = _run_layer(S7Layer, key=init_key, common=common, u=u, dt=dt)
    y_old, vars_old = _run_layer(LegacyS7, key=init_key, common=common, u=u, dt=dt)

    # The parameter trees (and init RNG paths) should now be identical because
    # the rewrite uses the same Flax submodule and parameter names.
    diff = float(jnp.max(jnp.abs(y_new - y_old)))
    print(f"max |Δy| (full init parity) = {diff:.3e}")
    if diff > 1e-5:
        print("PARAM TREE NEW:", jax.tree_util.tree_map(lambda x: x.shape, vars_new["params"]))
        print("PARAM TREE OLD:", jax.tree_util.tree_map(lambda x: x.shape, vars_old["params"]))
        raise SystemExit(f"FULL PARITY FAILED: max diff = {diff:.3e}")
    print("FULL PARITY OK")


if __name__ == "__main__":
    main()
