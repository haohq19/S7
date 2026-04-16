"""Frozen-reference regression test for the S7 layer.

Compares the rewritten s7.ssm.S7 forward pass against hardcoded numerical
reference values (generated once from the parity-verified code). This
replaces the original live-comparison test that imported the legacy
event_ssm.ssm.S7 module (now deleted).
"""

from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from s7.ssm import S7


def main():
    L, H_in = 64, 8
    P, block_size = 16, 16

    common = dict(
        H_in=H_in, H_out=H_in, P=P, block_size=block_size,
        C_init="lecun_normal", discretization="async",
        dt_min=1e-3, dt_max=1e-1,
        log_a=False, stablessm_a=False, bidirectional=False,
        conj_sym=False, clip_eigs=True, input_dependent=True,
    )

    rng = jax.random.PRNGKey(0)
    _, data_key = jax.random.split(rng)
    u = jax.random.normal(data_key, (L, H_in))
    dt = jnp.ones((L,))

    layer = S7(**common)
    variables = layer.init(rng, u, dt)  # use rng (not split) to match ref generation
    y = layer.apply(variables, u, dt)

    # Frozen reference values (generated 2026-04-16 from the parity-verified code)
    REF_SUM = -11.642549514770508
    REF_ABS_MAX = 2.9626331329345703

    sum_diff = abs(float(jnp.sum(y)) - REF_SUM)
    max_diff = abs(float(jnp.max(jnp.abs(y))) - REF_ABS_MAX)

    print(f"sum diff: {sum_diff:.3e}  max diff: {max_diff:.3e}")
    if sum_diff > 1e-4 or max_diff > 1e-4:
        raise SystemExit(f"REGRESSION: sum_diff={sum_diff:.3e} max_diff={max_diff:.3e}")
    print("SSM REGRESSION TEST OK")


if __name__ == "__main__":
    main()
