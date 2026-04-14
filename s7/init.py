"""HiPPO-LegS initialization and parameter init helpers for the S7 SSM.

Adapted from the annotated-S4 reference (Albert Gu et al.), with the diagonal-
plus-low-rank (DPLR) decomposition that S5/S7 use to initialize the diagonal
state matrix Λ from a normal eigendecomposition of the HiPPO-LegS operator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import eigh


# ---------------------------------------------------------------------------
# HiPPO-LegS construction
# ---------------------------------------------------------------------------

def make_hippo(N: int) -> jnp.ndarray:
    """Return the (N×N) HiPPO-LegS state matrix (negated)."""
    p = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = p[:, None] * p[None, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A


def make_nplr_hippo(N: int):
    """Return (A, P, B) — the rank-1 NPLR factorization of HiPPO-LegS."""
    A = make_hippo(N)
    P = jnp.sqrt(jnp.arange(N) + 0.5)
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return A, P, B


def make_dplr_hippo(N: int):
    """Return (Λ, P, B_conj, V, B_orig) — diagonal-plus-low-rank HiPPO-LegS.

    The state matrix S = A + P P^T is normal, so it diagonalizes as
    S = V diag(Λ) V^*. We use only Λ (and V to project B into the eigen basis).
    """
    A, P, B = make_nplr_hippo(N)
    S = A + P[:, None] * P[None, :]

    # The real part of Λ collapses to a single value for HiPPO-LegS.
    S_diag = jnp.diagonal(S)
    lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)
    lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return lambda_real + 1j * lambda_imag, P, B, V, B_orig


# ---------------------------------------------------------------------------
# Step-size (Δ) initialization
# ---------------------------------------------------------------------------

def log_step_initializer(dt_min: float = 0.001, dt_max: float = 0.1):
    """Return an initializer that samples log(Δ) uniformly in [log dt_min, log dt_max]."""
    log_min = jnp.log(dt_min)
    log_max = jnp.log(dt_max)

    def init(key, shape):
        return random.uniform(key, shape) * (log_max - log_min) + log_min

    return init


def init_log_steps(key, spec):
    """Init array of per-state log-Δ. ``spec`` is ``(H, dt_min, dt_max)``."""
    H, dt_min, dt_max = spec
    init_fn = log_step_initializer(dt_min=dt_min, dt_max=dt_max)
    return jax.vmap(lambda k: init_fn(k, (1,)))(random.split(key, H))


def compute_inv_dt(key, features: int, dt_min: float, dt_max: float):
    """Sample Δ uniformly in log-space, then return softplus⁻¹(Δ).

    Used as the bias of the Δ-projection so that softplus(bias) lands inside
    the desired range at initialization, before any input contribution.
    """
    rand = random.uniform(key, (features,))
    log_dt = rand * (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)
    dt = jnp.clip(jnp.exp(log_dt), min=1e-4)
    return dt + jnp.log(-jnp.expm1(-dt))


def uniform_init(minval: float, maxval: float):
    def init(key, shape, dtype=jnp.float32):
        return random.uniform(key, shape, dtype, minval, maxval)
    return init


def inv_dt_bias_init(dt_min: float, dt_max: float):
    def init(key, shape, dtype=jnp.float32):
        return compute_inv_dt(key, shape[0], dt_min, dt_max)
    return init


# ---------------------------------------------------------------------------
# B and C initialization in the HiPPO eigen basis
# ---------------------------------------------------------------------------

def init_VinvB(init_fun, rng, shape, Vinv):
    """Init B in the HiPPO eigen basis: sample real B then return V⁻¹B as (re, im)."""
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    return jnp.stack([VinvB.real, VinvB.imag], axis=-1)


def init_CV(init_fun, rng, shape, V):
    """Init C in the HiPPO eigen basis: sample complex C then return CV as (re, im)."""
    C = init_fun(rng, shape)
    C_complex = C[..., 0] + 1j * C[..., 1]
    CV = C_complex @ V
    return jnp.stack([CV.real, CV.imag], axis=-1)


def trunc_standard_normal(key, shape):
    """Per-row truncated standard normal — matches S5's reference initializer."""
    H = shape[0]
    keys = random.split(key, H)
    return jax.vmap(lambda k: lecun_normal()(k, (1,) + shape[1:])[0])(keys)
