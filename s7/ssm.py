"""S7 selective state space model (Flax module).

The S7 layer is a diagonal complex-Λ SSM (S5-style) with a *selective* time
step Δ — i.e. Δ is computed from the input via a small projection. The
underlying Λ, B, C parameters are LTI; selectivity enters only through Δ(u).

Per the S7 paper (arxiv 2410.03464) the discretized matrices Λ̄ and B̄ are
functions of u because they are functions of Δ(u), even though Λ and B
themselves are fixed parameters. This is the "Simplified" half of S7 vs. Mamba.

Math (zero-order hold, unit time step):
    Λ̄ₖ      = exp(Λ · Δₖ · Δtₖ)
    γ̄ₖ      = Λ⁻¹ · (Λ̄ₖ - I)            (or its event-time variant — see below)
    xₖ       = Λ̄ₖ · xₖ₋₁ + γ̄ₖ · (B uₖ)
    yₖ       = Re(C xₖ) + D uₖ

For event streams S7 introduces an *async* discretization where the discrete
state-transition uses the inter-event interval Δtₖ but the input gain γ̄ uses
only Δₖ — see :func:`discretize_async`.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import glorot_normal, lecun_normal, normal
from jax.scipy.linalg import block_diag

from s7.init import (
    init_CV,
    init_VinvB,
    init_log_steps,
    inv_dt_bias_init,
    make_dplr_hippo,
    trunc_standard_normal,
    uniform_init,
)


# ---------------------------------------------------------------------------
# Discretization rules
# ---------------------------------------------------------------------------

def discretize_zoh(Lambda, step_delta, time_delta):
    """Zero-order hold (the standard S5/S4 discretization).

    ``time_delta`` is collapsed into ``step_delta`` (i.e. the effective Δ is
    ``step_delta * time_delta``), producing both Λ̄ and γ̄ from the same Δ.
    """
    Delta = step_delta * time_delta
    Lambda_bar = jnp.exp(Lambda * Delta)
    gamma_bar = (1 / Lambda) * (Lambda_bar - 1.0)
    return Lambda_bar, gamma_bar


def discretize_dirac(Lambda, step_delta, time_delta):
    """Dirac-input discretization: γ̄ = 1, Λ̄ = exp(Λ·Δ·Δt)."""
    Lambda_bar = jnp.exp(Lambda * step_delta * time_delta)
    return Lambda_bar, jnp.asarray(1.0)


def discretize_async(Lambda, step_delta, time_delta):
    """Asynchronous (event-time) discretization for irregularly sampled streams.

    Λ̄ uses the inter-event interval Δtₖ (so the state decays for as long as the
    physical wait between events), but the input gain γ̄ uses only the learned
    Δ — modeling each event as a unit Dirac whose *integration window* is set
    by Δ alone. This is paper Eq. 12.
    """
    Lambda_bar = jnp.exp(Lambda * step_delta * time_delta)
    gamma_bar = (1 / Lambda) * (jnp.exp(Lambda * step_delta) - 1.0)
    return Lambda_bar, gamma_bar


_DISCRETIZE_FNS = {
    "zoh": discretize_zoh,
    "dirac": discretize_dirac,
    "async": discretize_async,
}


# ---------------------------------------------------------------------------
# Parallel scan
# ---------------------------------------------------------------------------

@jax.vmap
def _scan_op(q_i, q_j):
    """Binary op for the linear-recurrence parallel scan with diagonal Λ."""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, Bu, C_tilde, conj_sym: bool, bidirectional: bool = False):
    """Run the recurrence xₖ = Λ̄ₖ xₖ₋₁ + (Bu)ₖ via parallel scan, then project to y."""
    _, xs = jax.lax.associative_scan(_scan_op, (Lambda_bar, Bu))
    if bidirectional:
        _, xs2 = jax.lax.associative_scan(_scan_op, (Lambda_bar, Bu), reverse=True)
        xs = jnp.concatenate((xs, xs2), axis=-1)

    project = (lambda x: 2 * (C_tilde @ x).real) if conj_sym else (lambda x: (C_tilde @ x).real)
    return jax.vmap(project)(xs)


# ---------------------------------------------------------------------------
# Δ projection (the "Selective" head of S7)
# ---------------------------------------------------------------------------

class DeltaProj(nn.Module):
    """Linear map  uₖ ↦ pre-softplus Δₖ ∈ ℝ^P_local.

    Param names ``step_proj_kernel`` / ``step_proj_bias`` and the wrapping
    submodule name ``step_proj`` are kept identical to the legacy code so
    Flax derives bit-identical init RNGs and any old checkpoint loads cleanly.
    """
    features: int
    kernel_init: Callable
    bias_init: Callable

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param("step_proj_kernel", self.kernel_init,
                            (inputs.shape[-1], self.features))
        bias = self.param("step_proj_bias", self.bias_init, (self.features,))
        return inputs @ kernel + bias


# ---------------------------------------------------------------------------
# The S7 layer itself
# ---------------------------------------------------------------------------

class S7(nn.Module):
    """A single selective S7 SSM layer (Λ-only selectivity, diagonal complex Λ).

    Shape conventions:
        H_in   input feature dim
        H_out  output feature dim (typically == H_in)
        P      complex state dim (block-diagonal HiPPO init, ``P // block_size`` blocks)
        local_P P/2 if conj_sym else P  (we exploit the eigenpair symmetry of HiPPO-LegS)
    """

    H_in: int
    H_out: int
    P: int
    block_size: int
    discretization: str
    dt_min: float
    dt_max: float
    bidirectional: bool
    C_init: str = "lecun_normal"
    conj_sym: bool = True
    clip_eigs: bool = False
    step_rescale: float = 1.0
    input_dependent: bool = True
    log_a: bool = False
    stablessm_a: bool = False
    a_scale: float = 1.0
    b_offset: float = 0.5

    def setup(self):
        assert not (self.log_a and self.stablessm_a), \
            "log_a and stablessm_a are mutually exclusive A-reparameterizations."
        assert self.discretization in _DISCRETIZE_FNS, \
            f"unknown discretization '{self.discretization}'"

        # ---- HiPPO-LegS Λ in DPLR form -----------------------------------
        Lambda_full, _, _, V_full, _ = make_dplr_hippo(self.block_size)

        num_blocks = self.P // self.block_size
        block_size = self.block_size // 2 if self.conj_sym else self.block_size
        local_P = self.P // 2 if self.conj_sym else self.P
        self.local_P = local_P

        Lambda = Lambda_full[:block_size]
        V = V_full[:, :block_size]
        V_inv = V.conj().T

        # block-diagonal repetition of the HiPPO eigenstructure
        Lambda = jnp.tile(Lambda, num_blocks)
        V = block_diag(*([V] * num_blocks))
        V_inv = block_diag(*([V_inv] * num_blocks))

        if self.log_a:
            Lambda = jnp.log1p(Lambda)
        elif self.stablessm_a:
            Lambda = jnp.sqrt((-1 / Lambda - self.b_offset) / self.a_scale)

        # ---- Diagonal Λ as a learnable complex parameter -----------------
        self.Lambda_re = self.param("Lambda_re", lambda *_: Lambda.real, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda *_: Lambda.imag, (None,))

        # ---- Selective Δ projection (only learnable when input_dependent) -
        if self.input_dependent:
            dt_rank = math.ceil(self.H_in / 16)
            init_std = dt_rank ** -0.5 * self.step_rescale
            self.delta_proj = DeltaProj(
                features=local_P,
                kernel_init=uniform_init(-init_std, init_std),
                bias_init=inv_dt_bias_init(self.dt_min, self.dt_max),
                name="step_proj",
            )

        # ---- B (input matrix) — fixed Flax param in V_inv-rotated basis --
        self.B_param = self.param(
            "B",
            lambda rng, shape: init_VinvB(lecun_normal(), rng, shape, V_inv),
            (self.P, self.H_in),
        )

        # ---- C (output matrix) — selectable init scheme ------------------
        c_local = 2 * local_P if self.bidirectional else local_P
        if self.C_init == "complex_normal":
            self.C_param = self.param(
                "C", normal(stddev=0.5 ** 0.5), (self.H_out, c_local, 2)
            )
        else:
            init_fn = trunc_standard_normal if self.C_init == "trunc_standard_normal" else lecun_normal()
            if self.bidirectional:
                self.C1_param = self.param(
                    "C1", lambda rng, shape: init_CV(init_fn, rng, shape, V),
                    (self.H_out, local_P, 2),
                )
                self.C2_param = self.param(
                    "C2", lambda rng, shape: init_CV(init_fn, rng, shape, V),
                    (self.H_out, local_P, 2),
                )
            else:
                self.C_param = self.param(
                    "C", lambda rng, shape: init_CV(init_fn, rng, shape, V),
                    (self.H_out, local_P, 2),
                )

        # ---- Per-state log Δ used when input_dependent=False -------------
        self.log_step = self.param(
            "log_step", init_log_steps, (local_P, self.dt_min, self.dt_max)
        )

        # ---- D (feedthrough) ---------------------------------------------
        if self.H_in == self.H_out:
            self.D = self.param("D", normal(stddev=1.0), (self.H_in,))
        else:
            self.D = self.param("D", glorot_normal(), (self.H_out, self.H_in))

        self._discretize_fn = _DISCRETIZE_FNS[self.discretization]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_lambda(self):
        if self.clip_eigs:
            Lambda_re = jnp.clip(self.Lambda_re, None, -1e-4)
        else:
            Lambda_re = self.Lambda_re
        Lambda = Lambda_re + 1j * self.Lambda_im
        if self.log_a:
            return -jnp.exp(Lambda)
        if self.stablessm_a:
            return -jnp.sqrt((-1 - self.b_offset * Lambda) / (self.a_scale * Lambda))
        return Lambda

    def _resolve_C(self):
        if self.C_init == "complex_normal":
            C = self.C_param
            return C[..., 0] + 1j * C[..., 1]
        if self.bidirectional:
            C1 = self.C1_param[..., 0] + 1j * self.C1_param[..., 1]
            C2 = self.C2_param[..., 0] + 1j * self.C2_param[..., 1]
            return jnp.concatenate((C1, C2), axis=-1)
        return self.C_param[..., 0] + 1j * self.C_param[..., 1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(self, u, integration_timesteps):
        """Run the SSM on a single sequence.

        :param u: (L, H_in) — input sequence (already embedded).
        :param integration_timesteps: (L,) — Δt between consecutive events.
        :return: (L, H_out)
        """
        Lambda = self._resolve_lambda()
        B = self.B_param[..., 0] + 1j * self.B_param[..., 1]
        C = self._resolve_C()

        # Fuse discretization and Bu projection into a single vmap closure so
        # XLA can reuse buffers instead of materializing gamma_bar for the full
        # sequence — matters at DVS128's L ≈ 5·10⁵ where that tensor is ~260 MB.
        if self.input_dependent:
            step = jax.nn.softplus(self.delta_proj(u))  # (L, local_P)

            def step_fn(u_t, dt, step_t):
                Lambda_bar, gamma_bar = self._discretize_fn(
                    Lambda, self.step_rescale * step_t, dt
                )
                return Lambda_bar, gamma_bar * (B @ u_t)

            Lambda_bar, Bu = jax.vmap(step_fn)(u, integration_timesteps, step)
        else:
            log_step = jnp.exp(self.log_step[:, 0])

            def step_fn(u_t, dt):
                Lambda_bar, gamma_bar = self._discretize_fn(
                    Lambda, self.step_rescale * log_step, dt
                )
                return Lambda_bar, gamma_bar * (B @ u_t)

            Lambda_bar, Bu = jax.vmap(step_fn)(u, integration_timesteps)

        y = apply_ssm(Lambda_bar, Bu, C, conj_sym=self.conj_sym, bidirectional=self.bidirectional)

        if self.H_in == self.H_out:
            Du = jax.vmap(lambda x: self.D * x)(u)
        else:
            Du = jax.vmap(lambda x: self.D @ x)(u)
        return y + Du


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def build_s7(
    *,
    C_init: str,
    dt_min: float,
    dt_max: float,
    conj_sym: bool,
    clip_eigs: bool,
    log_a: bool,
    stablessm_a: bool,
    input_dependent: bool,
    bidirectional: bool,
    a: float = 1.0,
    b: float = 0.5,
):
    """Return a partially-applied :class:`S7` ready for ``layers.SequenceLayer``.

    The remaining shape arguments (``H_in``, ``H_out``, ``P``, …) are filled in
    by the surrounding ``SequenceLayer``.
    """
    return partial(
        S7,
        C_init=C_init,
        dt_min=dt_min,
        dt_max=dt_max,
        conj_sym=conj_sym,
        clip_eigs=clip_eigs,
        log_a=log_a,
        stablessm_a=stablessm_a,
        input_dependent=input_dependent,
        bidirectional=bidirectional,
        a_scale=a,
        b_offset=b,
    )
