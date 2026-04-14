"""Building blocks that wrap an :class:`s7.ssm.S7Layer` into a residual block.

A :class:`SequenceLayer` is one residual block: ``norm → SSM → GELU+gated dense →
dropout → +skip → optional event pooling``.

A :class:`SequenceStage` chains ``layers_per_stage`` blocks at the same
resolution; the *first* block in a stage may also pool to reduce sequence
length and/or change the feature width.
"""

from __future__ import annotations

from functools import partial

from flax import linen as nn


# ---------------------------------------------------------------------------
# Event pooling — used between stages to downsample event streams
# ---------------------------------------------------------------------------

class EventPooling(nn.Module):
    """Downsample an event sequence by an integer stride.

    Three modes:
        ``"last"``     — drop in-between events, keep one per stride window.
        ``"avgpool"``  — uniform mean within each window.
        ``"timepool"`` — Δt-weighted mean within each window (the default for S7).

    The integration timesteps are summed within each window so the downstream
    SSM still sees a consistent total time budget.
    """

    stride: int = 1
    mode: str = "last"
    eps: float = 1e-6

    def __call__(self, x, integration_timesteps):
        if self.stride <= 1:
            return x, integration_timesteps

        L = (integration_timesteps.shape[0] // self.stride) * self.stride
        x = x[:L]
        dt = integration_timesteps[:L]
        new_dt = dt.reshape(-1, self.stride).sum(axis=1)
        d_model = x.shape[-1]

        if self.mode == "last":
            return x[::self.stride], new_dt
        if self.mode == "avgpool":
            return x.reshape(-1, self.stride, d_model).mean(axis=1), new_dt
        if self.mode == "timepool":
            weight = dt[:, None] + self.eps
            num = (x * weight).reshape(-1, self.stride, d_model).sum(axis=1)
            den = weight.reshape(-1, self.stride, 1).sum(axis=1)
            return num / den, new_dt
        raise ValueError(f"unknown pooling mode: {self.mode}")


# ---------------------------------------------------------------------------
# Single residual block
# ---------------------------------------------------------------------------

class SequenceLayer(nn.Module):
    """Residual block wrapping a single S7 SSM call."""

    ssm: nn.Module                # partial(S7Layer, ...)  — see s7.ssm.build_s7
    discretization: str
    dropout: float
    d_model_in: int
    d_model_out: int
    d_ssm: int
    block_size: int
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_mode: str = "last"

    def _norm(self):
        if self.batchnorm:
            return nn.BatchNorm(momentum=self.bn_momentum, axis_name="batch")
        return nn.LayerNorm()

    @nn.compact
    def __call__(self, x, integration_timesteps=None, train: bool = True):
        skip = x

        if self.prenorm:
            n = self._norm()
            x = n(x, use_running_average=not train) if self.batchnorm else n(x)

        x = self.ssm(
            H_in=self.d_model_in,
            H_out=self.d_model_out,
            P=self.d_ssm,
            block_size=self.block_size,
            step_rescale=self.step_rescale,
            discretization=self.discretization,
        )(x, integration_timesteps)

        # Gated GELU MLP
        x1 = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not train)(nn.gelu(x))
        x1 = nn.Dense(self.d_model_out)(x1)
        x = x * nn.sigmoid(x1)
        x = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not train)(x)

        if self.d_model_in != self.d_model_out:
            skip = nn.Dense(self.d_model_out)(skip)
        x = x + skip

        if self.pooling_stride > 1:
            x, integration_timesteps = EventPooling(
                stride=self.pooling_stride, mode=self.pooling_mode,
            )(x, integration_timesteps)

        if not self.prenorm:
            n = self._norm()
            x = n(x, use_running_average=not train) if self.batchnorm else n(x)

        return x, integration_timesteps


# ---------------------------------------------------------------------------
# Stage — N residual blocks at the same resolution
# ---------------------------------------------------------------------------

class SequenceStage(nn.Module):
    """N stacked :class:`SequenceLayer` blocks. Pooling/widening only on block 0."""

    ssm: nn.Module
    discretization: str
    d_model_in: int
    d_model_out: int
    d_ssm: int
    ssm_block_size: int
    layers_per_stage: int
    dropout: float = 0.0
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_mode: str = "last"

    @nn.compact
    def __call__(self, x, integration_timesteps, train: bool = True):
        block = partial(
            SequenceLayer,
            ssm=self.ssm,
            discretization=self.discretization,
            dropout=self.dropout,
            d_ssm=self.d_ssm,
            block_size=self.ssm_block_size,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )

        # First block: optional pooling + width change.
        x, integration_timesteps = block(
            d_model_in=self.d_model_in,
            d_model_out=self.d_model_out,
            pooling_stride=self.pooling_stride,
            pooling_mode=self.pooling_mode,
        )(x, integration_timesteps, train=train)

        # Subsequent blocks: identity width, no pooling.
        for _ in range(self.layers_per_stage - 1):
            x, integration_timesteps = block(
                d_model_in=self.d_model_out,
                d_model_out=self.d_model_out,
                pooling_stride=1,
            )(x, integration_timesteps, train=train)

        return x, integration_timesteps
