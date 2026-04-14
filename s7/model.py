"""Top-level S7 sequence models: encoder backbone + classification/retrieval heads."""

from __future__ import annotations

from typing import List

import jax
import jax.numpy as jnp
from flax import linen as nn

from s7.layers import SequenceStage


# ---------------------------------------------------------------------------
# Pooling helpers used by the classification head
# ---------------------------------------------------------------------------

def masked_meanpool(x, length):
    """Mean over the first ``length`` positions of an L×D sequence."""
    L = x.shape[0]
    mask = jnp.arange(L) < length
    return jnp.sum(mask[:, None] * x, axis=0) / length


def masked_timepool(x, length, integration_timesteps, eps: float = 1e-6):
    """Δt-weighted mean over the first ``length`` positions of an L×D sequence."""
    L = x.shape[0]
    mask = jnp.arange(L) < length
    weight = integration_timesteps[:, None] + eps
    total_t = jnp.sum(integration_timesteps)
    integral = jnp.sum(mask[:, None] * x * weight, axis=0)
    return integral / total_t


# ---------------------------------------------------------------------------
# Encoder backbone — embedding / linear projection + N stages
# ---------------------------------------------------------------------------

class StackedEncoder(nn.Module):
    """Embedding (or Dense) + N :class:`SequenceStage` blocks.

    Between stages, sequence length is divided by ``pooling_stride`` and the
    feature width is multiplied by ``state_expansion_factor``.
    """

    ssm: nn.Module
    discretization: str
    d_model: int
    d_ssm: int
    ssm_block_size: int
    num_stages: int
    num_layers_per_stage: int
    num_embeddings: int = 0
    dropout: float = 0.0
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_mode: str = "last"
    state_expansion_factor: int = 1
    encoder_type: str = "embed"

    def setup(self):
        if self.encoder_type == "embed":
            assert self.num_embeddings > 0, "num_embeddings required for embed encoder"
            self.encoder = nn.Embed(num_embeddings=self.num_embeddings, features=self.d_model)
        elif self.encoder_type == "dense":
            self.encoder = nn.Dense(features=self.d_model)
        else:
            raise ValueError(f"unknown encoder_type: {self.encoder_type}")

        stages: List[SequenceStage] = []
        d_in = self.d_model
        d_out = self.d_model
        d_ssm = self.d_ssm
        downsampling = 1

        for stage in range(self.num_stages):
            downsampling *= self.pooling_stride
            stages.append(
                SequenceStage(
                    ssm=self.ssm,
                    discretization=self.discretization,
                    d_model_in=d_in,
                    d_model_out=d_out,
                    d_ssm=d_ssm,
                    ssm_block_size=self.ssm_block_size,
                    layers_per_stage=self.num_layers_per_stage,
                    dropout=self.dropout,
                    prenorm=self.prenorm,
                    batchnorm=self.batchnorm,
                    bn_momentum=self.bn_momentum,
                    step_rescale=self.step_rescale,
                    pooling_stride=self.pooling_stride,
                    pooling_mode=self.pooling_mode,
                )
            )
            d_ssm = self.state_expansion_factor * d_ssm
            d_out = self.state_expansion_factor * d_in
            if stage > 0:
                d_in = self.state_expansion_factor * d_in

        self.stages = stages
        self.total_downsampling = downsampling

    def __call__(self, x, integration_timesteps, train: bool):
        x = self.encoder(x)
        for stage in self.stages:
            x, integration_timesteps = stage(x, integration_timesteps, train=train)
        return x, integration_timesteps


# ---------------------------------------------------------------------------
# Classification head (event/sequence classification)
# ---------------------------------------------------------------------------

class ClassificationModel(nn.Module):
    """Backbone + pool + Dense head for sequence classification."""

    ssm: nn.Module
    discretization: str
    num_classes: int
    d_model: int
    d_ssm: int
    ssm_block_size: int
    num_stages: int
    num_layers_per_stage: int
    num_embeddings: int = 0
    dropout: float = 0.2
    classification_mode: str = "pool"      # one of: pool, timepool, last, none
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_mode: str = "last"
    state_expansion_factor: int = 1
    encoder_type: str = "embed"

    def setup(self):
        self.encoder = StackedEncoder(
            ssm=self.ssm,
            discretization=self.discretization,
            d_model=self.d_model,
            d_ssm=self.d_ssm,
            ssm_block_size=self.ssm_block_size,
            num_stages=self.num_stages,
            num_layers_per_stage=self.num_layers_per_stage,
            num_embeddings=self.num_embeddings,
            dropout=self.dropout,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            pooling_stride=self.pooling_stride,
            pooling_mode=self.pooling_mode,
            state_expansion_factor=self.state_expansion_factor,
            encoder_type=self.encoder_type,
        )
        self.decoder = nn.Dense(self.num_classes)

    def __call__(self, x, integration_timesteps, length, train: bool = True):
        length = length // self.encoder.total_downsampling

        x, integration_timesteps = self.encoder(x, integration_timesteps, train=train)

        if self.classification_mode == "pool":
            x = masked_meanpool(x, length)
        elif self.classification_mode == "timepool":
            x = masked_timepool(x, length, integration_timesteps)
        elif self.classification_mode == "last":
            x = x[-1]
        elif self.classification_mode == "none":
            pass
        else:
            raise ValueError(
                f"classification_mode must be one of pool|timepool|last|none, got {self.classification_mode!r}"
            )

        return self.decoder(x)


BatchClassificationModel = nn.vmap(
    ClassificationModel,
    in_axes=(0, 0, 0, None),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "batch_stats": None,
                   "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
    axis_name="batch",
)


# ---------------------------------------------------------------------------
# Retrieval head (LRA Retrieval / AAN)
# ---------------------------------------------------------------------------

class _RetrievalDecoder(nn.Module):
    d_model: int
    d_output: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model)(x)
        x = nn.gelu(x)
        return nn.Dense(self.d_output)(x)


class RetrievalModel(nn.Module):
    """Document-pair classification (LRA Retrieval): encode each document
    with the backbone, mean-pool, then feed [a, b, a-b, a*b] to a 2-layer MLP.

    Unlike :class:`ClassificationModel` this module operates directly on a
    *batch* — the inner backbone is vmapped, not the outer model.
    """

    ssm: nn.Module
    discretization: str
    num_classes: int
    d_model: int
    d_ssm: int
    ssm_block_size: int
    num_stages: int
    num_layers_per_stage: int
    num_embeddings: int = 0
    dropout: float = 0.2
    classification_mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_mode: str = "last"
    state_expansion_factor: int = 1
    encoder_type: str = "embed"

    def setup(self):
        BatchEncoder = nn.vmap(
            StackedEncoder,
            in_axes=(0, 0, None),
            out_axes=(0, 0),
            variable_axes={"params": None, "dropout": None, "batch_stats": None,
                           "cache": 0, "prime": None},
            split_rngs={"params": False, "dropout": True},
            axis_name="batch",
        )
        self.encoder = BatchEncoder(
            ssm=self.ssm,
            discretization=self.discretization,
            d_model=self.d_model,
            d_ssm=self.d_ssm,
            ssm_block_size=self.ssm_block_size,
            num_stages=self.num_stages,
            num_layers_per_stage=self.num_layers_per_stage,
            num_embeddings=self.num_embeddings,
            dropout=self.dropout,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            pooling_stride=self.pooling_stride,
            pooling_mode=self.pooling_mode,
            state_expansion_factor=self.state_expansion_factor,
            encoder_type=self.encoder_type,
        )
        BatchDecoder = nn.vmap(
            _RetrievalDecoder, in_axes=0, out_axes=0,
            variable_axes={"params": None}, split_rngs={"params": False},
        )
        self.decoder = BatchDecoder(d_model=self.d_model, d_output=self.num_classes)

    def __call__(self, x, integration_timesteps, length, train):
        length = length // self.encoder.total_downsampling
        x, _ = self.encoder(x, integration_timesteps, train)
        if self.classification_mode == "pool":
            x = jax.vmap(masked_meanpool)(x, length)
        elif self.classification_mode == "last":
            x = x[:, -1]
        else:
            raise ValueError(f"unsupported classification_mode for retrieval: {self.classification_mode}")

        a, b = jnp.split(x, 2, axis=0)
        features = jnp.concatenate([a, b, a - b, a * b], axis=-1)
        return self.decoder(features)
