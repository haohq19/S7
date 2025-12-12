import jax
import flax.linen as nn
from functools import partial
from typing import Tuple




class EventPooling(nn.Module):
    """
    Subsampling layer for event sequences.
    """
    stride: int = 1
    mode: str = "last"
    eps: float = 1e-6

    def __call__(self, x, integration_timesteps):
        """
        Compute the pooled (L/stride)xH output given an LxH input.
        :param x: input sequence (L, d_model)
        :param integration_timesteps: the integration timesteps for the SSM
        :return: output sequence (L/stride, d_model)
        """
        if self.stride == 1:
            raise ValueError("Stride 1 not supported for pooling")

        else:
            remaining_timesteps = (len(integration_timesteps) // self.stride) * self.stride
            new_integration_timesteps = integration_timesteps[:remaining_timesteps].reshape(-1, self.stride).sum(axis=1)
            x = x[:remaining_timesteps]
            d_model = x.shape[-1]

            if self.mode == 'last':
                x = x[::self.stride]
                return x, new_integration_timesteps
            elif self.mode == 'avgpool':
                x = x.reshape(-1, self.stride, d_model).mean(axis=1)
                return x, new_integration_timesteps
            elif self.mode == 'timepool':
                weight = integration_timesteps[:remaining_timesteps, None] + self.eps
                x = (x * weight).reshape(-1, self.stride, d_model).sum(axis=1)
                x = x / weight.reshape(-1, self.stride, 1).sum(axis=1)
                return x, new_integration_timesteps
            else:
                raise NotImplementedError("Pooling mode: {} not implemented".format(self.stride))



class SequenceLayer(nn.Module):
    """
    Defines a single event-ssm layer, with SSM or sLSTM, nonlinearity,
    dropout, batch/layer norm, etc.
    """
    ssm: nn.Module
    discretization: str
    dropout: float
    d_model_in: int
    d_model_out: int
    d_ssm: int
    block_size: int
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_mode: str = "last"


    @nn.compact
    def __call__(self, x, integration_timesteps=None, train: bool = True):
        skip = x

        if self.prenorm:
            norm = nn.BatchNorm(momentum=self.bn_momentum, axis_name='batch') if self.batchnorm else nn.LayerNorm()
            x = norm(x, use_running_average=not train) if self.batchnorm else norm(x)

        x = self.ssm(
                H_in=self.d_model_in,
                H_out=self.d_model_out,
                P=self.d_ssm,
                block_size=self.block_size,
                step_rescale=self.step_rescale,
                discretization=self.discretization,
                stride=self.pooling_stride,
                pooling_mode=self.pooling_mode,
            )(x, integration_timesteps)

            
        # Non-linear activation function
        x1 = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not train)(nn.gelu(x))
        x1 = nn.Dense(self.d_model_out)(x1)
        x = x * nn.sigmoid(x1)
        x = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not train)(x)

        if self.d_model_in != self.d_model_out:
            skip = nn.Dense(self.d_model_out)(skip)

        x = x + skip

        if self.pooling_stride > 1:
            pool = EventPooling(stride=self.pooling_stride, mode=self.pooling_mode)
            x, integration_timesteps = pool(x, integration_timesteps)

        if not self.prenorm:
            norm = nn.BatchNorm(momentum=self.bn_momentum, axis_name='batch') if self.batchnorm else nn.LayerNorm()
            x = norm(x, use_running_average=not train) if self.batchnorm else norm(x)

        return x, integration_timesteps
    


class SequenceStage(nn.Module):
    """
    Defines a block of EventSSM layers with the same hidden size and event-resolution.
    """
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
    state_expansion_factor: int = 1


    @nn.compact
    def __call__(self, x, integration_timesteps, train: bool = True):

        EventSSMLayer = partial(
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

        # First layer with pooling
        x, integration_timesteps = EventSSMLayer(
            d_model_in=self.d_model_in,
            d_model_out=self.d_model_out,
            pooling_stride=self.pooling_stride,
            pooling_mode=self.pooling_mode,
        )(x, integration_timesteps, train=train)

        # Further layers without pooling
        for l in range(self.layers_per_stage - 1):
            x, integration_timesteps = EventSSMLayer(
                d_model_in=self.d_model_out,
                d_model_out=self.d_model_out,
                pooling_stride=1,
            )(x, integration_timesteps, train=train)

        return x, integration_timesteps
