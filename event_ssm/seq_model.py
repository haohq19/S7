import jax
import jax.numpy as np
from flax import linen as nn
from .layers import SequenceStage


class StackedEncoderModel(nn.Module):
    """
    Defines a stack of S7 layers to be used as an encoder.

    :param ssm: the SSM to be used (i.e. S7 ssm)
    :param discretization: the discretization to be used for the SSM
    :param d_model: the feature size of the layer inputs and outputs. We usually refer to this size as H
    :param d_ssm: the size of the state space model. We usually refer to this size as P
    :param ssm_block_size: the block size of the state space model
    :param num_stages: the number of S7 layers to stack
    :param num_layers_per_stage: the number of EventSSM layers to stack
    :param num_embeddings: the number of embeddings to use
    :param dropout: dropout rate
    :param prenorm: whether to use layernorm before the module or after it
    :param batchnorm: If True, use batchnorm instead of layernorm
    :param bn_momentum: momentum for batchnorm
    :param step_rescale: rescale the integration timesteps by this factor
    :param pooling_stride: stride for subsampling
    :param pooling_every_n_layers: pool every n layers
    :param pooling_mode: pooling mode (last, avgpool, timepool)
    :param state_expansion_factor: factor to expand the state space model
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
    pooling_every_n_layers: int = 1
    pooling_mode: str = "last"
    state_expansion_factor: int = 1
    encoder_type: str = "embed"


    def setup(self):
        """
        Initializes a linear encoder and the stack of EventSSM layers.
        """
        assert self.num_embeddings > 0

        if self.encoder_type == "embed":
            print("Using embed encoder")
            self.encoder = nn.Embed(num_embeddings=self.num_embeddings, features=self.d_model)
        elif self.encoder_type == "dense":
            print("Using dense encoder")
            self.encoder = nn.Dense(features=self.d_model)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        # generate strides for the model
        stages = []
        d_model_in = self.d_model
        d_model_out = self.d_model
        d_ssm = self.d_ssm
        total_downsampling = 1
        for stage in range(self.num_stages):
            # pool from the first layer but don't expand the state dim for the first layer
            total_downsampling *= self.pooling_stride

            stages.append(
                SequenceStage(
                    ssm=self.ssm,
                    discretization=self.discretization,
                    d_model_in=d_model_in,
                    d_model_out=d_model_out,
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
            d_model_out = self.state_expansion_factor * d_model_in

            if stage > 0:
                d_model_in = self.state_expansion_factor * d_model_in

        self.stages = stages
        self.total_downsampling = total_downsampling

    def __call__(self, x, integration_timesteps, train: bool):

        # Encode input
        x = self.encoder(x)
        print("encoded input shape = ", x.shape)

        # Pass through stages
        for i, stage in enumerate(self.stages):
            #Apply layer SSM
            x, integration_timesteps = stage(
                x, integration_timesteps, train=train
            )


        return x, integration_timesteps



def masked_meanpool(x, lengths):
    """
    Helper function to perform mean pooling across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length.

    :param x: input sequence (L, d_model)
    :param lengths: the original length of the sequence before padding
    :return: mean pooled output sequence (d_model)
    """
    L = x.shape[0]
    mask = np.arange(L) < lengths
    return np.sum(mask[..., None]*x, axis=0)/lengths


## ??? What is the purpose of timepool ??? 
def timepool(x, integration_timesteps):
    """
    Helper function to perform weighted mean across the sequence length.
    Means are weighted with the integration time steps

    :param x: input sequence (L, d_model)
    :param integration_timesteps: the integration timesteps for the SSM
    :return: time pooled output sequence (d_model)
    """
    T = np.sum(integration_timesteps, axis=0)
    integral = np.sum(x * integration_timesteps[..., None], axis=0)
    return integral / T


def masked_timepool(x, lengths, integration_timesteps, eps=1e-6):
    """
    Helper function to perform weighted mean across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length. Means are weighted with the integration time steps

    :param x: input sequence (L, d_model)
    :param lengths: the original length of the sequence before padding
    :param integration_timesteps: the integration timesteps for the SSM
    :param eps: small value to avoid division by zero
    :return: time pooled output sequence (d_model)
    """
    L = x.shape[0]
    mask = np.arange(L) < lengths
    T = np.sum(integration_timesteps)

    # integrate with time weighting
    weight = integration_timesteps[..., None] + eps
    integral = np.sum(mask[..., None] * x * weight, axis=0)
    return integral / T


# Here we call vmap to parallelize across a batch of input sequences
batch_masked_meanpool = jax.vmap(masked_meanpool)


class ClassificationModel(nn.Module):
    """
    EventSSM classificaton sequence model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S7 layers), mean pooling
    across the sequence length, a linear decoder, and a softmax operation.

    :param ssm: the SSM to be used (i.e. S7 ssm)
    :param discretization: the discretization to be used for the SSM (zoh, dirac, async)
    :param num_classes: the number of classes for the classification task
    :param d_model: the feature size of the layer inputs and outputs. We usually refer to this size as H
    :param d_ssm: the size of the state space model. We usually refer to this size as P
    :param ssm_block_size: the block size of the state space model
    :param num_stages: the number of S7 layers to stack
    :param num_layers_per_stage: the number of EventSSM layers to stack
    :param num_embeddings: the number of embeddings to use
    :param dropout: dropout rate
    :param classification_mode: the classification mode (pool, timepool, last)
    :param prenorm: whether to use layernorm before the module or after it
    :param batchnorm: If True, use batchnorm instead of layernorm
    :param bn_momentum: momentum for batchnorm
    :param step_rescale: rescale the integration timesteps by this factor
    :param pooling_stride: stride for subsampling
    :param pooling_every_n_layers: pool every n layers
    :param pooling_mode: pooling mode (last, avgpool, timepool)
    :param state_expansion_factor: factor to expand the state space model
    :param encoder_type: type of encoder to use on the inputs (embed or dense)
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
    pooling_every_n_layers: int = 1
    pooling_mode: str = "last"
    state_expansion_factor: int = 1
    encoder_type: str = "embed"

    

    def setup(self):
        """
        Initializes the stacked EventSSM encoder and a linear decoder.
        """
        self.encoder = StackedEncoderModel(
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
            pooling_every_n_layers=self.pooling_every_n_layers,
            pooling_mode=self.pooling_mode,
            state_expansion_factor=self.state_expansion_factor,
            encoder_type=self.encoder_type,

        )
        self.decoder = nn.Dense(self.num_classes)



    def __call__(self, x, integration_timesteps, length, train=True):
        """
        Compute the size num_classes log softmax output given a
        Lxd_input input sequence.

        :param x: input sequence (L, d_input)
        :param integration_timesteps: the integration timesteps for the SSM
        :param length: the original length of the sequence before padding
        :param train: If True, applies dropout and batch norm from batch statistics

        :return: output (num_classes)
        """

        # if the sequence is downsampled we need to adjust the length
        length = length // self.encoder.total_downsampling
        
        # run encoder backbone
        x, integration_timesteps = self.encoder(x, integration_timesteps, train=train)
        
        if self.classification_mode in ["pool"]:
            # Perform mean pooling across time
            x = masked_meanpool(x, length)

        elif self.classification_mode in ["timepool"]:
            # Perform mean pooling across time weighted by integration time steps
            x = masked_timepool(x, length, integration_timesteps)

        elif self.classification_mode in ["last"]:
            # Just take the last state
            x = x[-1]
        elif self.classification_mode in ["none"]:
            # Just take the last state
            x = x
        else:            
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)

        return x 


# Here we call vmap to parallelize across a batch of input sequences
BatchClassificationModel = nn.vmap(
    ClassificationModel,
    in_axes=(0, 0, 0, None),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, 
    axis_name='batch'
)

## Retrieval Model

# For Document matching task (e.g. AAN)
class RetrievalDecoder(nn.Module):
    """
    Defines the decoder to be used for document matching tasks,
    e.g. the AAN task. This is defined as in the S4 paper where we apply
    an MLP to a set of 4 features. The features are computed as described in
    Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
    Args:
        d_output    (int32):    the output dimension, i.e. the number of classes
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                    we usually refer to this size as H
    """
    d_model: int
    d_output: int

    def setup(self):
        """
        Initializes 2 dense layers to be used for the MLP.
        """
        self.layer1 = nn.Dense(self.d_model)
        self.layer2 = nn.Dense(self.d_output)

    def __call__(self, x):
        """
        Computes the input to be used for the softmax function given a set of
        4 features. Note this function operates directly on the batch size.
        Args:
             x (float32): features (bsz, 4*d_model)
        Returns:
            output (float32): (bsz, d_output)
        """
        x = self.layer1(x)
        x = nn.gelu(x)
        return self.layer2(x)

class RetrievalModel(nn.Module):
    """ S7 Retrieval classification model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S7 layers), mean pooling
    across the sequence length, constructing 4 features which are fed into a MLP,
    and a softmax operation. Note that unlike the standard classification model above,
    the apply function of this model operates directly on the batch of data (instead of calling
    vmap on this model).
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
    pooling_every_n_layers: int = 1
    pooling_mode: str = "last"
    state_expansion_factor: int = 1

    def setup(self):
        """
        Initializes the S7 stacked encoder and the retrieval decoder. Note that here we
        vmap over the stacked encoder model to work well with the retrieval decoder that
        operates directly on the batch.
        """
        BatchEncoderModel = nn.vmap(
            StackedEncoderModel,
            in_axes=(0, 0, None),
            out_axes=(0, 0),
            variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
            split_rngs={"params": False, "dropout": True}, axis_name='batch'

        )
    
        self.encoder = BatchEncoderModel(
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
            pooling_every_n_layers=self.pooling_every_n_layers,
            pooling_mode=self.pooling_mode,
            state_expansion_factor=self.state_expansion_factor,
        )

        BatchRetrievalDecoder = nn.vmap(
            RetrievalDecoder,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )

        self.decoder = BatchRetrievalDecoder(
                                d_model=self.d_model,
                                d_output=self.num_classes
                                          )
    
    def __call__(self, x, integration_timesteps, length, train):
        """
        Compute the size num_classes log softmax output given a
        Lxd_input input sequence.

        :param x: input sequence (L, d_input)
        :param integration_timesteps: the integration timesteps for the SSM
        :param length: the original length of the sequence before padding
        :param train: If True, applies dropout and batch norm from batch statistics

        :return: output (num_classes)
        """
        
        length = length // self.encoder.total_downsampling
        
        x, integration_timesteps = self.encoder(x, integration_timesteps, train)
        # Apply classification head
        if self.classification_mode in ["pool"]:
            # Perform mean pooling across time
            x = batch_masked_meanpool(x, length)

        elif self.classification_mode in ["last"]:
            # Just take the last state
            x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")
        x0, x1 = np.split(x, 2)
        features = np.concatenate([x0, x1, x0-x1, x0*x1], axis=-1)
        x = self.decoder(features)
        return x
    


