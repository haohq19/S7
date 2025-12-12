from functools import partial
import jax
import jax.numpy as np
from jax.scipy.linalg import block_diag

from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal, glorot_normal

from .ssm_init import * 

from .layers import EventPooling
import math
from typing import Any, Callable, Sequence


def discretize_zoh(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    This is the default discretization method used by many SSM works including S5.

    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Delta = step_delta * time_delta
    Lambda_bar = np.exp(Lambda * Delta)
    gamma_bar = (1/Lambda * (Lambda_bar-Identity))
    return Lambda_bar, gamma_bar


def discretize_dirac(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    with dirac delta input spikes.
    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Delta = step_delta * time_delta
    Lambda_bar = np.exp(Lambda * Delta)
    gamma_bar = 1.0
    return Lambda_bar, gamma_bar


def discretize_async(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    with dirac delta input spikes and appropriate input normalization.

    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Identity = np.ones_like(Lambda)
    Lambda_bar = np.exp(Lambda * step_delta * time_delta)
    gamma_bar = (1/Lambda * (np.exp(Lambda * step_delta)-Identity))

    return Lambda_bar, gamma_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """
    Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.

    :param q_i: tuple containing A_i and Bu_i at position i (P,), (P,)
    :param q_j: tuple containing A_j and Bu_j at position j (P,), (P,)
    :return: new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_elements, Bu_elements, C_tilde, conj_sym, bidirectional=False):
    """
    Compute the LxH output of discretized SSM given an LxH input.

    :param Lambda_elements: (complex64) discretized state matrix (L, P)
    :param Bu_elements: (complex64) discretized inputs projected to state space (L, P)
    :param C_tilde: (complex64) output matrix (H, P)
    :param conj_sym: (bool) whether conjugate symmetry is enforced
    :return: ys: (float32) the SSM outputs (S5 layer preactivations) (L, H)
    """

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,(Lambda_elements, Bu_elements),reverse=True)
        xs = np.concatenate((xs, xs2), axis=-1)

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)
    
                
def compute_inv_dt(key, features, dt_min, dt_max):
    # Generate random values
    rand_values = jax.random.uniform(key, (features,))
    
    # Compute dt
    dt = np.exp(
        rand_values * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = np.clip(dt, a_min=1e-4)

    # Compute inverse of softplus
    inv_dt = dt + np.log(-np.expm1(-dt))

    return inv_dt


def weight_init(minval, maxval):
    def init(key, shape, dtype=np.float32):
        return jax.random.uniform(key, shape, dtype, minval, maxval)
    return init


def bias_init(dt_min, dt_max):
    def init(key, shape, dtype=np.float32):
        return compute_inv_dt(key, shape[0], dt_min, dt_max)
    return init


class SimpleDense(nn.Module):
    features: int
    kernel_init: Callable
    bias_init: Callable
    name: str = None

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(f'{self.name}_kernel',
                            self.kernel_init,  # Initialization function
                            (inputs.shape[-1], self.features))  # Shape info.
        y = np.dot(inputs, kernel)
        bias = self.param(f'{self.name}_bias', self.bias_init, (self.features,))
        y = y + bias
        return y


class S7(nn.Module):
    H_in: int
    H_out: int
    P: int
    block_size: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    log_a: bool
    stablessm_a: bool
    bidirectional: bool
    conj_sym: bool = True
    clip_eigs: bool = False
    step_rescale: float = 1.0
    stride: int = 1
    pooling_mode: str = "last"
    input_dependent: bool = True
    a: float = 1.0
    b: float = 0.5

    """
    Event-based S5 module
    
    :param H_in: int, SSM input dimension
    :param H_out: int, SSM output dimension
    :param P: int, SSM state dimension
    :param block_size: int, block size for block-diagonal state matrix
    :param C_init: str, initialization method for output matrix C
    :param discretization: str, discretization method for event-based SSM
    :param dt_min: float, minimum value of log timestep
    :param dt_max: float, maximum value of log timestep
    :param conj_sym: bool, whether to enforce conjugate symmetry in the state space operator
    :param clip_eigs: bool, whether to clip eigenvalues of the state space operator
    :param step_rescale: float, rescale factor for step size
    :param stride: int, stride for subsampling layer
    :param pooling_mode: str, pooling mode for subsampling layer
    :param log_a: bool, whether to learn state transition matrix A in log-space
    :param stablessm_a: bool, whether to apply a stable parameterization for A
    :param bidirectional: bool, whether to use a bidirectional SSM
    :param input_dependent: bool, whether the parameters B and C depend on the input
    :param a: float, scaling factor for stable parameterization of A
    :param b: float, offset for stable parameterization of A
    """

    def setup(self):
        """
        Initializes parameters once and performs discretization each time the SSM is applied to a sequence
        """
        if self.bidirectional:
            print("Bidirectional Model")
        else:
            print("Unidirectional Model")


        if self.input_dependent:
            print("Input dependent")
        else:
            print("LTI")

        if self.log_a == True:
            print("Learning A in log scale")
        elif self.stablessm_a == True:
            print("Learning A with StableSSM formula")
        else:
            print("Not learning A with log or StableSSM formula")

        assert not (self.log_a and self.stablessm_a), "Both log_a and stablessm_a cannot be true at the same time."
    
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(self.block_size)

        # Initialize state matrix A using approximation to HiPPO-LegS matrix

        num_blocks = self.P // self.block_size
        block_size = self.block_size // 2 if self.conj_sym else self.block_size
        local_P = self.P // 2 if self.conj_sym else self.P
        self.local_P = local_P
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T 

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * np.ones((num_blocks, block_size))).ravel()

        
        V = block_diag(*([V] * num_blocks))
        Vinv = block_diag(*([Vc] * num_blocks))

        state_str = f"SSM: {self.H_in} -> {self.P} -> {self.H_out}"
        if self.stride > 1:
            state_str += f" (stride {self.stride} with pooling mode {self.pooling_mode})"
        print(state_str)

        if self.log_a:
            Lambda = np.log1p(Lambda)
        elif self.stablessm_a:
            Lambda = np.sqrt(( -1/Lambda - self.b ) / self.a)


        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: Lambda.real, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: Lambda.imag, (None,))

        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        
        if self.input_dependent:  
            self.dt_rank = math.ceil(self.H_in/16)
            dt_init_std = self.dt_rank**-0.5 * self.step_rescale
            key = jax.random.PRNGKey(0)
            kernel_initializer = weight_init(-dt_init_std,dt_init_std)
            bias_initializer = bias_init(self.dt_min, self.dt_max)
            self.step_proj = SimpleDense(features=local_P, 
                        kernel_init=kernel_initializer,
                        bias_init=bias_initializer, name = "step_proj")
        
        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (self.P, self.H_in)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init, rng, shape, Vinv),
                            B_shape)
        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H_out, self.P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H_out, self.P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(self.C_init))
        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.H_out, 2 * local_P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.H_out, local_P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]
        else:
            if self.bidirectional:
                self.C1 = self.param("C1",
                                    lambda rng, shape: init_CV(C_init, rng, shape, V),
                                    C_shape)
                self.C2 = self.param("C2",
                                    lambda rng, shape: init_CV(C_init, rng, shape, V),
                                    C_shape)

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate((C1, C2), axis=-1)

            else:
                self.C = self.param("C",
                                    lambda rng, shape: init_CV(C_init, rng, shape, V),
                                    C_shape)
                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        self.log_step = self.param("log_step",
                            init_log_steps,
                            (local_P, self.dt_min, self.dt_max))


        # Initialize feedthrough (D) matrix
        if self.H_in == self.H_out:
            self.D = self.param("D", normal(stddev=1.0), (self.H_in,))
        else:
            self.D = self.param("D", glorot_normal(), (self.H_out, self.H_in))

        self.pool = EventPooling(stride=self.stride, mode=self.pooling_mode)

        # Discretize
        if self.discretization in ["zoh"]:
            self.discretize_fn = discretize_zoh
        elif self.discretization in ["dirac"]:
            self.discretize_fn = discretize_dirac
        elif self.discretization in ["async"]:
            self.discretize_fn = discretize_async
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, input_sequence, integration_timesteps):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence using a parallel scan.

        :param input_sequence: (float32) input sequence (L, H)
        :param integration_timesteps: (float32) integration timesteps (L,)
        :return: (float32) output sequence (L, H)
        """
        if self.log_a:
            Lambda = -np.exp(self.Lambda)
        elif self.stablessm_a:
            Lambda = -np.sqrt((-1 - self.b * self.Lambda)/(self.a*self.Lambda))
        else:
            Lambda = self.Lambda
       

        if self.input_dependent:
            step = self.step_proj(input_sequence)
            step = jax.nn.softplus(step)
        B = self.B[..., 0] + 1j * self.B[..., 1]
        C = self.C_tilde
        

        def discretize_and_project_inputs_input_dep(u, _timestep, log_step):
            step = self.step_rescale * log_step
            Lambda_bar, gamma_bar = self.discretize_fn(Lambda, step, _timestep)
            Bu = gamma_bar * (B @ u)
            return Lambda_bar, Bu
        
        def discretize_and_project_inputs(u, _timestep):
            step = self.step_rescale * np.exp(self.log_step[:, 0])
            Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda, step, _timestep)
            Bu = gamma_bar * (B @ u)
            return Lambda_bar, Bu

        if self.input_dependent:
            Lambda_bar_elements, b = jax.vmap(discretize_and_project_inputs_input_dep)(input_sequence, integration_timesteps, step)
        else:
            Lambda_bar_elements, b = jax.vmap(discretize_and_project_inputs)(input_sequence, integration_timesteps)

        y = apply_ssm(
            Lambda_bar_elements,
            b,
            C,
            self.conj_sym,
            bidirectional=self.bidirectional,
        )
        if self.H_in == self.H_out:
            Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        else:
            Du = jax.vmap(lambda u: self.D @ u)(input_sequence)
        return y + Du


def init_S5SSM(
        C_init,
        dt_min,
        dt_max,
        conj_sym,
        clip_eigs,
        log_a,
        stablessm_a,
        input_dependent,
        bidirectional,
        a=1.0,
        b=0.5,
):
    """
    Convenience function that will be used to initialize the SSM.
    Same arguments as defined in S5SSM above.
    """
    return partial(S7,
                   C_init=C_init,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   log_a=log_a,
                   input_dependent=input_dependent,
                   stablessm_a=stablessm_a,
                   bidirectional=bidirectional,
                   a=a,
                   b=b,
                   )
