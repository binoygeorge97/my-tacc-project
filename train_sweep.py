import os
import time
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import ray
from flax import serialization
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================================================
# 0. INSERT YOUR MODEL DEFINITIONS HERE
# Paste StackedModelRegression, S4LayerEnsemble, and batched_reg_runner here.
# (They were missing from your snippet but called in safe_train_regression).
# =========================================================================
import jax
from jax.numpy.linalg import inv, matrix_power
from jax.nn.initializers import normal, zeros, ones
import flax.nnx as nnx
import optax

# --- Helper Functions (JAX-compatible) ---

def scan_SSM(Ab, Bb, Cb, u, x0):
    """Run the SSM state-space equation."""
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)



def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """Initializer for the log_step parameter."""
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)
    return init


def causal_convolution(u, K):
    #jax.debug.print("DEBUG: u shape={} | K shape={}", u.shape, K.shape)
    #print("DEBUG: u shape={} | K shape={}", u.shape, K.shape)
    assert K.shape[0] == u.shape[0]
    ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
    Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
    out = ud * Kd
    return jnp.fft.irfft(out)[: u.shape[0]]

def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)


def init(x):
    def _init(key, shape):
        assert shape == x.shape
        return x

    return _init


def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    # Check skew symmetry
    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = jnp.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = jnp.sqrt(jnp.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return nhippo, P, B


def make_HiPPO(N):
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A

@jax.jit
def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = jnp.fft.ifft(atRoots, L).reshape(L)
    return out.real


def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    # Convert parameters to matrices
    B = B[:, jnp.newaxis]
    Ct = C[jnp.newaxis, :]

    N = Lambda.shape[0]
    A = jnp.diag(Lambda) - P[:, jnp.newaxis] @ Q[:, jnp.newaxis].conj().T
    I = jnp.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


class S4LayerEnsemble(nnx.Module):
    def __init__(self, N: int, l_max: int, D_MODEL: int, decode: bool, *, rngs: nnx.Rngs):
        self.N, self.decode, self.l_max, self.D_MODEL = N, decode, l_max, D_MODEL
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        init_C, init_D, init_log_step = normal(stddev=0.5**0.5), ones, log_step_initializer()
        vmap_in_axes = (0, None)
        vmap_init_A_re = jax.vmap(init_A_re, in_axes=vmap_in_axes)
        vmap_init_A_im = jax.vmap(init_A_im, in_axes=vmap_in_axes)
        vmap_init_P = jax.vmap(init_P, in_axes=vmap_in_axes)
        vmap_init_B = jax.vmap(init_B, in_axes=vmap_in_axes)
        vmap_init_C = jax.vmap(init_C, in_axes=vmap_in_axes)
        vmap_init_D = jax.vmap(init_D, in_axes=vmap_in_axes)
        vmap_init_log_step = jax.vmap(init_log_step, in_axes=vmap_in_axes)
        keys = jax.random.split(rngs.params(), 7)
        lr_meta = {'lr': 0.1}
        self.Lambda_re = nnx.Param(vmap_init_A_re(jax.random.split(keys[0], D_MODEL), (N,)), metadata=lr_meta)
        self.Lambda_im = nnx.Param(vmap_init_A_im(jax.random.split(keys[1], D_MODEL), (N,)), metadata=lr_meta)
        self.P = nnx.Param(vmap_init_P(jax.random.split(keys[2], D_MODEL), (N,)), metadata=lr_meta)
        self.B = nnx.Param(vmap_init_B(jax.random.split(keys[3], D_MODEL), (N,)), metadata=lr_meta)
        self.C_real_imag = nnx.Param(vmap_init_C(jax.random.split(keys[4], D_MODEL), (N, 2)), metadata=lr_meta)
        self.D = nnx.Param(vmap_init_D(jax.random.split(keys[5], D_MODEL), (1,)), metadata=lr_meta)
        self.log_step = nnx.Param(vmap_init_log_step(jax.random.split(keys[6], D_MODEL), (1,)), metadata=lr_meta)

        # --- NO MORE self.x_k_1 ---
        # if self.decode:
        #     self.x_k_1 = nnx.Variable(jnp.zeros((D_MODEL, N,), dtype=jnp.complex64))

    # --- __call__ signature has changed ---
    def __call__(self, u, x_k_1):
        """
        Takes in a single state vector x_k_1 [N,]
        Returns a single output y_s [L,] and new state x_k [N,]
        """
        dt_min, dt_max = 0.001, 1.0
        step = jnp.exp(self.log_step.value)
        step = jnp.clip(step, dt_min, dt_max)

        Lambda = jnp.clip(self.Lambda_re.value, None, -1e-4) + 1j * self.Lambda_im.value
        C_complex = self.C_real_imag.value[..., 0] + 1j * self.C_real_imag.value[..., 1]
        #step = jnp.exp(self.log_step.value)

        if not self.decode:
            # CNN mode is stateless, so we ignore x_k_1 and return it unchanged
            K = kernel_DPLR(Lambda, self.P.value, self.P.value, self.B.value, C_complex, step, self.l_max)
            y_s = causal_convolution(u, K) + self.D.value * u
            return y_s, x_k_1 # Return state unchanged
        else:
            # RNN mode uses and returns state
            Ab, Bb, Cb = discrete_DPLR(Lambda, self.P.value, self.P.value, self.B.value, C_complex, step, self.l_max)
            u_r = u[:, jnp.newaxis]
            x_k, y_s = scan_SSM(Ab, Bb, Cb, u_r, x_k_1) # Use passed-in state

            # --- DO NOT MUTATE SELF ---
            # self.x_k_1.value = x_k

            # --- Return the output and the new state ---
            return y_s.reshape(-1).real + self.D.value * u, x_k


class SequenceBlockNNX(nnx.Module):
    def __init__(self,
                 layer_cls: type[nnx.Module],
                 layer_args: dict,
                 d_model: int,
                 dropout: float,
                 prenorm: bool = True,
                 glu: bool = True,
                 decode: bool = False,
                 *, rngs: nnx.Rngs):

        self.d_model = d_model
        self.prenorm = prenorm
        self.glu = glu
        self.decode = decode
        self.dropout_rate = dropout

        self.seq = layer_cls(
            **layer_args,
            D_MODEL=d_model,
            decode=decode,
            rngs=rngs
        )

        # Mixing Layers
        keys = jax.random.split(rngs.params(), 3)
        self.norm = nnx.LayerNorm(d_model, rngs=nnx.Rngs(params=keys[0]))
        self.out = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(params=keys[1]))
        if self.glu:
            self.out2 = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(params=keys[2]))

        self.drop = nnx.Dropout(dropout, broadcast_dims=[0])

    def __call__(self, x, s4_state, *, rngs: nnx.Rngs = None, training: bool = True):
        skip = x

        if self.prenorm:
            x = self.norm(x)

        # --- ROBUST FIX: Manual JAX Vmap ---
        # 1. Split the S4 layer into Graph (Static) and Params (Data)
        seq_graph, seq_params = nnx.split(self.seq)

        # 2. Define a Pure Function for ONE channel
        def run_one_channel(params_slice, u_slice, state_slice):
            # Reconstruct the layer for this single channel
            single_layer = nnx.merge(seq_graph, params_slice)
            # Run it
            return single_layer(u_slice, state_slice)

        # 3. Use standard JAX vmap
        # seq_params: Axis 0 corresponds to D_MODEL (H)
        # x (Input): Axis 1 corresponds to H -> (L, H)
        # s4_state: Axis 0 corresponds to H -> (H, N)
        x, new_s4_state = jax.vmap(
            run_one_channel,
            in_axes=(0, 1, 0),  # Map over params(0), input(1), state(0)
            out_axes=(1, 0)     # Stack output(1), new_state(0)
        )(seq_params, x, s4_state)

        # -----------------------------------

        x = nnx.gelu(x)

        if training and rngs:
             x = self.drop(x, rngs=rngs)

        if self.glu:
            gate = jax.nn.sigmoid(self.out2(x))
            x = self.out(x) * gate
        else:
            x = self.out(x)

        if training and rngs:
            x = self.drop(x, rngs=rngs)

        x = skip + x

        if not self.prenorm:
            x = self.norm(x)

        return x, new_s4_state


class StackedModelRegression(nnx.Module):
    def __init__(self,
                 layer_cls: type[nnx.Module],
                 layer_args: dict,
                 d_input: int,
                 d_output: int,
                 d_model: int,
                 n_layers: int,
                 prenorm: bool = True,
                 dropout: float = 0.0,
                 decode: bool = False,
                 *, rngs: nnx.Rngs):

        self.d_model = d_model
        self.d_output = d_output
        self.n_layers = n_layers
        self.prenorm = prenorm
        self.decode = decode
        self.dropout = dropout

        keys = jax.random.split(rngs.params(), 3)

        # 1. Linear Encoder (No Embeddings!)
        # Projects 1 feature (sine value) -> d_model (Hidden)
        self.encoder = nnx.Linear(d_input, d_model, rngs=nnx.Rngs(params=keys[0]))

        # 2. Linear Decoder
        # Projects d_model -> 1 output value
        self.decoder = nnx.Linear(d_model, d_output, rngs=nnx.Rngs(params=keys[1]))

        layer_keys = jax.random.split(keys[2], n_layers)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(
                SequenceBlockNNX(
                    layer_cls=layer_cls,
                    layer_args=layer_args,
                    d_model=d_model,
                    dropout=dropout,
                    prenorm=prenorm,
                    decode=decode,
                    glu=True,
                    rngs=nnx.Rngs(params=layer_keys[i])
                )
            )

    def __call__(self, x, states=None, *, rngs: nnx.Rngs = None, training: bool = True):
        # x shape: (B, L, 1) or (L, 1)

        # --- FIX 1: Handle Rank-1 Input ---
        was_1d = False
        if x.ndim == 1:
            x = x[jnp.newaxis, :]
            was_1d = True

        # # Causal Padding for CNN mode
        # if not self.decode:
        #     x = jnp.pad(x[:-1], [(1, 0), (0, 0)])

        # --- NO NORMALIZATION (Input is already standard float) ---

        x = self.encoder(x)
        current_states = states if states is not None else [None] * self.n_layers

        new_states = []
        for layer, state in zip(self.layers, current_states):
            x, new_s = layer(x, state, rngs=rngs, training=training)
            new_states.append(new_s)

        x = self.decoder(x)

        # --- FIX 2: NO SOFTMAX (Regression Output) ---
        output = x

        if was_1d:
            output = output.squeeze(0)

        return output, new_states

    # Add the init_state helper for inference
    def init_state(self, N: int):
        return [jnp.zeros((self.d_model, N), dtype=jnp.complex64) for _ in range(self.n_layers)]


# 1. VMAP RUNNERS (The "BatchStackedModel" replacement)

batched_reg_runner = nnx.vmap(
    lambda m, x, k, is_train: m(x, states=None, rngs=nnx.Rngs(dropout=k), training=is_train),
    in_axes=(nnx.StateAxes({nnx.Param: None}), 0, 0, None),
    out_axes=0
)




# =========================================================================
# 1. THE PHYSICS ENGINE & DATALOADER
# =========================================================================
def get_discrete_matrices(A_continuous, dt=0.01):
    B_i = np.array([[0.0], [1.0]])
    B = np.block([
        [B_i, np.zeros((2,1)), np.zeros((2,1))],
        [np.zeros((2,1)), B_i, np.zeros((2,1))],
        [np.zeros((2,1)), np.zeros((2,1)), B_i]
    ])
    I = np.eye(6)
    inv_term = np.linalg.inv(I - (dt / 2.0) * A_continuous)
    Ad = inv_term @ (I + (dt / 2.0) * A_continuous)
    Bd = inv_term @ B * dt
    return Ad, Bd

def fast_vectorized_aprbs(batch_size, length, min_val, max_val, hold_prob, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    random_amps = rng.uniform(min_val, max_val, size=(batch_size, 3, length))
    switches = rng.rand(batch_size, 3, length) > hold_prob
    switches[:, :, 0] = True
    signal = np.zeros((batch_size, 3, length))
    current_amp = random_amps[:, :, 0]
    for k in range(length):
        current_amp = np.where(switches[:, :, k], random_amps[:, :, k], current_amp)
        signal[:, :, k] = current_amp
    return signal

def generate_microgrid_data_fast(Ad, Bd, batch_size, length=100, seed=42):
    rng = np.random.RandomState(seed)
    U_signals = fast_vectorized_aprbs(batch_size, length, -1.0, 1.0, hold_prob=0.8, rng=rng)
    batch_inputs = np.zeros((batch_size, length, 9))
    batch_targets = np.zeros((batch_size, length, 6))
    X_current = np.zeros((batch_size, 6))

    for k in range(length):
        U_k = U_signals[:, :, k]
        batch_inputs[:, k, 0:6] = X_current
        batch_inputs[:, k, 6:9] = U_k
        X_next = X_current.dot(Ad.T) + U_k.dot(Bd.T)
        batch_targets[:, k, :] = X_next
        X_current = X_next
    return batch_inputs, batch_targets

def create_microgrid_dataloaders(Ad, Bd, bsz=32, L=100):
    n_train = 30000
    n_test = 200
    total_samples = n_train + n_test
    all_inputs, all_targets = generate_microgrid_data_fast(Ad, Bd, batch_size=total_samples, length=L)

    train_in, train_out = all_inputs[:n_train], all_targets[:n_train]
    test_in, test_out = all_inputs[n_train:], all_targets[n_train:]

    trainloader = [(train_in[i*bsz:(i+1)*bsz], train_out[i*bsz:(i+1)*bsz]) for i in range(n_train // bsz)]
    testloader = [(test_in[i*bsz:(i+1)*bsz], test_out[i*bsz:(i+1)*bsz]) for i in range(n_test // bsz)]
    return trainloader, testloader, 9, 6

DatasetMetadata = {
    "microgrid": {
        "input_labels": ["e_1", "r_1", "e_2", "r_2", "e_3", "r_3", "u_1", "u_2", "u_3"],
        "output_labels": ["Next e_1", "Next r_1", "Next e_2", "Next r_2", "Next e_3", "Next r_3"],
        "dt": 0.01
    }
}

# =========================================================================
# 2. UTILS & TRAIN LOOP
# =========================================================================
def create_optimizer(model, base_lr, weight_decay, total_steps):
    if total_steps > 0:
        schedule_fn = lambda lr: optax.cosine_onecycle_schedule(peak_value=lr, transition_steps=total_steps, pct_start=0.1)
    else:
        schedule_fn = lambda lr: optax.constant_schedule(lr)
    tx = optax.adamw(learning_rate=schedule_fn(base_lr), weight_decay=weight_decay)
    return nnx.Optimizer(model, tx, wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, x_batch, y_batch, dropout_keys):
    def loss_fn(model):
        predictions, _ = batched_reg_runner(model, x_batch, dropout_keys, True)
        return jnp.mean((predictions - y_batch) ** 2), predictions
    (loss, preds), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss

@nnx.jit
def eval_step(model, x_batch, y_batch):
    B = x_batch.shape[0]
    dummy_keys = jax.random.split(jax.random.PRNGKey(0), B)
    predictions, _ = batched_reg_runner(model, x_batch, dummy_keys, False)
    return jnp.mean((predictions - y_batch) ** 2)

def validate(model, testloader):
    losses = [eval_step(model, jnp.array(x), jnp.array(y)) for x, y in testloader]
    return np.mean(losses)

def train_epoch(rng, model, optimizer, trainloader):
    batch_losses = []
    for batch in tqdm(trainloader, desc="Training", disable=True):
        inputs, targets = jnp.array(batch[0]), jnp.array(batch[1])
        rng, drop_rng = jax.random.split(rng)
        batch_keys = jax.random.split(drop_rng, inputs.shape[0])
        batch_losses.append(train_step(model, optimizer, inputs, targets, batch_keys))
    return rng, np.mean(batch_losses)

def save_model(model, config, filename="s4_model.msgpack"):
    model_state = nnx.state(model, nnx.Param).to_pure_dict()
    byte_data = serialization.to_bytes({'model_state': model_state, 'config': config})
    with open(filename, 'wb') as f:
        f.write(byte_data)

def load_model_regression(filename, d_input_arg=None, d_output_arg=None):
    with open(filename, 'rb') as f:
        byte_data = f.read()
    raw_structure = serialization.msgpack_restore(byte_data)
    config = raw_structure['config']
    
    d_input = d_input_arg if d_input_arg is not None else config.get('d_input', 1)
    d_output = d_output_arg if d_output_arg is not None else config.get('d_output', 1)
    l_max = config['model'].get('l_max', 100)
    s4_N = config['model'].get('N', 64)

    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    model = StackedModelRegression(
        layer_cls=S4LayerEnsemble, layer_args={'N': s4_N, 'l_max': l_max},
        d_input=d_input, d_output=d_output,
        d_model=config['model']['d_model'], n_layers=config['model']['n_layers'],
        dropout=config['model']['dropout'], prenorm=config['model']['prenorm'],
        decode=True, rngs=rngs
    )
    current_state_dict = nnx.state(model, nnx.Param).to_pure_dict()
    restored = serialization.from_bytes({'model_state': current_state_dict, 'config': config}, byte_data)
    nnx.update(model, restored['model_state'])
    return model

def safe_train_regression(dataset, layer, seed, model_cfg, train_cfg, Ad, Bd, unique_save_path):
    key = jax.random.PRNGKey(seed)
    key, model_rng, train_rng = jax.random.split(key, 3)

    trainloader, testloader, d_input, d_output = create_microgrid_dataloaders(Ad=Ad, Bd=Bd, bsz=train_cfg['bsz'], L=model_cfg.get('l_max', 100))
    
    rngs = nnx.Rngs(params=model_rng, dropout=0)
    stacked_args = model_cfg.copy()
    s4_N, l_max = stacked_args.pop('N'), stacked_args.pop('l_max')
    stacked_args.pop('embedding', None)

    model = StackedModelRegression(layer_cls=S4LayerEnsemble, layer_args={'N': s4_N, 'l_max': l_max}, d_input=d_input, d_output=d_output, decode=False, rngs=rngs, **stacked_args)
    optimizer = create_optimizer(model, base_lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'], total_steps=len(trainloader)*train_cfg['epochs'])

    best_loss = 1e9
    for epoch in range(train_cfg['epochs']):
        train_rng, train_loss = train_epoch(train_rng, model, optimizer, trainloader)
        test_loss = validate(model, testloader)
        
        # --- NEW: Log epoch-level data to W&B ---
        import wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss
        })
        # ----------------------------------------

        if test_loss < best_loss:
            best_loss = test_loss
            os.makedirs(os.path.dirname(unique_save_path), exist_ok=True)
            save_model(model, {'dataset': dataset, 'layer': layer, 'model': model_cfg, 'train': train_cfg}, unique_save_path)
    return model, best_loss

# =========================================================================
# 3. SWEEP DEFINITIONS & RAY WORKER
# =========================================================================
def get_sweep_configs():
    hp = {"d_model": 128, "n_layers": 2, "dropout": 0.0, "prenorm": True, "lr": 1e-3, "batch_size": 32, "epochs": 30, "N": 64, "l_max": 100}
    hp_marginal = hp.copy()
    hp_marginal["d_model"] = 64
    Z = np.zeros((2, 2))
    configs = []

    # Matrix 1
    A1 = np.block([[np.array([[-3.5, -2.4], [0.0, 0.0]]), np.array([[0.0, 0.03], [0.0, 0.0]]), np.array([[0.0, 0.06], [0.0, 0.0]])],
                   [Z, np.array([[-3.5, -2.3], [0.0, 0.0]]), Z], [Z, Z, np.array([[-5.2, -5.3], [0.0, 0.0]])]])
    configs.append([{"matrix_id": "1", "A_continuous": A1}, hp_marginal])


    # Matrix 1A
    A4_1a = np.array([[-2.5, -2.0],
               [ 0.0,  0.0]])
    A5_1a = np.array([[-3.0, -1.5],
                   [ 0.0,  0.0]])
    A6_1a = np.array([[-4.0, -2.2],
                   [ 0.0,  0.0]])
    A1a = np.block([
        [A4_1a, Z, Z],
        [Z, A5_1a, Z],
        [Z, Z, A6_1a]
    ])
    configs.append([{"matrix_id": "1a", "A_continuous": A1a}, hp_marginal])


    # Matrix 1B
    A4_1b = np.array([[-1.8, -3.0],
               [ 0.0,  0.0]])
    A5_1b = np.array([[-2.2, -2.8],
                   [ 0.0,  0.0]])
    A6_1b = np.array([[-3.5, -4.1],
                   [ 0.0,  0.0]])
    A1b = np.block([
        [A4_1b, Z, Z],
        [Z, A5_1b, Z],
        [Z, Z, A6_1b]
    ])
    configs.append([{"matrix_id": "1b", "A_continuous": A1b}, hp_marginal])

    
    # Matrix 2
    A2 = np.block([[np.array([[1.0, 0.5], [0.0, 2.0]]), Z, Z], [Z, np.array([[1.5, 0.5], [0.0, 3.0]]), Z], [Z, Z, np.array([[0.5, 0.5], [0.0, 4.0]])]])
    configs.append([{"matrix_id": 2, "A_continuous": A2}, hp])
    
    # ==========================================
    # 3. Oscillatory unstable (complex eigenvalues)
    # ==========================================
    A4_3 = np.array([[ 1.0,  2.0], [-2.0,  1.0]])
    A5_3 = np.array([[ 2.0,  5.0], [-5.0,  2.0]])
    A6_3 = np.array([[ 1.5, 10.0], [-10.0, 1.5]])

    A3 = np.block([
        [A4_3, Z, Z],
        [Z, A5_3, Z],
        [Z, Z, A6_3]
    ])
    configs.append([{"matrix_id": 3, "A_continuous": A3}, hp])

    # ==========================================
    # 4. Non-normal Jordan block
    # ==========================================
    A4_4 = np.array([[1.0, 100.0], [0.0, 1.0]])
    A5_4 = np.array([[1.5,  80.0], [0.0, 1.5]])
    A6_4 = np.array([[2.0, 120.0], [0.0, 2.0]])

    A4 = np.block([
        [A4_4, Z, Z],
        [Z, A5_4, Z],
        [Z, Z, A6_4]
    ])
    configs.append([{"matrix_id": 4, "A_continuous": A4}, hp])

    # ==========================================
    # 5. Weakly coupled unstable (transition case)
    # ==========================================
    A4_5 = np.array([[1.0, 1.0], [0.0, 2.0]])
    A5_5 = np.array([[1.5, 0.7], [0.0, 3.0]])
    A6_5 = np.array([[0.5, 1.2], [0.0, 4.0]])
    H45_5 = np.array([[ 0.3, -0.2], [ 0.0,  0.4]])
    H46_5 = np.array([[ 0.1,  0.0], [ 0.0, -0.3]])
    H56_5 = np.array([[ 0.25, -0.15], [ 0.0,   0.2]])

    A5 = np.block([
        [A4_5, H45_5, H46_5],
        [Z,    A5_5,  H56_5],
        [Z,    Z,     A6_5 ]
    ])
    configs.append([{"matrix_id": 5, "A_continuous": A5}, hp])

    # ==========================================
    # 6. Ill-conditioned eigenvectors
    # ==========================================
    A4_6 = np.array([[1.0, 1000.0], [0.001,   1.0]])
    A5_6 = np.array([[1.5,  800.0], [0.002,   1.5]])
    A6_6 = np.array([[2.0, 1200.0], [0.0015,  2.0]])

    A6 = np.block([
        [A4_6, Z, Z],
        [Z, A5_6, Z],
        [Z, Z, A6_6]
    ])
    configs.append([{"matrix_id": 6, "A_continuous": A6}, hp])

    # ==========================================
    # 7. Mixed stable / unstable modes
    # ==========================================
    A4_7 = np.array([[-2.0, 0.5], [ 0.0, 0.5]])
    A5_7 = np.array([[-1.0, 0.5], [ 0.0, 1.5]])
    A6_7 = np.array([[-3.0, 0.5], [ 0.0, 2.0]])

    A7 = np.block([
        [A4_7, Z, Z],
        [Z, A5_7, Z],
        [Z, Z, A6_7]
    ])
    configs.append([{"matrix_id": 7, "A_continuous": A7}, hp])

    return configs

import wandb

@ray.remote(num_gpus=0.2)
def train_single_model(matrix_dict, hp_dict):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    matrix_id, A_continuous = matrix_dict["matrix_id"], matrix_dict["A_continuous"]
    
    # 1. Initialize W&B for this specific Ray worker
    run = wandb.init(
        project="tacc-microgrid-s4-sweep", # Name your project here
        name=f"matrix_{matrix_id}",   # Name the specific run
        config={**hp_dict, "matrix_id": matrix_id}
    )

    model_cfg = {k: hp_dict[k] for k in ["d_model", "n_layers", "N", "l_max", "dropout", "prenorm"]}
    model_cfg["embedding"] = False
    train_cfg = {"epochs": hp_dict["epochs"], "bsz": hp_dict["batch_size"], "lr": hp_dict["lr"], "weight_decay": 0.0}
    unique_save_path = f"checkpoints/sweep/mat{matrix_id}_best_model.msgpack"

    Ad, Bd = get_discrete_matrices(A_continuous)
    
    # 2. Run your training
    trained_model, final_mse = safe_train_regression(
        "microgrid", "s4", 42, model_cfg, train_cfg, Ad, Bd, unique_save_path
    )
    
    wandb.log({"final_test_mse": final_mse})

    # --- CRITICAL FIX: Load the RNN (decode=True) version of the model ---
    rnn_model = load_model_regression(unique_save_path, d_input_arg=9, d_output_arg=6)
    # ---------------------------------------------------------------------

    # --- Run Evaluation INSIDE the worker before W&B closes ---
    plot_title = f"Matrix {matrix_id} | d_model={model_cfg['d_model']} | Test MSE: {final_mse:.6f}"
    run_evaluation(
        model=rnn_model,  # <--- Pass the RNN model here!
        Ad=Ad, 
        Bd=Bd, 
        d_model=model_cfg['d_model'], 
        n_layers=model_cfg['n_layers'], 
        dataset_name="microgrid", 
        custom_title=plot_title
    )

    # 3. Close the run
    wandb.finish()
    
    return {"matrix_id": matrix_id, "mse": final_mse, "path": unique_save_path}


# =========================================================================
# 4. HEADLESS PLOTTING & EVALUATION
# =========================================================================
def visualize_system_plots(inputs, targets, preds, dataset_name="microgrid", n_plot=3, max_channels=4, custom_title=""):
    inputs, targets, preds = np.array(inputs), np.array(targets), np.array(preds)
    meta = DatasetMetadata.get(dataset_name, {})
    dt = meta.get("dt", 0.01)
    
    in_labels = meta.get("input_labels", [f"Input Ch {d}" for d in range(inputs.shape[-1])])
    out_labels = meta.get("output_labels", [f"Output Ch {d}" for d in range(targets.shape[-1])])
    time_arr = np.arange(inputs.shape[1]) * dt

    fig, axes = plt.subplots(n_plot, 2, figsize=(16, 4 * n_plot), squeeze=False)
    fig.suptitle(custom_title, fontsize=16, fontweight='bold')

    for i in range(n_plot):
        ax_in, ax_out = axes[i, 0], axes[i, 1]
        
        # Plot Inputs
        for d in range(min(inputs.shape[-1], max_channels)):
            ax_in.plot(time_arr, inputs[i, :, d], alpha=0.7, label=in_labels[d] if d < len(in_labels) else f"In {d}")
        ax_in.set_title(f"Sample {i}: Inputs")
        ax_in.grid(True, alpha=0.3)
        ax_in.legend(loc='upper right')

        # Plot Outputs
        total_mse = 0.0
        for d in range(min(targets.shape[-1], max_channels)):
            label_name = out_labels[d] if d < len(out_labels) else f"Out {d}"
            ax_out.plot(time_arr, targets[i, :, d], '-', linewidth=2, alpha=0.5, label=f'True: {label_name}')
            ax_out.plot(time_arr, preds[i, :, d], '--', linewidth=1.5, label=f'Pred: {label_name}')
            total_mse += np.mean((targets[i, :, d] - preds[i, :, d])**2)
            
        ax_out.set_title(f"Sample {i}: Outputs (Avg MSE: {total_mse/targets.shape[-1]:.5f})")
        ax_out.grid(True, alpha=0.3)
        ax_out.legend(loc='upper right')

    plt.tight_layout()
    
    # MODIFIED FOR TACC: Save to file instead of plt.show()
    safe_title = custom_title.replace(" | ", "_").replace("=", "").replace(" ", "_")
    save_path = f"{safe_title}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # --- NEW: Upload the image to W&B if a run is currently active ---
    import wandb
    if wandb.run is not None:
        wandb.log({"Evaluation_Plots": wandb.Image(save_path)})
    # -----------------------------------------------------------------
    
    plt.close(fig)
    print(f"[*] Saved evaluation plot to {save_path} and logged to W&B")


def run_evaluation(model, Ad, Bd, d_model, n_layers, dataset_name="microgrid", custom_title=""):
    print(f"[*] Running Step-by-Step RNN Inference for: {custom_title}")

    l_max, bsz = 100, 32
    _, testloader, _, _ = create_microgrid_dataloaders(Ad, Bd, bsz=bsz, L=l_max)

    inputs_u, targets_y = jnp.array(testloader[0][0]), jnp.array(testloader[0][1])
    H_dim, N_dim = d_model, 64 

    @nnx.jit
    def step_by_step_inference(model, inputs):
        B_batch, L, D = inputs.shape
        inputs_t = jnp.transpose(inputs, (1, 0, 2))
        init_states = [jnp.zeros((B_batch, H_dim, N_dim), dtype=jnp.complex64) for _ in range(n_layers)]

        def scan_step(carry, x_t):
            model_carry, current_states_batch = carry
            def single_sample_step(m, x, s):
                pred, new_s = m(x, states=s, training=False)
                return pred, new_s

            runner = nnx.vmap(
                single_sample_step,
                in_axes=(nnx.StateAxes({nnx.Param: None}), 0, 0),
                out_axes=(0, 0)
            )
            pred_batch, new_states_batch = runner(model_carry, x_t, current_states_batch)
            return (model_carry, new_states_batch), pred_batch

        initial_carry = (model, init_states)
        _, preds_t = nnx.scan(scan_step, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(initial_carry, inputs_t)
        return jnp.transpose(preds_t, (1, 0, 2))

    preds_y = step_by_step_inference(model, inputs_u)
    visualize_system_plots(inputs_u, targets_y, preds_y, n_plot=3, dataset_name=dataset_name, custom_title=custom_title)


# =========================================================================
# 5. MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    import os
    
    # 1. Connect to Ray and pass W&B key to workers
    wandb_key = os.environ.get("WANDB_API_KEY")
    ray_env = {"env_vars": {"WANDB_API_KEY": wandb_key}} if wandb_key else {}

    if "RAY_ADDRESS" in os.environ:
        ray.init(address="auto", runtime_env=ray_env)
        print("[*] Connected to Slurm Ray Cluster")
    else:
        ray.init(ignore_reinit_error=True, runtime_env=ray_env)

    # 2. Execute the Sweep
    print("[*] Launching Ray Sweep...")
    experiments = get_sweep_configs()
    futures = [train_single_model.remote(mat, hp) for mat, hp in experiments]
    results = ray.get(futures)
    
    # 3. Save Sweep Results
    df = pd.DataFrame(results)
    csv_path = "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Sweep Complete! Results saved to {csv_path}")
    
    # # 4. Immediately Evaluate Top Models
    # top_models = df.sort_values(by="mse", ascending=True).head(7)
    # print("\n🏆 Top Models from the Sweep:")
    # print(top_models[['matrix_id', 'mse', 'path']])

    # matrix_lookup = {exp[0]["matrix_id"]: exp[0]["A_continuous"] for exp in experiments}
    # meta = DatasetMetadata.get("microgrid", {})
    # d_in = len(meta.get("input_labels", [1]))
    # d_out = len(meta.get("output_labels", [1]))

    # for index, row in top_models.iterrows():
    #     mat_id = int(row['matrix_id'])
    #     ckpt_path = row['path']
    #     print(f"\n=======================================================")
    #     print(f"[*] Evaluating Winner {index+1}: Matrix {mat_id}")

    #     if not os.path.exists(ckpt_path):
    #         print(f"❌ Checkpoint not found at {ckpt_path}")
    #         continue

    #     A_continuous = matrix_lookup[mat_id]
    #     Ad, Bd = get_discrete_matrices(A_continuous)
    #     model = load_model_regression(ckpt_path, d_input_arg=d_in, d_output_arg=d_out)
        
    #     plot_title = f"Rank {index+1} | Matrix {mat_id} | d_model={model.d_model} | Test MSE: {row['mse']:.6f}"
        
    #     run_evaluation(
    #         model=model, Ad=Ad, Bd=Bd, 
    #         d_model=model.d_model, n_layers=model.n_layers, 
    #         dataset_name="microgrid", custom_title=plot_title
    #     )
