# model/controller.py

import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization
from model.s4_code import StackedModelRegression, S4LayerEnsemble

class NeuralEnvironment(nnx.Module):
    def __init__(self, model, formatter_fn):
        self.model = model
        self.formatter_fn = formatter_fn

    def initialize_carry(self, batch_size=None):
        H_dim = self.model.d_model
        N_dim = self.model.layers[0].seq.N
        n_layers = self.model.n_layers

        if batch_size is None:
            return [jnp.zeros((H_dim, N_dim), dtype=jnp.complex64) for _ in range(n_layers)]
        else:
            return [jnp.zeros((batch_size, H_dim, N_dim), dtype=jnp.complex64) for _ in range(n_layers)]

    def __call__(self, u_curr, y_curr, carry):
        x = self.formatter_fn(u_curr, y_curr)
        y_next, new_carry = self.model(x, states=carry, training=False)
        return y_next, new_carry

class GRUController(nnx.Module):
    def __init__(self, d_y, d_u, max_action, rngs):
        self.encoder = nnx.Linear(d_y * 2, 128, rngs=rngs)
        self.gru = nnx.GRUCell(128, 128, rngs=rngs)
        self.hidden_layer = nnx.Linear(128, 256, rngs=rngs)
        self.hidden_layer2 = nnx.Linear(256, 256, rngs=rngs)
        self.hidden_layer3 = nnx.Linear(256, 128, rngs=rngs)
        self.head = nnx.Linear(128, d_u, rngs=rngs)
        self.max_action = max_action

    def __call__(self, y_curr, y_target, carry):
        x = jnp.concatenate([y_curr, y_target], axis=-1)
        skip = self.encoder(x)
        new_carry, hidden = self.gru(carry, skip)
        hidden = nnx.leaky_relu(self.hidden_layer(hidden))
        hidden = nnx.leaky_relu(self.hidden_layer2(hidden))
        hidden = nnx.swish(self.hidden_layer3(hidden))
        u = jax.nn.tanh(self.head(hidden + skip*0.05)) * self.max_action
        return u, new_carry

    def initialize_carry(self, batch_size):
        return self.gru.initialize_carry(input_shape=(batch_size, 128), rngs=nnx.Rngs(0))

def Trained_System_Model(filename, d_input_arg, d_output_arg, format_fn):
    print(f"[*] Reading {filename}...")
    with open(filename, 'rb') as f:
        byte_data = f.read()

    raw_structure = serialization.msgpack_restore(byte_data)
    config = raw_structure['config']

    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    base_model = StackedModelRegression(
        layer_cls=S4LayerEnsemble,
        layer_args={'N': config['model'].get('N', 64), 'l_max': config['model'].get('l_max', 100)},
        d_input=d_input_arg,
        d_output=d_output_arg,
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        dropout=config['model']['dropout'],
        prenorm=config['model']['prenorm'],
        decode=True,
        rngs=rngs
    )

    current_state_dict = nnx.state(base_model, nnx.Param).to_pure_dict()
    template = {'model_state': current_state_dict, 'config': config}
    restored = serialization.from_bytes(template, byte_data)
    nnx.update(base_model, restored['model_state'])
    return NeuralEnvironment(base_model, format_fn)
