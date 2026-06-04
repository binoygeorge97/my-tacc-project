import os
import glob
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax import serialization
import matplotlib.pyplot as plt

# =========================================================================
# 0. LOCAL MODULE IMPORTS
# =========================================================================
from model.s4_code import StackedModelRegression, S4LayerEnsemble
from data.dataloader import get_discrete_matrices, create_microgrid_dataloaders, DatasetMetadata
from data.systems import get_sweep_configs

# =========================================================================
# 1. UTILS
# =========================================================================
def load_model_regression(filename, d_input_arg=9, d_output_arg=6):
    with open(filename, 'rb') as f:
        byte_data = f.read()
    raw_structure = serialization.msgpack_restore(byte_data)
    config = raw_structure['config']
    
    l_max = config['model'].get('l_max', 100)
    s4_N = config['model'].get('N', 64)

    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    model = StackedModelRegression(
        layer_cls=S4LayerEnsemble, layer_args={'N': s4_N, 'l_max': l_max},
        d_input=d_input_arg, d_output=d_output_arg,
        d_model=config['model']['d_model'], n_layers=config['model']['n_layers'],
        dropout=config['model']['dropout'], prenorm=config['model']['prenorm'],
        decode=True, rngs=rngs
    )
    current_state_dict = nnx.state(model, nnx.Param).to_pure_dict()
    restored = serialization.from_bytes({'model_state': current_state_dict, 'config': config}, byte_data)
    nnx.update(model, restored['model_state'])
    return model, config

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
    safe_title = custom_title.replace(" | ", "_").replace("=", "").replace(" ", "_")
    save_path = f"plots/{safe_title}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[*] Saved evaluation plot to {save_path}")

# =========================================================================
# 2. INFERENCE RUNNER
# =========================================================================
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

            runner = nnx.vmap(single_sample_step, in_axes=(nnx.StateAxes({nnx.Param: None}), 0, 0), out_axes=(0, 0))
            pred_batch, new_states_batch = runner(model_carry, x_t, current_states_batch)
            return (model_carry, new_states_batch), pred_batch

        initial_carry = (model, init_states)
        _, preds_t = nnx.scan(scan_step, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(initial_carry, inputs_t)
        return jnp.transpose(preds_t, (1, 0, 2))

    preds_y = step_by_step_inference(model, inputs_u)
    visualize_system_plots(inputs_u, targets_y, preds_y, n_plot=3, dataset_name=dataset_name, custom_title=custom_title)

# =========================================================================
# 3. BATCH EVALUATION EXECUTION
# =========================================================================
if __name__ == "__main__":
    checkpoint_dir = "checkpoints/sweep/"
    model_files = glob.glob(os.path.join(checkpoint_dir, "*.msgpack"))
    
    if not model_files:
        print("[!] No checkpoints found in", checkpoint_dir)
        exit()

    # Get the raw continuous matrices so we can re-discretize them for the test loader
    experiments = {str(cfg[0]['matrix_id']): cfg[0]['A_continuous'] for cfg in get_sweep_configs()}

    for filepath in model_files:
        print(f"\n[*] Evaluating {filepath}...")
        
        # 1. Load Model & Config
        rnn_model, config_dict = load_model_regression(filepath, d_input_arg=9, d_output_arg=6)
        
        # 2. Retrieve corresponding Physics Matrices
        matrix_id = str(config_dict['model'].get('matrix_id', "unknown"))
        A_continuous = experiments.get(matrix_id)
        
        if A_continuous is None:
            print(f"[!] Warning: Could not find physics matrix for ID {matrix_id}. Skipping.")
            continue
            
        Ad, Bd = get_discrete_matrices(A_continuous)
        
        # 3. Generate Plot
        plot_title = f"Matrix {matrix_id} | d_model={config_dict['model']['d_model']}"
        run_evaluation(
            model=rnn_model, 
            Ad=Ad, Bd=Bd, 
            d_model=config_dict['model']['d_model'], 
            n_layers=config_dict['model']['n_layers'], 
            custom_title=plot_title
        )
