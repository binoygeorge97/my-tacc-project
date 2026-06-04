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
# 0. LOCAL MODULE IMPORTS
# =========================================================================
# Import from the 'model' folder
from model.s4_code import (
    StackedModelRegression, 
    S4LayerEnsemble, 
    batched_reg_runner
)

# Import from the 'data' folder
from data.dataloader import (
    get_discrete_matrices,
    create_microgrid_dataloaders,
    DatasetMetadata
)

from data.systems import get_sweep_configs


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
    #ray_env = {"env_vars": {"WANDB_API_KEY": wandb_key}} if wandb_key else {}
    # --- CRITICAL FIX: Add working_dir ---
    ray_env = {
        "working_dir": ".",  # This tells Ray to copy data/ and model/ to all nodes
        "env_vars": {"WANDB_API_KEY": wandb_key} if wandb_key else {}
    }
    # -------------------------------------

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
