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

# --- CRITICAL FIX 1: Force headless backend for TACC before importing pyplot ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

# =========================================================================
# 0. LOCAL MODULE IMPORTS
# =========================================================================
from model.s4_code import (
    StackedModelRegression, 
    S4LayerEnsemble, 
    batched_reg_runner
)
from data.dataloader import (
    get_discrete_matrices,
    create_microgrid_dataloaders,
    DatasetMetadata
)
from data.systems import get_sweep_configs

# =========================================================================
# 1. LQR CONTROL SOLVER
# =========================================================================
def compute_lqr_gain(Ad, Bd, Q_weight=1.0, R_weight=0.1):
    """
    Computes the discrete-time LQR feedback gain matrix K via DARE iteration.
    System: x_{t+1} = Ad * x_t + Bd * u_t
    Cost:   J = sum(x_t^T * Q * x_t + u_t^T * R * u_t)
    Law:    u_t = -K * x_t
    """
    nx = Ad.shape[0]  
    nu = Bd.shape[1]  
    
    Q = np.eye(nx) * Q_weight
    R = np.eye(nu) * R_weight
    
    P = np.copy(Q)
    max_iter = 1000
    tol = 1e-6
    
    for _ in range(max_iter):
        term1 = Ad.T @ P @ Bd
        term2 = np.linalg.inv(R + Bd.T @ P @ Bd)
        P_next = Ad.T @ P @ Ad - term1 @ term2 @ Bd.T @ P @ Ad + Q
        
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next

    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
    return K

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
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss
        })

        if test_loss < best_loss:
            best_loss = test_loss
            os.makedirs(os.path.dirname(unique_save_path), exist_ok=True)
            save_model(model, {'dataset': dataset, 'layer': layer, 'model': model_cfg, 'train': train_cfg}, unique_save_path)
    return model, best_loss

# # =========================================================================
# # 3. RAY WORKER
# # =========================================================================
# @ray.remote(num_gpus=0.2)
# def train_single_model(matrix_dict, hp_dict):
#     matrix_id, A_continuous = matrix_dict["matrix_id"], matrix_dict["A_continuous"]
    
#     run = wandb.init(
#         project="tacc-microgrid-s4-sweep", 
#         name=f"matrix_{matrix_id}_lqr",   
#         config={**hp_dict, "matrix_id": matrix_id, "controller": "LQR"}
#     )

#     model_cfg = {k: hp_dict[k] for k in ["d_model", "n_layers", "N", "l_max", "dropout", "prenorm"]}
#     model_cfg["embedding"] = False
#     train_cfg = {"epochs": hp_dict["epochs"], "bsz": hp_dict["batch_size"], "lr": hp_dict["lr"], "weight_decay": 0.0}
#     unique_save_path = f"checkpoints/sweep/mat{matrix_id}_best_model.msgpack"

#     Ad, Bd = get_discrete_matrices(A_continuous)
    
#     trained_model, final_mse = safe_train_regression(
#         "microgrid", "s4", 42, model_cfg, train_cfg, Ad, Bd, unique_save_path
#     )
    
#     wandb.log({"final_sys_id_mse": final_mse})

#     rnn_model = load_model_regression(unique_save_path, d_input_arg=9, d_output_arg=6)

#     print(f"[*] Calculating optimal LQR controller gains for Matrix {matrix_id}...")
#     K_gain = compute_lqr_gain(Ad, Bd, Q_weight=1.0, R_weight=0.1)

#     plot_title = f"Closed-Loop LQR Control | Matrix {matrix_id} | d_model={model_cfg['d_model']}"
#     run_lqr_evaluation(
#         model=rnn_model, 
#         Ad=Ad, 
#         Bd=Bd, 
#         K=K_gain,
#         d_model=model_cfg['d_model'], 
#         n_layers=model_cfg['n_layers'], 
#         dataset_name="microgrid", 
#         custom_title=plot_title
#     )

#     # Force a final sync block to let the background thread push the image asset
#     wandb.finish()
#     return {"matrix_id": matrix_id, "mse": final_mse, "path": unique_save_path}


# # =========================================================================
# # 4. CLOSED-LOOP ROLLOUT & PLOTTING
# # =========================================================================
# def visualize_lqr_plots(controlled_inputs, states, dataset_name="microgrid", n_plot=3, max_channels=4, custom_title=""):
#     controlled_inputs, states = np.array(controlled_inputs), np.array(states)
#     meta = DatasetMetadata.get(dataset_name, {})
#     dt = meta.get("dt", 0.01)
    
#     in_labels = meta.get("input_labels", [f"Control Actuator {d}" for d in range(controlled_inputs.shape[-1])])
#     out_labels = meta.get("output_labels", [f"State Ch {d}" for d in range(states.shape[-1])])
#     time_arr = np.arange(states.shape[1]) * dt

#     fig, axes = plt.subplots(n_plot, 2, figsize=(16, 4 * n_plot), squeeze=False)
#     fig.suptitle(custom_title, fontsize=14, fontweight='bold')

#     for i in range(n_plot):
#         ax_in, ax_out = axes[i, 0], axes[i, 1]
        
#         # Plot Generated LQR Control Inputs (u_t)
#         for d in range(min(controlled_inputs.shape[-1], max_channels)):
#             ax_in.plot(time_arr, controlled_inputs[i, :, d], alpha=0.8, label=in_labels[d])
#         ax_in.set_title(f"Sample {i}: LQR Control Action ($u_t$)")
#         ax_in.grid(True, alpha=0.3)
#         ax_in.legend(loc='upper right')

#         # Plot Regulated Plant States (y_t decaying to zero)
#         for d in range(min(states.shape[-1], max_channels)):
#             ax_out.plot(time_arr, states[i, :, d], '-', linewidth=2, alpha=0.8, label=out_labels[d])
            
#         ax_out.set_title(f"Sample {i}: Regulated Plant States ($y_t$)")
#         ax_out.grid(True, alpha=0.3)
#         ax_out.legend(loc='upper right')

#     plt.tight_layout()
    
#     # Clean the title string for file naming
#     safe_title = custom_title.replace(" | ", "_").replace("=", "").replace(" ", "_").replace("$", "").replace("^", "")
#     save_dir = "plots"
#     os.makedirs(save_dir, exist_ok=True)
#     relative_path = os.path.join(save_dir, f"{safe_title}.png")
    
#     # --- CRITICAL FIX 2: Compute Absolute Paths to map from Ray Virtual Environments ---
#     abs_save_path = os.path.abspath(relative_path)
#     plt.savefig(abs_save_path, bbox_inches='tight', dpi=300)
    
#     # --- CRITICAL FIX 3: Push using the absolute file path destination ---
#     if wandb.run is not None:
#         wandb.log({"Closed_Loop_LQR_Plots": wandb.Image(abs_save_path)})
#         print(f"[*] Successfully logged plot to W&B run: {wandb.run.name}")
        
#     plt.close(fig)
#     print(f"[*] Saved closed-loop control plot locally to {abs_save_path}")

# def run_lqr_evaluation(model, Ad, Bd, K, d_model, n_layers, dataset_name="microgrid", custom_title=""):
#     print(f"[*] Simulating Closed-Loop LQR + S4 System Rollout...")

#     l_max, bsz = 100, 32
#     _, testloader, _, _ = create_microgrid_dataloaders(Ad, Bd, bsz=bsz, L=l_max)

#     targets_y = jnp.array(testloader[0][1]) 
#     initial_states = targets_y[:, 0, :] 
    
#     H_dim, N_dim = d_model, 64 
#     K_jax = jnp.array(K)

#     @nnx.jit
#     def closed_loop_scan(model, x0):
#         B_batch = x0.shape[0]
#         init_s4_states = [jnp.zeros((B_batch, H_dim, N_dim), dtype=jnp.complex64) for _ in range(n_layers)]

#         def lqr_step(carry, _):
#             model_carry, current_s4_states, y_prev = carry
            
#             # 1. Compute control law: u_t = -K * y_{t-1}
#             # y_prev is (B_batch, 6), K_jax.T is (6, 3) -> u_t is (B_batch, 3)
#             u_t = -jnp.matmul(y_prev, K_jax.T)
            
#             # --- THE FIX: Concatenate state and control action ---
#             # The S4 model was trained with d_input=9, expecting [x_t, u_t]
#             s4_input = jnp.concatenate([y_prev, u_t], axis=-1)
#             # ----------------------------------------------------
            
#             def single_sample_step(m, x, s):
#                 pred, new_s = m(x, states=s, training=False)
#                 return pred, new_s

#             vmap_runner = nnx.vmap(
#                 single_sample_step, 
#                 in_axes=(nnx.StateAxes({nnx.Param: None}), 0, 0), 
#                 out_axes=(0, 0)
#             )
            
#             # 2. Pass the concatenated 9-dimensional vector into the S4 plant
#             y_next, next_s4_states = vmap_runner(model_carry, s4_input, current_s4_states)
            
#             return (model_carry, next_s4_states, y_next), (u_t, y_next)

#         initial_carry = (model, init_s4_states, x0)
#         _, (inputs_u_history, states_y_history) = nnx.scan(
#             lqr_step, 
#             in_axes=(nnx.Carry, 0), 
#             out_axes=(nnx.Carry, 0)
#         )(initial_carry, jnp.arange(l_max))
        
#         return jnp.transpose(inputs_u_history, (1, 0, 2)), jnp.transpose(states_y_history, (1, 0, 2))

#     u_rollout, y_rollout = closed_loop_scan(model, initial_states)
#     visualize_lqr_plots(u_rollout, y_rollout, dataset_name=dataset_name, n_plot=3, custom_title=custom_title)


# =========================================================================
# 3. RAY WORKER
# =========================================================================
@ray.remote(num_gpus=0.2)
def train_single_model(matrix_dict, hp_dict):
    matrix_id, A_continuous = matrix_dict["matrix_id"], matrix_dict["A_continuous"]
    
    run = wandb.init(
        project="tacc-microgrid-s4-sweep", 
        name=f"matrix_{matrix_id}_lqr",   
        config={**hp_dict, "matrix_id": matrix_id, "controller": "LQR"},
        # --- CRITICAL FIX 1: Run W&B in a thread so Ray doesn't kill it prematurely ---
        settings=wandb.Settings(start_method="thread") 
    )

    model_cfg = {k: hp_dict[k] for k in ["d_model", "n_layers", "N", "l_max", "dropout", "prenorm"]}
    model_cfg["embedding"] = False
    train_cfg = {"epochs": hp_dict["epochs"], "bsz": hp_dict["batch_size"], "lr": hp_dict["lr"], "weight_decay": 0.0}
    unique_save_path = f"checkpoints/sweep/mat{matrix_id}_best_model.msgpack"

    Ad, Bd = get_discrete_matrices(A_continuous)
    
    trained_model, final_mse = safe_train_regression(
        "microgrid", "s4", 42, model_cfg, train_cfg, Ad, Bd, unique_save_path
    )

    rnn_model = load_model_regression(unique_save_path, d_input_arg=9, d_output_arg=6)

    print(f"[*] Calculating optimal LQR controller gains for Matrix {matrix_id}...")
    K_gain = compute_lqr_gain(Ad, Bd, Q_weight=1.0, R_weight=0.1)

    plot_title = f"Closed-Loop LQR Control | Matrix {matrix_id} | d_model={model_cfg['d_model']}"
    
    # Catch the MATPLOTLIB FIGURE directly from the evaluation function
    fig = run_lqr_evaluation(
        model=rnn_model, 
        Ad=Ad, 
        Bd=Bd, 
        K=K_gain,
        d_model=model_cfg['d_model'], 
        n_layers=model_cfg['n_layers'], 
        dataset_name="microgrid", 
        custom_title=plot_title
    )

    print(f"[*] Uploading memory-buffered plot to W&B run: {run.name}")
    # --- CRITICAL FIX 2: Pass the in-memory figure to W&B, bypassing the filesystem ---
    run.log({
        "final_sys_id_mse": final_mse,
        "Closed_Loop_LQR_Plots": wandb.Image(fig)
    })
    
    plt.close(fig) # Free up the memory now that W&B has buffered it
    run.finish()
    return {"matrix_id": matrix_id, "mse": final_mse, "path": unique_save_path}


# =========================================================================
# 4. CLOSED-LOOP ROLLOUT & PLOTTING
# =========================================================================
def visualize_lqr_plots(controlled_inputs, states, dataset_name="microgrid", n_plot=3, custom_title=""):
    controlled_inputs, states = np.array(controlled_inputs), np.array(states)
    meta = DatasetMetadata.get(dataset_name, {})
    dt = meta.get("dt", 0.01)
    
    all_in_labels = meta.get("input_labels", [f"Input Ch {d}" for d in range(9)])
    in_labels = all_in_labels[-controlled_inputs.shape[-1]:] 
    out_labels = meta.get("output_labels", [f"State Ch {d}" for d in range(states.shape[-1])])
    time_arr = np.arange(states.shape[1]) * dt

    fig, axes = plt.subplots(n_plot, 2, figsize=(16, 4 * n_plot), squeeze=False)
    fig.suptitle(custom_title, fontsize=14, fontweight='bold')

    for i in range(n_plot):
        ax_in, ax_out = axes[i, 0], axes[i, 1]
        
        for d in range(controlled_inputs.shape[-1]):
            ax_in.plot(time_arr, controlled_inputs[i, :, d], alpha=0.8, label=in_labels[d])
        ax_in.set_title(f"Sample {i}: LQR Control Action ($u_t$)")
        ax_in.grid(True, alpha=0.3)
        ax_in.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax_in.legend(loc='upper right')

        for d in range(states.shape[-1]):
            ax_out.plot(time_arr, states[i, :, d], '-', linewidth=2, alpha=0.8, label=out_labels[d])
            
        ax_out.set_title(f"Sample {i}: Regulated Plant States ($y_t$)")
        ax_out.grid(True, alpha=0.3)
        ax_out.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax_out.legend(loc='upper right')

    plt.tight_layout()
    
    # Still save a backup copy to your TACC drive
    safe_title = custom_title.replace(" | ", "_").replace("=", "").replace(" ", "_").replace("$", "").replace("^", "")
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)
    abs_save_path = os.path.abspath(os.path.join(save_dir, f"{safe_title}.png"))
    plt.savefig(abs_save_path, bbox_inches='tight', dpi=300)
    print(f"[*] Saved closed-loop control plot locally to {abs_save_path}")
    
    # --- CRITICAL FIX 3: Return the actual FIGURE OBJECT instead of the file path ---
    return fig

def run_lqr_evaluation(model, Ad, Bd, K, d_model, n_layers, dataset_name="microgrid", custom_title=""):
    print(f"[*] Simulating Closed-Loop LQR + S4 System Rollout...")

    l_max, bsz = 200, 32
    _, testloader, _, _ = create_microgrid_dataloaders(Ad, Bd, bsz=bsz, L=l_max)

    targets_y = jnp.array(testloader[0][1]) 
    initial_states = targets_y[:, 0, :] 
    
    H_dim, N_dim = d_model, 64 
    K_jax = jnp.array(K)

    @nnx.jit
    def closed_loop_scan(model, x0):
        B_batch = x0.shape[0]
        init_s4_states = [jnp.zeros((B_batch, H_dim, N_dim), dtype=jnp.complex64) for _ in range(n_layers)]

        def lqr_step(carry, _):
            model_carry, current_s4_states, y_prev = carry
            
            u_t = -jnp.matmul(y_prev, K_jax.T)
            s4_input = jnp.concatenate([y_prev, u_t], axis=-1)
            
            def single_sample_step(m, x, s):
                pred, new_s = m(x, states=s, training=False)
                return pred, new_s

            vmap_runner = nnx.vmap(
                single_sample_step, 
                in_axes=(nnx.StateAxes({nnx.Param: None}), 0, 0), 
                out_axes=(0, 0)
            )
            
            y_next, next_s4_states = vmap_runner(model_carry, s4_input, current_s4_states)
            
            return (model_carry, next_s4_states, y_next), (u_t, y_next)

        initial_carry = (model, init_s4_states, x0)
        _, (inputs_u_history, states_y_history) = nnx.scan(
            lqr_step, 
            in_axes=(nnx.Carry, 0), 
            out_axes=(nnx.Carry, 0)
        )(initial_carry, jnp.arange(l_max))
        
        return jnp.transpose(inputs_u_history, (1, 0, 2)), jnp.transpose(states_y_history, (1, 0, 2))

    u_rollout, y_rollout = closed_loop_scan(model, initial_states)
    
    # Pass the figure object up the chain
    return visualize_lqr_plots(u_rollout, y_rollout, dataset_name=dataset_name, n_plot=3, custom_title=custom_title)



# =========================================================================
# 5. MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    wandb_key = os.environ.get("WANDB_API_KEY")
    
    ray_env = {
        "working_dir": ".",  
        "env_vars": {
            "WANDB_API_KEY": wandb_key,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.10"
        }
    }

    if "RAY_ADDRESS" in os.environ:
        ray.init(address="auto", runtime_env=ray_env)
        print("[*] Connected to Slurm Ray Cluster")
    else:
        ray.init(ignore_reinit_error=True, runtime_env=ray_env)

    print("[*] Launching Distributed LQR Control Sweep...")
    experiments = get_sweep_configs()
    futures = [train_single_model.remote(mat, hp) for mat, hp in experiments]
    results = ray.get(futures)
    
    df = pd.DataFrame(results)
    csv_path = "lqr_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ LQR Control Sweep Complete! Summary logs cataloged in {csv_path}")
