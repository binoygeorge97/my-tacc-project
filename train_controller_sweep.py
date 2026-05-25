# train_controller_sweep.py

import os
# THESE MUST COME BEFORE ANY OTHER IMPORTS!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"

import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import matplotlib.pyplot as plt
from flax import serialization
import ray
import wandb
import time

# --- Local TACC Imports ---
from model.controller import GRUController, Trained_System_Model
from data.dataloader import get_discrete_matrices, get_sweep_configs

@nnx.jit
def train_ctrl_step(ctrl, env_model, optimizer, y_targets, y_initial):
    batch_size = y_targets.shape[0]
    initial_s4_carry = env_model.initialize_carry(batch_size=batch_size)
    initial_ctrl_carry = ctrl.initialize_carry(batch_size)

    env_graph, env_params = nnx.split(env_model, nnx.Param)

    def loss_fn(current_ctrl):
        def scan_step(carry, target_t_batch):
            y_curr, s4_c, ctrl_c = carry
            u_cmd, new_ctrl_c = current_ctrl(y_curr, target_t_batch, ctrl_c)

            def pure_env_step(p, u, y, c):
                m = nnx.merge(env_graph, p)
                return m(u, y, c)

            y_next, new_s4_c = jax.vmap(
                pure_env_step,
                in_axes=(None, 0, 0, 0),
                out_axes=(0, 0)
            )(env_params, u_cmd, y_curr, s4_c)

            y_next = jnp.real(y_next)
            step_loss = jnp.mean((y_next - target_t_batch) ** 2)
            return (y_next, new_s4_c, new_ctrl_c), step_loss

        initial_carry = (y_initial, initial_s4_carry, initial_ctrl_carry)
        targets_seq = jnp.transpose(y_targets, (1, 0, 2))
        _, step_losses = jax.lax.scan(scan_step, initial_carry, targets_seq)
        return jnp.mean(step_losses)

    loss, grad = nnx.value_and_grad(loss_fn)(ctrl)
    optimizer.update(ctrl, grad)
    return loss

def save_controller(ctrl, max_u_val, filename):
    ctrl_state = nnx.state(ctrl, nnx.Param).to_pure_dict()
    checkpoint_data = {'model_state': ctrl_state, 'config': {'max_u': max_u_val}}
    byte_data = serialization.to_bytes(checkpoint_data)
    with open(filename, 'wb') as f:
        f.write(byte_data)

@ray.remote(num_gpus=0.15) # TACC Optimized Ratio
def train_single_controller(matrix_id, A_continuous, s4_ckpt_path, max_u_val):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"
    
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    import wandb 

    # --- FIX: Bring the step compilation into the remote worker scope ---
    from train_controller_sweep import train_ctrl_step

    run = wandb.init(
        project="tacc-microgrid-s4-sweep",
        group="gru_saturation_sweep",
        name=f"mat{matrix_id}_maxu{max_u_val}",
        config={"matrix_id": matrix_id, "max_action": max_u_val, "epochs": 40, "batch_size": 32, "seq_len": 100},
        reinit=True
    )

    Ad, Bd = get_discrete_matrices(A_continuous)
    d_x, d_u, d_y = 6, 3, 6

    env_model = Trained_System_Model(
        s4_ckpt_path, d_input_arg=9, d_output_arg=d_y,
        format_fn=lambda u, y: jnp.concatenate([y, u], axis=-1)
    )

    ctrl = GRUController(d_y=d_y, d_u=d_u, max_action=max_u_val, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(ctrl, optax.adam(1e-4), wrt=nnx.Param)

    L_seq, B_size = 100, 32

    for i in range(40):
        y_targets = jnp.zeros((B_size, L_seq, d_y))
        y_initial = jax.random.uniform(jax.random.PRNGKey(i), (B_size, d_y), minval=-1.0, maxval=1.0)
        loss = train_ctrl_step(ctrl, env_model, optimizer, y_targets, y_initial)
        wandb.log({"train/mse_loss": float(loss), "epoch": i})

    def jax_simulate_real_step(x_prev, u_curr):
        return jnp.dot(Ad, x_prev) + jnp.dot(Bd, u_curr)

    @jax.jit
    def fast_test_loop(ctrl_model, x_initial_state, test_targets):
        c_carry = ctrl_model.initialize_carry(batch_size=1)
        y_initial = x_initial_state.flatten()[jnp.newaxis, :]

        def test_scan(carry, target_t):
            y_curr, c_c, x_c = carry
            target_t_batch = target_t[jnp.newaxis, :]
            u_cmd, new_c_c = ctrl_model(y_curr, target_t_batch, c_c)
            u_mat = jnp.transpose(u_cmd)
            x_next = jax_simulate_real_step(x_c, u_mat)
            y_next = x_next.flatten()[jnp.newaxis, :]
            return (y_next, new_c_c, x_next), y_next

        initial_carry = (y_initial, c_carry, x_initial_state)
        _, y_history = jax.lax.scan(test_scan, initial_carry, test_targets)
        return jnp.squeeze(y_history, axis=1)

    test_L = 150
    y_target_test = jnp.zeros((test_L, d_y))
    x_real_init = jnp.array([[1.0], [-0.8], [0.5], [-0.5], [1.2], [-1.0]])

    y_actual_hist = np.array(fast_test_loop(ctrl, x_real_init, y_target_test))
    final_test_mse = float(np.mean(y_actual_hist[-50:] ** 2))

    # --- FIX: Generate Plot Directly Inside Active Worker Thread ---
    t_test = np.linspace(0, test_L * 0.01, test_L)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Matrix {matrix_id} | max_u = {max_u_val} | Final Test MSE: {final_test_mse:.2f}", fontsize=16, fontweight='bold')

    for y_dim in range(d_y): 
        ax1.plot(t_test, y_actual_hist[:, y_dim], alpha=0.8)
    ax1.plot(t_test, y_target_test[:, 0], 'k--', linewidth=2)
    ax1.set_title("System States over Time")
    ax1.grid(True, alpha=0.3)

    # Re-run control loop briefly to extract historical command actions for plotting
    u_hist = []
    c_carry = ctrl.initialize_carry(batch_size=1)
    y_curr_p = x_real_init.flatten()
    for t in range(test_L):
        u_cmd, c_carry = ctrl(y_curr_p, y_target_test[t], c_carry)
        u_flat = np.array(u_cmd).flatten()
        u_hist.append(u_flat)
        x_real_next = jax_simulate_real_step(x_real_init, u_flat.reshape(d_u, 1)) # dummy step for plotting path
        y_curr_p = np.array(x_real_next).flatten()
    u_hist = np.array(u_hist)

    for u_dim in range(d_u): 
        ax2.plot(t_test, u_hist[:, u_dim], alpha=0.8)
    ax2.axhline(max_u_val, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(-max_u_val, color='red', linestyle=':', alpha=0.5)
    ax2.set_title("Controller Action")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot_path = f"eval_mat{matrix_id}_maxu{max_u_val}.png"
    plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
    
    # Send metrics and images sequentially through the active socket
    wandb.log({
        "test/sim_to_real_mse": final_test_mse,
        "evaluation/rollout_plot": wandb.Image(save_plot_path)
    })
    plt.close(fig)
    # -----------------------------------------------------------------

    run.finish() # Closes securely

    save_path = f"checkpoints/controllers/gru_mat{matrix_id}_maxu{max_u_val}.msgpack"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_controller(ctrl, max_u_val, save_path)

    return {"matrix_id": matrix_id, "max_u": max_u_val, "sim_to_real_mse": final_test_mse, "ctrl_path": save_path}


def load_gru_controller(ckpt_path, d_y, d_u, max_action):
    with open(ckpt_path, 'rb') as f:
        byte_data = f.read()
    ctrl = GRUController(d_y=d_y, d_u=d_u, max_action=max_action, rngs=nnx.Rngs(0))
    current_state_dict = nnx.state(ctrl, nnx.Param).to_pure_dict()
    template = {'model_state': current_state_dict, 'config': {'max_u': max_action}}
    restored = serialization.from_bytes(template, byte_data)
    nnx.update(ctrl, restored['model_state'])
    return ctrl

# def plot_gru_results():
#     csv_path = "gru_sweep_results.csv"
#     df = pd.read_csv(csv_path)
#     experiments = get_sweep_configs()
#     matrix_lookup = {str(exp[0]["matrix_id"]): exp[0]["A_continuous"] for exp in experiments}

#     d_x, d_u, d_y, test_L, dt = 6, 3, 6, 150, 0.01
#     t_test = np.linspace(0, test_L * dt, test_L)

#     for index, row in df.iterrows():
#         mat_id = str(row['matrix_id'])
#         max_u = float(row['max_u'])
#         ckpt_path = row['ctrl_path']
#         mse = float(row['sim_to_real_mse'])

#         run = wandb.init(
#             project="tacc-microgrid-s4-sweep",
#             group="gru_saturation_sweep",
#             job_type="evaluation", 
#             name=f"eval_mat{mat_id}_maxu{max_u}",
#             config={"matrix_id": mat_id, "max_u": max_u, "test_mse": mse},
#             reinit=True
#         )

#         Ad, Bd = get_discrete_matrices(matrix_lookup[mat_id])
#         ctrl = load_gru_controller(ckpt_path, d_y, d_u, max_u)
#         ctrl_carry = ctrl.initialize_carry(batch_size=1)

#         y_target_test = jnp.zeros((test_L, d_y))
#         x_real = jnp.array([[1.0], [-0.8], [0.5], [-0.5], [1.2], [-1.0]])
#         y_curr = x_real.flatten()

#         y_actual_hist, u_hist = [], []

#         for t in range(test_L):
#             u_cmd, ctrl_carry = ctrl(y_curr, y_target_test[t], ctrl_carry)
#             u_flat = np.array(u_cmd).flatten()
#             u_hist.append(u_flat)
#             x_real = jnp.dot(Ad, x_real) + jnp.dot(Bd, u_flat.reshape(d_u, 1))
#             y_curr = np.array(x_real).flatten()
#             y_actual_hist.append(y_curr)

#         y_actual_hist, u_hist = np.array(y_actual_hist), np.array(u_hist)

#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#         fig.suptitle(f"Matrix {mat_id} | max_u = {max_u} | Final Test MSE: {mse:.2f}", fontsize=16, fontweight='bold')

#         for y_dim in range(d_y): ax1.plot(t_test, y_actual_hist[:, y_dim], alpha=0.8)
#         ax1.plot(t_test, y_target_test[:, 0], 'k--', linewidth=2)
#         ax1.set_title("System States over Time")
#         ax1.grid(True, alpha=0.3)

#         for u_dim in range(d_u): ax2.plot(t_test, u_hist[:, u_dim], alpha=0.8)
#         ax2.axhline(max_u, color='red', linestyle=':', alpha=0.5)
#         ax2.axhline(-max_u, color='red', linestyle=':', alpha=0.5)
#         ax2.set_title("Controller Action")
#         ax2.grid(True, alpha=0.3)

#         # --- TACC Headless Plotting Fix ---
#         save_plot_path = f"eval_mat{mat_id}_maxu{max_u}.png"
#         plt.tight_layout()
#         plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
        
#         wandb.log({"evaluation/rollout_plot": wandb.Image(save_plot_path)})
#         plt.close(fig)
#         run.finish()

if __name__ == "__main__":
    ray.shutdown()
    wandb_key = os.environ.get("WANDB_API_KEY")
    
    # --- TACC Slurm Ray Cluster Fix ---
    ray_env = {
        "working_dir": ".", 
        "env_vars": {"WANDB_API_KEY": wandb_key} if wandb_key else {}
    }
    if "RAY_ADDRESS" in os.environ:
        ray.init(address="auto", runtime_env=ray_env)
        print("[*] Connected to Slurm Ray Cluster")
    else:
        ray.init(ignore_reinit_error=True, runtime_env=ray_env)
    # ----------------------------------

    print("\n[*] Initializing Stage 2: GRU Controller Sweep...")
    df_s4 = pd.read_csv("sweep_results.csv")
    experiments = get_sweep_configs()
    matrix_lookup = {str(exp[0]["matrix_id"]): exp[0]["A_continuous"] for exp in experiments}

    gru_jobs = []
    unique_matrices = df_s4['matrix_id'].unique()

    for mat_id in unique_matrices:
        mat_id = str(mat_id)
        best_s4_row = df_s4[df_s4['matrix_id'] == mat_id].sort_values(by="mse").iloc[0]
        best_s4_path = best_s4_row['path']
        
        for max_u in [30.0, 50.0, 100.0, 150.0]:
            gru_jobs.append({"mat_id": mat_id, "A_cont": matrix_lookup[mat_id], "s4_path": best_s4_path, "max_u": max_u})

    print(f"[*] Launching {len(gru_jobs)} parallel GRU training jobs...")
    futures = [train_single_controller.remote(j["mat_id"], j["A_cont"], j["s4_path"], j["max_u"]) for j in gru_jobs]
    df_ctrl = pd.DataFrame(ray.get(futures))
    df_ctrl.to_csv("gru_sweep_results.csv", index=False)

    print("\n✅ GRU Sweep Complete!")
    #plot_gru_results()
