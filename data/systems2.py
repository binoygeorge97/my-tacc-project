import numpy as np
import itertools

def get_sweep_configs():
    # 1. Define your base hyperparameters (keeping your updated l_max=200)
    base_hp = {
        "dropout": 0.0, 
        "prenorm": True, 
        "lr": 1e-3, 
        "batch_size": 32, 
        "epochs": 70, 
        "l_max": 200
    }
    
    # 2. Define the grid search parameters
    N_vals = [32, 64, 128]
    d_model_vals = [32, 64, 128]
    n_layers_vals = [1, 2, 4]

    Z = np.zeros((2, 2))
    
    # ==========================================
    # 3. Define Physics Matrices
    # ==========================================
    # Matrix 1
    A1 = np.block([[np.array([[-3.5, -2.4], [0.0, 0.0]]), np.array([[0.0, 0.03], [0.0, 0.0]]), np.array([[0.0, 0.06], [0.0, 0.0]])],
                   [Z, np.array([[-3.5, -2.3], [0.0, 0.0]]), Z], [Z, Z, np.array([[-5.2, -5.3], [0.0, 0.0]])]])

    # Matrix 1A
    A4_1a = np.array([[-2.5, -2.0], [ 0.0,  0.0]])
    A5_1a = np.array([[-3.0, -1.5], [ 0.0,  0.0]])
    A6_1a = np.array([[-4.0, -2.2], [ 0.0,  0.0]])
    A1a = np.block([[A4_1a, Z, Z], [Z, A5_1a, Z], [Z, Z, A6_1a]])

    # Matrix 2
    A2 = np.block([[np.array([[1.0, 0.5], [0.0, 2.0]]), Z, Z], [Z, np.array([[1.5, 0.5], [0.0, 3.0]]), Z], [Z, Z, np.array([[0.5, 0.5], [0.0, 4.0]])]])

    # Matrix 3 (Oscillatory unstable)
    A4_3 = np.array([[ 1.0,  2.0], [-2.0,  1.0]])
    A5_3 = np.array([[ 2.0,  5.0], [-5.0,  2.0]])
    A6_3 = np.array([[ 1.5, 10.0], [-10.0, 1.5]])
    A3 = np.block([[A4_3, Z, Z], [Z, A5_3, Z], [Z, Z, A6_3]])

    # Matrix 4 (Non-normal Jordan block)
    A4_4 = np.array([[1.0, 100.0], [0.0,   1.0]])
    A5_4 = np.array([[1.5,  80.0], [0.0,   1.5]])
    A6_4 = np.array([[2.0, 120.0], [0.0,   2.0]])
    A4 = np.block([[A4_4, Z, Z], [Z, A5_4, Z], [Z, Z, A6_4]])

    # Matrix 5 (Weakly coupled unstable)
    A4_5 = np.array([[1.0, 1.0], [0.0, 2.0]])
    A5_5 = np.array([[1.5, 0.7], [0.0, 3.0]])
    A6_5 = np.array([[0.5, 1.2], [0.0, 4.0]])
    H45_5 = np.array([[ 0.3, -0.2], [ 0.0,  0.4]])
    H46_5 = np.array([[ 0.1,  0.0], [ 0.0, -0.3]])
    H56_5 = np.array([[ 0.25, -0.15], [ 0.0,   0.2]])
    A5 = np.block([[A4_5, H45_5, H46_5], [Z, A5_5, H56_5], [Z, Z, A6_5]])

    # Matrix 6 (Ill-conditioned)
    A4_6 = np.array([[1.0, 1000.0], [0.001,   1.0]])
    A5_6 = np.array([[1.5,  800.0], [0.002,   1.5]])
    A6_6 = np.array([[2.0, 1200.0], [0.0015,  2.0]])
    A6 = np.block([[A4_6, Z, Z], [Z, A5_6, Z], [Z, Z, A6_6]])

    # Matrix 7 (Mixed stable / unstable)
    A4_7 = np.array([[-2.0, 0.5], [ 0.0, 0.5]])
    A5_7 = np.array([[-1.0, 0.5], [ 0.0, 1.5]])
    A6_7 = np.array([[-3.0, 0.5], [ 0.0, 2.0]])
    A7 = np.block([[A4_7, Z, Z], [Z, A5_7, Z], [Z, Z, A6_7]])

    # 4. Create an active list of matrices you want to sweep over
    matrices_to_test = [
        {"matrix_id": "1", "A_continuous": A1},
        {"matrix_id": "2", "A_continuous": A2},
        {"matrix_id": "3", "A_continuous": A3}
        # {"matrix_id": "4", "A_continuous": A4},
        # {"matrix_id": "5", "A_continuous": A5},
        # {"matrix_id": "6", "A_continuous": A6},
        # {"matrix_id": "7", "A_continuous": A7},
        # {"matrix_id": "1a", "A_continuous": A1a}
    ]

    # ==========================================
    # 5. Generate Combinatorial Grid
    # ==========================================
    configs = []
    
    for mat in matrices_to_test:
        # itertools.product creates every possible combination of your 3 arrays
        for N, d_model, n_layers in itertools.product(N_vals, d_model_vals, n_layers_vals):
            
            # Copy the base dict so we don't accidentally overwrite it
            current_hp = base_hp.copy()
            current_hp["N"] = N
            current_hp["d_model"] = d_model
            current_hp["n_layers"] = n_layers
            
            # Append the combo to the execution queue
            configs.append([mat, current_hp])

    return configs
