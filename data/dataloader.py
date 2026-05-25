# dataloader.py
import numpy as np

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

    # Matrix 2A
    A4_2a = np.array([[1.2, 0.3],
               [0.0, 2.2]])

    A5_2a = np.array([[1.7, 0.4],
                   [0.0, 3.5]])
    
    A6_2a = np.array([[0.8, 0.6],
                   [0.0, 4.5]])
    A2a = np.block([
        [A4_2a, Z, Z],
        [Z, A5_2a, Z],
        [Z, Z, A6_2a]
    ])
    configs.append([{"matrix_id": "2a", "A_continuous": A2a}, hp])

    # Matrix 2B
    A4_2b = np.array([[0.9, 0.7],
               [0.0, 1.8]])

    A5_2b = np.array([[1.3, 0.8],
                   [0.0, 2.7]])
    
    A6_2b = np.array([[1.1, 0.5],
                   [0.0, 3.9]])
    A2b = np.block([
        [A4_2b, Z, Z],
        [Z, A5_2b, Z],
        [Z, Z, A6_2b]
    ])
    configs.append([{"matrix_id": "2b", "A_continuous": A2b}, hp])
    
    
    # ==========================================
    # 3. Oscillatory unstable (complex eigenvalues)
    # ==========================================
    A4_3 = np.array([[ 1.0,  2.0],
                     [-2.0,  1.0]])

    A5_3 = np.array([[ 2.0,  5.0],
                     [-5.0,  2.0]])

    A6_3 = np.array([[ 1.5, 10.0],
                     [-10.0, 1.5]])

    A3 = np.block([
        [A4_3, Z, Z],
        [Z, A5_3, Z],
        [Z, Z, A6_3]
    ])

    configs.append([{"matrix_id": "3", "A_continuous": A3}, hp])


    # Matrix 3A
    A4_3a = np.array([[ 1.2,  3.0],
                      [-3.0,  1.2]])

    A5_3a = np.array([[ 2.5,  4.0],
                      [-4.0,  2.5]])

    A6_3a = np.array([[ 1.8,  6.0],
                      [-6.0,  1.8]])

    A3a = np.block([
        [A4_3a, Z, Z],
        [Z, A5_3a, Z],
        [Z, Z, A6_3a]
    ])

    configs.append([{"matrix_id": "3a", "A_continuous": A3a}, hp])


    # Matrix 3B
    A4_3b = np.array([[ 0.7,  5.0],
                      [-5.0,  0.7]])

    A5_3b = np.array([[ 1.9,  7.0],
                      [-7.0,  1.9]])

    A6_3b = np.array([[ 1.2,  9.0],
                      [-9.0,  1.2]])

    A3b = np.block([
        [A4_3b, Z, Z],
        [Z, A5_3b, Z],
        [Z, Z, A6_3b]
    ])

    configs.append([{"matrix_id": "3b", "A_continuous": A3b}, hp])


    # ==========================================
    # 4. Non-normal Jordan block
    # ==========================================
    A4_4 = np.array([[1.0, 100.0],
                     [0.0,   1.0]])

    A5_4 = np.array([[1.5,  80.0],
                     [0.0,   1.5]])

    A6_4 = np.array([[2.0, 120.0],
                     [0.0,   2.0]])

    A4_case = np.block([
        [A4_4, Z, Z],
        [Z, A5_4, Z],
        [Z, Z, A6_4]
    ])

    configs.append([{"matrix_id": "4", "A_continuous": A4_case}, hp])


    # Matrix 4A
    A4_4a = np.array([[1.0, 150.0],
                      [0.0,   1.0]])

    A5_4a = np.array([[1.3, 100.0],
                      [0.0,   1.3]])

    A6_4a = np.array([[1.8, 180.0],
                      [0.0,   1.8]])

    A4a = np.block([
        [A4_4a, Z, Z],
        [Z, A5_4a, Z],
        [Z, Z, A6_4a]
    ])

    configs.append([{"matrix_id": "4a", "A_continuous": A4a}, hp])


    # Matrix 4B
    A4_4b = np.array([[0.8, 200.0],
                      [0.0,   0.8]])

    A5_4b = np.array([[1.2, 120.0],
                      [0.0,   1.2]])

    A6_4b = np.array([[1.6, 220.0],
                      [0.0,   1.6]])

    A4b = np.block([
        [A4_4b, Z, Z],
        [Z, A5_4b, Z],
        [Z, Z, A6_4b]
    ])

    configs.append([{"matrix_id": "4b", "A_continuous": A4b}, hp])


    # ==========================================
    # 5. Weakly coupled unstable transition case
    # ==========================================
    A4_5 = np.array([[1.0, 1.0],
                     [0.0, 2.0]])

    A5_5 = np.array([[1.5, 0.7],
                     [0.0, 3.0]])

    A6_5 = np.array([[0.5, 1.2],
                     [0.0, 4.0]])

    H45_5 = np.array([[ 0.3, -0.2],
                      [ 0.0,  0.4]])

    H46_5 = np.array([[ 0.1,  0.0],
                      [ 0.0, -0.3]])

    H56_5 = np.array([[ 0.25, -0.15],
                      [ 0.0,   0.2]])

    A5_case = np.block([
        [A4_5, H45_5, H46_5],
        [Z,    A5_5,  H56_5],
        [Z,    Z,     A6_5 ]
    ])

    configs.append([{"matrix_id": "5", "A_continuous": A5_case}, hp])


    # Matrix 5A
    A4_5a = np.array([[1.1, 0.9],
                      [0.0, 2.2]])

    A5_5a = np.array([[1.6, 0.6],
                      [0.0, 3.2]])

    A6_5a = np.array([[0.7, 1.0],
                      [0.0, 4.2]])

    H45_5a = np.array([[0.2, -0.1],
                       [0.0,  0.3]])

    H46_5a = np.array([[0.1,  0.2],
                       [0.0, -0.2]])

    H56_5a = np.array([[0.15, -0.05],
                       [0.0,   0.2]])

    A5a = np.block([
        [A4_5a, H45_5a, H46_5a],
        [Z,     A5_5a,  H56_5a],
        [Z,     Z,      A6_5a ]
    ])

    configs.append([{"matrix_id": "5a", "A_continuous": A5a}, hp])


    # Matrix 5B
    A4_5b = np.array([[1.3, 0.8],
                      [0.0, 2.5]])

    A5_5b = np.array([[1.8, 0.6],
                      [0.0, 3.0]])

    A6_5b = np.array([[0.6, 1.1],
                      [0.0, 4.1]])

    H45_5b = np.array([[0.25, -0.15],
                       [0.0,   0.35]])

    H46_5b = np.array([[0.05,  0.1],
                       [0.0,  -0.25]])

    H56_5b = np.array([[0.2, -0.1],
                       [0.0,  0.3]])

    A5b = np.block([
        [A4_5b, H45_5b, H46_5b],
        [Z,     A5_5b,  H56_5b],
        [Z,     Z,      A6_5b ]
    ])

    configs.append([{"matrix_id": "5b", "A_continuous": A5b}, hp])


    # ==========================================
    # 6. Ill-conditioned eigenvectors
    # ==========================================
    A4_6 = np.array([[1.0, 1000.0],
                     [0.001,   1.0]])

    A5_6 = np.array([[1.5,  800.0],
                     [0.002,   1.5]])

    A6_6 = np.array([[2.0, 1200.0],
                     [0.0015,  2.0]])

    A6_case = np.block([
        [A4_6, Z, Z],
        [Z, A5_6, Z],
        [Z, Z, A6_6]
    ])

    configs.append([{"matrix_id": "6", "A_continuous": A6_case}, hp])


    # Matrix 6A
    A4_6a = np.array([[1.0, 1500.0],
                      [0.001,   1.0]])

    A5_6a = np.array([[1.4, 1000.0],
                      [0.002,   1.4]])

    A6_6a = np.array([[1.9, 1800.0],
                      [0.0015,  1.9]])

    A6a = np.block([
        [A4_6a, Z, Z],
        [Z, A5_6a, Z],
        [Z, Z, A6_6a]
    ])

    configs.append([{"matrix_id": "6a", "A_continuous": A6a}, hp])


    # Matrix 6B
    A4_6b = np.array([[1.1, 1200.0],
                      [0.0008, 1.1]])

    A5_6b = np.array([[1.6, 900.0],
                      [0.0015, 1.6]])

    A6_6b = np.array([[2.1, 1600.0],
                      [0.0012, 2.1]])

    A6b = np.block([
        [A4_6b, Z, Z],
        [Z, A5_6b, Z],
        [Z, Z, A6_6b]
    ])

    configs.append([{"matrix_id": "6b", "A_continuous": A6b}, hp])


    # ==========================================
    # 7. Mixed stable / unstable modes
    # ==========================================
    A4_7 = np.array([[-2.0, 0.5],
                     [ 0.0, 0.5]])

    A5_7 = np.array([[-1.0, 0.5],
                     [ 0.0, 1.5]])

    A6_7 = np.array([[-3.0, 0.5],
                     [ 0.0, 2.0]])

    A7 = np.block([
        [A4_7, Z, Z],
        [Z, A5_7, Z],
        [Z, Z, A6_7]
    ])

    configs.append([{"matrix_id": "7", "A_continuous": A7}, hp])


    # Matrix 7A
    A4_7a = np.array([[-1.5, 0.6],
                      [ 0.0, 0.7]])

    A5_7a = np.array([[-0.8, 0.5],
                      [ 0.0, 1.7]])

    A6_7a = np.array([[-2.5, 0.4],
                      [ 0.0, 2.2]])

    A7a = np.block([
        [A4_7a, Z, Z],
        [Z, A5_7a, Z],
        [Z, Z, A6_7a]
    ])

    configs.append([{"matrix_id": "7a", "A_continuous": A7a}, hp])


    # Matrix 7B
    A4_7b = np.array([[-2.2, 0.7],
                      [ 0.0, 0.9]])

    A5_7b = np.array([[-1.3, 0.6],
                      [ 0.0, 1.2]])

    A6_7b = np.array([[-2.8, 0.5],
                      [ 0.0, 2.5]])

    A7b = np.block([
        [A4_7b, Z, Z],
        [Z, A5_7b, Z],
        [Z, Z, A6_7b]
    ])

    configs.append([{"matrix_id": "7b", "A_continuous": A7b}, hp])

    return configs
