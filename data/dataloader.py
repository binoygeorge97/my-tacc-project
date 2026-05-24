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
