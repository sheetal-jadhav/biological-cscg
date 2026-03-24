import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# NETWORK PARAMETERS
# -----------------------
vhat = 1000
tau = 20
beta = 0.02
eta0 = 10
tau_ref = 5
ref_time = 10
baseline = 20

N = 200              # afferent neurons
K = 100              # recurrent neurons
W_std = 3.5
V_std = 2.5

input_fr = 100       # Hz
num_trials = 2000    # learning trials
each_obs_time = 20   # ms per pattern

np.random.seed(7)
b = baseline * np.ones(K)

# -----------------------
# PATTERNS AND SEQUENCES
# -----------------------
P = 9
PATTERN_NAMES = ['A','B','C','D','delay','a','b','c','d']

seq1 = np.array([0, 1, 4, 5, 6])  # A B delay a b
seq2 = np.array([2, 3, 4, 7, 8])  # C D delay c d

each_input = 10
x_rate = np.zeros((P, N))
for p_each in range(P):
    active = np.random.choice(N, size=each_input, replace=False)
    x_rate[p_each, active] = input_fr

# -----------------------
# RUN SEQUENCE (SPIKES ONLY)
# -----------------------
def run_one_sequence(pattern_ids, W, V, learn=True):
    x = np.zeros(N)
    refractory = np.zeros(K)
    refractory_input = np.zeros(N)
    y_all = np.zeros(K)
    time_since_spike = 7 * np.ones(K)

    spikes_rec_all = []

    for curr_P in range(len(pattern_ids)):
        pattern_id = pattern_ids[curr_P]
        for t in range(each_obs_time):
            rate_vec = x_rate[pattern_id, :]
            spiking_input = np.where(
                np.random.uniform(0, 1, N) < (rate_vec / 1000.0)
            )

            for neur in range(N):
                if neur in spiking_input[0] and refractory_input[neur] == 0:
                    x[neur] = np.exp(-1/tau) * (1 + x[neur])
                    refractory_input[neur] = ref_time
                else:
                    x[neur] = np.exp(-1/tau) * x[neur]

            eta = eta0 * np.exp(-time_since_spike / tau_ref)
            ubar = (W @ x) + (V @ y_all) + b - eta
            v = vhat * np.exp(ubar) / np.sum(np.exp(ubar))

            spiking_neurons = np.where(
                np.random.uniform(0, 1, K) < (v / 1000.0)
            )
            spikes_vec = np.zeros(K)
            spikes_vec[spiking_neurons[0]] = 1.0
            spikes_rec_all.append(spikes_vec.copy())

            for neur in range(K):
                if neur in spiking_neurons[0] and refractory[neur] == 0:
                    refractory[neur] = ref_time
                    time_since_spike[neur] = -1
                    y_all[neur] = np.exp(-1/tau) * (1 + y_all[neur])

                    if learn:
                        W[neur, :] = np.maximum(
                            W[neur, :] + beta * (np.exp(-W[neur, :]) * x - 0.1),
                            0.0,
                        )
                        V[neur, :] = np.maximum(
                            V[neur, :] + beta * (np.exp(-V[neur, :]) * y_all - 0.1),
                            0.0,
                        )
                        V[neur, neur] = 0.0
                else:
                    y_all[neur] = np.exp(-1/tau) * y_all[neur]

            refractory[refractory > 0] -= 1
            refractory_input[refractory_input > 0] -= 1
            time_since_spike += 1

    return np.array(spikes_rec_all), W, V

# -----------------------
# RUN SEQUENCE AND RETURN vmean PER EPOCH (TEST TRIALS)
# -----------------------
def run_one_sequence_with_rates(pattern_ids, W, V, learn=False):
    x = np.zeros(N)
    refractory = np.zeros(K)
    refractory_input = np.zeros(N)
    y_all = np.zeros(K)
    time_since_spike = 7 * np.ones(K)

    vmean_trial = []

    for curr_P in range(len(pattern_ids)):
        pattern_id = pattern_ids[curr_P]
        v_all = []
        for t in range(each_obs_time):
            rate_vec = x_rate[pattern_id, :]
            spiking_input = np.where(
                np.random.uniform(0, 1, N) < (rate_vec / 1000.0)
            )

            for neur in range(N):
                if neur in spiking_input[0] and refractory_input[neur] == 0:
                    x[neur] = np.exp(-1/tau) * (1 + x[neur])
                    refractory_input[neur] = ref_time
                else:
                    x[neur] = np.exp(-1/tau) * x[neur]

            eta = eta0 * np.exp(-time_since_spike / tau_ref)
            ubar = (W @ x) + (V @ y_all) + b - eta
            v = vhat * np.exp(ubar) / np.sum(np.exp(ubar))
            v_all.append(v)

            spiking_neurons = np.where(
                np.random.uniform(0, 1, K) < (v / 1000.0)
            )

            for neur in range(K):
                if neur in spiking_neurons[0] and refractory[neur] == 0:
                    refractory[neur] = ref_time
                    time_since_spike[neur] = -1
                    y_all[neur] = np.exp(-1/tau) * (1 + y_all[neur])

                    if learn:
                        W[neur, :] = np.maximum(
                            W[neur, :] + beta * (np.exp(-W[neur, :]) * x - 0.1),
                            0.0,
                        )
                        V[neur, :] = np.maximum(
                            V[neur, :] + beta * (np.exp(-V[neur, :]) * y_all - 0.1),
                            0.0,
                        )
                        V[neur, neur] = 0.0
                else:
                    y_all[neur] = np.exp(-1/tau) * y_all[neur]

            refractory[refractory > 0] -= 1
            refractory_input[refractory_input > 0] -= 1
            time_since_spike += 1

        vmean_trial.append(np.mean(np.array(v_all), axis=0))

    return np.array(vmean_trial), W, V

# -----------------------
# RASTER PLOT
# -----------------------
def plot_raster(spikes_rec, pattern_ids, title):
    T = spikes_rec.shape[0]
    times = np.arange(T)
    neuron_ids = []
    times_ms = []
    for t in range(T):
        active = np.where(spikes_rec[t] > 0.5)[0]
        if active.size > 0:
            neuron_ids.extend(active.tolist())
            times_ms.extend([times[t]] * active.size)

    plt.figure(figsize=(10, 4))
    if len(times_ms) > 0:
        plt.scatter(times_ms, neuron_ids, s=2, color='k', marker='|')

    for pos, pid in enumerate(pattern_ids):
        t0 = pos * each_obs_time
        t1 = (pos + 1) * each_obs_time
        plt.axvline(t0, color='gray', linestyle='--', linewidth=0.6)
        plt.text((t0 + t1) / 2, -2, PATTERN_NAMES[pid],
                 ha='center', va='top', fontsize=9)

    plt.ylim(-5, K)
    plt.xlabel("time (ms)")
    plt.ylabel("recurrent neuron")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -----------------------
# TRAINING + TEST + CORRELATIONS
# -----------------------
vmean_all_rep = []
trials_all_rep = []

for sim_rep in range(1):
    W = np.maximum(np.random.normal(0, W_std, (K, N)), 0)
    V = np.maximum(np.random.normal(0, V_std, (K, K)), 0)
    np.fill_diagonal(V, 0)

    W_init = W.copy()
    V_init = V.copy()

    # BEFORE learning
    spikes_seq1_before, _, _ = run_one_sequence(seq1, W_init.copy(), V_init.copy(), learn=False)
    spikes_seq2_before, _, _ = run_one_sequence(seq2, W_init.copy(), V_init.copy(), learn=False)
    plot_raster(spikes_seq1_before, seq1, "Seq1 (AB delay ab) BEFORE learning")
    plot_raster(spikes_seq2_before, seq2, "Seq2 (CD delay cd) BEFORE learning")

    # Learning
    vmean = []
    for rep in range(num_trials):
        if rep % 100 == 0:
            print(rep)
        if np.random.rand() < 0.5:
            curr_seq = seq1
        else:
            curr_seq = seq2
        _, W, V = run_one_sequence(curr_seq, W, V, learn=True)

    # TEST PHASE: vmean per epoch, many trials
    num_test_trials = 200
    vmean_seq1 = []
    vmean_seq2 = []

    for trial in range(num_test_trials):
        v1, _, _ = run_one_sequence_with_rates(seq1, W.copy(), V.copy(), learn=False)
        vmean_seq1.append(v1)
        v2, _, _ = run_one_sequence_with_rates(seq2, W.copy(), V.copy(), learn=False)
        vmean_seq2.append(v2)

    vmean_seq1 = np.array(vmean_seq1)  # (T1, 5, K)
    vmean_seq2 = np.array(vmean_seq2)  # (T2, 5, K)

    # AFTER learning rasters (optional)
    spikes_seq1_after, _, _ = run_one_sequence(seq1, W.copy(), V.copy(), learn=False)
    spikes_seq2_after, _, _ = run_one_sequence(seq2, W.copy(), V.copy(), learn=False)
    plot_raster(spikes_seq1_after, seq1, "Seq1 (AB delay ab) AFTER learning")
    plot_raster(spikes_seq2_after, seq2, "Seq2 (CD delay cd) AFTER learning")

    # Population-code correlations
    mean_seq1 = np.mean(vmean_seq1, axis=0)  # (5, K)
    mean_seq2 = np.mean(vmean_seq2, axis=0)  # (5, K)

    corrmat = np.zeros((len(seq1), len(seq2)))
    for i in range(len(seq1)):
        a = mean_seq1[i]
        a_c = a - a.mean()
        for j in range(len(seq2)):
            b = mean_seq2[j]
            b_c = b - b.mean()
            num = np.dot(a_c, b_c)
            den = np.linalg.norm(a_c) * np.linalg.norm(b_c)
            corrmat[i, j] = num / den if den > 0 else 0.0

    plt.figure(figsize=(5, 4))
    plt.imshow(corrmat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
    plt.colorbar(label='corr')
    plt.xticks(range(len(seq2)), [PATTERN_NAMES[p] for p in seq2], rotation=45)
    plt.yticks(range(len(seq1)), [PATTERN_NAMES[p] for p in seq1])
    plt.title('Seq1 vs Seq2 population correlations (after learning)')
    plt.tight_layout()
    plt.show()

    vmean_all_rep.append(vmean)
    trials_all_rep.append(None)

    
'''
   # -----------------------
    # POPULATION CODE CORRELATION MATRIX
    # -----------------------
    mean_seq1 = np.mean(vmean_seq1, axis=0)  # (len(seq1), K)
    mean_seq2 = np.mean(vmean_seq2, axis=0)  # (len(seq2), K)

    corrmat = np.zeros((len(seq1), len(seq2)))
    for i in range(len(seq1)):
        a = mean_seq1[i]
        a_c = a - a.mean()
        for j in range(len(seq2)):
            b = mean_seq2[j]
            b_c = b - b.mean()
            num = np.dot(a_c, b_c)
            den = np.linalg.norm(a_c) * np.linalg.norm(b_c)
            corrmat[i, j] = num / den if den > 0 else 0.0

    plt.figure(figsize=(5, 4))
    plt.imshow(corrmat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
    plt.colorbar(label='correlation')
    plt.xticks(range(len(seq2)), [PATTERN_NAMES[p] for p in seq2], rotation=45)
    plt.yticks(range(len(seq1)), [PATTERN_NAMES[p] for p in seq1])
    plt.title('Seq1 vs Seq2 population correlations (after learning)')
    plt.tight_layout()
    plt.show()
'''
 
