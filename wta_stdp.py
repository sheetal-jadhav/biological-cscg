
import numpy as np
import matplotlib.pyplot as plt

# NETWORK PARAMETERS

vhat = 2000
tau = 20
beta = 0.05
eta0 = 30
tau_ref = 5
ref_time = 10
baseline = 8

N = 200              # afferent neurons
K = 100              # recurrent neurons
W_std = 3.5
V_std = 2.5

input_fr = 100       # Hz
num_trials = 2000   # learning trials
each_obs_time = 50   # ms per patter
checkpoints = [500, 1000, 2000]  # only 3 quick checks

np.random.seed(7)
b = baseline * np.ones(K)

# SQUENCE GENERATION
P = 9
PATTERN_NAMES = ['A','B','C','D','delay','a','b','c','d']

seq1 = np.array([0,1,4,5,6])  # A B delay a b
seq2 = np.array([2,3,4,7,8])  # C D delay c d
seq3 = np.array([1,0,4,6,5])  # B A delay b a
seq4 = np.array([3,2,4,8,7])  # D C delay d c

SEQUENCES = [seq1, seq2, seq3, seq4]

each_input = 10
x_rate = np.zeros((P, N))
for p_each in range(P):
    active = np.random.choice(N, size=each_input, replace=False)
    x_rate[p_each, active] = input_fr


# RUN SEQUENCE (SPIKES ONLY)

def run_one_sequence(pattern_ids, W, V, learn=True):
    x = np.zeros(N)
    refractory = np.zeros(K)
    refractory_input = np.zeros(N)
    y_all = np.zeros(K)
    time_since_spike = 10 * np.ones(K)

    spikes_rec_all = []

    for curr_P in range(len(pattern_ids)):
        pattern_id = pattern_ids[curr_P]
        for t in range(each_obs_time):
            rate_vec = x_rate[pattern_id, :]
            spiking_input = np.where(np.random.uniform(0, 1, N) < (rate_vec / 1000.0))

            for neur in range(N):
                if neur in spiking_input[0] and refractory_input[neur] == 0:
                    x[neur] = np.exp(-1/tau) * (1 + x[neur])
                    refractory_input[neur] = ref_time
                else:
                    x[neur] = np.exp(-1/tau) * x[neur]

            eta = eta0 * np.exp(-time_since_spike / tau_ref)
            ubar = (W @ x) + (V @ y_all) + b - eta
            v = vhat * np.exp(ubar) / np.sum(np.exp(ubar))

            spiking_neurons = np.where(np.random.uniform(0, 1, K) < (v / 1000.0))
            spikes_vec = np.zeros(K)
            spikes_vec[spiking_neurons[0]] = 1.0
            spikes_rec_all.append(spikes_vec.copy())

            for neur in range(K):
                if neur in spiking_neurons[0] and refractory[neur] == 0:
                    refractory[neur] = ref_time
                    time_since_spike[neur] = -1
                    y_all[neur] = np.exp(-1/tau) * (1 + y_all[neur])

                    if learn:
                        W[neur, :] = np.maximum(W[neur, :] + beta * (np.exp(-W[neur, :]) * x - 0.1),0.0)
                        V[neur, :] = np.maximum(V[neur, :] + beta * (np.exp(-V[neur, :]) * y_all - 0.1),0.0)
                        V[neur, neur] = 0.0
                else:
                    y_all[neur] = np.exp(-1/tau) * y_all[neur]

            refractory[refractory > 0] -= 1
            refractory_input[refractory_input > 0] -= 1
            time_since_spike += 1

    return np.array(spikes_rec_all), W, V


# RUN SEQUENCE AND RETURN vmean PER EPOCH (TEST TRIALS)
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
                            W[neur, :] + beta * (np.exp(-W[neur, :]) * x - 1),
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


# NEURON SORTING FROM TEST TRIALS
def compute_neuron_order_from_tests(seq_ids, W, V, num_test_trials=200):
    """
    Run many test trials (no learning) for a given sequence and
    return neuron_order sorted by peak mean activity time, plus mean_rate_time.
    """
    all_spikes = []

    for trial in range(num_test_trials):
        spikes_rec, _, _ = run_one_sequence(seq_ids, W.copy(), V.copy(), learn=False)
        all_spikes.append(spikes_rec)

    all_spikes = np.array(all_spikes)           # (num_trials, T, K)
    mean_rate_time = all_spikes.mean(axis=0)    # (T, K)

    peak_times = np.argmax(mean_rate_time, axis=0)  # (K,)
    neuron_order = np.argsort(peak_times)

    return neuron_order, mean_rate_time


# SORTED RASTER PLOT FOR ONE TRIAL
def plot_sorted_raster_one_trial(spikes_rec, pattern_ids, neuron_order, title):
    spikes_sorted = spikes_rec[:, neuron_order]   # reorder neurons

    T = spikes_sorted.shape[0]
    times = np.arange(T)
    neuron_ids = []
    times_ms = []
    for t in range(T):
        active = np.where(spikes_sorted[t] > 0.5)[0]
        if active.size > 0:
            neuron_ids.extend(active.tolist())
            times_ms.extend([times[t]] * active.size)

    plt.figure(figsize=(10, 4))
    if len(times_ms) > 0:
        plt.scatter(times_ms, neuron_ids, s=2, color='b', marker='.')

    for pos, pid in enumerate(pattern_ids):
        t0 = pos * each_obs_time
        t1 = (pos + 1) * each_obs_time
        plt.axvline(t0, color='gray', linestyle='--', linewidth=0.6)
        plt.text((t0 + t1) / 2, -5, PATTERN_NAMES[pid],
                 ha='center', va='top', fontsize=9)

    plt.ylim(-5, K)
    plt.xlabel("time (ms)")
    plt.ylabel("neuron ID")
    plt.title(title)
    plt.tight_layout()
    plt.show()
def print_checkpoint_metrics(W, V, rep):
    print(f"\n🔍 CHECKPOINT rep={rep}")
    neuron_order, mean_rate_time = compute_neuron_order_from_tests(
        seq1, W.copy(), V.copy(), num_test_trials=50)

    # Assembly sizes
    peak_times  = np.argmax(mean_rate_time, axis=0)
    assembly_id = np.digitize(peak_times / each_obs_time, np.arange(6))
    print(f"  Assemblies (A,B,d,a,b): {np.bincount(assembly_id)[1:6]}")

    # Per-sequence sizes
    for i, s in enumerate(SEQUENCES):
        no, mrt = compute_neuron_order_from_tests(s, W.copy(), V.copy(), num_test_trials=30)
        pt  = np.argmax(mrt, axis=0)
        aid = np.digitize(pt / each_obs_time, np.arange(6))
        label = "-".join(PATTERN_NAMES[p] for p in s)
        print(f"  Seq{i+1} ({label}): {np.bincount(aid)[1:6]}")

    # Forward weights
    V_sorted = V[np.ix_(neuron_order, neuron_order)]
    for lag in [3, 5]:
        fwd = np.diag(V_sorted, k=lag)
        print(f"  Lag-{lag} fwd: {np.mean(fwd):.3f}±{np.std(fwd):.3f}")

    print(f"  V mean={np.mean(V):.3f}  W mean={np.mean(W):.3f}")

    # V matrix plot
    plt.figure(figsize=(5,4))
    plt.imshow(V_sorted, cmap='viridis', aspect='equal',
               vmin=0, vmax=max(np.percentile(V_sorted, 98), 0.01))
    plt.colorbar(label='weight')
    plt.title(f"V_sorted @ rep={rep}")
    plt.tight_layout()
    plt.show()

# MAIN TRAINING + SORTED RASTERS
print(f"Starting {num_trials} trials | checkpoints={checkpoints}")
for sim_rep in range(1):
    W = np.maximum(np.random.normal(0, W_std, (K, N)), 0)
    V = np.maximum(np.random.normal(0, V_std, (K, K)), 0)
    np.fill_diagonal(V, 0)

    # Learning
    for rep in range(num_trials):
        if rep % 100 == 0:
            print("train #", rep)
        idx = np.random.randint(len(SEQUENCES))  # 0..3
        curr_seq = SEQUENCES[idx]
        _, W, V = run_one_sequence(curr_seq, W, V, learn=True)


    # FINAL FULL EVAL (your original)
    print("🏁 FINAL EVAL")
    neuron_order, mean_rate_time = compute_neuron_order_from_tests(seq1, W, V, 200)
    
    for i, s in enumerate(SEQUENCES):
        spikes_test, _, _ = run_one_sequence(s, W.copy(), V.copy(), learn=False)
        label = "-".join(PATTERN_NAMES[p] for p in s)
        plot_sorted_raster_one_trial(spikes_test, s, neuron_order, f"FINAL Seq{i+1} ({label})")

    # FINAL V_sorted
    V_sorted = V[np.ix_(neuron_order, neuron_order)]
    plt.figure(figsize=(6, 5))
    plt.imshow(V_sorted, cmap='viridis', aspect='equal')
    plt.colorbar(label='weight')
    plt.xlabel('pre (sorted)')
    plt.ylabel('post (sorted)')
    plt.title('FINAL V_sorted (2K trials)')
    plt.tight_layout()
    plt.show()
    
    # FINAL METRICS
    peak_times = np.argmax(mean_rate_time, axis=0)
    assembly_id = np.digitize(peak_times / each_obs_time, np.arange(6))
    print("🏆 FINAL Assemblies:", np.bincount(assembly_id)[1:6])
    print("🏆 FINAL Fwd lag5:", np.mean(np.diag(V_sorted, k=5)))


    # Compute neuron order from many test trials of Seq1
    neuron_order, mean_rate_time = compute_neuron_order_from_tests(
        seq1, W.copy(), V.copy(), num_test_trials=200
    )

    for i, s in enumerate(SEQUENCES):
        spikes_test, _, _ = run_one_sequence(s, W.copy(), V.copy(), learn=False)
        label = "-".join(PATTERN_NAMES[p] for p in s)
        plot_sorted_raster_one_trial(spikes_test, s, neuron_order,
                                     f"Seq{i+1} ({label}) AFTER learning")


    # TEST PHASE FOR CORRELATIONS

    num_test_trials = 200  # or more

    vmean_all = []  # one entry per sequence

    for s in SEQUENCES:
        vmean_seq = []
        for trial in range(num_test_trials):
            v_trial, _, _ = run_one_sequence_with_rates(s, W.copy(), V.copy(), learn=False)
            vmean_seq.append(v_trial)
        vmean_all.append(np.array(vmean_seq))  # shape (trials, len(s), K)

    # LATERAL WEIGHTS MATRIX 
    # reorder V by neuron_order (same ordering as in sorted raster)
    V_sorted = V[np.ix_(neuron_order, neuron_order)]   # post x pre

    plt.figure(figsize=(6, 5))
    plt.imshow(V_sorted, cmap='viridis', aspect='equal')
    plt.colorbar(label='synaptic weight')
    plt.xlabel('presynaptic neuron')
    plt.ylabel('postsynaptic neuron')
    plt.title('Lateral weights after learning')
    plt.tight_layout()
    plt.show()

# Assembly separation
peak_times = np.argmax(mean_rate_time, axis=0)
assembly_id = np.digitize(peak_times, np.linspace(0, each_obs_time*5, 6))
print("Neurons per assembly:", np.bincount(assembly_id))

# V chain strength (should be >2 on subdiagonal)
off_diag = np.diag(V_sorted, k=5)  # lag ~50ms/5-10 neurons
print("Forward weights:", np.mean(off_diag))

# Check all 4 sequences' peak distributions
for i, s in enumerate(SEQUENCES):
    no, mrt = compute_neuron_order_from_tests(s, W.copy(), V.copy(), num_test_trials=30)
    pt = np.argmax(mrt, axis=0)
    aid = np.digitize(pt / each_obs_time, np.arange(6))
    print(f"  Seq{i+1} assembly sizes: {np.bincount(aid)[1:6]}")

 