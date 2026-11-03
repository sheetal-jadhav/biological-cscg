import numpy as np
import matplotlib.pyplot as plt

# ========== PARAMETERS ==========
vhat          = 500
tau           = 20
beta          = 0.1
eta0          = 20
tau_ref       = 5
ref_time      = 10
baseline      = 3

N             = 100
K             = 100
W_std         = 3.5
V_std         = 2.5

input_fr      = 100
each_obs_time = 20
num_trials    = 2000
checkpoints   = [500, 1000, 2000]

np.random.seed(7)
b = baseline * np.ones(K)

# ========== SEQUENCES ==========
P             = 9
PATTERN_NAMES = ['A','B','C','D','delay','a','b','c','d']

seq1 = np.array([0,1,4,5,6])
seq2 = np.array([2,3,4,7,8])
seq3 = np.array([1,0,4,6,5])
seq4 = np.array([3,2,4,8,7])
SEQUENCES = [seq1, seq2, seq3, seq4]

each_input = 10
x_rate = np.zeros((P, N))
for p_each in range(P):
    active = np.random.choice(N, size=each_input, replace=False)
    x_rate[p_each, active] = input_fr


# ========== FUNCTIONS ==========

def run_one_sequence(pattern_ids, W, V, learn=True):
    x                = np.zeros(N)
    refractory       = np.zeros(K)
    refractory_input = np.zeros(N)
    y_all            = np.zeros(K)
    time_since_spike = 10 * np.ones(K)
    spikes_rec_all   = []

    for curr_P in range(len(pattern_ids)):
        pattern_id = pattern_ids[curr_P]
        for t in range(each_obs_time):

            # ---- FEEDFORWARD INPUT ----
            rate_vec      = x_rate[pattern_id, :]
            spiking_input = np.where(np.random.uniform(0,1,N) < (rate_vec / 1000.0))[0]
            for neur in range(N):
                if neur in spiking_input and refractory_input[neur] == 0:
                    x[neur] = np.exp(-1/tau) * (1 + x[neur])
                    refractory_input[neur] = ref_time
                else:
                    x[neur] = np.exp(-1/tau) * x[neur]

            # ---- MEMBRANE ----
            eta  = eta0 * np.exp(-time_since_spike / tau_ref)
            ubar = (W @ x) + (V @ y_all) + b - eta
            v    = vhat * np.exp(ubar) / np.sum(np.exp(ubar))

            spiking_neurons = np.where(np.random.uniform(0,1,K) < (v / 1000.0))[0]
            spikes_vec = np.zeros(K)
            spikes_vec[spiking_neurons] = 1.0
            spikes_rec_all.append(spikes_vec.copy())

            # ---- FREE NEURONS THAT SPIKED ----
            spiking_and_free = [n for n in spiking_neurons if refractory[n] == 0]

            # ---- STEP 1: WEIGHT UPDATE — uses y_all BEFORE this timestep's spikes ----
            if learn:
                for neur in spiking_and_free:
                    W[neur, :] = np.maximum(
                        W[neur, :] + beta * (np.exp(-W[neur, :]) * x - 0.1), 0.0)
                    V[neur, :] = np.maximum(
                        V[neur, :] + beta * (np.exp(-V[neur, :]) * y_all - 0.1), 0.0)
                    V[neur, neur] = 0.0

            # ---- STEP 2: UPDATE y_all AND REFRACTORY — AFTER weight update ----
            for neur in spiking_and_free:
                refractory[neur]       = ref_time
                time_since_spike[neur] = -1
                y_all[neur]            = np.exp(-1/tau) * (1 + y_all[neur])

            # ---- DECAY y_all for non-spiking neurons ----
            for neur in range(K):
                if neur not in spiking_and_free:
                    y_all[neur] = np.exp(-1/tau) * y_all[neur]

            refractory[refractory > 0]             -= 1
            refractory_input[refractory_input > 0] -= 1
            time_since_spike                        += 1

    return np.array(spikes_rec_all), W, V


def run_one_sequence_with_rates(pattern_ids, W, V, learn=False):
    x                = np.zeros(N)
    refractory       = np.zeros(K)
    refractory_input = np.zeros(N)
    y_all            = np.zeros(K)
    time_since_spike = 10 * np.ones(K)
    vmean_trial      = []

    for curr_P in range(len(pattern_ids)):
        pattern_id = pattern_ids[curr_P]
        v_all = []
        for t in range(each_obs_time):

            rate_vec      = x_rate[pattern_id, :]
            spiking_input = np.where(np.random.uniform(0,1,N) < (rate_vec / 1000.0))[0]
            for neur in range(N):
                if neur in spiking_input and refractory_input[neur] == 0:
                    x[neur] = np.exp(-1/tau) * (1 + x[neur])
                    refractory_input[neur] = ref_time
                else:
                    x[neur] = np.exp(-1/tau) * x[neur]

            eta  = eta0 * np.exp(-time_since_spike / tau_ref)
            ubar = (W @ x) + (V @ y_all) + b - eta
            v    = vhat * np.exp(ubar) / np.sum(np.exp(ubar))
            v_all.append(v)

            spiking_neurons = np.where(np.random.uniform(0,1,K) < (v / 1000.0))[0]
            spiking_and_free = [n for n in spiking_neurons if refractory[n] == 0]

            for neur in spiking_and_free:
                refractory[neur]       = ref_time
                time_since_spike[neur] = -1
                y_all[neur]            = np.exp(-1/tau) * (1 + y_all[neur])

            for neur in range(K):
                if neur not in spiking_and_free:
                    y_all[neur] = np.exp(-1/tau) * y_all[neur]

            refractory[refractory > 0]             -= 1
            refractory_input[refractory_input > 0] -= 1
            time_since_spike                        += 1

        vmean_trial.append(np.mean(np.array(v_all), axis=0))

    return np.array(vmean_trial), W, V


def compute_neuron_order_from_tests(seq_ids, W, V, num_test_trials=200):
    all_spikes = []
    for trial in range(num_test_trials):
        spikes_rec, _, _ = run_one_sequence(seq_ids, W.copy(), V.copy(), learn=False)
        all_spikes.append(spikes_rec)
    all_spikes     = np.array(all_spikes)
    mean_rate_time = all_spikes.mean(axis=0)
    peak_times     = np.argmax(mean_rate_time, axis=0)
    neuron_order   = np.argsort(peak_times)
    return neuron_order, mean_rate_time


def plot_sorted_raster_one_trial(spikes_rec, pattern_ids, neuron_order, title):
    spikes_sorted        = spikes_rec[:, neuron_order]
    T                    = spikes_sorted.shape[0]
    neuron_ids, times_ms = [], []
    for t in range(T):
        active = np.where(spikes_sorted[t] > 0.5)[0]
        if active.size > 0:
            neuron_ids.extend(active.tolist())
            times_ms.extend([t] * active.size)

    plt.figure(figsize=(10, 4))
    if len(times_ms) > 0:
        plt.scatter(times_ms, neuron_ids, s=2, color='b', marker='.')
    for pos, pid in enumerate(pattern_ids):
        t0 = pos * each_obs_time
        t1 = (pos + 1) * each_obs_time
        plt.axvline(t0, color='gray', linestyle='--', linewidth=0.6)
        plt.text((t0+t1)/2, -5, PATTERN_NAMES[pid], ha='center', va='top', fontsize=9)
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

    peak_times  = np.argmax(mean_rate_time, axis=0)
    assembly_id = np.digitize(peak_times / each_obs_time, np.arange(6))
    print(f"  Assemblies (A,B,d,a,b): {np.bincount(assembly_id)[1:6]}")

    for i, s in enumerate(SEQUENCES):
        no, mrt = compute_neuron_order_from_tests(s, W.copy(), V.copy(), num_test_trials=30)
        pt  = np.argmax(mrt, axis=0)
        aid = np.digitize(pt / each_obs_time, np.arange(6))
        label = "-".join(PATTERN_NAMES[p] for p in s)
        print(f"  Seq{i+1} ({label}): {np.bincount(aid)[1:6]}")

    V_sorted = V[np.ix_(neuron_order, neuron_order)]
    for lag in [3, 5]:
        fwd = np.diag(V_sorted, k=lag)
        print(f"  Lag-{lag} fwd: {np.mean(fwd):.3f}±{np.std(fwd):.3f}")

    print(f"  V mean={np.mean(V):.3f}  W mean={np.mean(W):.3f}")

    plt.figure(figsize=(5,4))
    plt.imshow(V_sorted, cmap='viridis', aspect='equal',
               vmin=0, vmax=max(np.percentile(V_sorted, 98), 0.01))
    plt.colorbar(label='weight')
    plt.title(f"V_sorted @ rep={rep}")
    plt.tight_layout()
    plt.show()


# ========== TRAINING ==========
print(f"Starting {num_trials} trials | checkpoints={checkpoints}")

W = np.maximum(np.random.normal(0, W_std, (K, N)), 0)
V = np.maximum(np.random.normal(0, V_std, (K, K)), 0)
np.fill_diagonal(V, 0)

for rep in range(num_trials):
    if rep % 500 == 0:
        print(f"  train #{rep}")

    idx      = np.random.randint(len(SEQUENCES))
    curr_seq = SEQUENCES[idx]
    _, W, V  = run_one_sequence(curr_seq, W.copy(), V.copy(), learn=True)

    if rep + 1 in checkpoints:
        print_checkpoint_metrics(W, V, rep + 1)


# ========== FINAL EVAL ==========
print("\n🏁 FINAL EVAL")
neuron_order, mean_rate_time = compute_neuron_order_from_tests(seq1, W, V, 200)

for i, s in enumerate(SEQUENCES):
    spikes_test, _, _ = run_one_sequence(s, W.copy(), V.copy(), learn=False)
    label = "-".join(PATTERN_NAMES[p] for p in s)
    plot_sorted_raster_one_trial(spikes_test, s, neuron_order, f"FINAL Seq{i+1} ({label})")

V_sorted = V[np.ix_(neuron_order, neuron_order)]
plt.figure(figsize=(6, 5))
plt.imshow(V_sorted, cmap='viridis', aspect='equal')
plt.colorbar(label='weight')
plt.xlabel('pre (sorted)')
plt.ylabel('post (sorted)')
plt.title('FINAL V_sorted')
plt.tight_layout()
plt.show()

vmean_all = []
for s in SEQUENCES:
    vmean_seq = []
    for trial in range(200):
        v_trial, _, _ = run_one_sequence_with_rates(s, W.copy(), V.copy(), learn=False)
        vmean_seq.append(v_trial)
    vmean_all.append(np.array(vmean_seq))

peak_times  = np.argmax(mean_rate_time, axis=0)
assembly_id = np.digitize(peak_times / each_obs_time, np.arange(6))
print("🏆 FINAL Assemblies:", np.bincount(assembly_id)[1:6])
print("🏆 FINAL Fwd lag-5:", f"{np.mean(np.diag(V_sorted, k=5)):.3f}")
print("🏆 FINAL V mean:",    f"{np.mean(V):.3f}")

for i, s in enumerate(SEQUENCES):
    no, mrt = compute_neuron_order_from_tests(s, W.copy(), V.copy(), num_test_trials=30)
    pt  = np.argmax(mrt, axis=0)
    id = np.digitize(pt / each_obs_time, np.arange(6))