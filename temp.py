

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ========== PARAMETERS ==========
vhat          = 300
tau           = 20
beta          = 0.05
eta0          = 40
tau_ref       = 10      # soft decay; potassium challe recovery or relative refractory
ref_time      = 10       # hard block # sodium channel inactivation or absolute refractory 
baseline      = 5

N             = 200
K             = 100
W_std         = 3.5
V_std         = 2.5

input_fr      = 100
each_obs_time = 50
num_trials    = 2000
checkpoints   = [500,1000,1500]

# ── AXONAL DELAY ─────────────────────────────────────────────────────────
rec_delay_ms  = 5    # mean recurrent axonal delay in timesteps
#   When neuron k spikes at time t, its effect on y_all
#   is not felt until t + rec_delay_ms timesteps later.
#   This matches Kappel's mean_rec_delay=10ms scaled to our 1ms steps.
# ─────────────────────────────────────────────────────────────────────────

np.random.seed(7)
b = baseline * np.ones(K)

# ========== SEQUENCES ==========
P             = 9
PATTERN_NAMES = ['A','B','C','D','delay','a','b','c','d']

seq1 = np.array([0,1,4,5,6])
seq2 = np.array([2,3,4,7,8])
seq3 = np.array([1,0,4,6,5])
seq4 = np.array([3,2,4,8,7])
SEQUENCES  = [seq1, seq2, seq3, seq4]
SEQ_LABELS = ["A→B→delay→a→b",
              "C→D→delay→c→d",
              "B→A→delay→b→a",
              "D→C→delay→d→c"]

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
    time_since_spike = np.ones(K)
    spikes_rec_all   = []

    # ── AXONAL DELAY BUFFER ───────────────────────────────────────────────
    # Each entry is a spike vector that will be applied to y_all
    # rec_delay_ms steps in the future.
    # deque of length rec_delay_ms — oldest entry is applied at current step.
    delay_buffer = deque(
        [np.zeros(K) for _ in range(rec_delay_ms)],
        maxlen=rec_delay_ms
    )
    # ─────────────────────────────────────────────────────────────────────

    for curr_P in range(len(pattern_ids)):
        pattern_id = pattern_ids[curr_P]

        for t in range(each_obs_time):

            # ---- FEEDFORWARD INPUT ----
            rate_vec      = x_rate[pattern_id, :]
            spiking_input = np.where(
                np.random.uniform(0,1,N) < (rate_vec / 1000.0))[0]
            for neur in range(N):
                if neur in spiking_input and refractory_input[neur] == 0:
                    x[neur] = np.exp(-1/tau) * (1 + x[neur])
                    refractory_input[neur] = ref_time
                else:
                    x[neur] = np.exp(-1/tau) * x[neur]

            # ── APPLY DELAYED SPIKES TO y_all ────────────────────────────
            # Pull the oldest entry from the buffer — these are spikes that
            # fired rec_delay_ms steps ago and are now arriving at dendrites.
            delayed_spikes = delay_buffer[0]   # oldest = leftmost in deque
            for neur in range(K):
                if delayed_spikes[neur] > 0:
                    y_all[neur] = np.exp(-1/tau) * (1 + y_all[neur])
                else:
                    y_all[neur] = np.exp(-1/tau) * y_all[neur]
            # ─────────────────────────────────────────────────────────────

            # ---- MEMBRANE ----
            eta  = eta0 * np.exp(-time_since_spike / tau_ref)
            ubar = (W @ x) + (V @ y_all) + b - eta
            v    = vhat * np.exp(ubar) / np.sum(np.exp(ubar))

            spiking_neurons  = np.where(
                np.random.uniform(0,1,K) < (v / 1000.0))[0]
            spikes_vec       = np.zeros(K)
            spikes_vec[spiking_neurons] = 1.0
            spikes_rec_all.append(spikes_vec.copy())

            spiking_and_free = [n for n in spiking_neurons if refractory[n] == 0]

            # ---- WEIGHT UPDATE — uses y_all from DELAYED spikes ----
            if learn:
                for neur in spiking_and_free:
                    W[neur, :] = np.maximum(
                        W[neur, :] + beta*(np.exp(-W[neur,:])*x - 0.1), 0.0)
                    V[neur, :] = np.maximum(
                        V[neur, :] + beta*(np.exp(-V[neur,:])*y_all - 0.1), 0.0)
                    V[neur, neur] = 0.0

            # ---- UPDATE REFRACTORY + time_since_spike ----
            for neur in spiking_and_free:
                refractory[neur]       = ref_time
                time_since_spike[neur] = -1
            # Note: y_all is now updated via delay buffer, NOT here directly

            refractory[refractory > 0]             -= 1
            refractory_input[refractory_input > 0] -= 1
            time_since_spike                        += 1

            # ── PUSH CURRENT SPIKES INTO DELAY BUFFER ────────────────────
            # These will arrive at dendrites rec_delay_ms steps from now.
            new_entry = np.zeros(K)
            new_entry[spiking_and_free] = 1.0
            delay_buffer.append(new_entry)
            # ─────────────────────────────────────────────────────────────

    return np.array(spikes_rec_all), W, V


def run_one_sequence_with_rates(pattern_ids, W, V, learn=False):
    x                = np.zeros(N)
    refractory       = np.zeros(K)
    refractory_input = np.zeros(N)
    y_all            = np.zeros(K)
    time_since_spike = 10 * np.ones(K)
    vmean_trial      = []

    delay_buffer = deque(
        [np.zeros(K) for _ in range(rec_delay_ms)],
        maxlen=rec_delay_ms
    )

    for curr_P in range(len(pattern_ids)):
        pattern_id = pattern_ids[curr_P]
        v_all = []

        for t in range(each_obs_time):

            rate_vec      = x_rate[pattern_id, :]
            spiking_input = np.where(
                np.random.uniform(0,1,N) < (rate_vec / 1000.0))[0]
            for neur in range(N):
                if neur in spiking_input and refractory_input[neur] == 0:
                    x[neur] = np.exp(-1/tau) * (1 + x[neur])
                    refractory_input[neur] = ref_time
                else:
                    x[neur] = np.exp(-1/tau) * x[neur]

            # apply delayed spikes to y_all
            delayed_spikes = delay_buffer[0]
            for neur in range(K):
                if delayed_spikes[neur] > 0:
                    y_all[neur] = np.exp(-1/tau) * (1 + y_all[neur])
                else:
                    y_all[neur] = np.exp(-1/tau) * y_all[neur]

            eta  = eta0 * np.exp(-time_since_spike / tau_ref)
            ubar = (W @ x) + (V @ y_all) + b - eta
            v    = vhat * np.exp(ubar) / np.sum(np.exp(ubar))
            v_all.append(v)

            spiking_neurons  = np.where(
                np.random.uniform(0,1,K) < (v / 1000.0))[0]
            spiking_and_free = [n for n in spiking_neurons if refractory[n] == 0]

            for neur in spiking_and_free:
                refractory[neur]       = ref_time
                time_since_spike[neur] = -1

            refractory[refractory > 0]             -= 1
            refractory_input[refractory_input > 0] -= 1
            time_since_spike                        += 1

            new_entry = np.zeros(K)
            new_entry[spiking_and_free] = 1.0
            delay_buffer.append(new_entry)

        vmean_trial.append(np.mean(np.array(v_all), axis=0))

    return np.array(vmean_trial), W, V


def compute_neuron_order_from_tests(seq_ids, W, V, num_test_trials=200):
    all_spikes = []
    for trial in range(num_test_trials):
        spikes_rec, _, _ = run_one_sequence(
            seq_ids, W.copy(), V.copy(), learn=False)
        all_spikes.append(spikes_rec)
    all_spikes     = np.array(all_spikes)
    mean_rate_time = all_spikes.mean(axis=0)
    peak_times     = np.argmax(mean_rate_time, axis=0)
    neuron_order   = np.argsort(peak_times)
    return neuron_order, mean_rate_time


def plot_sorted_raster_one_trial(spikes_rec, pattern_ids,
                                  neuron_order, title):
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
        plt.axvspan(t0, t1, alpha=0.08, color=f'C{pid%10}', zorder=0)
        plt.axvline(t0, color='gray', linestyle='--', linewidth=0.6)
        plt.text((t0+t1)/2, -5, PATTERN_NAMES[pid],
                 ha='center', va='top', fontsize=9, fontweight='bold')
    plt.ylim(-6, K)
    plt.xlabel("time (ms)")
    plt.ylabel("neuron (sorted)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def print_checkpoint_metrics(W, V, rep):
    print(f"\n🔍 CHECKPOINT rep={rep}")
    neuron_order, mean_rate_time = compute_neuron_order_from_tests(
        seq1, W.copy(), V.copy(), num_test_trials=50)

    peak_times  = np.argmax(mean_rate_time, axis=0)
    assembly_id = np.digitize(peak_times / each_obs_time, np.arange(6))
    print(f"  Assemblies (seq1): {np.bincount(assembly_id)[1:6]}")

    for i, s in enumerate(SEQUENCES):
        no, mrt = compute_neuron_order_from_tests(
            s, W.copy(), V.copy(), num_test_trials=30)
        pt  = np.argmax(mrt, axis=0)
        aid = np.digitize(pt / each_obs_time, np.arange(6))
        label = "→".join(PATTERN_NAMES[p] for p in s)
        print(f"  Seq{i+1} ({label}): {np.bincount(aid)[1:6]}")

    V_sorted = V[np.ix_(neuron_order, neuron_order)]
    for lag in [3, 5]:
        fwd = np.diag(V_sorted, k=lag)
        print(f"  Lag-{lag} fwd: {np.mean(fwd):.3f} ± {np.std(fwd):.3f}")
    print(f"  V mean={np.mean(V):.3f}  W mean={np.mean(W):.3f}")

    # check A→B chain strength directly
    t_A = np.where(assembly_id == 1)[0]   # neurons peaking at symbol-0
    t_B = np.where(assembly_id == 2)[0]   # neurons peaking at symbol-1
    if len(t_A) > 0 and len(t_B) > 0:
        AB = V[np.ix_(t_B, t_A)].mean()
        print(f"  A→B chain V[B,A] = {AB:.4f}")

    plt.figure(figsize=(5,4))
    plt.imshow(V_sorted, cmap='viridis', aspect='equal',
               vmin=0, vmax=max(np.percentile(V_sorted, 98), 0.01))
    plt.colorbar(label='weight')
    for lag, col in [(3,'red'),(5,'orange')]:
        xs = np.arange(K - lag)
        plt.plot(xs, xs+lag, color=col, lw=1.0, ls='--', alpha=0.7,
                 label=f'lag-{lag}')
    plt.legend(fontsize=7)
    plt.title(f"V_sorted @ rep={rep}  (rec_delay={rec_delay_ms}ms)")
    plt.tight_layout()
    plt.show()


# ========== TRAINING ==========
print(f"Starting {num_trials} trials | rec_delay={rec_delay_ms}ms "
      f"| tau={tau} | checkpoints={checkpoints}")

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
neuron_order, mean_rate_time = compute_neuron_order_from_tests(
    seq1, W, V, 200)

for i, s in enumerate(SEQUENCES):
    spikes_test, _, _ = run_one_sequence(
        s, W.copy(), V.copy(), learn=False)
    label = "→".join(PATTERN_NAMES[p] for p in s)
    plot_sorted_raster_one_trial(
        spikes_test, s, neuron_order,
        f"FINAL Seq{i+1} ({label})  rec_delay={rec_delay_ms}ms")

V_sorted = V[np.ix_(neuron_order, neuron_order)]
plt.figure(figsize=(6,5))
plt.imshow(V_sorted, cmap='viridis', aspect='equal')
plt.colorbar(label='weight')
for lag, col in [(3,'red'),(5,'orange')]:
    xs = np.arange(K-lag)
    plt.plot(xs, xs+lag, color=col, lw=1.2, ls='--', alpha=0.8,
             label=f'lag-{lag}={np.mean(np.diag(V_sorted,k=lag)):.3f}')
plt.legend(fontsize=8)
plt.xlabel('pre (sorted)')
plt.ylabel('post (sorted)')
plt.title(f'FINAL V_sorted  (rec_delay={rec_delay_ms}ms, tau={tau})')
plt.tight_layout()
plt.show()

peak_times  = np.argmax(mean_rate_time, axis=0)
assembly_id = np.digitize(peak_times / each_obs_time, np.arange(6))
print("🏆 FINAL Assemblies:", np.bincount(assembly_id)[1:6])
print("🏆 FINAL Fwd lag-5:", f"{np.mean(np.diag(V_sorted,k=5)):.3f}")
print("🏆 FINAL V mean:",    f"{np.mean(V):.3f}")

for i, s in enumerate(SEQUENCES):
    no, mrt = compute_neuron_order_from_tests(
        s, W.copy(), V.copy(), num_test_trials=30)
    pt  = np.argmax(mrt, axis=0)
    aid = np.digitize(pt / each_obs_time, np.arange(6))
    label = "→".join(PATTERN_NAMES[p] for p in s)
    print(f"  Seq{i+1} ({label}): {np.bincount(aid)[1:6]}")
