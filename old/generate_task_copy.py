import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d

class PatternGenerator:
    def __init__(self, num_inputs=200, seed=42):
        self.num_inputs = num_inputs
        self.rng = np.random.default_rng(seed)
    '''
     def generate_patterns(self, n=9):
        patterns = np.zeros((n, self.num_inputs))
        for i in range(n):
            raw = self.rng.beta(0.2, 0.8, self.num_inputs)
            patterns[i] = 5.0 + raw * 95.0
        return patterns
    '''
    
    def generate_patterns(self, n=9):
        patterns = np.zeros((n, self.num_inputs))
        for i in range(n):
            active = self.rng.choice(self.num_inputs, size=40, replace=False)
            patterns[i, active] = 100.0  # Hz
        return patterns

    
    def pattern_to_spikes(self, rates, dur):
        spikes = []
        for nid, r in enumerate(rates):
            if r > 0:
                t = 0
                while t < dur:
                    t += self.rng.exponential(1000.0 / r)
                    if t < dur:
                        spikes.append([nid, t])
        if not spikes:
            return np.zeros((2, 0))
        spikes = np.array(spikes).T
        return spikes[:, np.argsort(spikes[1])]
    
    def generate_sequence(self, patterns, seq_ids, dur=50.0):
        all_spikes = []
        times = []
        t = 0
        for pid in seq_ids:
            d = max(10, self.rng.normal(dur, 5))
            sp = self.pattern_to_spikes(patterns[pid], d)
            if sp.shape[1] > 0:
                sp[1, :] += t
                all_spikes.append(sp)
            times.append((t, t + d, pid))
            t += d
        if not all_spikes:
            return np.zeros((2, 0)), times
        return np.hstack(all_spikes), times
    

#-----------------------------------------------------------
# PLOTS TO TEST THE CODE
# -----------------------------------------------------------

'''
spikes, seg_times = gen.generate_sequence(patterns, [0,1,4,5,6], dur=50.0)
# spikes shape (2, n_spikes): spikes[0] neuron ids, spikes[1] times (ms)
print("segments:", seg_times)
print("num spikes:", spikes.shape[1])

# ...existing code...
spikes, seg_times = gen.generate_sequence(patterns, [0,1,4,5,6], dur=50.0)
print("segments:", seg_times)
print("num spikes:", spikes.shape[1])

# raster plot from spikes (spikes: [neuron_ids; times_ms])
neuron_ids = spikes[0].astype(int)
times_ms = spikes[1].astype(float)  # times are in ms

plt.figure(figsize=(10, 4))
plt.scatter(times_ms, neuron_ids, s=2, color="k", marker="|")
# annotate segment boundaries
for (t0, t1, pid) in seg_times:
    plt.axvline(t0, color="gray", linestyle="--", linewidth=0.6)
    plt.text((t0 + t1) / 2, -2, f"{PATTERN_NAMES[pid]}", ha="center", va="top", fontsize=9)
plt.xlabel("time (ms)")
plt.ylabel("neuron id")
plt.ylim(-5, gen.num_inputs)  # adjust if using different num_inputs
plt.title("Input spike raster")
plt.tight_layout()
plt.show()

# ...existing code...
# plot rasters for all sequences
n_seqs = len(SEQUENCES)
fig, axes = plt.subplots(n_seqs, 1, figsize=(10, 2.5 * n_seqs), sharex=True)
if n_seqs == 1:
    axes = [axes]

for i, seq_ids in enumerate(SEQUENCES):
    spikes, seg_times = gen.generate_sequence(patterns, seq_ids, dur=PATTERN_DURATION)
    neuron_ids = spikes[0].astype(int)
    times_ms = spikes[1].astype(float)

    ax = axes[i]
    if spikes.shape[1] > 0:
        ax.scatter(times_ms, neuron_ids, s=2, color="k", marker="|")

    # annotate segment boundaries and labels
    for (t0, t1, pid) in seg_times:
        ax.axvline(t0, color="gray", linestyle="--", linewidth=0.6)
        ax.text((t0 + t1) / 2, -2, f"{PATTERN_NAMES[pid]}", ha="center", va="top", fontsize=9)

    ax.set_ylim(-5, N_AFFERENT)
    ax.set_ylabel("neuron")
    seq_label = "-".join([PATTERN_NAMES[p] for p in seq_ids])
    ax.set_title(f"Sequence {i+1}: {seq_label}")

axes[-1].set_xlabel("time (ms)")
plt.tight_layout()
plt.show()
# ...existing
'''