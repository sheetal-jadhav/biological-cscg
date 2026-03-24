"""
COMPLETE Figure 2 Reproduction Script
Includes ALL panels: B-E (rasters), F (PSTH), G (rank correlation)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

from codebase.wta_stdp import WTANetwork
from codebase.old.sequence_generation import create_sequence_task

print("="*70)
print("FIGURE 2 REPRODUCTION - Sequence Memory Task")
print("Kappel et al. (2014)")
print("="*70)


# ============================================================================
# STEP 1: CREATE TASK
# ============================================================================

print("\nSTEP 1: Creating sequence task...")

task_data = create_sequence_task(
    num_inputs=200,
    input_rate=15.0,
    pattern_length=0.050,  # 50ms
    hold_length=0.150,     # 150ms
    dt=0.001,
    seed=42
)

print(f"✓ Created {task_data['num_sequences']} sequences")


# ============================================================================
# STEP 2: CREATE NETWORK
# ============================================================================

print("\nSTEP 2: Creating WTA network...")

network = WTANetwork(
    num_neurons=100,
    num_inputs=200,
    output_rate=200.0,
    input_rate=15.0,
    tau_x_r=0.002,
    tau_z_r=0.002,
    tau_x_f=0.02,
    tau_z_f=0.02,
    eta=0.005,
    dt=0.001,
    seed=42
)

print(f"✓ {network}")


# ============================================================================
# STEP 3: TRAINING
# ============================================================================

def train_network(network, task_data, num_iterations=5000, verbose=True):
    """Train network on sequences with STDP."""
    sequences = task_data['sequences']
    num_sequences = task_data['num_sequences']

    if verbose:
        print(f"\nSTEP 3: Training for {num_iterations} iterations...")

    for iteration in range(num_iterations):
        # Randomly select sequence
        seq_id = np.random.randint(num_sequences)
        spike_train = sequences[seq_id]

        # Reset network state
        network.reset_state()

        # Simulate sequence
        for t in range(len(spike_train)):
            input_spikes = spike_train[t]
            network.step(input_spikes, learn=True)

        # Print progress
        if verbose and (iteration + 1) % 500 == 0:
            print(f"  Iteration {iteration + 1}/{num_iterations}")

    if verbose:
        print("✓ Training complete!")

# Train the network
train_network(network, task_data, num_iterations=5000)


# ============================================================================
# STEP 4: TESTING
# ============================================================================

def test_network(network, spike_train, num_trials=20):
    """Test network on multiple trials."""
    trial_results = []

    for trial in range(num_trials):
        network.reset_state()

        spike_times = []
        spike_neurons = []

        for t in range(len(spike_train)):
            input_spikes = spike_train[t]
            output_spikes = network.step(input_spikes, learn=False)

            if np.any(output_spikes):
                neurons = np.where(output_spikes)[0]
                spike_times.extend([t * network.dt] * len(neurons))
                spike_neurons.extend(neurons)

        trial_results.append({
            'spike_times': np.array(spike_times),
            'spike_neurons': np.array(spike_neurons)
        })

    return trial_results

print("\nSTEP 4: Testing network...")

# Test on first sequence (AB-hold-ab)
seq_id = 0
spike_train = task_data['sequences'][seq_id]
trial_results = test_network(network, spike_train, num_trials=20)

print(f"✓ Completed 20 test trials")


# ============================================================================
# STEP 5: COMPUTE PSTH
# ============================================================================

def compute_psth(trial_results, num_neurons, duration, dt=0.001, bin_size=0.005):
    """Compute Peri-Stimulus Time Histogram."""
    time_bins = np.arange(0, duration + bin_size, bin_size)
    num_bins = len(time_bins) - 1

    psth = np.zeros((num_neurons, num_bins))

    for result in trial_results:
        spike_times = result['spike_times']
        spike_neurons = result['spike_neurons']

        for neuron in range(num_neurons):
            neuron_spikes = spike_times[spike_neurons == neuron]
            if len(neuron_spikes) > 0:
                hist, _ = np.histogram(neuron_spikes, bins=time_bins)
                psth[neuron] += hist

    # Convert to rate (Hz)
    psth = psth / (len(trial_results) * bin_size)

    # Smooth
    psth = gaussian_filter1d(psth, sigma=2, axis=1)

    return psth, time_bins[:-1]

print("\nSTEP 5: Computing PSTH...")

duration = len(spike_train) * network.dt
psth, time_bins = compute_psth(trial_results, network.num_neurons, duration)

# Sort neurons by peak time
peak_times = np.argmax(psth, axis=1)
sorted_idx = np.argsort(peak_times)
psth_sorted = psth[sorted_idx]

print(f"✓ PSTH computed and neurons sorted")


# ============================================================================
# STEP 6: COMPUTE RANK CORRELATIONS (Panel G)
# ============================================================================

def compute_rank_correlations(trial_results, sorted_idx):
    """
    Compute Spearman rank correlation for each trial.

    This measures how consistently neurons fire in the same order
    across trials compared to the expected order (from PSTH sorting).

    High correlation (~0.8-0.9) means reliable sequence replay.
    """
    correlations = []

    for trial_idx, result in enumerate(trial_results):
        spike_times = result['spike_times']
        spike_neurons = result['spike_neurons']

        if len(spike_neurons) == 0:
            continue

        # Get first spike time for each neuron
        unique_neurons = []
        first_spike_times = []

        for neuron in range(len(sorted_idx)):
            neuron_id = sorted_idx[neuron]
            neuron_spikes = spike_times[spike_neurons == neuron_id]

            if len(neuron_spikes) > 0:
                unique_neurons.append(neuron)
                first_spike_times.append(neuron_spikes[0])

        if len(unique_neurons) < 2:
            continue

        # Compute rank correlation between expected order and actual order
        expected_order = np.array(unique_neurons)
        actual_order_idx = np.argsort(first_spike_times)
        actual_order = expected_order[actual_order_idx]

        # Spearman correlation
        if len(actual_order) >= 2:
            corr, _ = spearmanr(expected_order, actual_order)
            if not np.isnan(corr):
                correlations.append(corr)

    return np.array(correlations)

print("\nSTEP 6: Computing rank correlations...")

rank_correlations = compute_rank_correlations(trial_results, sorted_idx)

print(f"✓ Computed {len(rank_correlations)} rank correlations")
print(f"  Mean correlation: {np.mean(rank_correlations):.3f}")
print(f"  Std correlation: {np.std(rank_correlations):.3f}")


# ============================================================================
# STEP 7: PLOTTING COMPLETE FIGURE 2
# ============================================================================

print("\nSTEP 7: Generating complete Figure 2...")

fig = plt.figure(figsize=(15, 11))
gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[1,1,1,1,0.8])

# Select 4 neurons for individual rasters (Panels B-E)
selected_neurons = sorted_idx[[15, 35, 60, 80]]

# Panels B-E: Individual neuron rasters
for i, neuron_id in enumerate(selected_neurons):
    ax = fig.add_subplot(gs[i, 0])

    for trial_idx, result in enumerate(trial_results):
        spike_times = result['spike_times']
        spike_neurons = result['spike_neurons']
        neuron_spikes = spike_times[spike_neurons == neuron_id]

        if len(neuron_spikes) > 0:
            ax.scatter(neuron_spikes * 1000, [trial_idx] * len(neuron_spikes),
                      s=5, c='black', marker='|', linewidths=1)

    ax.set_xlim([0, duration * 1000])
    ax.set_ylim([-1, 20])
    ax.set_ylabel('Trial', fontsize=10)

    if i == 3:
        ax.set_xlabel('Time (ms)', fontsize=10)
    else:
        ax.set_xticklabels([])

    # Label as B, C, D, E
    panel_label = chr(66 + i)  # B=66, C=67, etc.
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')

    ax.set_title(f'Neuron {neuron_id}', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Panel F: PSTH heatmap
ax_psth = fig.add_subplot(gs[0:4, 1])

im = ax_psth.imshow(psth_sorted, aspect='auto', cmap='hot',
                    extent=[0, duration * 1000, 0, network.num_neurons],
                    origin='lower', vmin=0, vmax=200, interpolation='bilinear')

ax_psth.set_xlabel('Time (ms)', fontsize=11)
ax_psth.set_ylabel('Neuron (sorted by peak time)', fontsize=11)
ax_psth.text(-0.08, 1.02, 'F', transform=ax_psth.transAxes,
             fontsize=14, fontweight='bold', va='top')

# Mark pattern boundaries
pattern_boundaries = [0, 50, 100, 250, 300, 350]
for t in pattern_boundaries:
    ax_psth.axvline(t, color='white', linestyle='--', linewidth=0.8, alpha=0.5)

# Add pattern labels
pattern_labels_pos = [(25, 'A'), (75, 'B'), (175, 'hold'), (275, 'a'), (325, 'b')]
for pos, label in pattern_labels_pos:
    ax_psth.text(pos, network.num_neurons * 1.02, label, 
                 ha='center', va='bottom', fontsize=9, color='white',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

# Colorbar
cbar = plt.colorbar(im, ax=ax_psth, fraction=0.046, pad=0.04)
cbar.set_label('Firing Rate (Hz)', fontsize=10)

# Panel G: Rank correlation histogram
ax_corr = fig.add_subplot(gs[4, :])

if len(rank_correlations) > 0:
    bins = np.linspace(-1, 1, 21)
    ax_corr.hist(rank_correlations, bins=bins, color='steelblue', 
                 edgecolor='black', alpha=0.7)

    # Add mean line
    mean_corr = np.mean(rank_correlations)
    ax_corr.axvline(mean_corr, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_corr:.3f}')
    ax_corr.legend(loc='upper left', fontsize=10)

ax_corr.set_xlabel('Rank Correlation', fontsize=11)
ax_corr.set_ylabel('Number of Trials', fontsize=11)
ax_corr.set_xlim([-1.05, 1.05])
ax_corr.text(-0.04, 1.1, 'G', transform=ax_corr.transAxes,
             fontsize=14, fontweight='bold', va='top')
ax_corr.spines['top'].set_visible(False)
ax_corr.spines['right'].set_visible(False)

plt.suptitle(f'Figure 2: Sequence Memory Task - {task_data["sequence_names"][seq_id]}', 
             fontsize=14, fontweight='bold', y=0.995)

# Save figure
plt.savefig('figure2_complete.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_complete.pdf', bbox_inches='tight')

print("✓ Figure saved as figure2_complete.png and figure2_complete.pdf")

plt.show()

print("\n" + "="*70)
print("COMPLETE! ALL PANELS REPRODUCED")
print("="*70)
print("\nFigure 2 components:")
print("  ✓ Panels B-E: Individual neuron rasters (4 neurons)")
print("  ✓ Panel F: Population PSTH heatmap")
print("  ✓ Panel G: Rank correlation histogram")
print(f"\nResults:")
print(f"  - Mean rank correlation: {np.mean(rank_correlations):.3f}")
print(f"  - Training iterations: 5000")
print(f"  - Number of trials: 20")
print("\nFor better results:")
print("  - Increase num_iterations to 10000-20000")
print("  - Run multiple times with different seeds")
print("="*70)
