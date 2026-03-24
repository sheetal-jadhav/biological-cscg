"""
Pattern Generation for Sequence Memory Task
Based on generate_rate_pattern.m from hmm-stdp repository

This generates input patterns using beta distribution as specified in default_options.m
"""

import numpy as np
from typing import List, Optional


def generate_beta_patterns(
    num_patterns: int,
    num_inputs: int,
    input_rate: float = 15.0,
    alpha: float = 0.2,
    beta: float = 0.8,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate rate patterns using beta distribution.

    From default_options.m:
    - pattern_type: 'beta'
    - pat_alpha: 0.2
    - pat_beta: 0.8
    - input_rate: 15 Hz per neuron

    From do_learning_task.m (gen_patterns function):
    ```matlab
    pat_beta_dist_mean = (pat_alpha/(pat_alpha+pat_beta));
    max_in_rate = input_rate/pat_beta_dist_mean;
    pats_{i_} = max_in_rate*betarnd(pat_alpha, pat_beta, [num_inputs, 1]);
    ```

    Parameters:
    -----------
    num_patterns : int
        Number of distinct patterns to generate
    num_inputs : int
        Number of input neurons (afferent neurons)
    input_rate : float
        Mean input rate per neuron (Hz)
    alpha : float
        Alpha parameter for beta distribution
    beta : float
        Beta parameter for beta distribution
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    patterns : list of ndarray
        List of rate patterns, each of shape (num_inputs,)
    """
    rng = np.random.RandomState(seed)

    # Calculate mean of beta distribution
    beta_mean = alpha / (alpha + beta)

    # Scale to achieve desired input rate
    max_rate = input_rate / beta_mean

    patterns = []
    for i in range(num_patterns):
        # Draw from beta distribution and scale
        pattern = max_rate * rng.beta(alpha, beta, num_inputs)
        patterns.append(pattern)

    return patterns


def generate_poisson_spike_train(
    rate_pattern: np.ndarray,
    duration: float,
    dt: float = 0.001,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Poisson spike train from rate pattern.

    Converts continuous rates to discrete spike times using inhomogeneous
    Poisson process.

    Parameters:
    -----------
    rate_pattern : ndarray
        Firing rates for each input neuron (num_inputs,) in Hz
    duration : float
        Duration of spike train in seconds
    dt : float
        Time step in seconds (default 1ms)
    seed : int, optional
        Random seed

    Returns:
    --------
    spike_train : ndarray
        Binary spike train of shape (num_steps, num_inputs)
        spike_train[t, i] = 1 if neuron i spikes at time t
    """
    rng = np.random.RandomState(seed)

    num_steps = int(duration / dt)
    num_inputs = len(rate_pattern)

    spike_train = np.zeros((num_steps, num_inputs), dtype=bool)

    # Generate spikes at each time step using Poisson process
    for t in range(num_steps):
        spike_probs = rate_pattern * dt
        spike_probs = np.clip(spike_probs, 0, 1)  # Ensure valid probabilities
        spike_train[t] = rng.rand(num_inputs) < spike_probs

    return spike_train


# Test the functions
if __name__ == "__main__":
    print("="*70)
    print("Testing Pattern Generation")
    print("="*70)

    # Generate 9 patterns (A, B, C, D, hold, a, b, c, d)
    patterns = generate_beta_patterns(
        num_patterns=9,
        num_inputs=200,
        input_rate=15.0,
        alpha=0.2,
        beta=0.8,
        seed=42
    )

    print(f"\nGenerated {len(patterns)} patterns")
    print(f"Each pattern has {len(patterns[0])} input neurons")
    print(f"\nPattern statistics:")

    for i, pattern in enumerate(patterns[:5]):  # Show first 5
        labels = ['A', 'B', 'C', 'D', 'hold', 'a', 'b', 'c', 'd']
        print(f"  Pattern {labels[i]}: mean={np.mean(pattern):.2f} Hz, "
              f"max={np.max(pattern):.2f} Hz, "
              f"min={np.min(pattern):.2f} Hz")

    # Generate spike train for one pattern
    print(f"\nGenerating spike train for pattern A (duration=50ms)...")
    spike_train = generate_poisson_spike_train(
        patterns[0],
        duration=0.050,  # 50ms
        dt=0.001,
        seed=42
    )

    print(f"Spike train shape: {spike_train.shape}")
    print(f"Total spikes: {np.sum(spike_train)}")
    print(f"Spikes per ms: {np.sum(spike_train) / 50:.1f}")

    print(f"\n{'='*70}")
    print("Pattern generation working! ✓")
    print("="*70)
