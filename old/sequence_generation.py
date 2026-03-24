


"""
Sequence Generation for Memory Task

Creates sequences like: AB-hold-ab, BA-hold-ba, CD-hold-cd, DC-hold-dc
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from codebase.old.pattern_generation import generate_beta_patterns, generate_poisson_spike_train


def create_sequence_task(
    num_inputs: int = 200,
    input_rate: float = 15.0,
    pattern_length: float = 0.050,  # 50ms
    hold_length: float = 0.150,     # 150ms
    dt: float = 0.001,
    seed: Optional[int] = None
) -> Dict:
    """
    Create the sequence memory task from run_experiment.m

    Pattern indices: 0=A, 1=B, 2=C, 3=D, 4=hold, 5=a, 6=b, 7=c, 8=d

    Parameters:
    -----------
    num_inputs : int
        Number of input neurons
    input_rate : float
        Mean input rate (Hz)
    pattern_length : float
        Duration of each pattern (s)
    hold_length : float
        Duration of hold/delay period (s)
    dt : float
        Time step (s)
    seed : int, optional
        Random seed

    Returns:
    --------
    task_data : dict
        Dictionary containing:
        - 'patterns': List of 9 rate patterns
        - 'sequences': List of 4 spike train sequences
        - 'sequence_ids': Pattern IDs for each sequence
        - 'labels': Pattern labels
        - 'times': Time array for each sequence
        - 'durations': Duration info
    """
    # Generate 9 patterns
    print("Generating 9 patterns (A, B, C, D, hold, a, b, c, d)...")
    patterns = generate_beta_patterns(
        num_patterns=9,
        num_inputs=num_inputs,
        input_rate=input_rate,
        alpha=0.2,
        beta=0.8,
        seed=seed
    )

    # Pattern labels
    labels = ['A', 'B', 'C', 'D', 'hold', 'a', 'b', 'c', 'd']

    # Sequence definitions (0-indexed)
    # AB-hold-ab, BA-hold-ba, CD-hold-cd, DC-hold-dc
    sequence_ids = [
        [0, 1, 4, 5, 6],  # AB-hold-ab
        [1, 0, 4, 6, 5],  # BA-hold-ba
        [2, 3, 4, 7, 8],  # CD-hold-cd
        [3, 2, 4, 8, 7],  # DC-hold-dc
    ]

    sequence_names = [
        'AB-hold-ab',
        'BA-hold-ba',
        'CD-hold-cd',
        'DC-hold-dc'
    ]

    print(f"Creating {len(sequence_ids)} sequences...")

    # Generate spike trains for each sequence
    sequences = []
    times = []

    rng = np.random.RandomState(seed)

    for seq_idx, seq_ids in enumerate(sequence_ids):
        print(f"  {seq_idx+1}. {sequence_names[seq_idx]}")

        segments = []
        current_time = 0.0

        for pat_id in seq_ids:
            # Determine duration
            if pat_id == 4:  # Hold period
                duration = hold_length
            else:
                duration = pattern_length

            # Generate spike train for this pattern
            spike_segment = generate_poisson_spike_train(
                patterns[pat_id],
                duration=duration,
                dt=dt,
                seed=rng.randint(1000000)
            )

            segments.append(spike_segment)
            current_time += duration

        # Concatenate all segments
        spike_train = np.vstack(segments)
        time_array = np.arange(len(spike_train)) * dt

        sequences.append(spike_train)
        times.append(time_array)

    task_data = {
        'patterns': patterns,
        'sequences': sequences,
        'sequence_ids': sequence_ids,
        'sequence_names': sequence_names,
        'labels': labels,
        'times': times,
        'num_sequences': len(sequences),
        'pattern_length': pattern_length,
        'hold_length': hold_length,
        'dt': dt
    }

    return task_data


# Test the function
if __name__ == "__main__":
    print("="*70)
    print("Testing Sequence Generation")
    print("="*70)

    task_data = create_sequence_task(
        num_inputs=200,
        input_rate=15.0,
        pattern_length=0.050,  # 50ms
        hold_length=0.150,     # 150ms
        dt=0.001,
        seed=42
    )

    print(f"\nTask created successfully!")
    print(f"\nSequence information:")
    print(f"  Number of sequences: {task_data['num_sequences']}")

    for i, name in enumerate(task_data['sequence_names']):
        seq = task_data['sequences'][i]
        time = task_data['times'][i]
        total_spikes = np.sum(seq)
        duration_ms = time[-1] * 1000

        print(f"\n  {i+1}. {name}")
        print(f"     Duration: {duration_ms:.0f} ms")
        print(f"     Total spikes: {total_spikes}")
        print(f"     Spike rate: {total_spikes / (duration_ms/1000) / 200:.1f} Hz per input")

    print(f"\n{'='*70}")
    print("Sequence generation working! ✓")
    print("="*70)
