
import numpy as np

# =========================
# Hyperparameters
# =========================
N = 200                       # number of neurons / units
n_obs = 6                     # number of observation populations (edit to match your task)
P = n_obs                     # no extra reward population

n_trials = 200
trial_len = 40                # timesteps per trial
dt = 1.0

beta = 0.005                  # learning rate
tau_x = 10.0                  # pre trace decay
tau_y = 10.0                  # post trace decay
tau_mem = 10.0                # membrane / activity decay

w_in_scale = 0.3
w_rec_scale = 0.01
noise_scale = 0.01
temp = 1.0

rng = np.random.default_rng(0)

# =========================
# Task structure
# =========================
# Example: two context-dependent sequences with different starts
# Replace these with your actual room / maze sequences.
#
# Trial type 0: TL start
# Trial type 1: TR start
#
# Each number is an observation id in [0, n_obs-1]

seq_trial0 = [0, 2, 3, 4, 5, 1]
seq_trial1 = [1, 2, 3, 4, 5, 0]

trial_types = rng.integers(0, 2, size=n_trials)

# =========================
# Inputs
# =========================
# x_all[trial, time, neuron] = input rate/probability proxy
# We assign a random subpopulation to each observation.

obs_patterns = np.zeros((n_obs, N))
units_per_obs = max(5, N // 10)

for obs in range(n_obs):
    idx = rng.choice(N, size=units_per_obs, replace=False)
    obs_patterns[obs, idx] = 1.0

# Optional overlap between observations can be added if desired
obs_patterns /= np.maximum(obs_patterns.sum(axis=1, keepdims=True), 1.0)

x_all = np.zeros((n_trials, trial_len, N))
obs_id_all = np.zeros((n_trials, trial_len), dtype=int)

for tr in range(n_trials):
    base_seq = seq_trial0 if trial_types[tr] == 0 else seq_trial1
    seq = (base_seq * ((trial_len // len(base_seq)) + 1))[:trial_len]
    for t, obs in enumerate(seq):
        obs_id_all[tr, t] = obs
        x_all[tr, t, :] = 25.0 * obs_patterns[obs]   # same scale as original probability/rate style

# =========================
# Network parameters
# =========================
W_in = rng.normal(0.0, w_in_scale, size=(N, N))
W_rec = rng.normal(0.0, w_rec_scale, size=(N, N))
np.fill_diagonal(W_rec, 0.0)

# Traces / state
x_trace = np.zeros(N)         # presynaptic trace
y_trace = np.zeros(N)         # postsynaptic trace
u = np.zeros(N)               # membrane / activation state

# Logging
spikes_all = np.zeros((n_trials, trial_len, N), dtype=np.int8)
winner_all = np.full((n_trials, trial_len), -1, dtype=int)

# =========================
# Helpers
# =========================
def softmax(z, temp=1.0):
    z = z / max(temp, 1e-8)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

# =========================
# Training loop
# =========================
for tr in range(n_trials):
    # reset per trial if you want trial-separated dynamics
    x_trace[:] = 0.0
    y_trace[:] = 0.0
    u[:] = 0.0

    for t in range(trial_len):
        sensory_drive = x_all[tr, t, :]

        # Bernoulli sensory spikes from current observation only
        sensory_spikes = (rng.random(N) < (sensory_drive / 1000.0)).astype(float)

        # Decay traces
        x_trace *= np.exp(-dt / tau_x)
        y_trace *= np.exp(-dt / tau_y)
        u *= np.exp(-dt / tau_mem)

        # Update pre trace from sensory input
        x_trace += sensory_spikes

        # Recurrent competition / WTA-like selection
        drive = W_in @ x_trace + W_rec @ y_trace + noise_scale * rng.standard_normal(N)
        probs = softmax(drive, temp=temp)

        # One winner spike per timestep
        winner = rng.choice(N, p=probs)
        post_spikes = np.zeros(N)
        post_spikes[winner] = 1.0

        # Update post trace
        y_trace += post_spikes
        u += post_spikes

        spikes_all[tr, t, :] = post_spikes.astype(np.int8)
        winner_all[tr, t] = winner

        # =========================
        # Hebbian / STDP-like updates
        # =========================
        neur = winner

        # Input weights: pre = sensory trace, post = winner
        W_in[neur, :] += beta * (np.exp(-W_in[neur, :]) * x_trace - 0.1)

        # Recurrent weights: pre = previous active trace, post = winner
        W_rec[neur, :] += beta * (np.exp(-W_rec[neur, :]) * y_trace - 0.1)

        # No self-recurrence
        W_rec[neur, neur] = 0.0

# =========================
# Simple decoding / analysis
# =========================
# Average winner identity by observation and trial type
mean_winner_by_obs_type = {}

for typ in [0, 1]:
    for obs in range(n_obs):
        mask = (trial_types[:, None] == typ) & (obs_id_all == obs)
        winners = winner_all[mask]
        if winners.size > 0:
            mean_winner_by_obs_type[(typ, obs)] = np.bincount(winners, minlength=N) / winners.size
        else:
            mean_winner_by_obs_type[(typ, obs)] = np.zeros(N)

print("Training complete.")
print("W_in shape:", W_in.shape)
print("W_rec shape:", W_rec.shape)