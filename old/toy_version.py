import numpy as np

"""
This function simulates a recurrant WTA spiking network in continuous time 
over a given window, including refractory effects and STDP style weight updates
The code is a python translation of original MATLAB code by David Kappel

OVER ALL STEPS
1. the function takes a network, spike data, and a time index range
2. it samples the number of spikes from a poisson process with a given rate..lambda 
3. for each sampled time follwing steps happen
    - update synaptic traces from input and output spikes
    - compute membrane potenial using a refractory term
    - compute WTA spike probabilities 
    - sample which neuron spikes and store them
    - accumulate weight update in STDP fashion

Intuition in one line --
This is an event‑based sampler: it runs a spike‑response‑model‑like recurrent WTA network 
with refractory dynamics, draws spikes according to instantaneous softmax firing probabilities, 
and computes local learning signals that can be interpreted as Bayesian / importance‑weighted STDP updates 
for the network parameters.
"""

# steps followed in the matlab code 

# STEP 1: Time window and number of spikes

# STEP 2: Initilize recurrent spikes and state variables 

# STEP 3: Variance tracking for adaptive learning rates

# STEP 4: Flags and reset spike counts

# STEP 5: main loop over output spikes

# STEP 6: update feedforward PSP traces

# STEP 7: update recurrent PSP traces

# STEP 8: decay PSP to current time and compute effective PSP

# STEP 9: compute membrane potential including refractory period

# STEP 10: spike probabilties WTA

# STEP 11: ...did not understand 

# STEP 12: draw a spike and update book keeping 

# STEP 13: compute STDP style wight increments

# STEP 14: 

# --------------
class WTANetwork:
    def __init__(self, n_cir=100, n_in=200, lam=2000, seed=42):
        self.N = n_cir
        self.N_in = n_in
        self.lam = lam
        self.rng = np.random.default_rng(seed)
        
        # 🐛 BUG #2 FIX: Weights in LOG-SPACE (MATLAB does this!)
        # MATLAB: W_exp = exp(W)
        self.W_ff = self.rng.uniform(-2.5, -2.0, (self.N, self.N_in)).astype(np.float32)
        self.W_lat = self.rng.uniform(-3.5, -3.0, (self.N, self.N)).astype(np.float32)
        np.fill_diagonal(self.W_lat, -10.0)  # Very negative = no self-connection
        
        # Learning rates (per-synapse like MATLAB)
        self.eta_W = np.full((self.N, self.N_in), ETA_W, dtype=np.float32)
        self.eta_V = np.full((self.N, self.N), ETA_V, dtype=np.float32)
        
        self.h_ff = np.zeros((self.N_in, 2), dtype=np.float32)
        self.h_lat = np.zeros((self.N, 2), dtype=np.float32)
        self.last_t = np.zeros(self.N, dtype=np.float32)
        self.spikes = []
    
    def reset(self):
        self.h_ff[:] = 0
        self.h_lat[:] = 0
        self.last_t[:] = 0
        self.spikes = []
    
    def update_traces(self, dt):
        self.h_ff[:, 0] *= np.exp(-dt / TAU_RISE)
        self.h_ff[:, 1] *= np.exp(-dt / TAU_FALL)
        self.h_lat[:, 0] *= np.exp(-dt / TAU_RISE)
        self.h_lat[:, 1] *= np.exp(-dt / TAU_FALL)
    
    def compute_u(self, t):
        d_ff = self.h_ff[:, 0] - self.h_ff[:, 1]
        d_lat = self.h_lat[:, 0] - self.h_lat[:, 1]
        
        # Convert from log-space (MATLAB line 90-92)
        W_ff_exp = np.exp(np.clip(self.W_ff, -10, 3))
        W_lat_exp = np.exp(np.clip(self.W_lat, -10, 2))
        
        u = W_ff_exp @ d_ff + W_lat_exp @ d_lat
        dt = t - self.last_t
        u += W_REFRAC * np.exp(-dt / TAU_REFRAC)
        return u
    
    def softmax(self, u):
        exp_u = np.exp(u - u.max())
        return (exp_u / exp_u.sum()) * self.lam
    
    def stdp(self, k):
        """
        🐛 BUG #3 FIX: EXACT MATLAB STDP (lines 196, 218-219)
        """
        d_ff = self.h_ff[:, 0] - self.h_ff[:, 1]
        d_lat = self.h_lat[:, 0] - self.h_lat[:, 1]
        
        # Convert to exp-space
        W_ff_exp = np.exp(np.clip(self.W_ff[k], -10, 3))
        W_lat_exp = np.exp(np.clip(self.W_lat[k], -10, 2))
        
        # Feedforward STDP (MATLAB line 196)
        # net.d_W(k,:) = ... + eta_W.*(d_hX' - W_exp(k,:)) ./ max(eta_W, W_exp(k,:))
        dW = self.eta_W[k] * (d_ff - W_ff_exp) / np.maximum(self.eta_W[k], W_ff_exp)
        self.W_ff[k] += dW
        self.W_ff[k] = np.clip(self.W_ff[k], -8.0, 2.0)
        
        # Lateral STDP (MATLAB lines 218-219, homeostatic version)
        # Line 218: net.d_V(k,:) = ... + 2*eta_V.*(d_hZ' - V_exp(k,:)) ./ max(eta_V, V_exp(k,:))
        dV_out = 2 * self.eta_V[k] * (d_lat - W_lat_exp) / np.maximum(self.eta_V[k], W_lat_exp)
        self.W_lat[k] += dV_out
        
        # Line 219: net.d_V(:,k) = ... - eta_V.*d_hZ (homeostatic)
        dV_in = -self.eta_V[:, k] * d_lat
        self.W_lat[:, k] += dV_in
        
        self.W_lat[k] = np.clip(self.W_lat[k], -8.0, 1.0)
        np.fill_diagonal(self.W_lat, -10.0)
    
    def simulate(self, inp_spikes, T, learn=True):
        n_sp = self.rng.poisson(self.lam * T / 1000)
        times = np.sort(self.rng.uniform(0, T, n_sp))
        idx = 0
        n_inp = inp_spikes.shape[1]
        
        for t in times:
            while idx < n_inp and inp_spikes[1, idx] <= t:
                nid = int(inp_spikes[0, idx])
                self.h_ff[nid] += 1
                idx += 1
            
            self.update_traces(0.1)
            u = self.compute_u(t)
            rates = self.softmax(u)
            k = self.rng.choice(self.N, p=rates / rates.sum())
            
            self.spikes.append((t, k))
            self.last_t[k] = t
            if learn:
                self.stdp(k)
            self.h_lat[k] += 1  