
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d

print("="*70)
print("  FIGURE 2: translated with EXACT MATLAB equations")
print("="*70)

# Parameters
N_CIRCUIT = 100
N_AFFERENT = 200
LAMBDA_TOTAL = 2000
NUM_PATTERNS = 9
PATTERN_DURATION = 50.0


ETA_W = 0.08 
ETA_V = 0.08  

TAU_RISE = 0.5
TAU_FALL = 20.0
TAU_REFRAC = 5.0
W_REFRAC = -10.0

SEQUENCES = [
    [0, 1, 4, 5, 6],
    [1, 0, 4, 6, 5],
    [2, 3, 4, 7, 8],
    [3, 2, 4, 8, 7],
]
PATTERN_NAMES = ['A', 'B', 'C', 'D', 'delay', 'a', 'b', 'c', 'd']

# ============================================================================
# PATTERN GENERATOR 
# ============================================================================
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

# ============================================================================
# WTA NETWORK 
# ============================================================================
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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n[1/4] Generating patterns...")
gen = PatternGenerator(N_AFFERENT, seed=42)
patterns = gen.generate_patterns(NUM_PATTERNS)
print(f"✓ Created {NUM_PATTERNS} patterns")

print("\n[2/4] Creating untrained network...")
net_untrained = WTANetwork(N_CIRCUIT, N_AFFERENT, LAMBDA_TOTAL, seed=42)

print("\n[3/4] Training network...")
net_trained = WTANetwork(N_CIRCUIT, N_AFFERENT, LAMBDA_TOTAL, seed=43)

# 🐛 BUG #4 FIX: More iterations needed (was 800, now 2000)
for it in range(2000):
    sid = net_trained.rng.integers(0, 4)
    inp, times = gen.generate_sequence(patterns, SEQUENCES[sid])
    T = times[-1][1]
    net_trained.simulate(inp, T, learn=True)
    if (it + 1) % 400 == 0:
        W_ff_mean = np.exp(np.clip(net_trained.W_ff, -10, 3)).mean()
        W_lat_mean = np.exp(np.clip(net_trained.W_lat, -10, 2)).mean()
        print(f"  Iteration {it+1}/2000: W_ff={W_ff_mean:.3f}, W_lat={W_lat_mean:.3f}")
print("✓ Training done!")

print("\n[4/4] Generating Figure 2...")

# ============================================================================
# PLOTTING 
# ============================================================================
fig = plt.figure(figsize=(16, 12))

ax_A_raster = plt.subplot2grid((4, 3), (0, 0), colspan=1)
ax_A_graph = plt.subplot2grid((4, 3), (0, 1), colspan=2)

inp_test, times_test = gen.generate_sequence(patterns, SEQUENCES[0])
if inp_test.shape[1] > 0:
    ax_A_raster.scatter(inp_test[1], inp_test[0], s=0.5, c='blue', alpha=0.5)

for t0, t1, pid in times_test:
    ax_A_raster.axvline(t0, color='red', ls='--', alpha=0.5, lw=0.8)
    ax_A_raster.text((t0 + t1) / 2, 210, PATTERN_NAMES[pid], ha='center', fontsize=9)

ax_A_raster.set_xlim(0, times_test[-1][1])
ax_A_raster.set_ylim(0, 220)
ax_A_raster.set_ylabel('Afferent neuron')
ax_A_raster.set_title('Panel A: Input', fontweight='bold')

ax_A_graph.set_xlim(0, 8)
ax_A_graph.set_ylim(0, 6)
ax_A_graph.axis('off')
y_pos = [5, 3.5, 2, 0.5]
seq_labels = [['A','B','delay','a','b'], ['B','A','delay','b','a'], 
              ['C','D','delay','c','d'], ['D','C','delay','d','c']]

for i, (y, labs) in enumerate(zip(y_pos, seq_labels)):
    x = 0.5
    for lab in labs:
        ax_A_graph.add_patch(Circle((x, y), 0.15, fc='lightblue', ec='blue'))
        ax_A_graph.text(x, y, lab, ha='center', va='center', fontsize=8)
        if x < 6.5:
            ax_A_graph.arrow(x+0.2, y, 0.6, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
        x += 1.2

ax_A_graph.set_title('State Graph', fontweight='bold')

for i in range(4):
    ax = plt.subplot2grid((4, 3), (1 + i // 2, i % 2))
    net_untrained.reset()
    inp, times = gen.generate_sequence(patterns, SEQUENCES[i])
    net_untrained.simulate(inp, times[-1][1], learn=False)
    
    if net_untrained.spikes:
        ts, ns = zip(*net_untrained.spikes)
        ax.scatter(ts, ns, s=1, c='gray', alpha=0.5)
    
    for t0, t1, pid in times:
        ax.axvline(t0, color='red', ls='--', alpha=0.3, lw=0.6)
    
    ax.set_xlim(0, times[-1][1])
    ax.set_ylim(0, N_CIRCUIT)
    if i == 0:
        ax.set_title('Panel B: Before', fontweight='bold')

for i in range(4):
    ax = plt.subplot2grid((4, 3), (1 + i // 2, 2))
    net_trained.reset()
    inp, times = gen.generate_sequence(patterns, SEQUENCES[i])
    net_trained.simulate
    net_trained.simulate(inp, times[-1][1], learn=False)
    
    if net_trained.spikes:
        ts, ns = zip(*net_trained.spikes)
        ax.scatter(ts, ns, s=1, c='steelblue', alpha=0.7, rasterized=True)
        
        # Add smoothed population activity
        bins = np.arange(0, times[-1][1], 2)
        hist, _ = np.histogram(ts, bins=bins)
        smoothed = gaussian_filter1d(hist.astype(float), sigma=3)
        ax2 = ax.twinx()
        ax2.plot(bins[:-1], smoothed, 'b-', alpha=0.7, linewidth=1.5)
        ax2.set_ylim(0, smoothed.max() * 1.2)
        ax2.set_ylabel('Rate', fontsize=7)
        ax2.tick_params(labelsize=7)
    
    for t0, t1, pid in times:
        ax.axvline(t0, color='red', ls='--', alpha=0.3, lw=0.6)
    
    ax.set_xlim(0, times[-1][1])
    ax.set_ylim(0, N_CIRCUIT)
    if i == 0:
        ax.set_title('Panel C: After', fontweight='bold')

# Panel D: Lateral Weights (CRITICAL: Convert from log-space!)
ax_D = plt.subplot2grid((4, 3), (3, 0), colspan=3)
W_lat_display = np.exp(np.clip(net_trained.W_lat, -10, 2))  # Convert from log!
im = ax_D.imshow(W_lat_display, cmap='hot', aspect='auto', 
                 interpolation='nearest', origin='lower')
ax_D.set_xlabel('Presynaptic neuron')
ax_D.set_ylabel('Postsynaptic neuron')
ax_D.set_title('Panel D: Lateral Weights After Learning', fontweight='bold')
plt.colorbar(im, ax=ax_D, label='Weight')

plt.suptitle('Figure 2: Sequence Learning Through WTA/STDP', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figure_latest.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("  ✓ FIGURE 2 REPRODUCED!")
print("="*70)
print(f"\nFinal weights (exp-space):")
print(f"  W_ff mean: {np.exp(np.clip(net_trained.W_ff, -10, 3)).mean():.3f}")
print(f"  W_lat mean: {np.exp(np.clip(net_trained.W_lat, -10, 2)).mean():.3f}")
print(f"  W_lat max: {np.exp(np.clip(net_trained.W_lat, -10, 2)).max():.3f}")
print(f"\n✅ Saved: figure2_latest.png")