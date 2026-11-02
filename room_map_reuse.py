'''
Reusing a learned room map in a recoloured room (3x3, action-augmented Hebbian network).

For each of three rooms (non-aliased, cells 0 & 8 aliased, four corners aliased):
  1. Room 1: a network learns the room from a random walk
     (Hebbian rule, one V[a] per action, surprise correction on).
  2. Room 2: same layout and same aliasing pattern, but every colour changed.
  3. Two ways to learn room 2, on the same random walk:
       - from scratch: a new network, all weights learned, surprise correction on
       - map reused: room-1 transitions V[a] frozen, input weights W reset
         to small values and relearned
  4. While learning, before every step, the network's probability for the NEXT
     colour is recorded (from its transitions V[a] and a running record of the
     colour each neuron fires for). Plotted with the fraction of cells visited.

This is a vectorised (fast) version of the same equations as
hebbian_rnn_actions_room.py, so the whole figure takes about 1-2 minutes.
Needs numpy and matplotlib.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------- network parameters (same as the loop script) ----------------
vhat, tau, beta = 500, 20, 0.1
eta0, tau_ref, ref_time, baseline = 10, 5, 10, 10
N, K, W_std, V_std = 90, 50, 3.5, 0.3
input_fr, T = 100, 20                       # Hz, ms per step
gamma, V_max, g_adapt, tau_adapt, z_thr = 3.0, 10.0, 5.0, 20.0, 1.0
W_reuse_std = 0.3                           # input weights are reset to small values when the map is reused

n_room1_steps, n_room2_steps = 8000, 6000
ACTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])          # up, down, left, right

ROOMS = {'Non-aliased':          np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
         'Cells 0 & 8 aliased':  np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0]]),
         'Four corners aliased': np.array([[0, 1, 0], [3, 4, 5], [0, 7, 0]])}
NEW_COLOURS = {'Non-aliased':          [3, 7, 0, 5, 8, 1, 6, 2, 4],   # colour c in room 1 -> NEW_COLOURS[c] in room 2
               'Cells 0 & 8 aliased':  [5, 2, 7, 0, 3, 6, 1, 4],
               'Four corners aliased': [4, 2, 6, 7, 0, 3, 5, 1]}
COLOURS = ['#6B21A8', '#F472B6', '#EAB308', '#93C5FD', '#9CA3AF', '#0D9488', '#16A34A', '#D97706', '#991B1B']


def random_walk(room, n, rng):
    H, Wd = room.shape
    r, c = rng.integers(H), rng.integers(Wd)
    obs, acts, cells = [], [], []
    for _ in range(n):
        obs.append(room[r, c]); cells.append(r * Wd + c)
        a = rng.integers(4); acts.append(a)
        nr, nc = r + ACTIONS[a][0], c + ACTIONS[a][1]
        if 0 <= nr < H and 0 <= nc < Wd:                      # wall bump = stay
            r, c = nr, nc
    return np.array(obs), np.array(acts), np.array(cells)


def run(room, obs, acts, W, V, rng, learn_W=True, learn_V=True, surprise=True):
    '''Network with action-selected V[a]. Returns W, V and, for every step,
    the probability the network gave to the colour seen at the next step.'''
    P = room.max() + 1
    per = N // P
    x_rate = np.zeros((P, N))
    for p in range(P):
        x_rate[p, p * per:(p + 1) * per] = input_fr
    x, y = np.zeros(N), np.zeros(K)
    ref, ref_in, tss = np.zeros(K), np.zeros(N), 7 * np.ones(K)
    dec, b = np.exp(-1 / tau), baseline * np.ones(K)
    n = len(obs)
    vmean, y_start = np.zeros((n, K)), np.zeros((n, K))
    adapt, mu, var = np.zeros(K), 0.5, 0.01
    colour_count = np.full((K, P), 1e-3)                 # readout: colour seen while each neuron is active
    p_next = np.full(n, np.nan)

    for s, o in enumerate(obs):
        a = acts[s - 1] if s > 0 else rng.integers(4)
        y_start[s] = y
        acc = 0
        for t in range(T):
            spk_in = (rng.random(N) < x_rate[o] / 1000) & (ref_in == 0)
            x = dec * (x + spk_in); ref_in[spk_in] = ref_time
            u = W @ x + V[a] @ y + b - eta0 * np.exp(-tss / tau_ref) - adapt
            u -= u.max()
            v = vhat * np.exp(u) / np.exp(u).sum()
            acc = acc + v
            spk = (rng.random(K) < v / 1000) & (ref == 0)
            y = dec * (y + spk)
            if spk.any():
                i = np.where(spk)[0]
                if learn_W:
                    W[i] = np.maximum(W[i] + beta * (np.exp(-W[i]) * x - 0.1), 0)
                if learn_V:
                    V[a][i] = np.clip(V[a][i] + beta * (np.exp(-V[a][i]) * y - 0.1), 0, V_max)
                    V[a][i, i] = 0
            ref[spk] = ref_time; tss[spk] = -1
            ref[ref > 0] -= 1; ref_in[ref_in > 0] -= 1; tss += 1
        vmean[s] = acc / T
        now = vmean[s] / vmean[s].max()
        colour_count[:, o] += now

        # probability given to the NEXT colour, using the current weights
        if s < n - 1:
            Q = np.exp(V[acts[s]]); Q = Q / Q.sum(0, keepdims=True)       # each neuron's prediction of the next neurons
            q_next = Q @ now / now.sum()
            colour_given_neuron = colour_count / colour_count.sum(1, keepdims=True)
            p_next[s] = (q_next @ colour_given_neuron)[obs[s + 1]]

        # surprise correction (same as USE_SURPRISE in the loop script)
        adapt *= np.exp(-1 / tau_adapt)
        if surprise and learn_V and s >= 2:
            a_out, a_in = acts[s - 1], acts[s - 2]
            Q = np.exp(V[a_out]); Q = Q / Q.sum(0, keepdims=True)
            before = vmean[s - 1] / vmean[s - 1].max()
            predicted_well = now @ Q
            S = 1 - np.sum(before * predicted_well) / np.sum(before)
            error = (1 - predicted_well) - np.sum(before * (1 - predicted_well)) / np.sum(before)
            V[a_in] = np.clip(V[a_in] - gamma * S * np.outer(before * error, y_start[s - 1]), 0, V_max)
            np.fill_diagonal(V[a_in], 0)
            z = (S - mu) / np.sqrt(var)
            mu += 0.01 * (S - mu); var += 0.01 * ((S - mu) ** 2 - var)
            if z > z_thr:
                adapt += g_adapt * before * np.maximum(error, 0)
    return W, V, p_next


def fresh_weights(rng, w_std=W_std):
    W = np.maximum(rng.normal(0, w_std, (K, N)), 0)
    V = np.maximum(rng.normal(0, V_std, (4, K, K)), 0)
    for a in range(4):
        np.fill_diagonal(V[a], 0)
    return W, V


# ---------------- run all three rooms ----------------
results = {}
for i_room, (name, room1) in enumerate(ROOMS.items()):
    print(name)
    rng = np.random.default_rng(i_room)
    # 1. learn room 1
    W1, V1 = fresh_weights(rng)
    o, a, _ = random_walk(room1, n_room1_steps, rng)
    W1, V1, _ = run(room1, o, a, W1, V1, rng)
    # 2. room 2: same layout, new colours
    room2 = np.array(NEW_COLOURS[name])[room1]
    o2, a2, c2 = random_walk(room2, n_room2_steps, rng)
    explored = np.array([len(set(c2[:s + 1])) / room2.size for s in range(n_room2_steps)])
    # 3a. from scratch
    W, V = fresh_weights(rng)
    _, _, p_scratch = run(room2, o2, a2, W, V, rng)
    # 3b. map reused: V[a] frozen, W reset small and relearned
    W, _ = fresh_weights(rng, W_reuse_std)
    _, _, p_reuse = run(room2, o2, a2, W, V1.copy(), rng, learn_V=False, surprise=False)
    results[name] = dict(room1=room1, room2=room2, explored=explored, scratch=p_scratch, reuse=p_reuse)
    for k_ in ['scratch', 'reuse']:
        p = results[name][k_]
        print(f'  {k_:8s}: first 200 steps {np.nanmean(p[:200]):.2f}, last 1000 steps {np.nanmean(p[-1000:]):.2f}')

# ---------------- figure ----------------
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 4, width_ratios=[0.6, 0.6, 2, 2], wspace=0.35, hspace=0.45)
win = 50
handles = labels = None
for r, (name, d) in enumerate(results.items()):
    P = max(d['room1'].max(), d['room2'].max()) + 1
    for j, (rm, ttl) in enumerate([(d['room1'], 'Room 1\n(map learned here)'), (d['room2'], 'Room 2\nsame layout, new colours')]):
        ax = fig.add_subplot(gs[r, j])
        ax.imshow(rm, cmap=ListedColormap(COLOURS[:P]), vmin=-0.5, vmax=P - 0.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=10)
        if j == 0:
            ax.set_ylabel(name, fontsize=12, fontweight='bold')
    chance = 1 / len(np.unique(d['room2']))
    for j, (key, ttl) in enumerate([('scratch', 'New network learns room 2 from scratch\n(all weights learned, surprise correction on)'),
                                    ('reuse', 'Room-1 map reused in room 2\n(transitions V[a] frozen, only input weights W relearned)')]):
        ax = fig.add_subplot(gs[r, 2 + j])
        p = d[key]
        sm = np.convolve(np.nan_to_num(p, nan=np.nanmean(p)), np.ones(win) / win, mode='valid')
        ax.plot(np.arange(len(sm)) + win, sm, color='k', lw=1.5, label='prob. of next colour (50-step average)')
        ax.plot(d['explored'], color='red', lw=2, label='fraction of room explored')
        ax.axhline(chance, color='0.5', ls=':', label='chance level (1 / number of colours)')
        ax.set_ylim(0, 1.05); ax.set_xlim(0, len(p))
        ax.set_xlabel('number of actions'); ax.set_ylabel('probability'); ax.set_title(ttl, fontsize=11)
        ax.text(0.98, 0.04, f'mean over last 1000 steps: {np.nanmean(p[-1000:]):.2f}\nfirst 200 steps: {np.nanmean(p[:200]):.2f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9, bbox=dict(fc='w', ec='0.7'))
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        ins = ax.inset_axes([0.42, 0.42, 0.3, 0.36])            # zoom on the first 300 actions
        ins.plot(np.arange(len(sm)) + win, sm, color='k', lw=1)
        ins.plot(d['explored'], color='red', lw=1.5)
        ins.axhline(chance, color='0.5', ls=':')
        ins.set_xlim(0, 300); ins.set_ylim(0, 1.05); ins.tick_params(labelsize=7)
        ins.text(0.97, 0.06, 'zoom: first 300 actions', transform=ins.transAxes, ha='right', fontsize=7)
fig.suptitle('Reusing a learned room map in a recoloured room (3\u00d73, action-augmented Hebbian network)\n'
             'black = probability the network assigns to the next colour; red = fraction of the 9 cells visited',
             fontsize=14, fontweight='bold')
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.6, 0.945), ncol=3, fontsize=10, frameon=False)
plt.savefig('/Users/sheetaljadhav/Desktop/CompNeuroUniverse/CSCG_project/codebase/proof_of_concept/outputs/room_map_reuse.png', dpi=105, bbox_inches='tight')
plt.show()
print('Saved: room_map_reuse.png')