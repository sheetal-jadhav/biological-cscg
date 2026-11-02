'''
Splitter cells in a figure-of-eight T-maze (like Fig. 4a-c of George et al. 2021),
with the action-augmented Hebbian network and the surprise correction.

The maze (letters = observations):
    red route : A B C D E F G J K L
    blue route: L M N D E F G H I A        (laps alternate: red, blue, red, ...)
The stem D E F G is shared: both routes see exactly the same letters there.
There is ONE action ("forward") everywhere, so the turn at G (to J or H) can only be
predicted by remembering which route the agent is on. A splitter cell is a stem
neuron that fires differently on red and blue laps.

Three versions are trained on the same laps, for several seeds:
    plain    : Hebbian rule only (the original Sun et al. network; 100 neurons)
    surprise : Hebbian rule + surprise correction (as in hebbian_rnn_actions_room.py)
    frozen V : recurrent weights never learn (control: anything here is just leftover input)
Then, with frozen weights:
    - route decoding at each stem letter (can the route be read out from the population?)
    - splitter cells: stem neurons with route selectivity > 0.5
Figures:
    tmaze_example.png : maze, learned graph and clone activity (version EXAMPLE_VERSION)
    tmaze_summary.png : route decoding along the stem for every seed and version

Vectorised (fast) version of the same equations as the room scripts.
Runtime: about 35 s per seed per version (about 2 minutes for one seed).
Results are cached in tmaze_cache.pkl so a rerun only computes what is missing.
Needs numpy and matplotlib.
'''
import os, pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------------- parameters ----------------
vhat, tau, beta = 500, 20, 0.1
eta0, tau_ref, ref_time, baseline = 10, 5, 10, 10
K, W_std, V_std, input_fr, T = 100, 3.5, 0.3, 100, 20     ## K = 100 recurrent neurons, as in the original hebbian_rnn.py
gamma, V_max, g_adapt, tau_adapt, z_thr = 3.0, 10.0, 5.0, 20.0, 1.0
SEEDS = [4]     ## which seeds to run; for ONE seed write e.g.  SEEDS = [2]
EXAMPLE_SEED = None          ## seed shown in the example figure; None = the best of SEEDS
EXAMPLE_VERSION = 'plain'    ## version shown in the example figure: 'plain', 'surprise' or 'frozen V'
N_TRAIN_LOOPS = 2000         ## one loop = red lap + blue lap = 18 steps -> 36,000 training steps
N_TEST_LOOPS = 30
SELECTIVITY_THRESHOLD = 0.5  ## |r_red - r_blue| / (r_red + r_blue) above this = splitter cell

LETTERS = list('ABCDEFGHIJKLMN'); P = len(LETTERS); N = 10 * P      # 10 input neurons per letter
RED, BLUE, STEM = list('ABCDEFGJKL'), list('MNDEFGHI'), list('DEFG')
ROUTE_COLOURS = ['#DC2626', '#2563EB']


def make_laps(n_loops):
    obs, route = [], []
    for _ in range(n_loops):
        for l in RED:  obs.append(LETTERS.index(l)); route.append(0)
        for l in BLUE: obs.append(LETTERS.index(l)); route.append(1)
    return np.array(obs), np.array(route)


def run(obs, W, V, rng, learn_W=True, learn_V=True, surprise=True):
    '''Same network as the room scripts, with a single action (V[0]).'''
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
    for s, o in enumerate(obs):
        y_start[s] = y
        acc = 0
        for t in range(T):
            spk_in = (rng.random(N) < x_rate[o] / 1000) & (ref_in == 0)
            x = dec * (x + spk_in); ref_in[spk_in] = ref_time
            u = W @ x + V[0] @ y + b - eta0 * np.exp(-tss / tau_ref) - adapt
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
                    V[0][i] = np.clip(V[0][i] + beta * (np.exp(-V[0][i]) * y - 0.1), 0, V_max)
                    V[0][i, i] = 0
            ref[spk] = ref_time; tss[spk] = -1
            ref[ref > 0] -= 1; ref_in[ref_in > 0] -= 1; tss += 1
        vmean[s] = acc / T
        now = vmean[s] / vmean[s].max()
        adapt *= np.exp(-1 / tau_adapt)
        if surprise and learn_V and s >= 2:            # surprise correction
            Q = np.exp(V[0]); Q = Q / Q.sum(0, keepdims=True)
            before = vmean[s - 1] / vmean[s - 1].max()
            predicted_well = now @ Q
            S = 1 - np.sum(before * predicted_well) / np.sum(before)
            error = (1 - predicted_well) - np.sum(before * (1 - predicted_well)) / np.sum(before)
            V[0] = np.clip(V[0] - gamma * S * np.outer(before * error, y_start[s - 1]), 0, V_max)
            np.fill_diagonal(V[0], 0)
            z = (S - mu) / np.sqrt(var)
            mu += 0.01 * (S - mu); var += 0.01 * ((S - mu) ** 2 - var)
            if z > z_thr:
                adapt += g_adapt * before * np.maximum(error, 0)
    return W, V, vmean


def evaluate(W, V):
    obs, route = make_laps(N_TEST_LOOPS)
    _, _, vm = run(obs, W.copy(), V.copy(), np.random.default_rng(100), learn_W=False, learn_V=False, surprise=False)
    vm, obs, route = vm[18:], obs[18:], route[18:]                    # drop the first loop (no history yet)
    decoding = {}
    for l in STEM:                                                    # nearest-centroid route decoding, 20 random halves
        m = obs == LETTERS.index(l); X, yv = vm[m], route[m]; accs = []
        for rep in range(20):
            r_ = np.random.default_rng(rep).permutation(len(X)); tr, te = r_[:len(X) // 2], r_[len(X) // 2:]
            c0, c1 = X[tr][yv[tr] == 0].mean(0), X[tr][yv[tr] == 1].mean(0)
            pred = (np.linalg.norm(X[te] - c1, axis=1) < np.linalg.norm(X[te] - c0, axis=1)).astype(int)
            accs.append(np.mean(pred == yv[te]))
        decoding[l] = float(np.mean(accs))
    rates = np.zeros((K, P, 2))                                      # mean rate per neuron, letter, route
    for l in range(P):
        for r in range(2):
            m = (obs == l) & (route == r)
            if m.any(): rates[:, l, r] = vm[m].mean(0)
    peak_letter = rates.max(2).argmax(1)
    stem_ids = [LETTERS.index(l) for l in STEM]
    stem_cells = [k for k in range(K) if peak_letter[k] in stem_ids and rates[k].max() > 20]
    sel = {k: abs(rates[k, peak_letter[k], 0] - rates[k, peak_letter[k], 1]) / (rates[k, peak_letter[k]].sum() + 1e-9)
           for k in stem_cells}
    n_split = int(sum(v > SELECTIVITY_THRESHOLD for v in sel.values()))
    return dict(decoding=decoding, n_stem=len(stem_cells), n_splitter=n_split, rates=rates,
                vm=vm, obs=obs, route=route, selectivity=sel)


def train(version, seed):
    rng = np.random.default_rng(seed)
    W = np.maximum(rng.normal(0, W_std, (K, N)), 0)
    V = np.maximum(rng.normal(0, V_std, (1, K, K)), 0); np.fill_diagonal(V[0], 0)
    obs, _ = make_laps(N_TRAIN_LOOPS)
    if version == 'surprise': W, V, _ = run(obs, W, V, rng)
    elif version == 'plain':  W, V, _ = run(obs, W, V, rng, surprise=False)
    else:                     W, V, _ = run(obs, W, V, rng, learn_V=False, surprise=False)
    return W, V


# ---------------- train and evaluate (with cache) ----------------
## make sure the example seed is also run
if EXAMPLE_SEED is not None and EXAMPLE_SEED not in SEEDS:
    SEEDS.append(EXAMPLE_SEED)

CACHE = 'tmaze_cache.pkl'
cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}
VERSIONS = ['plain', 'surprise', 'frozen V']
for version in VERSIONS:
    for seed in SEEDS:
        if (version, seed) in cache:
            continue
        W, V = train(version, seed)
        res = evaluate(W, V)
        res['W'], res['V'] = W, V
        cache[(version, seed)] = res
        pickle.dump(cache, open(CACHE, 'wb'))
        d = res['decoding']
        print(f'{version:9s} seed {seed}: route decoding D {d["D"]:.2f} E {d["E"]:.2f} F {d["F"]:.2f} G {d["G"]:.2f} | '
              f'splitter cells {res["n_splitter"]}/{res["n_stem"]} stem cells', flush=True)

# ---------------- summary figure ----------------
fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
for version, col in zip(VERSIONS, ['#059669', '#7C3AED', '#D97706']):
    D = np.array([[cache[(version, s)]['decoding'][l] for l in STEM] for s in SEEDS])
    for row in D:
        ax[0].plot(range(4), row, color=col, alpha=0.25, lw=1)
    ax[0].plot(range(4), D.mean(0), color=col, lw=3, marker='o', label=f'{version} (mean of {len(SEEDS)} seeds)')
ax[0].axhline(0.5, color='k', ls=':', label='chance')
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([f'{l}\n(stem {i + 1})' for i, l in enumerate(STEM)])
ax[0].set_ylim(0.3, 1.05); ax[0].set_ylabel('route decoding accuracy'); ax[0].legend(fontsize=9)
ax[0].set_title('Can the route (red or blue) be read out along the shared stem?\nthin lines = single seeds')
width = 0.27
for i, version in enumerate(VERSIONS):
    ns = [cache[(version, s)]['n_splitter'] for s in SEEDS]
    ax[1].bar(np.arange(len(SEEDS)) + (i - 1) * width, ns, width, label=version, color=['#059669', '#7C3AED', '#D97706'][i])
ax[1].set_xticks(range(len(SEEDS))); ax[1].set_xticklabels(SEEDS); ax[1].set_xlabel('seed'); ax[1].set_ylabel('number of splitter cells')
ax[1].set_title(f'Splitter cells per seed (stem neurons with route selectivity > {SELECTIVITY_THRESHOLD})')
ax[1].legend()
fig.suptitle('Figure-of-eight T-maze: route information along the shared stem D\u2192E\u2192F\u2192G', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('tmaze_summary.png', dpi=110, bbox_inches='tight')

# ---------------- example figure: best surprise seed ----------------
best = EXAMPLE_SEED if EXAMPLE_SEED is not None else max(SEEDS, key=lambda s: cache[(EXAMPLE_VERSION, s)]['decoding']['G'])
res = cache[(EXAMPLE_VERSION, best)]
rates, vm, obs, route = res['rates'], res['vm'], res['obs'], res['route']
POS = {'A': (0, 4), 'I': (1, 4), 'H': (2, 4), 'G': (3, 3.3), 'J': (4, 4), 'K': (5, 4), 'L': (6, 4),
       'B': (0.8, 2.6), 'C': (1.6, 1.4), 'F': (3, 2.4), 'E': (3, 1.4), 'D': (3, 0.4), 'M': (5.2, 2.6), 'N': (4.4, 1.4)}

fig = plt.figure(figsize=(22, 7.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.4, 0.7, 0.7], wspace=0.3)
## (1) maze
ax = fig.add_subplot(gs[0])
for route_letters, col, off in [(RED, ROUTE_COLOURS[0], -0.06), (['L'] + BLUE + ['A'], ROUTE_COLOURS[1], 0.06)]:
    for l1, l2 in zip(route_letters[:-1], route_letters[1:]):
        (x1, y1), (x2, y2) = POS[l1], POS[l2]
        ax.annotate('', xy=(x2 + off, y2 + off), xytext=(x1 + off, y1 + off),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
for l, (x_, y_) in POS.items():
    ax.text(x_, y_, l, ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='circle', fc='#FDE68A' if l in STEM else 'white', ec='k'))
ax.set_xlim(-0.6, 6.6); ax.set_ylim(-0.3, 4.6); ax.axis('off')
ax.set_title('Maze: red route A\u2192L, blue route L\u2192A\nshared stem D E F G (yellow)', fontsize=12)

## (2) learned graph: used neurons placed at their letter, coloured by the route they prefer
ax = fig.add_subplot(gs[1])
winner = vm.argmax(1)
used = [k for k in range(K) if np.mean(winner == k) >= 0.01]
neuron_letter = {k: rates[k].max(1).argmax() for k in used}
by_letter = {}
for k in used:
    by_letter.setdefault(neuron_letter[k], []).append(k)
npos, ncol = {}, {}
for l_id, ks in by_letter.items():
    x0, y0 = POS[LETTERS[l_id]]
    for j, k in enumerate(sorted(ks, key=lambda k: rates[k, l_id, 0] - rates[k, l_id, 1])):
        npos[k] = (x0 + (j - (len(ks) - 1) / 2) * 0.32, y0)
        r_red, r_blue = rates[k, l_id]
        s_ = (r_red - r_blue) / (r_red + r_blue + 1e-9)
        ncol[k] = ROUTE_COLOURS[0] if s_ > SELECTIVITY_THRESHOLD else ROUTE_COLOURS[1] if s_ < -SELECTIVITY_THRESHOLD else '#9CA3AF'
counts = np.zeros((K, K)); route_of = np.zeros((K, K, 2))
for s in range(len(winner) - 1):
    counts[winner[s], winner[s + 1]] += 1; route_of[winner[s], winner[s + 1], route[s + 1]] += 1
for k in used:
    total = sum(counts[k, j] for j in used if j != k)
    for j in used:
        if j != k and counts[k, j] >= 3 and counts[k, j] / max(total, 1) >= 0.1:
            col = ROUTE_COLOURS[int(route_of[k, j].argmax())] if route_of[k, j].max() / route_of[k, j].sum() > 0.8 else '#9CA3AF'
            ax.annotate('', xy=npos[j], xytext=npos[k],
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.3, shrinkA=9, shrinkB=9, connectionstyle='arc3,rad=0.15'))
for k in used:
    ax.scatter(*npos[k], s=260, c=ncol[k], edgecolors='k', zorder=3)
for l, (x_, y_) in POS.items():
    ax.text(x_, y_ + 0.32, l, ha='center', fontsize=11, fontweight='bold')
ax.set_xlim(-0.6, 6.6); ax.set_ylim(-0.3, 4.7); ax.axis('off')
ax.set_title(f'Learned graph ({EXAMPLE_VERSION}, seed {best}): one dot per active neuron, at its letter\n'
             'red/blue = fires mainly on that route; grey = both routes', fontsize=12)

## (3) clone activity for one red lap and one blue lap (neurons sorted by when they peak)
order_key = []
for k in used:
    l_id = neuron_letter[k]; r_ = int(rates[k, l_id].argmax())
    seq = RED if r_ == 0 else BLUE
    t_in_loop = (seq.index(LETTERS[l_id]) if LETTERS[l_id] in seq else 0) + (0 if r_ == 0 else len(RED))
    order_key.append((t_in_loop, k))
order = [k for _, k in sorted(order_key)]
start = 18 * 5                                        # one full loop from the middle of the test walk
lap = vm[start:start + 18][:, order]
lap = lap / lap.max(1, keepdims=True)
for j, (sl, ttl, col, labels) in enumerate([(slice(0, 10), 'red lap', ROUTE_COLOURS[0], RED),
                                            (slice(10, 18), 'blue lap', ROUTE_COLOURS[1], BLUE)]):
    ax = fig.add_subplot(gs[2 + j])
    ax.imshow(lap[sl].T, aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    for side in ax.spines.values():
        side.set_edgecolor(col); side.set_linewidth(4)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    for tick, l in zip(ax.get_xticklabels(), labels):
        if l in STEM: tick.set_fontweight('bold'); tick.set_color('#B45309')
    ax.set_yticks([]); ax.set_xlabel('time (letter seen)'); ax.set_ylabel('neurons (sorted by peak)' if j == 0 else '')
    ax.set_title(f'Clone activity, {ttl}', fontsize=12)
d = res['decoding']
which = f'seed {best}' if EXAMPLE_SEED is not None else f'best of {len(SEEDS)} seeds (seed {best})'
fig.suptitle(f'Figure-of-eight T-maze, {which}, {EXAMPLE_VERSION} network: route decoding on the stem '
             f'D {d["D"]:.2f}, E {d["E"]:.2f}, F {d["F"]:.2f}, G {d["G"]:.2f};  '
             f'{res["n_splitter"]} of {res["n_stem"]} stem neurons are splitter cells', fontsize=13, fontweight='bold')
example_file = f'tmaze_test{best}.png' if EXAMPLE_SEED is not None else 'tmaze_example.png'
plt.savefig('/Users/sheetaljadhav/Desktop/CompNeuroUniverse/CSCG_project/codebase/proof_of_concept/outputs/tmaze_test', dpi=105, bbox_inches='tight')
plt.show()
print('Saved: tmaze_summary.png,', example_file)