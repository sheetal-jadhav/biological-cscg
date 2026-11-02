'''
Hebbian RNN (Sun et al., 2025) with ACTIONS and a SURPRISE third factor,
learning a 3x3 room (aliased or non-aliased) from a random walk.


  A. Input: random walk in a room; the agent sees the colour of its cell.
  B. Actions: one recurrent matrix per action, V[a] (like CSCG's T[a]).
     While the agent is in a cell, only V[a] for the action that brought
     it there is used and learned.
  C. Surprise third factor (switch on/off with USE_SURPRISE), three parts:

     1. SURPRISE SIGNAL. After every step, each neuron that was active in the
        previous step is asked: "did your outgoing weights predict who is active
        now?" Its outgoing V[a] weights are turned into a probability over the
        next neurons (normalised over the receiving neurons). The population
        surprise S (0 = perfectly predicted, 1 = completely wrong) is the
        activity-weighted average failure.
        Why normalised: Hebbian V stores "where did I come from"; without
        normalising, a neuron that stands for two places sends strong drive
        to all successors of both and is never surprised.

     2. ELIGIBILITY TRACE + THIRD FACTOR. The weights that put a neuron into
        its state in the previous step (its V[a] row, times the activity that
        was present at the start of that step) are the eligible synapses.
        They are changed by  - gamma * S * (relative prediction error).
        Neurons that predicted worse than the others lose those weights,
        better ones gain them. S gates how big the change is.

     3. WINNER SUPPRESSION. When S jumps above its running average
        (z-score > z_thr), the previous winners that predicted badly get an
        adaptation current that decays over tau_adapt steps. While they are
        adapted, uncommitted neurons can win and be recruited as new clones.

  D. A weight ceiling V_max for the recurrent weights.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


# Parameters (from orginial code)

vhat = 500          ## scales firing rate
tau = 20            ## 20 ms exponential decay time constant
beta = 0.1          ## learning rate
eta0 = 10           ## refractory reduction scaling
tau_ref = 5         ## refractory reduction time constant
ref_time = 10       ## refractory time
baseline = 10

N = 90              ## afferent neurons (10 per colour for 9 colours)
K = 50              ## recurrent neurons
W_std = 3.5
V_std = 0.3         ## small initial recurrent weights (untrained V[a] rows stay quiet)

input_fr = 100      ## input neuron firing rate (Hz)
each_obs_time = 20  ## each observation is shown for 20 ms

num_train_steps = 8000   ## NEW: training random walk length
num_test_steps = 1500    ## NEW: test walk length (weights frozen)


#surprise third factor

USE_SURPRISE = True      ## False = plain Hebbian network with actions
gamma = 3.0              ## strength of the surprise-gated correction (point 2)
V_max = 10.0             ## ceiling for recurrent weights
g_adapt = 5.0            ## strength of winner suppression (point 3)
tau_adapt = 20.0         ## suppression decays over this many steps
z_thr = 1.0              ## suppression is triggered when surprise z-score > z_thr


# The room. Each number is the colour seen in that cell. Pick ONE.

#room = np.array([[0, 1, 2],
#                  [3, 4, 5],
#                  [6, 7, 8]])        ## non-aliased

#room = np.array([[0, 1, 2],
#                 [3, 4, 5],
#                 [6, 7, 0]])          ## aliased: cells 0 and 8 look the same


room = np.array([[0, 1, 0],
                  [3, 4, 5],
                 [0, 7, 0]])        ## aliased: all four corners look the same

n_rows, n_cols = room.shape
n_cells = n_rows * n_cols

actions = np.array([[-1, 0],   ## 0 = up
                    [ 1, 0],   ## 1 = down
                    [ 0,-1],   ## 2 = left
                    [ 0, 1]])  ## 3 = right
action_names = ['up', 'down', 'left', 'right']
n_actions = 4

np.random.seed(7)
b = baseline * np.ones(K)



# Random walk. Bumping into a wall = staying in the same cell.

def random_walk(num_steps):
    row, col = np.random.randint(n_rows), np.random.randint(n_cols)
    obs_seq, act_seq, cell_seq = [], [], []
    for step in range(num_steps):
        obs_seq.append(room[row, col])
        cell_seq.append(row * n_cols + col)
        a = np.random.randint(n_actions)
        act_seq.append(a)
        new_row, new_col = row + actions[a][0], col + actions[a][1]
        if 0 <= new_row < n_rows and 0 <= new_col < n_cols:
            row, col = new_row, new_col
    return np.array(obs_seq), np.array(act_seq), np.array(cell_seq)


def next_cell(c, a):
    r, col = divmod(c, n_cols)
    nr, nc = r + actions[a][0], col + actions[a][1]
    if 0 <= nr < n_rows and 0 <= nc < n_cols:
        return nr * n_cols + nc
    return c



# Input firing rates for each colour (as the original x_rate)

P = room.max() + 1
each_input = int(np.floor(N / P))
x_rate = np.zeros((P, N))
for p_each in range(P):
    x_rate[p_each, p_each * each_input:(1 + p_each) * each_input] = input_fr



# The network

def simulate(obs_seq, act_seq, W, V, learn):
    x = np.zeros(N)
    y_all = np.zeros(K)
    refractory = np.zeros(K)
    refractory_input = np.zeros(N)
    time_since_spike = 7 * np.ones(K)
    adaptation = np.zeros(K)         ## NEW: winner-suppression current (point 3)
    surprise_mean, surprise_var = 0.5, 0.01   ## NEW: running statistics of surprise
    vmean = []
    y_start = []                     ## NEW: recurrent activity at the start of each step
    surprise = np.zeros(len(obs_seq))         ## NEW: surprise S of every step
    n_suppressions = 0

    for step in range(len(obs_seq)):
        if step % 1000 == 0:
            print('  step', step)

        obs = obs_seq[step]
        if step == 0:
            a = np.random.randint(n_actions)
        else:
            a = act_seq[step - 1]    ## the action that brought the agent here picks V[a]

        y_start.append(y_all.copy())
        v_all = []
        for t in range(0, each_obs_time):

            ## Input neuron spike/no spike (as original)
            spiking_input = np.where(np.random.uniform(0, 1, N) < (x_rate[obs] / 1000))
            for neur in range(0, N):
                if neur in spiking_input[0] and refractory_input[neur] == 0:
                    x[neur] = np.exp(-1 / tau) * (1 + x[neur])
                    refractory_input[neur] = ref_time
                else:
                    x[neur] = np.exp(-1 / tau) * (x[neur])

            ## Firing rate of recurrent neurons (soft winner-take-all)
            eta = eta0 * np.exp(-time_since_spike / tau_ref)
            ubar = (W @ x) + (V[a] @ y_all) + b - eta - adaptation   ## NEW: V[a], adaptation
            ubar = ubar - np.max(ubar)                                ## numerical safety only
            v = vhat * np.exp(ubar) / (np.sum(np.exp(ubar)))
            v_all.append(v)
            spiking_neurons = np.where(np.random.uniform(0, 1, K) < v / 1000)

            for neur in range(0, K):
                if neur in spiking_neurons[0] and refractory[neur] == 0:
                    refractory[neur] = ref_time
                    time_since_spike[neur] = -1
                    y_all[neur] = np.exp(-1 / tau) * (1 + y_all[neur])
                    if learn:   ## weight update as original, only for V[a], with ceiling V_max
                        W[neur, :] = np.maximum(W[neur, :] + beta * (np.exp(-W[neur, :]) * x - 0.1), 0)
                        V[a, neur, :] = np.clip(V[a, neur, :] + beta * (np.exp(-V[a, neur, :]) * y_all - 0.1), 0, V_max)
                        V[a, neur, neur] = 0
                else:
                    y_all[neur] = np.exp(-1 / tau) * (y_all[neur])

            refractory[refractory > 0] = refractory[refractory > 0] - 1
            refractory_input[refractory_input > 0] = refractory_input[refractory_input > 0] - 1
            time_since_spike = time_since_spike + 1

        vmean.append(np.mean(np.array(v_all), axis=0))
        adaptation = adaptation * np.exp(-1 / tau_adapt)     ## suppression fades step by step

        ## ---------------- NEW: surprise third factor ----------------
        if USE_SURPRISE and step >= 2:
            a_out = act_seq[step - 1]    ## action from the previous cell to this one
            a_in = act_seq[step - 2]     ## action that brought the agent INTO the previous cell
            now = vmean[step] / np.max(vmean[step])              ## who is active now (0..1)
            before = vmean[step - 1] / np.max(vmean[step - 1])   ## who was active one step ago

            ## point 1: each previous neuron's forward prediction, and the population surprise
            predicted_well = np.zeros(K)
            for k in range(K):
                outgoing = np.exp(V[a_out][:, k])
                outgoing = outgoing / np.sum(outgoing)           ## normalised over receiving neurons
                predicted_well[k] = np.sum(outgoing * now)
            S = 1 - np.sum(before * predicted_well) / np.sum(before)
            surprise[step] = S

            ## each neuron's error relative to the other neurons that were active with it
            error = (1 - predicted_well) - np.sum(before * (1 - predicted_well)) / np.sum(before)

            if learn:
                ## point 2: eligible synapses = V[a_in] row x activity at the start of the previous step,
                ##          changed by the third factor S times the relative error
                for k in range(K):
                    V[a_in, k, :] = np.clip(V[a_in, k, :] - gamma * S * before[k] * error[k] * y_start[step - 1], 0, V_max)
                    V[a_in, k, k] = 0

                ## point 3: if S jumps above its running average, suppress the bad previous winners
                z = (S - surprise_mean) / np.sqrt(surprise_var)
                surprise_mean = surprise_mean + 0.01 * (S - surprise_mean)
                surprise_var = surprise_var + 0.01 * ((S - surprise_mean) ** 2 - surprise_var)
                if z > z_thr:
                    for k in range(K):
                        if error[k] > 0:
                            adaptation[k] = adaptation[k] + g_adapt * before[k] * error[k]
                    n_suppressions += 1

    if learn and USE_SURPRISE:
        print(f'  suppression triggered on {n_suppressions / len(obs_seq):.0%} of steps')
    return W, V, np.array(vmean), surprise



# Train, then test with frozen weights

W = np.maximum(np.random.normal(0, W_std, (K, N)), 0)
V = np.maximum(np.random.normal(0, V_std, (n_actions, K, K)), 0)
for a in range(n_actions):
    np.fill_diagonal(V[a], 0)

print('Training...')
obs_train, act_train, cell_train = random_walk(num_train_steps)
W, V, _, surprise_train = simulate(obs_train, act_train, W, V, learn=True)

print('Testing (weights frozen)...')
obs_test, act_test, cell_test = random_walk(num_test_steps)
_, _, vmean, surprise_test = simulate(obs_test, act_test, W.copy(), V.copy(), learn=False)


# Analysis

fields = np.zeros((n_cells, K))
for c in range(n_cells):
    fields[c] = vmean[cell_test == c].mean(axis=0)
pref_cell = np.argmax(fields, axis=0)
neuron_order = np.lexsort((-fields.max(axis=0), pref_cell))

print('=' * 60)
print('  surprise third factor   :', 'ON' if USE_SURPRISE else 'OFF')
print('  neurons per cell        :', np.bincount(pref_cell, minlength=n_cells))
tracking = np.mean(pref_cell[np.argmax(vmean, axis=1)] == cell_test)
print(f'  tracking accuracy       : {tracking:.2f}  (most active neuron belongs to the true cell)')

correct, total = 0, 0
for a in range(n_actions):
    for prev in range(n_cells):
        new = next_cell(prev, a)
        if new == prev or not np.any(pref_cell == new):
            continue
        strength = np.zeros(n_cells)
        for c in range(n_cells):
            if c != new and np.any(pref_cell == c):
                strength[c] = V[a][np.ix_(pref_cell == new, pref_cell == c)].mean()
        correct += (np.argmax(strength) == prev)
        total += 1
print(f'  V[a] points back correctly: {correct}/{total} moves')

split_values = []
for colour in range(P):
    cells_with_colour = np.where(room.flatten() == colour)[0]
    if len(cells_with_colour) < 2:
        continue
    m = len(cells_with_colour)
    neurons = np.where(np.isin(pref_cell, cells_with_colour))[0]
    rates = fields[np.ix_(cells_with_colour, neurons)]
    share = rates.max(axis=0) / rates.sum(axis=0)
    split = (share - 1 / m) / (1 - 1 / m)
    split_values = split
    print(f'  colour {colour} in cells {list(cells_with_colour)}: {len(neurons)} neurons, '
          f'median splitting index = {np.median(split):.2f}')
print('=' * 60)



# Plots

colours = ['#6B21A8', '#F472B6', '#EAB308', '#93C5FD', '#9CA3AF',
           '#0D9488', '#16A34A', '#D97706', '#991B1B']
fig, axes = plt.subplots(3, 4, figsize=(21, 14))
fig.suptitle(f"Action-augmented Hebbian RNN — surprise third factor {'ON' if USE_SURPRISE else 'OFF'}   "
             f"(tracking {tracking:.2f}, moves {correct}/{total}"
             + (f", splitting {np.median(split_values):.2f})" if len(split_values) else ")"),
             fontsize=14, fontweight='bold')

## room layout
ax = axes[0, 0]
ax.imshow(room, cmap=ListedColormap(colours[:P]), vmin=-0.5, vmax=P - 0.5)
for r in range(n_rows):
    for c in range(n_cols):
        ax.text(c, r, f'cell {r*n_cols+c}\nobs {room[r, c]}', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
ax.set_title('Room layout'); ax.set_xticks([]); ax.set_yticks([])

## place fields
ax = axes[0, 1]
im = ax.imshow(fields[:, neuron_order].T, aspect='auto', cmap='hot', interpolation='nearest')
ax.set_title('Firing rate per cell (neurons sorted)')
ax.set_xlabel('true cell'); ax.set_ylabel('neuron (sorted)')
plt.colorbar(im, ax=ax, fraction=0.046, label='Hz')

## splitting index of the shared colour
ax = axes[0, 2]
if len(split_values):
    ax.hist(split_values, bins=np.linspace(0, 1, 11), color=colours[0], edgecolor='k')
    ax.set_title('Shared-colour neurons: splitting index')
    ax.set_xlabel('0 = fires in all its cells, 1 = only one'); ax.set_ylabel('# neurons')
else:
    ax.axis('off')

## surprise during training (moving average)
ax = axes[0, 3]
if USE_SURPRISE:
    win = 200
    smooth = np.convolve(surprise_train, np.ones(win) / win, mode='valid')
    ax.plot(smooth, color='k')
    ax.set_title('Surprise S during training (200-step average)')
    ax.set_xlabel('training step'); ax.set_ylabel('S')
else:
    ax.axis('off')

## each V[a], sorted; red = true moves, grey = rows that can never be trained
counts = np.bincount(pref_cell, minlength=n_cells)
start = np.concatenate([[0], np.cumsum(counts)])
mid = (start[:-1] + start[1:]) / 2 - 0.5
for a in range(n_actions):
    ax = axes[1, a]
    ax.imshow(V[a][np.ix_(neuron_order, neuron_order)], aspect='auto', cmap='viridis', interpolation='nearest')
    for e in start[1:-1]:
        ax.axhline(e - 0.5, color='w', lw=0.4); ax.axvline(e - 0.5, color='w', lw=0.4)
    reachable = set()
    for prev in range(n_cells):
        new = next_cell(prev, a)
        reachable.add(new)
        if new != prev and counts[new] > 0 and counts[prev] > 0:
            ax.add_patch(Rectangle((start[prev] - 0.5, start[new] - 0.5), counts[prev], counts[new],
                                   fill=False, ec='red', lw=1.5, ls='--'))
    for c in range(n_cells):
        if c not in reachable and counts[c] > 0:
            ax.add_patch(Rectangle((-0.5, start[c] - 0.5), K, counts[c], color='grey', alpha=0.5))
    ax.set_xticks(mid); ax.set_xticklabels([f'c{c}' for c in range(n_cells)], fontsize=8)
    ax.set_yticks(mid); ax.set_yticklabels([f'c{c}' for c in range(n_cells)], fontsize=8)
    ax.set_title(f'V[{action_names[a]}]')
    ax.set_xlabel('came FROM'); ax.set_ylabel('ARRIVED in')

## aggregated learned map for each action (block means, cell x cell) vs truth
for a in range(n_actions):
    ax = axes[2, a]
    G = np.full((n_cells, n_cells), np.nan)
    for i in range(n_cells):
        for j in range(n_cells):
            if counts[i] > 0 and counts[j] > 0 and i != j:
                G[i, j] = V[a][np.ix_(pref_cell == i, pref_cell == j)].mean()
    ax.imshow(G, cmap='viridis')
    for prev in range(n_cells):
        new = next_cell(prev, a)
        if new != prev:
            ax.add_patch(Rectangle((prev - 0.5, new - 0.5), 1, 1, fill=False, ec='red', lw=1.5))
    ax.set_xticks(range(n_cells)); ax.set_yticks(range(n_cells))
    ax.set_title(f'V[{action_names[a]}] per cell (red = true move)')
    ax.set_xlabel('came FROM'); ax.set_ylabel('ARRIVED in')

plt.tight_layout()
fname = f"actions_room_surprise_#4_alaised_{'ON' if USE_SURPRISE else 'OFF'}.png"
plt.savefig('/Users/sheetaljadhav/Desktop/CompNeuroUniverse/CSCG_project/codebase/proof_of_concept/outputs/actions_room_surprise_#4_alaised', dpi=120, bbox_inches='tight')
plt.show()
print('Saved:', fname)