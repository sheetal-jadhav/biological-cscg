'''
Hebbian RNN (Sun et al., 2025) with ACTIONS, learning a 3x3 room from a random walk.

New changes as follows:

  1. Input: instead of two fixed trials, the agent does a random walk in a room
     and sees the colour (observation) of the cell it is standing in.
  2. Actions: like CSCG's T[a], there is one recurrent matrix per action:
     V[0] = up, V[1] = down, V[2] = left, V[3] = right.
     While the agent is in a cell, only V[a] is used and learned, where 'a' is the
     action that brought the agent into that cell.
  3. Initial recurrent weights are small (V_std = 0.3 instead of 2.5), because some
     (cell, action) pairs never happen (you cannot enter a top-row cell by moving
     down), and those rows of V[a] would otherwise keep large random weights forever.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Parameters 

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
V_std = 0.3         ## CHANGED: small initial recurrent weights (see note 3 above)

input_fr = 100      ## input neuron firing rate (Hz)
each_obs_time = 20  ## each observation is shown for 20 ms

num_train_steps = 6000   ## NEW: length of the training random walk
num_test_steps = 1500    ## NEW: length of the test walk (weights frozen)


# Pick ONE of the rooms below.Each number is the colour (observation) seen in that cell.

#room = np.array([[0, 1, 2],
#                 [3, 4, 5],       ## non-aliased: every cell has its own colour
#                 [6, 7, 8]])
                             


#room = np.array([[0, 1, 2],
#                 [3, 4, 5],
#                 [6, 7, 0]])        ## aliased: cell 0 and cell 8 look the same


room = np.array([[0, 1, 0],
                  [3, 4, 5],
                  [0, 7, 0]])        ## aliased: all four corners look the same

n_rows, n_cols = room.shape
n_cells = n_rows * n_cols

## NEW: the 4 actions, as (row change, column change)
actions = np.array([[-1, 0],   ## 0 = up
                    [ 1, 0],   ## 1 = down
                    [ 0,-1],   ## 2 = left
                    [ 0, 1]])  ## 3 = right
action_names = ['up', 'down', 'left', 'right']
n_actions = 4

np.random.seed(7)
b = baseline * np.ones(K)


# Random walk in the room
# Returns, for every step: the colour seen, the action taken next, the true cell.
# Bumping into a wall = staying in the same cell (CSCG convention).

def random_walk(num_steps):
    row, col = np.random.randint(n_rows), np.random.randint(n_cols)
    obs_seq, act_seq, cell_seq = [], [], []
    for step in range(num_steps):
        obs_seq.append(room[row, col])
        cell_seq.append(row * n_cols + col)
        a = np.random.randint(n_actions)          ## pick a random action
        act_seq.append(a)
        new_row, new_col = row + actions[a][0], col + actions[a][1]
        if 0 <= new_row < n_rows and 0 <= new_col < n_cols:
            row, col = new_row, new_col           ## move (otherwise stay)
    return np.array(obs_seq), np.array(act_seq), np.array(cell_seq)



# Input firing rates for each colour 

P = room.max() + 1                      ## number of different colours
each_input = int(np.floor(N / P))       ## input neurons per colour
x_rate = np.zeros((P, N))
for p_each in range(P):
    x_rate[p_each, p_each * each_input:(1 + p_each) * each_input] = input_fr



# The network. The loop body is the original code; the only difference is V[a] instead of V.

def simulate(obs_seq, act_seq, W, V, learn):
    x = np.zeros(N)
    y_all = np.zeros(K)
    refractory = np.zeros(K)
    refractory_input = np.zeros(N)
    time_since_spike = 7 * np.ones(K)
    vmean = []

    for step in range(len(obs_seq)):
        if step % 1000 == 0:
            print('  step', step)

        obs = obs_seq[step]
        ## NEW: the action that brought the agent here picks the recurrent matrix
        if step == 0:
            a = np.random.randint(n_actions)
        else:
            a = act_seq[step - 1]

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
            ubar = (W @ x) + (V[a] @ y_all) + b - eta          ## CHANGED: V -> V[a]
            v = vhat * np.exp(ubar) / (np.sum(np.exp(ubar)))
            v_all.append(v)
            spiking_neurons = np.where(np.random.uniform(0, 1, K) < v / 1000)

            for neur in range(0, K):
                if neur in spiking_neurons[0] and refractory[neur] == 0:   ## spiking neuron
                    refractory[neur] = ref_time
                    time_since_spike[neur] = -1
                    y_all[neur] = np.exp(-1 / tau) * (1 + y_all[neur])

                    ## Weight update (as original), only for V[a]
                    if learn:
                        W[neur, :] = np.maximum(W[neur, :] + beta * (np.exp(-W[neur, :]) * x - 0.1), 0)
                        V[a, neur, :] = np.maximum(V[a, neur, :] + beta * (np.exp(-V[a, neur, :]) * y_all - 0.1), 0)
                        V[a, neur, neur] = 0   ## no autapse
                else:
                    y_all[neur] = np.exp(-1 / tau) * (y_all[neur])         ## non-spiking neuron

            refractory[refractory > 0] = refractory[refractory > 0] - 1
            refractory_input[refractory_input > 0] = refractory_input[refractory_input > 0] - 1
            time_since_spike = time_since_spike + 1

        vmean.append(np.mean(np.array(v_all), axis=0))
    return W, V, np.array(vmean)



# Train, then test with frozen weights

W = np.maximum(np.random.normal(0, W_std, (K, N)), 0)                 ## feedforward
V = np.maximum(np.random.normal(0, V_std, (n_actions, K, K)), 0)      ## NEW: one V per action
for a in range(n_actions):
    np.fill_diagonal(V[a], 0)

print('Training...')
obs_train, act_train, cell_train = random_walk(num_train_steps)
W, V, _ = simulate(obs_train, act_train, W, V, learn=True)

print('Testing (weights frozen)...')
obs_test, act_test, cell_test = random_walk(num_test_steps)
_, _, vmean = simulate(obs_test, act_test, W.copy(), V.copy(), learn=False)



# Analysis 1: place fields. Mean firing rate of each neuron in each TRUE cell.
# (We label neurons by true cell, not by colour, so aliased cells stay separate.)

fields = np.zeros((n_cells, K))
for c in range(n_cells):
    fields[c] = vmean[cell_test == c].mean(axis=0)
pref_cell = np.argmax(fields, axis=0)                 ## each neuron's favourite cell
neuron_order = np.argsort(pref_cell)

print('=' * 55)
print('  neurons per cell :', np.bincount(pref_cell, minlength=n_cells))


# Analysis 2: what did each V[a] learn?
# The learning rule makes V[a][post, pre] large when 'pre' was active just
# before 'post' fired, i.e. "I arrived at post's cell from pre's cell by action a".
# For every real move (prev cell --a--> new cell), check which cell's neurons
# send the strongest V[a] input onto the new cell's neurons. It should be prev cell.

def next_cell(c, a):
    r, col = divmod(c, n_cols)
    nr, nc = r + actions[a][0], col + actions[a][1]
    if 0 <= nr < n_rows and 0 <= nc < n_cols:
        return nr * n_cols + nc
    return c

correct, total = 0, 0
for a in range(n_actions):
    for prev in range(n_cells):
        new = next_cell(prev, a)
        if new == prev or not np.any(pref_cell == new):
            continue                                   ## skip wall bumps / empty cells
        strength = np.zeros(n_cells)
        for c in range(n_cells):
            if c != new and np.any(pref_cell == c):
                strength[c] = V[a][np.ix_(pref_cell == new, pref_cell == c)].mean()
        correct += (np.argmax(strength) == prev)
        total += 1
print(f'  V[a] points back to the correct previous cell: {correct}/{total} moves')


# Analysis 3 (aliased rooms only): did the neurons for a shared colour split?
# Splitting index per neuron: 0 = fires equally in all cells with that colour,
#                             1 = fires in only one of them.

for colour in range(P):
    cells_with_colour = np.where(room.flatten() == colour)[0]
    if len(cells_with_colour) < 2:
        continue
    m = len(cells_with_colour)
    neurons = np.where(np.isin(pref_cell, cells_with_colour))[0]
    rates = fields[np.ix_(cells_with_colour, neurons)]
    share = rates.max(axis=0) / rates.sum(axis=0)
    split = (share - 1 / m) / (1 - 1 / m)
    print(f'  colour {colour} shared by cells {list(cells_with_colour)}: '
          f'{len(neurons)} neurons, median splitting index = {np.median(split):.2f}')
print('=' * 55)



# Plots

colours = ['#6B21A8', '#F472B6', '#EAB308', '#93C5FD', '#9CA3AF',
           '#0D9488', '#16A34A', '#D97706', '#991B1B']

fig, axes = plt.subplots(2, 4, figsize=(20, 9))

## room layout
ax = axes[0, 0]
ax.imshow(room, cmap=ListedColormap(colours[:P]), vmin=-0.5, vmax=P - 0.5)
for r in range(n_rows):
    for c in range(n_cols):
        ax.text(c, r, f'cell {r*n_cols+c}\nobs {room[r, c]}', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
ax.set_title('Room layout')
ax.set_xticks([]); ax.set_yticks([])

## place fields
ax = axes[0, 1]
im = ax.imshow(fields[:, neuron_order].T, aspect='auto', cmap='hot', interpolation='nearest')
ax.set_title('Firing rate per cell (neurons sorted)')
ax.set_xlabel('true cell'); ax.set_ylabel('neuron (sorted)')
plt.colorbar(im, ax=ax, fraction=0.046, label='Hz')

## V summed over actions, sorted
ax = axes[0, 2]
V_sum_sorted = V.sum(axis=0)[np.ix_(neuron_order, neuron_order)]
im = ax.imshow(V_sum_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
ax.set_title('Sum of V[a] over actions (sorted by cell)')
ax.set_xlabel('presynaptic (sorted)'); ax.set_ylabel('postsynaptic (sorted)')
plt.colorbar(im, ax=ax, fraction=0.046)
axes[0, 3].axis('off')

## each V[a], sorted by preferred cell; white lines separate cells
edges = np.cumsum(np.bincount(pref_cell, minlength=n_cells))[:-1]
for a in range(n_actions):
    ax = axes[1, a]
    im = ax.imshow(V[a][np.ix_(neuron_order, neuron_order)], aspect='auto',
                   cmap='viridis', interpolation='nearest')
    for e in edges:
        ax.axhline(e - 0.5, color='w', lw=0.4); ax.axvline(e - 0.5, color='w', lw=0.4)
    ax.set_title(f'V[{action_names[a]}]  (sorted by cell)')
    ax.set_xlabel('presynaptic: cell you came FROM'); ax.set_ylabel('postsynaptic: cell you ARRIVED in')

plt.tight_layout()
plt.savefig('/Users/sheetaljadhav/Desktop/CompNeuroUniverse/CSCG_project/codebase/proof_of_concept/outputs/hebbian_rnn_actions_room_#4_aliased.png', dpi=130, bbox_inches='tight')
plt.show()
print('Saved: hebbian_rnn_actions_room_#4_aliased.png')