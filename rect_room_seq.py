import numpy as np
import matplotlib.pyplot as plt
import igraph
import os
from matplotlib import cm, colors


from codebase.chmm_actions import CHMM, forwardE, datagen_structured_obs_room


custom_colors = (np.array([
    [214, 214, 214],
            [85, 35, 157],
            [253, 252, 144],
            [114, 245, 144],
            [151, 38, 20],
            [239, 142, 192],
            [214, 134, 48],
            [140, 194, 250],
            [72, 160, 162],
]
)/256
)

def plot_graph(chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30
):
    
    '''
    build and draw a directed graph showing tranisition 
    structure of the CHMM over decoded states 
    '''

    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    g = igraph.Graph.Adjacency((A > 0).tolist())
    node_labels = np.arange(x.max() + 1).repeat(n_clones)[v]
    if multiple_episodes:
        node_labels -= 1
    colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
    out = igraph.plot(
        g,
        output_file,
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=vertex_size,
        margin=50,
    )
    return out



def get_mess_fwd(chmm, x, pseudocount=0.0, pseudocount_E=0.0):

    '''
builds smoothed normalized per clone emission probabilties (E)
and normalized, avergaed transition probabilities (T). runs forward message passing to get 
per-time forward messages for the observed sequence x 
this is usefuls for decoding, visualization or downstream inference
STEPS
1. inputs chmm.n_clones,x, nonnegative scalars for additive smoothing
2. build emmission matrix E
3. prepare transition tensor T
4. run forward inference (forwardE)
5. return values - 

'''

    n_clones = chmm.n_clones
    E = np.zeros((n_clones.sum(), len(n_clones)))
    last = 0
    for c in range(len(n_clones)):
        E[last : last + n_clones[c], c] = 1
        last += n_clones[c]
    E += pseudocount_E
    norm = E.sum(1, keepdims=True)
    norm[norm == 0] = 1
    E /= norm
    T = chmm.C + pseudocount
    norm = T.sum(2, keepdims=True)
    norm[norm == 0] = 1
    T /= norm
    T = T.mean(0, keepdims=True)
    log2_lik, mess_fwd = forwardE(
        T.transpose(0, 2, 1), E, chmm.Pi_x, chmm.n_clones, x, x * 0, store_messages=True
    )
    return mess_fwd



def place_field(mess_fwd, rc, clone):

    '''
    computes sptial "place field" for a single hidden clone by averaging that clone's
    forward message weight over all visits to each (row,col) location
    '''
    assert mess_fwd.shape[0] == rc.shape[0] and clone < mess_fwd.shape[1]
    field = np.zeros(rc.max(0) + 1)
    count = np.zeros(rc.max(0) + 1, int)
    for t in range(mess_fwd.shape[0]):
        r, c = rc[t]
        field[r, c] += mess_fwd[t, clone]
        count[r, c] += 1
    count[count == 0] = 1
    return field / count

# -----------------------------------------------------------
# TRAINING CLONED HIDDEN MARKOV MODEL (chmm)
# -----------------------------------------------------------

# Train CHMM on random data
TIMESTEPS = 1000
OBS = 2
x = np.random.randint(OBS, size=(1000,))  # Observations. Replace with your data.
a = np.zeros(
    1000, dtype=np.int64
)  # If there are actions in your domain replace this. If not, keep the vector of zeros.
n_clones = (
    np.ones(OBS, dtype=np.int64) * 5
)  # Number of clones specifies the capacity for each observation.

x_test = np.random.randint(
    OBS, size=(1000,)
)  # Test observations. Replace with your data.
a_test = np.zeros(1000, dtype=np.int64)

chmm = CHMM(n_clones=n_clones, pseudocount=1e-10, x=x, a=a)  # Initialize the model
progression = chmm.learn_em_T(x, a, n_iter=100, term_early=False)  # Training

nll_per_prediction = chmm.bps(
    x_test, a_test
)  # Evaluate negative log-likelihood (base 2 log)
avg_nll = np.mean(nll_per_prediction)
avg_prediction_probability = 2 ** (-avg_nll)
print(avg_prediction_probability)


# -----------------------------------------------------------
# RECTANGULAR ROOM SEQUENCE
# -----------------------------------------------------------

room = np.array(
    [
        [1, 2, 3, 0, 3, 1, 1, 1],
        [1, 1, 3, 2, 3, 2, 3, 1],
        [1, 1, 2, 0, 1, 2, 1, 0],
        [0, 2, 1, 1, 3, 0, 0, 2],
        [3, 3, 1, 0, 1, 0, 3, 0],
        [2, 1, 2, 3, 3, 3, 2, 0],
    ]
)
n_emissions = room.max() + 1

a, x, rc = datagen_structured_obs_room(room, length=50000)

n_clones = np.ones(n_emissions, dtype=np.int64) * 70
chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
progression = chmm.learn_em_T(x, a, n_iter=1000)  # Training

# refine learning
chmm.pseudocount = 0.0
chmm.learn_viterbi_T(x, a, n_iter=100)

cmap = colors.ListedColormap(custom_colors[-4:])
plt.matshow(room, cmap=cmap)
plt.show()

graph = plot_graph(
    chmm, x, a, output_file="figures/rectangular_room_graph.pdf", cmap=cmap
)
graph

mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)

clone = 114
plt.matshow(place_field(mess_fwd, rc, clone))
plt.show()