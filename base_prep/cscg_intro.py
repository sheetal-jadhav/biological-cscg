
import numpy as np
import matplotlib.pyplot as plt
import igraph 
from chmm_actions import CHMM, forwardE, datagen_structured_obs_room
from matplotlib import cm, colors

import os

custom_colors = (
    np.array(
        [
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
    )
    / 256
)

if not os.path.exists("cscg_figures"):
    os.makedirs("cscg_figures")

## Utility functions

def plot_graph(
    chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30
):
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
    assert mess_fwd.shape[0] == rc.shape[0] and clone < mess_fwd.shape[1]
    field = np.zeros(rc.max(0) + 1)
    count = np.zeros(rc.max(0) + 1, int)
    for t in range(mess_fwd.shape[0]):
        r, c = rc[t]
        field[r, c] += mess_fwd[t, clone]
        count[r, c] += 1
    count[count == 0] = 1
    return field / count

# rectangular alised room with 4 unique obs

room = np.array(
    [
        [1, 2, 3, 0, 3,],
        [1, 1, 3, 2, 3,],
        [1, 1, 2, 0, 1,],
        [0, 2, 1, 1, 3,],
        [3, 3, 1, 0, 1,],
        [2, 1, 2, 3, 3,],
    ]
)


# plot room layout
cmap = colors.ListedColormap(custom_colors[-4:])
plt.matshow(room,cmap=cmap)
plt.title('Room Layout')
plt.savefig('cscg_figures/room_layout')

# generate data from the room and train the cscg

n_emissions = room.max()+1
a,x,rc = datagen_structured_obs_room(room,length=5000)
n_clones = np.ones(n_emissions,dtype=np.int64)*70
chmm = CHMM(n_clones=n_clones,pseudocount=2e-3,x=x,a=a,seed=4)
progression = chmm.learn_em_T(x,a,n_iter=100)
chmm.pseudocount = 0.0
chmm.learn_viterbi_T(x,a,n_iter=100)

#visualize the graphs

graph = plot_graph(chmm,x,a,output_file="cscg_figures/rect_room_graph.png",cmap=cmap)

#mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.0001) #calulate forward messages, used for place field
