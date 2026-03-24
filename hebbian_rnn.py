import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
vhat = 500 ## scales firing rate
tau = 20 ## 20ms exponential decay time constant
beta = 0.1  ## learning rate 0.05
prob_nonrw = 0.4 ## probability of not getting reward
eta0 = 10 ## refractory reduction scaling
tau_ref = 5 ## refractory reduction time constant
ref_time = 10 ## refractory time
baseline=10

N = 100 ## afferent neurons
K = 100 ## recurrent neurons
W_std = 3.5
V_std = 2.5

input_fr = 100  ## input neuron firing rates
num_trials = 2000
each_obs_time = 20 ## each observation is shown for 20ms

trial1x = np.array([1,1,1,1,1,1,2,2,2,2,1,1,1,4,4,1,1,1,5,5,1,1,6,0,0,0])
trial2x = np.array([1,1,1,1,1,1,3,3,3,3,1,1,1,4,4,1,1,1,5,5,1,1,6,0,0,0])
np.random.seed(7)

b = baseline*np.ones(K)

vmean_all_rep = [] ## firing rate of neurons
trials_all_rep = []
for sim_rep in range(0,1):
    
    ## Initialize weights
    W = np.maximum(np.random.normal(0,W_std,(K,N)),0) ## feedforward 
    V = np.maximum(np.random.normal(0,V_std,(K,K)),0)  ## recurrent 
    y = np.zeros(K) ## recurrent input
    
    ## Generate inputs for each sensory observation
    P = len(np.unique(np.concatenate((trial1x, trial2x)))) + 1 ## add one for reward
    each_input = int(np.floor(N/P)) ## num input neurons that will fire for each input observation
    x_rate= np.zeros((P,N))
    for p_each in range(P):
        x_rate[p_each, p_each*each_input: (1+p_each)*each_input] = input_fr
    
    ## Generate trials
    x_all = np.zeros((num_trials, len(trial1x), N))
    trials = np.random.choice(2,num_trials)
    for trial_number in range(0, num_trials):
        trial = trials[trial_number]
        if trial == 0:
            x_all[trial_number,:,:] = x_rate[trial1x]
        else:
            x_all[trial_number,:,:] = x_rate[trial2x]



    x = np.zeros(N)
    refractory = np.zeros(K)
    refractory_input = np.zeros(N)
    y_all = np.zeros(K)
    vmean = []
    time_since_spike = 7*np.ones(K)

    for rep in range(0, 1000):

        if rep%100==0:
            print(rep)
        for curr_P in range(0,len(trial1x)):
            v_all = []
            
            for t in range(0,each_obs_time):
                
                ## Find current input
                if (curr_P == 14) and trials[rep] == 0 and np.random.uniform()>prob_nonrw:
                    spiking_input = np.where(np.random.uniform(0,1,N)<(x_rate[P-1]/1000))

                elif  (curr_P == 19) and trials[rep] == 1 and np.random.uniform()>prob_nonrw:
                    spiking_input = np.where(np.random.uniform(0,1,N)<(x_rate[P-1]/1000)) 

                else:
                    spiking_input = np.where(np.random.uniform(0,1,N)<(x_all[rep, curr_P,:]/1000))

                ## Input neuron spike/no spike
                for neur in range(0,N):
                    if neur in spiking_input[0] and refractory_input[neur] == 0: ## ie. neuron spikes
                        x[neur] = np.exp(-1/tau)* (1+x[neur])
                        refractory_input[neur]= ref_time
                    else: ## neuron didn't spike
                        x[neur] = np.exp(-1/tau)* (x[neur])

                ## Calculate firing rate of recurrent neuron 
                eta = eta0*np.exp(-time_since_spike/tau_ref)
                ubar = (W@x) + (V@y_all) + b - eta
                inh = np.log(np.sum(np.exp(ubar)))
                v = vhat*np.exp(ubar)/(np.sum(np.exp(ubar)))
                v_all.append(v)
                spiking_neurons = np.where(np.random.uniform(0,1, K)<v/1000)

                for neur in range(0,K):
                    
                    if neur in spiking_neurons[0] and refractory[neur] == 0: ## spiking neurons

                        refractory[neur]= ref_time
                        time_since_spike[neur] = -1

                        y_all[neur] = np.exp(-1/tau)*(1+y_all[neur])
                        
                        ## Weight update
                        W[neur,:] = np.maximum(W[neur,:] + beta*(np.exp(-W[neur,:])*x-0.1),0)
                        V[neur,:] = np.maximum(V[neur,:] + beta*(np.exp(-V[neur,:])*y_all-0.1),0)
                        
                        V[neur, neur] = 0  ## no autapse


                    else:
                        y_all[neur] = np.exp(-1/tau)*(y_all[neur]) ## non-spiking neurons


                

                refractory[refractory>0] = refractory[refractory>0]-1
                refractory_input[refractory_input>0] = refractory_input[refractory_input>0]-1
                time_since_spike = time_since_spike + 1
            vmean.append(np.mean(np.array(v_all), axis = 0))




    vmean_all_rep.append(vmean)
    trials_all_rep.append(trials)

# ========== LATERAL WEIGHT MATRIX + STATS ==========

# --- Neuron ordering: sort by peak activation time across all timesteps ---
# Use vmean which is shape (num_timesteps_total, K)
vmean_arr = np.array(vmean)   # shape: (1000 * len(trial1x), K)

# Reshape to (num_reps, len(trial1x), K) then average over reps
vmean_3d   = vmean_arr.reshape(1000, len(trial1x), K)
vmean_mean = vmean_3d.mean(axis=0)          # shape: (len(trial1x), K)
peak_times = np.argmax(vmean_mean, axis=0)  # shape: (K,) — peak timestep per neuron
neuron_order = np.argsort(peak_times)       # sort neurons by when they peak

# --- Sort V matrix ---
V_sorted = V[np.ix_(neuron_order, neuron_order)]

# --- Stats ---
W_mean = np.mean(W)
V_mean = np.mean(V)
lag3   = np.mean(np.diag(V_sorted, k=3))
lag5   = np.mean(np.diag(V_sorted, k=5))

print("=" * 50)
print("  WEIGHT STATS (post-training)")
print("=" * 50)
print(f"  W_mean  = {W_mean:.4f}")
print(f"  V_mean  = {V_mean:.4f}")
print(f"  lag-3   = {lag3:.4f}   (V_sorted diagonal offset +3)")
print(f"  lag-5   = {lag5:.4f}   (V_sorted diagonal offset +5)")
print("=" * 50)

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Lateral Weight Matrix V — Post Training",
             fontsize=13, fontweight='bold')

# Raw V unsorted
im0 = axes[0].imshow(V, aspect='auto', cmap='viridis', interpolation='nearest')
axes[0].set_title("V  (unsorted)", fontsize=10)
axes[0].set_xlabel("Presynaptic neuron")
axes[0].set_ylabel("Postsynaptic neuron")
plt.colorbar(im0, ax=axes[0], fraction=0.046, label='weight')

# V sorted by peak activation time
im1 = axes[1].imshow(V_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
axes[1].set_title("V  (sorted by peak activation time)", fontsize=10)
axes[1].set_xlabel("Presynaptic (sorted)")
axes[1].set_ylabel("Postsynaptic (sorted)")
plt.colorbar(im1, ax=axes[1], fraction=0.046, label='weight')

# Annotate lag diagonals on sorted V
diag_ax = axes[1]
for lag, col in [(3, 'red'), (5, 'orange')]:
    xs = np.arange(K - lag)
    ys = xs + lag
    diag_ax.plot(xs, ys, color=col, linewidth=1.2, linestyle='--',
                 alpha=0.7, label=f'lag-{lag}={np.mean(np.diag(V_sorted,k=lag)):.3f}')
diag_ax.legend(fontsize=8, loc='upper right')

# Mean activation time heatmap per neuron
im2 = axes[2].imshow(
    vmean_mean[:, neuron_order].T,   # shape: (K, len(trial1x))
    aspect='auto', cmap='hot', interpolation='nearest'
)
axes[2].set_title("Mean firing rate over time\n(neurons sorted, averaged over reps)",
                  fontsize=9)
axes[2].set_xlabel("Timestep (observation index)")
axes[2].set_ylabel("Neuron (sorted)")

# Mark trial1x observation boundaries
obs_boundaries = np.arange(0, len(trial1x), 1)
for b_line in obs_boundaries:
    axes[2].axvline(b_line, color='white', linewidth=0.3, alpha=0.4)
plt.colorbar(im2, ax=axes[2], fraction=0.046, label='firing rate (Hz)')

plt.tight_layout()
plt.savefig("lateral_weights_sorted.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: lateral_weights_sorted.png")