# Translation of snn_sample_ctrfh.m to Python
# Continuous-time spiking HMM SEMS realistic ref. Homeostatic

import numpy as np
from typing import Tuple, Dict, Any

def wta_softmax(U):
    """Helper function for softmax computation with log-sum-exp trick"""
    U = np.asarray(U).flatten()
    max_U = np.max(U)
    exp_U = np.exp(U - max_U)
    sum_exp = np.sum(exp_U)
    P = exp_U / sum_exp
    A = max_U + np.log(sum_exp)
    return P, A

def wta_draw_k(P_t):
    """Draw a sample from probability distribution P_t"""
    k = np.random.choice(len(P_t), p=P_t)
    Z_i = np.zeros(len(P_t))
    Z_i[k] = 1
    return Z_i, k

def mk_stochastic(arr):
    """Make array stochastic (normalize to sum to 1)"""
    return arr / np.sum(arr)

def snn_sample_ctrfh(net: Dict[str, Any], data: Dict[str, Any], ct: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Draws a spike from SEM network - continuous time spiking HMM SEMS realistic ref. Homeostatic.
    
    Parameters:
    -----------
    net : dict
        A WTA-network dictionary with parameters and state
    data : dict
        A data structure to be simulated
    ct : array
        Current time index
    
    Returns:
    --------
    net : dict
        The (modified) network structure
    Z : ndarray
        Network output spikes [2, num_spikes]
    P : ndarray
        Output probabilities [num_neurons+1, num_spikes]
    
    Network parameters:
    -------------------
    tau_x_r : float (0.002) - time constant of EPSP window rise
    tau_z_r : float (0.002) - time constant of EPSP window rise
    tau_x_f : float (0.02) - time constant of EPSP window fall
    tau_z_f : float (0.02) - time constant of EPSP window fall
    tau_rf : float (0.005) - time constant of refractory window
    w_rf : float (-10) - strength of refractory window
    lambda_ : float (2000) - network spike rate
    mean_rec_delay : float (0.010) - mean delay on recurrent synapses
    std_rec_delay : float (0.000) - standard deviation of recurrent delay
    temperature : float (1) - sampling temperature
    iw_mode : str ('exact') - method to calculate importance weights
    use_variance_tracking : bool (False)
    groups : array (0) - assignment of neurons to groups
    """
    
    t = data['time'][ct[0]]
    time_range = data['time'][ct[-1]] - data['time'][ct[0]]
    
    # Generate spike times using Poisson process
    num_spikes = np.random.poisson(net['lambda'] * time_range)
    spike_times = np.sort(time_range * np.random.rand(num_spikes))
    
    # Initialize recurrent spikes
    rec_spikes = np.hstack([net['rec_spikes'], 
                            np.full((2, num_spikes), np.inf, dtype=np.float32)])
    num_rs = net['rec_spikes'].shape[1]
    
    # Initialize output arrays
    Z = np.zeros((2, num_spikes), dtype=np.float32)
    P = np.zeros((net['num_neurons'] + 1, num_spikes), dtype=np.float32)
    net['At'] = np.zeros((2, num_spikes), dtype=np.float32)
    
    # Initialize spike tracking
    last_input_spikes = np.full(net['num_inputs'], t)
    last_output_spikes = np.full(net['num_neurons'], t)
    
    hX = net['hX'].copy()
    hZ = net['hZ'].copy()
    
    # Exponential weights
    W_exp = np.exp(net['W'].astype(np.float64))
    V_exp = np.exp(net['V'].astype(np.float64))
    V0_exp = np.exp(net['V0'].astype(np.float64))
    
    # Initialize weight updates
    net['d_W'] = np.zeros((net['num_neurons'], net['num_inputs']))
    net['d_V'] = np.zeros((net['num_neurons'], net['num_neurons']))
    net['d_V0'] = np.zeros(net['num_neurons'])
    
    i = 0
    l = 0
    
    # Tracking arrays
    hX_all = np.zeros((net['num_inputs'], num_spikes))
    hZ_all = np.zeros((net['num_neurons'], num_spikes))
    A_v = np.zeros(num_spikes)
    A_w = np.zeros(num_spikes)
    
    # Setup groups
    if 'groups' not in net:
        net['groups'] = np.array([0])
    
    if net['groups'].size == 1 and net['groups'][0] == 0:
        group_idx = np.array([0, net['num_neurons']])
    else:
        group_idx = np.concatenate([[0], np.cumsum(net['groups']), [net['num_neurons']]])
    
    # Variance tracking setup
    if net.get('use_variance_tracking', False):
        SW_new = net['SW'].copy()
        QW_new = net['QW'].copy()
        SV_new = net['SV'].copy()
        QV_new = net['QV'].copy()
        S0_new = net['S0'].copy()
        Q0_new = net['Q0'].copy()
        
        net['eta_W'] = net['eta'] * (QW_new - SW_new**2) / (np.exp(-SW_new) + 1)
        net['eta_V'] = net['eta'] * (QV_new - SV_new**2) / (np.exp(-SV_new) + 1)
        net['eta_0'] = net['eta'] * (Q0_new - S0_new**2) / (np.exp(-S0_new) + 1)
    
    use_exact_iw = (net.get('iw_mode', 'exact') == 'exact')
    
    d_hZ = np.diff(hZ, axis=1).flatten()
    
    # Reset spike counter
    net['num_o'] = np.zeros(net['num_neurons'])
    
    try:
        for j in range(num_spikes):
            t = spike_times[j]
            
            # Process input spikes
            while (i < data['Xt'].shape[1]) and (t > data['Xt'][1, i]):
                n_id = int(data['Xt'][0, i])
                sp_t = data['Xt'][1, i]
                
                hX[n_id, 0] = hX[n_id, 0] * np.exp(-(sp_t - last_input_spikes[n_id]) / net['tau_x_r']) + 1
                hX[n_id, 1] = hX[n_id, 1] * np.exp(-(sp_t - last_input_spikes[n_id]) / net['tau_x_f']) + 1
                
                last_input_spikes[n_id] = sp_t
                i += 1
            
            # Process recurrent spikes
            while (l < rec_spikes.shape[1]) and (t > rec_spikes[1, l]):
                n_id = int(rec_spikes[0, l])
                sp_t = rec_spikes[1, l]
                
                hZ[n_id, 0] = hZ[n_id, 0] * np.exp(-(sp_t - last_output_spikes[n_id]) / net['tau_z_r']) + 1
                hZ[n_id, 1] = hZ[n_id, 1] * np.exp(-(sp_t - last_output_spikes[n_id]) / net['tau_z_f']) + 1
                
                last_output_spikes[n_id] = sp_t
                l += 1
            
            # Decay PSPs
            hZ[:, 0] = hZ[:, 0] * np.exp(-(t - last_output_spikes) / net['tau_z_r'])
            hZ[:, 1] = hZ[:, 1] * np.exp(-(t - last_output_spikes) / net['tau_z_f'])
            hX[:, 0] = hX[:, 0] * np.exp(-(t - last_input_spikes) / net['tau_x_r'])
            hX[:, 1] = hX[:, 1] * np.exp(-(t - last_input_spikes) / net['tau_x_f'])
            
            d_hX = np.diff(hX, axis=1).flatten()
            d_hZ = np.diff(hZ, axis=1).flatten()
            
            hX_all[:, j] = d_hX
            hZ_all[:, j] = d_hZ
            
            last_input_spikes[:] = t
            last_output_spikes[:] = t
            
            # Refractory potential
            u_rf = net['w_rf'] * np.exp(-(t - net['last_spike_t']) / net['tau_rf'])
            
            # Compute potentials
            U_v = net['V'] @ d_hZ + u_rf + net['V0']
            U_w = net['W'] @ d_hX
            U = U_w + U_v
            
            P_t_v, A_v[j] = wta_softmax(U_v)
            P_t_w, A_w[j] = wta_softmax(U_w)
            
            # Use groups for softmax
            P_t = np.zeros(net['num_neurons'])
            for g in range(len(group_idx) - 1):
                start_idx = group_idx[g]
                end_idx = group_idx[g + 1]
                P_t[start_idx:end_idx], A = wta_softmax(U[start_idx:end_idx])
            
            P_t = P_t / np.sum(P_t)
            
            # Importance weight calculation
            if use_exact_iw:
                A = A - A_v[j] - A_w[j]
            else:
                A = A - A_v[j]
            
            # Draw winner neuron
            Z_i, k = wta_draw_k(P_t)
            
            Z[:, j] = [k, t]
            P[:, j] = np.append(P_t, t)
            
            rec_spikes[:, j + num_rs] = [k, t + net['rec_delay'][k]]
            
            net['num_o'][k] += 1
            net['last_spike_t'][k] = t
            
            # UPDATE FOR W (feedforward weights)
            net['d_W'][k, :] += (net['eta_W'][k, :] * (d_hX - W_exp[k, :]) / 
                                 np.maximum(net['eta_W'][k, :], W_exp[k, :]))
            
            # UPDATE FOR V (recurrent weights) - homeostatic rule
            net['d_V'][k, :] += (2 * net['eta_V'][k, :] * (d_hZ - V_exp[k, :]) / 
                                 np.maximum(net['eta_V'][k, :], V_exp[k, :]))
            net['d_V'][:, k] -= net['eta_V'][:, k] * d_hZ
            
            # UPDATE FOR V0 (bias) - homeostatic update
            net['d_V0'] += 100 * net['eta_0'] * (1 / net['num_neurons'])
            net['d_V0'][k] -= 100 * net['eta_0'][k]
            
            net['At'][:, j] = [A, t]
            
            # Variance tracking updates
            if net.get('use_variance_tracking', False):
                SW_new[k, :] += net['eta_W'][k, :] * (net['W'][k, :] + net['d_W'][k, :] - SW_new[k, :])
                QW_new[k, :] += net['eta_W'][k, :] * ((net['W'][k, :] + net['d_W'][k, :])**2 - QW_new[k, :])
                
                SV_new[k, :] += net['eta_V'][k, :] * ((net['V'][k, :] + net['d_V'][k, :]) - SV_new[k, :])
                QV_new[k, :] += net['eta_V'][k, :] * ((net['V'][k, :] + net['d_V'][k, :])**2 - QV_new[k, :])
                
                S0_new += net['eta_0'] * ((net['V0'] + net['d_V0']) - S0_new)
                Q0_new += net['eta_0'] * ((net['V0'] + net['d_V0'])**2 - Q0_new)
    
    except Exception as e:
        print(f'There has been an error while sampling!\nExcluding run from training\n')
        print(f'Error: {str(e)}')
        net['R'] = -100000
        return net, Z, P
    
    # Update variance tracking
    if net.get('use_variance_tracking', False):
        net['SW_new'] = SW_new
        net['QW_new'] = QW_new
        net['SV_new'] = SV_new
        net['QV_new'] = QV_new
        net['S0_new'] = S0_new
        net['Q0_new'] = Q0_new
    
    # Clean up recurrent spikes
    net['rec_spikes'] = rec_spikes[:, l:]
    net['rec_spikes'][1, :] -= t
    net['last_spike_t'] -= t
    
    # Compute importance weight
    net['R'] = np.sum(net['At'][0, :])
    
    # Store additional information
    net['A_v'] = A_v
    net['A_w'] = A_w
    net['hX'] = hX
    net['hZ'] = hZ
    net['hX_all'] = hX_all
    net['hZ_all'] = hZ_all
    
    return net, Z, P

print("Translation complete! The function snn_sample_ctrfh has been translated to Python.")