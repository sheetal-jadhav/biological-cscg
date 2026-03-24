#Matlab snn_sample_hmmdt.m code in Jupyter

import numpy as np
import matplotlib.pyplot as plt

def snn_sample_hmmdt(net, data, ct):
    """
    Draws a spike from sem network - continuous time spiking hmm sems realistic refractory mechanim.

    net: A wta-network, see: wta-new()
    data: A data structure to be simulated.
    ct: Current time index.

    Returns:
    net: The (modified) network structure
    Z: Network output spikes
    """

    t = data['time'][ct[0]]

    time_range = data['time'][ct[-1]] - data['time'][ct[0]]

    num_spikes = round(net['sample_rate'] * time_range) - 1
    d_spikes = (time_range / num_spikes)
    spike_times = np.arange(data['time'][ct[0]] + d_spikes, data['time'][ct[-1]], d_spikes)

    rec_spikes = np.concatenate((net['rec_spikes'], np.array([np.inf, np.inf]).reshape((2, 1)) * np.ones((2, num_spikes))), axis=1)
    num_rs = net['rec_spikes'].shape[1]

    Z = np.zeros((2, num_spikes), dtype=np.float32)
    P = np.zeros((net['num_neurons'] + 1, num_spikes), dtype=np.float32)
    net['At'] = np.zeros((2, num_spikes), dtype=np.float32)

    last_input_spikes = np.tile(t, (net['num_inputs'], 1))
    last_output_spikes = np.tile(t, (net['num_neurons'], 1))

    hX = net['hX']
    hZ = net['hZ']

    W_exp = np.exp(net['W'].astype(np.float64))
    V_n_exp = np.exp(-net['V'].astype(np.float64))

    net['d_W'] = np.zeros((net['num_neurons'], net['num_inputs']), dtype=np.float32)
    net['d_V'] = np.zeros((net['num_neurons'], net['num_neurons']), dtype=np.float32)
    net['d_V0'] = np.zeros((net['num_neurons'], 1), dtype=np.float32)

    i = 1
    l = 1

    hX_all = np.zeros((net['num_inputs'], num_spikes), dtype=np.float32)
    hZ_all = np.zeros((net['num_neurons'], num_spikes), dtype=np.float32)

    if net['use_variance_tracking']:

        SW_new = net['SW']
        QW_new = net['QW']

        SV_new = net['SV']
        QV_new = net['QV']

        S0_new = net['S0']
        Q0_new = net['Q0']

        net['eta_W'] = net['eta'] * (QW_new - SW_new ** 2) / (np.exp(-SW_new) + 1)
        net['eta_V'] = net['eta'] * (QV_new - SV_new ** 2) / (np.exp(-SV_new) + 1)
        net['eta_0'] = net['eta'] * (Q0_new - S0_new ** 2) / (np.exp(-S0_new) + 1)
        

    try:
        for j in range(num_spikes):
            t = spike_times[j]

            while (i < data['Xt'].shape[1]) and (t > data['Xt'][1, i]):
                n_id = data['Xt'][0, i]
                sp_t = data['Xt'][1, i]

                hX[n_id, 0] = hX[n_id, 0] * np.exp(-float(sp_t - last_input_spikes[n_id]) / net['tau_x_r']) + 1
                hX[n_id, 1] = hX[n_id, 1] * np.exp(-float(sp_t - last_input_spikes[n_id]) / net['tau_x_f']) + 1

                last_input_spikes[n_id] = sp_t
                i += 1

            while (l < rec_spikes.shape[1]) and (t > rec_spikes[1, l]):
                n_id = rec_spikes[0, l]
                sp_t = rec_spikes[1, l]

                hZ[n_id, 0] = hZ[n_id, 0] * np.exp(-float(sp_t - last_output_spikes[n_id]) / net['tau_z_r']) + 1
                hZ[n_id, 1] = hZ[n_id, 1] * np.exp(-float(sp_t - last_output_spikes[n_id]) / net['tau_z_f']) + 1

                last_output_spikes[n_id] = sp_t
                l += 1

            hZ[:, 0] = hZ[:, 0] * np.exp(-float(t - last_output_spikes) / net['tau_z_r'])
            hZ[:, 1] = hZ[:, 1] * np.exp(-float(t - last_output_spikes) / net['tau_z_f'])
            hX[:, 0] = hX[:, 0] * np.exp(-float(t - last_input_spikes) / net['tau_x_r'])
            hX[:, 1] = hX[:, 1] * np.exp(-float(t - last_input_spikes) / net['tau_x_f'])

            d_hX = np.diff(hX, axis=1)
            d_hZ = np.diff(hZ, axis=1)

            hX_all[:, j] = d_hX.flatten()
            hZ_all[:, j] = d_hZ.flatten()

            last_input_spikes[:] = t
            last_output_spikes[:] = t

            u_rf = net['w_rf'] * np.exp(-float(t - net['last_spike_t']) / net['tau_rf'])

            U = np.dot(net['W'], d_hX) + np.dot(net['V'], d_hZ) + u_rf

            P_t = np.exp(U)
            exp_A = np.sum(P_t)
            A = np.log(exp_A)

            Z_i, k = wta_draw_k0(np.minimum(1, P_t))

            Z[:, j] = [k, t]
            P[:, j] = np.append(P_t, t)
            net['At'][:, j] = [A, t]
    except NameError:
        pass
    
        if (k > 0):
                rec_spikes[:, j + num_rs] = [k, t + net['rec_delay'][k]]

                net['num_o'][k] += 1

                net['last_spike_t'][k] = t

                net['d_W'][k, :] = net['d_W'][k, :] + net['eta_W'][k, :] * (d_hX.T - W_exp[k, :]) / np.maximum(net['eta_W'][k, :], W_exp[k, :])

                net['d_V'][:, k] = net['d_V'][:, k] - net['eta_V'][:, k]
                net['d_V'][k, :] = net['d_V'][k, :] + net['eta_V'][k, :] * np.minimum(V_n_exp[k, :] * d_hZ.T, net['eta_V'][k, :])

                if net['use_variance_tracking']:
                    SW_new[k, :] = SW_new[k, :] + net['eta_W'][k, :] * (net['W'][k, :] + net['d_W'][k, :] - SW_new[k, :])
                    QW_new[k, :] = QW_new[k, :] + net['eta_W'][k, :] * ((net['W'][k, :] + net['d_W'][k, :]) ** 2 - QW_new[k, :])

                    SV_new[k, :] = SV_new[k, :] + net['eta_V'][k, :] * ((net['V'][k, :] + net['d_V'][k, :]) - SV_new[k, :])
                    QV_new[k, :] = QV_new[k, :] + net['eta_V'][k, :] * ((net['V'][k, :] + net['d_V'][k, :]) ** 2 - QV_new[k, :])

        l = np.max([l, j + num_rs])
    except NameError:
            pass

    if net['use_variance_tracking']:
            net['SW_new'] = SW_new
            net['QW_new'] = QW_new

            net['SV_new'] = SV_new
            net['QV_new'] = QV_new

            net['S0_new'] = S0_new
            net['Q0_new'] = Q0_new

    net['rec_spikes'] = rec_spikes[:, l:]
    net['rec_spikes'][1, :] = net['rec_spikes'][1, :] - t
    net['last_spike_t'] = net['last_spike_t'] - t

    net['R'] = np.sum(net['At'][0, :])

    net['hX'] = hX
    net['hZ'] = hZ
    net['hX_all'] = hX_all
    net['hZ_all'] = hZ_all


    # ------

    