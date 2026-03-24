import numpy as np

def snn_sample_ctrf(net, data, ct):
    """
    Continuous-time WTA spiking network sampler with
    realistic refractory mechanism and recurrent delay.
    Faithful Python translation of Kappel's MATLAB snn_sample_ctrf.
    """

    # ================================================================
    # SECTION 1: SETUP & SPIKE TIME GENERATION
    # ================================================================

    # MATLAB: t = data.time(ct(1))
    # Get the start time of this simulation window
    t = data['time'][ct[0]]

    # MATLAB: time_range = data.time(ct(end)) - data.time(ct(1))
    # Duration of this simulation window
    time_range = data['time'][ct[-1]] - data['time'][ct[0]]

    # MATLAB: num_spikes = poissrnd(lambda * time_range)
    # How many spikes will the WTA circuit emit in this window?
    # Drawn from Poisson — the circuit fires at mean rate lambda (=2000 Hz)
    if net['fix_num_spikes']:
        num_spikes = int(round(net['lambda'] * time_range))
    else:
        num_spikes = np.random.poisson(net['lambda'] * time_range)

    # MATLAB: spike_times = sort(time_range * rand(1, num_spikes))
    # Draw num_spikes uniform random times in [0, time_range], then sort
    # These are the CONTINUOUS times at which WTA spikes will occur
    spike_times = np.sort(np.random.uniform(0, time_range, num_spikes))

    # MATLAB: rec_spikes = [net.rec_spikes, inf(2, num_spikes)]
    # Append placeholder columns (inf = not yet assigned) for new recurrent spikes
    # rec_spikes is 2×M: row0=neuron_id, row1=arrival_time
    if net['rec_spikes'].shape[1] > 0:
        rec_spikes = np.hstack([
            net['rec_spikes'],
            np.full((2, num_spikes), np.inf)
        ])
    else:
        rec_spikes = np.full((2, num_spikes), np.inf)

    # Number of pre-existing (carried-over) recurrent spikes from previous window
    num_rs = net['rec_spikes'].shape[1]

    # ================================================================
    # SECTION 2: INITIALISE OUTPUT ARRAYS & LOCAL TRACES
    # ================================================================

    Z   = np.zeros((2, num_spikes))          # output spike log: [neuron_id; time]
    P   = np.zeros((net['num_neurons']+1, num_spikes))  # probability log
    At  = np.zeros((2, num_spikes))          # log activity at each spike time

    # last_input_spikes: for each input neuron, time of its most recent spike
    # initialised to t (start of window) — means "no spike yet"
    last_input_spikes  = np.full(net['num_inputs'],  t)
    last_output_spikes = np.full(net['num_neurons'], t)

    # Copy EPSP traces from network state (carry-over from previous window)
    # hX: (num_inputs  × 2) — col0=rise trace, col1=fall trace
    # hZ: (num_neurons × 2) — col0=rise trace, col1=fall trace
    hX = net['hX'].copy()
    hZ = net['hZ'].copy()

    # Precompute exp(W), exp(V) for efficiency in weight update
    W_exp  = np.exp(net['W'].astype(float))
    V_exp  = np.exp(net['V'].astype(float))
    V0_exp = np.exp(net['V0'].astype(float))

    # Accumulate weight deltas — applied after all spikes (batch update)
    net['d_W']  = np.zeros((net['num_neurons'], net['num_inputs']))
    net['d_V']  = np.zeros((net['num_neurons'], net['num_neurons']))
    net['d_V0'] = np.zeros( net['num_neurons'])

    # i = pointer into data.Xt (input spike train, sorted by time)
    # l = pointer into rec_spikes (recurrent spike queue, sorted by arrival time)
    i = 0
    l = 0

    # Log EPSP values and log-partition values at each spike time (for learning)
    hX_all = np.zeros((net['num_inputs'],  num_spikes))
    hZ_all = np.zeros((net['num_neurons'], num_spikes))
    A_v    = np.zeros(num_spikes)   # log-partition from recurrent input only
    A_w    = np.zeros(num_spikes)   # log-partition from feedforward input only

    # ================================================================
    # SECTION 3: MAIN SPIKE LOOP — iterate over each WTA spike event
    # ================================================================

    for j in range(num_spikes):

        t = spike_times[j]   # current spike time

        # ------------------------------------------------------------
        # 3a. UPDATE FEEDFORWARD EPSPs (hX)
        # Process all INPUT spikes that arrived before current time t
        # MATLAB: while (i < size(data.Xt,2)) && (t > data.Xt(2,i))
        # data.Xt is 2×M: row0=neuron_id, row1=spike_time
        # ------------------------------------------------------------
        while (i < data['Xt'].shape[1]) and (t > data['Xt'][1, i]):

            n_id = int(data['Xt'][0, i])   # which input neuron spiked
            sp_t = data['Xt'][1, i]        # when it spiked

            # Decay the existing trace to sp_t, then add 1 (spike arrived)
            # Rise trace: hX[n,0] = hX[n,0] * exp(-dt/tau_x_r) + 1
            # Fall trace: hX[n,1] = hX[n,1] * exp(-dt/tau_x_f) + 1
            dt_in = sp_t - last_input_spikes[n_id]
            hX[n_id, 0] = hX[n_id, 0] * np.exp(-dt_in / net['tau_x_r']) + 1
            hX[n_id, 1] = hX[n_id, 1] * np.exp(-dt_in / net['tau_x_f']) + 1

            last_input_spikes[n_id] = sp_t
            i += 1

        # ------------------------------------------------------------
        # 3b. UPDATE RECURRENT EPSPs (hZ) FROM DELAYED RECURRENT SPIKES
        # Process all recurrent spikes whose DELAYED arrival time < t
        # This is where the 10ms delay actually takes effect:
        # a spike fired at t'=5ms arrives here at t'+rec_delay=15ms
        # ------------------------------------------------------------
        while (l < rec_spikes.shape[1]) and (t > rec_spikes[1, l]):

            n_id = int(rec_spikes[0, l])   # which output neuron's spike
            sp_t = rec_spikes[1, l]        # delayed arrival time

            dt_out = sp_t - last_output_spikes[n_id]
            hZ[n_id, 0] = hZ[n_id, 0] * np.exp(-dt_out / net['tau_z_r']) + 1
            hZ[n_id, 1] = hZ[n_id, 1] * np.exp(-dt_out / net['tau_z_f']) + 1

            last_output_spikes[n_id] = sp_t
            l += 1

        # ------------------------------------------------------------
        # 3c. DECAY ALL EPSP TRACES TO CURRENT TIME t
        # After processing all spikes up to t, decay everything to t
        # This gives the CURRENT value of all traces at exactly time t
        # ------------------------------------------------------------
        dt_out_all = t - last_output_spikes          # (num_neurons,)
        dt_in_all  = t - last_input_spikes           # (num_inputs,)

        hZ[:, 0] *= np.exp(-dt_out_all / net['tau_z_r'])
        hZ[:, 1] *= np.exp(-dt_out_all / net['tau_z_f'])
        hX[:, 0] *= np.exp(-dt_in_all  / net['tau_x_r'])
        hX[:, 1] *= np.exp(-dt_in_all  / net['tau_x_f'])

        # Reset "last spike" to t for next iteration's decay calculation
        last_input_spikes[:]  = t
        last_output_spikes[:] = t

        # ------------------------------------------------------------
        # 3d. COMPUTE ALPHA-SHAPED EPSP SIGNALS
        # d_hX = hX[:,1] - hX[:,0]  =  fall - rise  =  alpha shape
        # This is ZERO at spike time, peaks at ~tau_x_f, then decays
        # This is the actual synaptic drive seen by the neuron
        # ------------------------------------------------------------
        d_hX = hX[:, 1] - hX[:, 0]   # (num_inputs,)   feedforward EPSP
        d_hZ = hZ[:, 1] - hZ[:, 0]   # (num_neurons,)  recurrent EPSP

        hX_all[:, j] = d_hX
        hZ_all[:, j] = d_hZ

        # ------------------------------------------------------------
        # 3e. REFRACTORY TERM
        # u_rf = w_rf * exp(-(t - last_spike_t) / tau_rf)
        # w_rf = -10 → strong negative drive right after a spike
        # decays with tau_rf = 5ms → gone after ~25ms
        # ------------------------------------------------------------
        u_rf = net['w_rf'] * np.exp(
            -(t - net['last_spike_t']) / net['tau_rf'])   # (num_neurons,)

        # ------------------------------------------------------------
        # 3f. MEMBRANE POTENTIAL
        # U_v = recurrent contribution + refractory
        # U_w = feedforward contribution (scaled by w_temperature)
        # U   = total membrane potential
        # ------------------------------------------------------------
        U_v = net['V'] @ d_hZ + u_rf          # (num_neurons,)
        U_w = net['w_temperature'] * net['W'] @ d_hX  # (num_neurons,)
        U   = U_w + U_v

        # ------------------------------------------------------------
        # 3g. SOFTMAX FIRING PROBABILITY
        # P_t[k] = exp(U[k]) / sum(exp(U))
        # This is the WTA soft competition — all neurons compete
        # The log-partition A = log(sum(exp(U))) is saved for learning
        # ------------------------------------------------------------
        def wta_softmax(u):
            u_shifted = u - np.max(u)          # numerical stability
            exp_u = np.exp(u_shifted)
            A = np.log(np.sum(np.exp(u)))      # true log-partition
            return exp_u / np.sum(exp_u), A

        P_t_v, A_v[j] = wta_softmax(U_v)
        P_t_w, A_w[j] = wta_softmax(U_w)
        P_t,   A       = wta_softmax(U)

        # Importance weight correction (exact mode):
        # A = log Z_total - log Z_v - log Z_w
        # This corrects for the factored approximation
        A = A - A_v[j] - A_w[j]

        # ------------------------------------------------------------
        # 3h. DRAW SPIKE — sample winner neuron k from P_t
        # ------------------------------------------------------------
        k = np.random.choice(net['num_neurons'], p=P_t)
        Z_i = np.zeros(net['num_neurons'])
        Z_i[k] = 1.0

        Z[:, j] = [k, t]
        P[:, j] = np.append(P_t, t)

        # ------------------------------------------------------------
        # 3i. QUEUE DELAYED RECURRENT SPIKE
        # Neuron k fired at time t → its recurrent spike arrives
        # at t + rec_delay[k] (= t + 10ms)
        # Stored in rec_spikes for future iterations of this loop
        # or carried over to the next simulation window
        # ------------------------------------------------------------
        rec_spikes[:, j + num_rs] = [k, t + net['rec_delay'][k]]

        net['last_spike_t'][k] = t
        net['num_o'][k] += 1

        # ------------------------------------------------------------
        # 3j. STDP WEIGHT UPDATE
        # dW[k,i] = eta * (alpha_w * d_hX[i] - exp(W[k,i]))
        #           / max(eta, exp(W[k,i]))
        #
        # Hebbian term:  alpha_w * d_hX[i]  → potentiate if pre fired before post
        # Depression:    exp(W[k,i])         → always depress (weight-dependent)
        # Normaliser:    max(eta, exp(W))    → adaptive learning rate
        # ------------------------------------------------------------
        d_W_k = (net['eta_W'][k, :] *
                 (net['alpha_w'] * d_hX - W_exp[k, :]) /
                 np.maximum(net['eta_W'][k, :], W_exp[k, :]))

        d_V_k = (net['eta_V'][k, :] *
                 (net['alpha_v'] * d_hZ - V_exp[k, :]) /
                 np.maximum(net['eta_V'][k, :], V_exp[k, :]))

        # d_V0: bias update — pushes toward Z_i (one-hot of winner)
        d_V0  = (net['eta_0'] * (Z_i - V0_exp) /
                 np.maximum(net['eta_0'], V0_exp))

        At[:, j] = [A, t]

        # Accumulate deltas (applied as batch after all spikes)
        net['d_W'][k, :]  += d_W_k
        net['d_V'][k, :]  += d_V_k
        net['d_V0']       += d_V0

    # ================================================================
    # SECTION 4: POST-LOOP — SAVE STATE BACK TO NET
    # ================================================================

    # Keep only unprocessed recurrent spikes (those not yet arrived)
    # Shift their times relative to current t (for next window)
    net['rec_spikes'] = rec_spikes[:, l:]
    net['rec_spikes'][1, :] -= t

    # Shift last_spike_t relative to t (continuous time bookkeeping)
    net['last_spike_t'] -= t

    # Importance weight R = mean log-activity = learning signal
    net['R'] = np.mean(At[0, :]) if num_spikes > 0 else 0.0

    net['A_v'] = A_v
    net['A_w'] = A_w
    net['It']  = np.vstack([At[0, :] + A_v + A_w, At[1, :]])

    # Save EPSP traces for next window (continuous state)
    net['hX']     = hX
    net['hZ']     = hZ
    net['hX_all'] = hX_all
    net['hZ_all'] = hZ_all
    net['At']     = At

    return net, Z, P
