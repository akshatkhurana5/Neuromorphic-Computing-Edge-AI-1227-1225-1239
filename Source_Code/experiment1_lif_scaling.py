# Experiment 1 — LIF Neuron Scaling
# Validates Table II: Platform specs (neuron count vs compute)
# Run on Google Colab or locally after: pip install brian2 numpy matplotlib

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import time as pytime  # avoid clash with brian2's built-in 'time'

prefs.codegen.target = 'numpy'

neuron_counts = [1000, 10000, 50000, 100000]
spike_rates   = []
sim_times_ms  = []

for N in neuron_counts:
    start_scope()
    eqs = '''
    dv/dt = (I - v) / (10*ms) : 1
    I : 1
    '''
    G = NeuronGroup(N, eqs, threshold='v > 1',
                    reset='v = 0', method='exact')
    G.I = '0.9 + 0.2*rand()'
    G.v = 'rand()'
    M  = SpikeMonitor(G)
    SM = StateMonitor(G, 'v', record=[0, 1, 2])

    t0 = pytime.time()
    run(100*ms)
    elapsed = (pytime.time() - t0) * 1000

    rate = np.array(M.count).mean() / 0.1   # fix: wrap in np.array()
    spike_rates.append(rate)
    sim_times_ms.append(elapsed)
    print(f"N={N:>7,}  spikes/s={rate:.1f}  sim_time={elapsed:.0f}ms")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(neuron_counts, sim_times_ms, 'o-', color='steelblue')
axes[0].set_xlabel('Neuron count')
axes[0].set_ylabel('Simulation time (ms)')
axes[0].set_title('Scaling — LIF network size vs compute time')
axes[1].plot(neuron_counts, spike_rates, 's-', color='darkorange')
axes[1].set_xlabel('Neuron count')
axes[1].set_ylabel('Mean spike rate (Hz)')
axes[1].set_title('Spike activity remains stable at scale')
plt.tight_layout()
plt.savefig('fig_table2_scaling.png', dpi=150)
plt.show()
print("Saved: fig_table2_scaling.png")
