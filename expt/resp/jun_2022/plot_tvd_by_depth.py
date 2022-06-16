import sys

sys.path.append('../../..')

import pickle
import numpy as np
import cirq
import h5py
import json
import matplotlib.pyplot as plt


from fd_greens import histogram_to_array, CircuitStringConverter
from fd_greens.cirq_ver.postprocessing import process_bitstring_counts

# qubits = cirq.LineQubit.range(4)
# converter = CircuitStringConverter(qubits)
# results = pickle.load(open('qtrl_collections_3A_run0524_2_CZCS.pkl', 'rb'))
# results_567 = pickle.load(open('qtrl_collections_3A_run0524_2_567_CZCS.pkl', 'rb'))

def plot_tvd_by_depth(run_id):

    qubits = cirq.LineQubit.range(4)
    converter = CircuitStringConverter(qubits)

    pkl_data = pickle.load(open(f'resp_3A_run0613_{run_id}.pkl', 'rb'))
    results_expt = pkl_data[f'circ0d1u_by_depth']['results']

    with h5py.File('lih_resp_sim.h5', 'r') as h5file:
        qtrl_strings = json.loads(h5file[f'circ0d1u/zz'][()])
        circuit_sim = converter.convert_strings_to_circuit(qtrl_strings)

    n_qubits = 4
    pad_zero = ''
    
    fidelities = []
    positions_3q = []
    for i in range(len(circuit_sim)):
        moment = circuit_sim[i]
        if len(moment) == 1 and moment.operations[0].gate.num_qubits() == 3:
            positions_3q.append(i)
         
        circuit_moment = circuit_sim[:i + 1]
        circuit_moment += [cirq.measure(qubits[i]) for i in range(n_qubits)]

        result_moment = cirq.Simulator().run(circuit_moment, repetitions=5000)
        histogram = result_moment.multi_measurement_histogram(keys=[str(i) for i in range(n_qubits)])
        array_sim = histogram_to_array(histogram)

        array_expt = process_bitstring_counts(results_expt[i], pad_zero=pad_zero)
        tvd = np.sum(np.abs(array_sim - array_expt)) / 2

        fidelities.append(1 - tvd)

    plt.clf()
    fig, ax = plt.subplots()

    ax.plot(fidelities, marker='.')
    ax.plot(fidelities, color='r', ls='', marker='x', ms=10, mew=2,  markevery=positions_3q)
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Fidelity (1 - TVD)")
    fig.savefig(f"fid_by_depth_{run_id}.png", dpi=250, bbox_inches='tight')


if __name__ == '__main__':
    for run_id in [0, 1, 2, 3]:
        plot_tvd_by_depth(run_id)
