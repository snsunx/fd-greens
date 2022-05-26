import sys

from py import process
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

def plot_tvd_by_depth(circ_name):

    qubits = cirq.LineQubit.range(4)
    converter = CircuitStringConverter(qubits)

    results = pickle.load(open('qtrl_collections_3A_run0524_2_CZCS.pkl', 'rb'))


    h5file = h5py.File('lih_3A_sim.h5', 'r')
    qtrl_strings = json.loads(h5file[f'{circ_name}/transpiled'][()])
    circuit_sim = converter.convert_strings_to_circuit(qtrl_strings)
    results_expt = results[f'{circ_name}_by_depth']['results']

    if len(circ_name) == 6:
        n_qubits = 3
        pad_zero = 'end'
    else:
        n_qubits = 4
        pad_zero = ''
    
    fidelities = []
    for i in range(len(circuit_sim)):
        circuit_moment = circuit_sim[:i + 1]
        circuit_moment += [cirq.measure(qubits[i]) for i in range(n_qubits)]

        result_moment = cirq.Simulator().run(circuit_moment, repetitions=1000)
        histogram = result_moment.multi_measurement_histogram(keys=[str(i) for i in range(n_qubits)])
        array_sim = histogram_to_array(histogram)

        array_expt = process_bitstring_counts(results_expt[i], pad_zero=pad_zero)
        tvd = np.sum(np.abs(array_sim - array_expt)) / 2

        fidelities.append(1 - tvd)

    plt.clf()
    fig, ax = plt.subplots()

    ax.plot(fidelities, marker='o')
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Fidelity (1 - TVD)")
    fig.savefig(f"fid_{circ_name}.png", dpi=300, bbox_inches='tight')

    h5file.close()
    

if __name__ == '__main__':
    circ_names = ['circ0u', 'circ1u', 'circ01u', 'circ0d', 'circ1d', 'circ01d']

    for circ_name in circ_names:
        print('circ_name =', circ_name)
        plot_tvd_by_depth(circ_name)
