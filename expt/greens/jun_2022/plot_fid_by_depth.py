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
from fd_greens.cirq_ver.utilities import get_non_z_locations


def plot_fid_by_depth(run_id):

    qubits = cirq.LineQubit.range(4)
    converter = CircuitStringConverter(qubits)

    pkl_data = pickle.load(open(f'greens_3A_run0613_{run_id}.pkl', 'rb'))
    circuit_expt = pkl_data[f'circ01d_by_depth']['circs'][-1]
    results_expt = pkl_data[f'circ01d_by_depth']['results']

    with h5py.File('lih_3A_sim.h5', 'r') as h5file:
        qtrl_strings = json.loads(h5file[f'circ01d/zz'][()])
        circuit_sim = converter.convert_strings_to_circuit(qtrl_strings)

    print(f"{len(circuit_sim) = }")
    non_z_locations = get_non_z_locations(circuit_sim)
    non_z_locations.append(len(circuit_sim) - 1)
    print(len(non_z_locations))

    print(f"{len(circuit_expt) = }")
    non_z_locations_expt = get_non_z_locations(circuit_expt)
    non_z_locations_expt.append(len(circuit_expt) - 1)
    print(len(non_z_locations_expt))

    n_qubits = 4
    pad_zero = ''
    
    fidelities = []
    positions_3q = []
    for e, (i, j) in enumerate(zip(non_z_locations, non_z_locations_expt)):
        print(f"{i = }")
        moment = circuit_sim[i]
        if len(moment) == 1 and moment.operations[0].gate.num_qubits() == 3:
            positions_3q.append(e)
         
        circuit_moment = circuit_sim[:i + 1]
        circuit_moment += [cirq.measure(qubits[k]) for k in range(n_qubits)]

        result_moment = cirq.Simulator().run(circuit_moment, repetitions=5000)
        histogram = result_moment.multi_measurement_histogram(keys=[str(i) for i in range(n_qubits)])
        array_sim = histogram_to_array(histogram)

        array_expt = process_bitstring_counts(results_expt[j], pad_zero=pad_zero)
        tvd = np.sum(np.abs(array_sim - array_expt)) / 2

        fidelities.append(1 - tvd)

    plt.clf()
    fig, ax = plt.subplots()

    ax.plot(fidelities, marker='.')
    ax.plot(fidelities, color='r', ls='', marker='x', ms=10, mew=2,  markevery=positions_3q)
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Fidelity (1 - TVD)")
    fig.savefig(f"fid_by_depth_0616_{run_id}.png", dpi=250, bbox_inches='tight')


if __name__ == '__main__':
    for run_id in [0]:
        plot_fid_by_depth(run_id)
