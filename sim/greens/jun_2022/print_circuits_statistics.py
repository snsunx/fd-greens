import sys
sys.path.append('../../..')

import cirq
import h5py
import json
import numpy as np

from fd_greens import CircuitStringConverter, print_circuit_statistics

qubits = cirq.LineQubit.range(4)
converter = CircuitStringConverter(qubits)

with h5py.File('lih_3A_sim.h5', 'r') as h5file:
    for key in h5file.keys():
        if key[:4] == 'circ':
            print('=' * 25 + ' ' + key + ' ' + '=' * 25) 
            qtrl_strings = json.loads(h5file[f'{key}/zz'][()])
            circuit = converter.convert_strings_to_circuit(qtrl_strings)
            print_circuit_statistics(circuit)