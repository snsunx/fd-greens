import sys
sys.path.append('../../..')

import pickle
import numpy as np
import cirq
import json
import h5py

from fd_greens import CircuitStringConverter

qubits = cirq.LineQubit.range(4)
converter = CircuitStringConverter(qubits)

results = pickle.load(open('qtrl_collections_3A_run0524_2_CZCS.pkl', 'rb'))
# print(results['full']['labels'])

h5file = h5py.File('lih_3A_sim.h5', 'r')
for circ_name in ['circ0u', 'circ0d', 'circ1u', 'circ1d', 'circ01u', 'circ01d']:
	qtrl_strings = json.loads(h5file[f'{circ_name}/transpiled'][()])
	print(circ_name, len(qtrl_strings))
	# print(circ_name, len(h5file[f'{circ_name}/transpiled']))
