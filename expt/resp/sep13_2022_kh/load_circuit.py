import sys
sys.path.append('../../..')

import h5py
import json
import cirq

from fd_greens import CircuitStringConverter, print_circuit_statistics

# NOTE: Valid circuit names are circ0u, circ0d, circ1u, circ1d,
# circ0u0d, circ0u1u, circ0u1d, circ0d1u, circ0d1d, circ1u1d
circuit_name = "circ0u1d"

# Load the Qtrl strings
h5file = h5py.File("kh_resp_exact.h5", 'r')
strings = json.loads(h5file[f"{circuit_name}/transpiled"][()])

# Convert the Qtrl strings to circuit
qubits = cirq.LineQubit.range(4)
converter = CircuitStringConverter(qubits)
circuit = converter.convert_strings_to_circuit(strings)

# Print circuit statistics and circuit diagram
print_circuit_statistics(circuit)
print('')
print(circuit)
