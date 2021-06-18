import numpy as np
from scipy.linalg import expm
from qiskit import *
from recompilation_dev import *


n_qubits = 6
circ = QuantumCircuit(n_qubits)
statevector = np.zeros((2 ** n_qubits,))
statevector[0] = 1

A = np.kron(np.eye(2), np.random.rand(2 ** (n_qubits - 1), 2 ** (n_qubits - 1)))
U = expm(-1j * (A + A.T))
# print(np.allclose(U @ U.conj().T, np.eye(4)))

quimb_gates = recompile_with_statevector(statevector, U, n_gate_rounds=6)
for gate in quimb_gates:
    print(gate)
#fig = circ.psi.graph(color=['PSI0', 'H', 'CNOT', 'RY', 'CZ'], 
#                     show_tags=False, show_inds=False, 
#                     initial_layout='spectral', return_fig=True)
#fig.savefig('recompilation_dev_test.png')
apply_quimb_gates(quimb_gates, circ)
print(circ)