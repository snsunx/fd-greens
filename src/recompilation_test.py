import numpy as np
from scipy.linalg import expm
from qiskit import *
from recompilation import *
from tools import *

n_qubits = 6
n_gate_rounds = 6

circ = QuantumCircuit(n_qubits)
statevector = get_statevector(circ)
statevector_rev = reverse_qubit_order(statevector)

A = np.kron(np.eye(2), np.random.rand(2 ** (n_qubits - 1), 2 ** (n_qubits - 1)))
U = expm(-1j * (A + A.T))
U_rev = reverse_qubit_order(U)

quimb_gates = recompile_with_statevector(statevector, U, n_gate_rounds=n_gate_rounds)
quimb_gates_rev = recompile_with_statevector(statevector_rev, U_rev, n_gate_rounds=n_gate_rounds)
#fig = circ.psi.graph(color=['PSI0', 'H', 'CNOT', 'RY', 'CZ'], 
#                     show_tags=False, show_inds=False, 
#                     initial_layout='spectral', return_fig=True)
#fig.savefig('recompilation_dev_test.png')
circ1 = apply_quimb_gates(quimb_gates, circ.copy(), reverse=True)
print(circ1)
psi1 = get_statevector(circ1)
print(psi1)

circ1_rev = apply_quimb_gates(quimb_gates_rev, circ.copy())
print(circ1_rev)
psi1_rev = get_statevector(circ1_rev)
print(psi1_rev)


print(abs(psi1.conj().T @ psi1_rev))
"""
statevector_recompiled = get_statevector(circ)
statevector_recompiled /= (statevector_recompiled[0] / abs(statevector_recompiled[0]))
print(statevector_recompiled[:10])

statevector_original = U @ statevector
statevector_original /= (statevector_original[0] / abs(statevector_original[0]))
print(statevector_original[:10])

print(statevector_recompiled.conj().T @ statevector_original)
"""
