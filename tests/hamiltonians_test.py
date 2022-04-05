import numpy as np
from scipy.sparse.linalg import eigsh
from constants import HARTREE_TO_EV
from hamiltonians import *

hamiltonian = MolecularHamiltonian(
    [["Li", (0, 0, 0)], ["H", (0, 0, 1.6)]], "sto3g", occ_inds=[0], act_inds=[1, 2]
)
# hamiltonian = MolecularHamiltonian([['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]], 'sto3g')
hamiltonian.build()
print(type(hamiltonian.openfermion_op))
molecule = hamiltonian.molecule
print("n_orbitals =", molecule.n_orbitals)
print("n_electrons =", molecule.n_electrons)
print("n_qubits =", molecule.n_qubits)
print("-" * 80)
print("n_orb =", 2 * len(hamiltonian.act_inds))
print("n_occ =", molecule.n_electrons - 2 * len(hamiltonian.occ_inds))
# qiskit_op = hamiltonian.to_qiskit_qubit_operator()
# print(qiskit_op)
print(molecule.hf_energy * HARTREE_TO_EV)

arr = hamiltonian.to_array(return_type="sparse")
e, v = eigsh(arr, which="SA", k=3)
print(e * HARTREE_TO_EV)
