import numpy as np
from scipy.sparse.linalg import eigsh
from constants import HARTREE_TO_EV
from hamiltonians import *

geometry = 'Li 0 0 0; H 0 0 1.6'
basis = 'sto3g'
hamiltonian = MolecularHamiltonian(geometry, basis)
hamiltonian.to_openfermion_qubit_operator()
molecule = hamiltonian.molecule
for x in molecule.__dir__():
    if x[:2] == 'n_':
        print(x)
print('n_orbitals =', molecule.n_orbitals)
print('n_electrons =', molecule.n_electrons)
print('n_qubits =', molecule.n_qubits)
exit()
print(molecule.hf_energy * HARTREE_TO_EV)

arr = hamiltonian.to_array(return_type='sparse')
e, v = eigsh(arr, which='SA', k=3)
print(e * HARTREE_TO_EV)
