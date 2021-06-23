import sys
sys.path.append('../../src/')
from ansatze import *
import pickle
from tools import *
from openfermion.linalg import get_sparse_operator
from constants import *

ansatz = build_ne2_ansatz(4)
print(ansatz.parameters)
print(ansatz.parameters[0].name)

f = open('ansatz_params.pkl', 'rb')
params = pickle.load(f)
f.close()
print(params)

params_new = {}
for key, val in params.items():
    for p in ansatz.parameters:
        if key.name == p.name:
            params_new.update({p: val})

ansatz.assign_parameters(params_new, inplace=True)
psi = reverse_qubit_order(get_statevector(ansatz))
print(psi.shape)

hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]], 'sto3g', occupied_inds=[0], active_inds=[1, 2])
hamiltonian_arr = get_sparse_operator(hamiltonian.openfermion_op).toarray() * HARTREE_TO_EV
print(hamiltonian_arr.shape)

print(psi.conj().T @ hamiltonian_arr @ psi)
