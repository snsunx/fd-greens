from tools import *
from hamiltonians import *

def get_number_state_indices_test():
    inds = get_number_state_indices(4, 2)
    print(inds)

def number_state_eigensolver_test():
    geometry = 'H 0 0 0; H 0 0 0.74'
    basis = 'sto3g'
    hamiltonian = MolecularHamiltonian(geometry, basis)
    hamiltonian_mat = hamiltonian.to_array(array_type='sparse')
    eigvals, eigvecs = np.linalg.eigh(hamiltonian_mat.toarray())
    print(eigvals)

    for n_elec in [0, 1, 2, 3, 4]:
        eigvals, eigvecs = number_state_eigensolver(hamiltonian_mat, n_elec)
        print(n_elec, eigvals)

get_number_state_indices_test()
