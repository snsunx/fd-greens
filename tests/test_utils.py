from qiskit import *
from utils import *
from hamiltonians import *


def get_number_state_indices_test():
    inds = get_number_state_indices(4, 2)
    print(inds)


def number_state_eigensolver_test():
    geometry = "H 0 0 0; H 0 0 0.74"
    basis = "sto3g"
    hamiltonian = MolecularHamiltonian(geometry, basis)
    hamiltonian_mat = hamiltonian.to_array(array_type="sparse")
    eigvals, eigvecs = np.linalg.eigh(hamiltonian_mat.toarray())
    print(eigvals)

    for n_elec in [0, 1, 2, 3, 4]:
        eigvals, eigvecs = number_state_eigensolver(hamiltonian_mat, n_elec)
        print(n_elec, eigvals)


def get_quantum_instance_test():
    backend = Aer.get_backend("qasm_simulator")
    q_instance = get_quantum_instance(
        backend, noise_model_name="ibmq_jakarta", shots=8192
    )


def get_pauli_tuple_test():
    pauli_tuple = get_pauli_tuple(4, 2)
    print(pauli_tuple)


if __name__ == "__main__":
    get_pauli_tuple_test()
