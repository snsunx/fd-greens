"""Classical Transition Amplitudes Solver."""

from functools import reduce
from itertools import permutations

import numpy as np

from fd_greens.cirq_ver.qubit_indices import QubitIndices

from .molecular_hamiltonian import MolecularHamiltonian


np.set_printoptions(precision=6)

class ClassicalAmplitudesSolver:
    """Classical solver of the transition amplitudes."""

    def __init__(self, hamiltonian: MolecularHamiltonian, verbose: bool = True) -> None:
        """Initializes a ``ClassicalAmplitudesSolver`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
        """
        self.hamiltonian = hamiltonian
        self.n_orbitals = len(hamiltonian.active_indices) # Spatial orbitals
        self.n_qubits = 2 * self.n_orbitals
        self.verbose = verbose

        # Calculate the ground state.
        _, states = np.linalg.eigh(self.hamiltonian.matrix)
        self.state_gs = states[:, 0]
    
    def _get_a_operator(self, m: int, spin: str = '', dagger: bool = False) -> np.ndarray:
        """Returns a creation/annihilation operator."""
        assert spin in ['', 'u', 'd']
        n = 2 * self.n_orbitals
        I_matrix = np.eye(2)
        Z_matrix = np.diag([1.0, -1.0])
        a_matrix = np.array([[0.0, 1.0], [0.0, 0.0]])
        if dagger:
            a_matrix = a_matrix.T

        # When spin is '', interpret m as spin orbital index.
        if spin != '':
            m = 2 * m + (spin == 'd')

        matrices = [Z_matrix] * m + [a_matrix] + [I_matrix] * (n - m - 1)
        return reduce(np.kron, matrices)
    
    def _get_n_operator(self, i: int) -> np.ndarray:
        """Returns a number operator."""
        n = 2 * self.n_orbitals
        I_matrix = np.eye(2)
        Z_matrix = np.diag([1.0, -1.0])
        n_matrix = np.diag([0.0, 1.0])

        matrices = [Z_matrix] * i + [n_matrix] + [I_matrix] * (n - i - 1)
        return reduce(np.kron, matrices)

    def compute_B(self, spin: str = '') -> None:
        """Computes the B matrix."""
        assert spin in ['u', 'd', '']
        if spin in ['u', 'd']:
            qubit_indices = QubitIndices.get_eh_qubit_indices_dict(self.n_qubits, spin, system_only=True)
            indices_e = qubit_indices['e'].int
            indices_h = qubit_indices['h'].int
            dim_B = self.n_orbitals
        else:
            indices_e = [int(x, 2) for x in ['0111', '1011', '1101', '1110']]
            indices_h = [int(x, 2) for x in ['0001', '0010', '0100', '1000']]
            dim_B = 2 * self.n_orbitals

        hamiltonian_e = self.hamiltonian.matrix[indices_e][:, indices_e]
        hamiltonian_h = self.hamiltonian.matrix[indices_h][:, indices_h]

        # Calculate the (N±1)-electron eigenstates.
        _, states_e = np.linalg.eigh(hamiltonian_e)
        _, states_h = np.linalg.eigh(hamiltonian_h)
        n_states = states_e.shape[1]
        self.states_e = np.zeros((2 ** self.n_qubits, n_states), dtype=complex)
        self.states_h = np.zeros((2 ** self.n_qubits, n_states), dtype=complex)
        for i in range(n_states):
            self.states_e[indices_e, i] = states_e[:, i]
            self.states_h[indices_h, i] = states_h[:, i]
    
        # Calculate the B elements from the (N±1)-electron eigenstates.
        self.B = {subscript: np.zeros((dim_B, dim_B, n_states), dtype=complex) for subscript in ['e', 'h']}
        for m in range(dim_B):
            a_m = self._get_a_operator(m, spin, dagger=False)
            for n in range(dim_B):
                a_n_dag = self._get_a_operator(n, spin, dagger=True)

                self.B['e'][m, n] = (self.state_gs @ a_m @ self.states_e) * \
                    (self.states_e.conj().T @ a_n_dag @ self.state_gs)
                self.B['h'][m, n] = (self.state_gs @ a_n_dag @ self.states_h) * \
                    (self.states_h.conj().T @ a_m @ self.state_gs)

                self.B['e'][m, n][np.abs(self.B['e'][m, n]) < 1e-8] = 0.0
                self.B['h'][m, n][np.abs(self.B['h'][m, n]) < 1e-8] = 0.0
                if self.verbose:
                    print(f"B['e'][{m}, {n}] = ", self.B['e'][m, n])
                    print(f"B['h'][{m}, {n}] = ", self.B['h'][m, n])

    def compute_N(self) -> None:
        """Computes the N matrix."""
        qubit_indices = QubitIndices.get_excited_qubit_indices_dict(2 * self.n_orbitals, system_only=True)
        indices = qubit_indices['n'].int
        print(indices)

        hamiltonian_subspace = self.hamiltonian.matrix[indices][:, indices]
        _, states_es = np.linalg.eigh(hamiltonian_subspace)
        states_es = states_es[:, 1:]
        n_states = states_es.shape[1]

        self.states_es = np.zeros((2 ** self.n_qubits, n_states), dtype=complex)
        for i in range(n_states):
            self.states_es[indices, i] = states_es[:, i]


        self.N = {'n': np.zeros((2 * self.n_orbitals, 2 * self.n_orbitals, n_states))}
        for i in range(2 * self.n_orbitals):
            n_i = self._get_n_operator(i)
            for j in range(2 * self.n_orbitals):
                n_j = self._get_n_operator(j)

                self.N['n'][i, j] = (self.state_gs @ n_i @ self.states_es)\
                    * (self.states_es.conj().T @ n_j @ self.state_gs)
                
                self.N['n'][i, j][np.abs(self.N['n'][i, j]) < 1e-8] = 0.0
                if self.verbose:
                    print(f"N['n'][{i}, {j}] = ", self.N['n'][i, j])
