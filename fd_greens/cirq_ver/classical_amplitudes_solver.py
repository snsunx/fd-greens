"""Classical Transition Amplitudes Solver."""

from functools import reduce
from itertools import permutations

import numpy as np

from fd_greens.cirq_ver.qubit_indices import QubitIndices

from .molecular_hamiltonian import MolecularHamiltonian


np.set_printoptions(precision=6)

class ClassicalAmplitudesSolver:

    def __init__(self, hamiltonian: MolecularHamiltonian, fname: str):
        self.hamiltonian = hamiltonian
        self.h5fname = fname + '.h5'


        self.n_orbitals = len(hamiltonian.active_indices)

        e, v = np.linalg.eigh(self.hamiltonian.matrix)
        self.state_gs = v[:, 0]
        # self.state_gs = np.flip(self.state_gs)
        # self.state_gs[abs(self.state_gs) < 1e-8] = 0.0
        # print('self.states_gs =', self.state_gs)
        for i, x in enumerate(self.state_gs):
            if abs(x) > 1e-8:
                print(format(i, "04b"), format(x, ".6f"))
    
    def _get_a_operator(self, m: int, spin: str = '', dagger: bool = False) -> np.ndarray:
        """Returns a creation/annihilation operator."""
        assert spin in ['', 'u', 'd']
        n = 2 * self.n_orbitals
        I_matrix = np.eye(2)
        Z_matrix = np.diag([1.0, -1.0])
        a_matrix = np.array([[0.0, 1.0], [0.0, 0.0]])
        if dagger:
            a_matrix = a_matrix.T

        if spin != '':
            m = 2 * m + (spin == 'd')

        matrices = [Z_matrix] * m + [a_matrix] + [I_matrix] * (n - m - 1)
        return reduce(np.kron, matrices)

    def compute_B(self, spin: str = 'u'):
        """Computes the B matrix."""
        qubit_indices = QubitIndices.get_eh_qubit_indices_dict(4, spin, system_only=True)
        print(qubit_indices['e'])
        print(qubit_indices['h'])

        energies_e, states_e = np.linalg.eigh(qubit_indices['e'](self.hamiltonian.matrix))
        energies_h, states_h = np.linalg.eigh(qubit_indices['h'](self.hamiltonian.matrix))

        states_e = states_e
        states_h = np.flip(states_h)

        self.states_e = np.array([qubit_indices['e'].expand(states_e[:, i]) for i in range(2)]).T
        self.states_h = np.array([qubit_indices['h'].expand(states_h[:, i]) for i in range(2)]).T

        self.B = {subscript: np.zeros((self.n_orbitals, self.n_orbitals, 2), dtype=complex)
                  for subscript in ['e', 'h']}
        
        for m in range(self.n_orbitals):
            a_m = self._get_a_operator(m, spin, dagger=False)
            for n in range(self.n_orbitals):
                a_n_dag = self._get_a_operator(n, spin, dagger=True)
                # print(f"{self.states_e.shape}")
                # print(f"{a_n_dag.shape}")
                # print(f"{self.state_gs.shape}")
                self.B['e'][m, n] = (self.state_gs @ a_m @ self.states_e) * (self.states_e.conj().T @ a_n_dag @ self.state_gs)
                self.B['h'][m, n] = (self.state_gs @ a_n_dag @ self.states_h) * (self.states_h.conj().T @ a_m @ self.state_gs)

                print(f"self.B['e'][{m}, {n}] = ", self.B['e'][m, n])
            # print(f"self.B['h'][{m}, {m}] = ", self.B['h'][m, m])

    def compute_B1(self):
        """Computes the B matrix."""

        indices_e = [int(x, 2) for x in ['0111', '1011', '1101', '1110']]
        indices_h = [int(x, 2) for x in ['0001', '0010', '0100', '1000']]
        hamiltonian_e = self.hamiltonian.matrix[indices_e][:, indices_e]
        hamiltonian_h = self.hamiltonian.matrix[indices_h][:, indices_h]

        print(indices_e)
        print(indices_h)

        energies_e, states_e = np.linalg.eigh(hamiltonian_e)
        energies_h, states_h = np.linalg.eigh(hamiltonian_h)

        self.states_e = np.zeros((16, 4), dtype=complex)
        self.states_h = np.zeros((16, 4), dtype=complex)
        for i in range(4):
            self.states_e[indices_e, i] = states_e[:, i]
            self.states_h[indices_h, i] = states_h[:, i]
        # print(energies_e)
        # print(states_e)
        # print(energies_h)
        # print(states_h)

        self.B = {subscript: np.zeros((2 * self.n_orbitals, 2 * self.n_orbitals, 4), dtype=complex)
                  for subscript in ['e', 'h']}

        print('-' * 80)
        for j in range(4):
            print('j = ', j)
            a_n = self._get_a_operator(j, dagger=True)
            result = a_n @ self.state_gs
            for i, x in enumerate(result):
                if abs(x) > 1e-8:
                    print(i, format(i, "04b"), format(x, ".6f"))

        

        for m in range(2 * self.n_orbitals):
            a_m = self._get_a_operator(m, dagger=False)
            for n in range(2 * self.n_orbitals):
                a_n_dag = self._get_a_operator(n, dagger=True)
                self.B['e'][m, n] = (self.state_gs @ a_m @ self.states_e) * (self.states_e.conj().T @ a_n_dag @ self.state_gs)
                self.B['h'][m, n] = (self.state_gs @ a_n_dag @ self.states_h) * (self.states_h.conj().T @ a_m @ self.state_gs)


            result = self.B['e'][m, m]
            result[abs(result) < 1e-8] = 0.0
            print(f"self.B['e'][{m}, {m}] = ", result)
            # print(f"self.B['h'][{m}, {m}] = ", self.B['h'][m, m])


    def compute_N(self):
        pass

