"""
======================================================
Response Function (:mod:`fd_greens.response_function`)
======================================================
"""

import os
from itertools import product
from typing import Sequence

import h5py
import numpy as np

from .molecular_hamiltonian import MolecularHamiltonian
from .qubit_indices import QubitIndices
from .parameters import method_indices_pairs, basis_matrix
from .utilities import reverse_qubit_order

np.set_printoptions(precision=6)

class ResponseFunction:
    """Frequency-domain charge-charge response function."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        h5fname: str = 'lih',
        suffix: str = '',
        method: str = 'exact',
        verbose: bool = True
    ) -> None:
        """Initializes a ``ResponseFunction`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
            h5fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
            method: The method for extracting transition amplitudes.
            verbose: Whether to print out observable values.
        """
        self.hamiltonian = hamiltonian
        self.h5fname = h5fname + '.h5'
        self.suffix = suffix
        self.method = method
        self.verbose = verbose

        with h5py.File(self.h5fname, 'r') as h5file:
            self.energies = {'gs': h5file['gs/energy'][()],
                             'n': h5file['es/energies'][:]}
            self.state_vectors = {'n': h5file['es/states'][:]}

        self.n_states = {'n': self.state_vectors['n'].shape[1]}
        self.n_orbitals = len(hamiltonian.active_indices)
        self.orbital_labels = list(product(range(self.n_orbitals), ['u', 'd']))
        self.n_system_qubits = 2 * self.n_orbitals - len(dict(method_indices_pairs)['taper'])

        self.qubit_indices_dict = QubitIndices.get_excited_qubit_indices_dict(
            2 * self.n_orbitals, method_indices_pairs)

        self.N = {subscript: np.zeros((2 * self.n_orbitals, 2 * self.n_orbitals, self.n_states[subscript]),
                    dtype=complex) for subscript in ['n']}
        self.T = {subscript: np.zeros((2 * self.n_orbitals, 2 * self.n_orbitals, self.n_states[subscript[0]]), 
                    dtype=complex) for subscript in ['np', 'nm']}

        self._process_diagonal()
        self._process_off_diagonal()

    def _process_diagonal(self) -> None:
        """Post-processes diagonal transition amplitudes circuits."""
        h5file = h5py.File(self.h5fname, 'r+')
        for i in range(2 * self.n_orbitals):
            m, s = self.orbital_labels[i]
            circuit_label = f'circ{m}{s}'

            for subscript in ['n']:
                qubit_indices = self.qubit_indices_dict[subscript]
                state_vectors_exact = self.state_vectors[subscript]

                if self.method == 'exact':
                    state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f'psi{self.suffix}']
                    state_vector = reverse_qubit_order(state_vector)
                    state_vector = qubit_indices(state_vector)

                    self.N[subscript][i, i] = np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                elif self.method == 'tomo':
                    tomography_labels = [''.join(x) for x in product('xyz', repeat=2)]

                    array = []
                    for tomography_label in tomography_labels:
                        array_all = h5file[f"{circuit_label}/{tomography_label}"].attrs[f"counts{self.suffix}"]
                        array_all = reverse_qubit_order(array_all) # XXX
                        start_index = int(qubit_indices.ancilla.str[0], 2)
                        array_label =  array_all[start_index :: 2]  / np.sum(array_all)
                        array += list(array_label)
                    
                    density_matrix = np.linalg.lstsq(basis_matrix, array)[0]
                    density_matrix = density_matrix(2 ** self.n_system_qubits, 2 ** self.n_system_qubits, order='F')
                    density_matrix = qubit_indices.system(density_matrix)

                    self.N[subscript][i, i] = [
                        (state_vectors_exact[:, j].conj() @ density_matrix @ state_vectors_exact[:, j]).real
                        for j in range(self.n_states[subscript])]
                
                if self.verbose:
                    print(f'N[{subscript}][{i}, {i}] = {self.N[subscript][i, i]}')

        h5file.close()

    def _process_off_diagonal(self) -> None:
        """Post-processes off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")

        for i in range(2 * self.n_orbitals):
            m, s = self.orbital_labels[i]
            for j in range(i + 1, 2 * self.n_orbitals):
                m_, s_ = self.orbital_labels[j]
                circuit_label = f'circ{m}{s}{m_}{s_}'

                for subscript in ['np', 'nm']:
                    qubit_indices = self.qubit_indices_dict[subscript]
                    state_vectors_exact = self.state_vectors[subscript[0]]

                    if self.method == 'exact':
                        state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f'psi{self.suffix}']
                        state_vector = reverse_qubit_order(state_vector) # XXX
                        state_vector = qubit_indices(state_vector)

                        self.T[subscript][i, j] = self.T[subscript][j, i] = np.abs(
                            state_vectors_exact.conj().T @ state_vector) ** 2

                    elif self.method == 'tomo':
                        tomography_labels = [''.join(x) for x in product('xyz', repeat=2)]

                        array = []
                        for tomography_label in tomography_labels:
                            array_all = h5file[f"{circuit_label}/{tomography_label}"].attrs[f"counts{self.suffix}"]
                            array_all = reverse_qubit_order(array_all) # XXX
                            start_index = int(qubit_indices.ancilla.str[0], 2)
                            array_label =  array_all[start_index :: 2 ** 2]  / np.sum(array_all)
                            array += list(array_label)

                        density_matrix = np.linalg.lstsq(basis_matrix, array)[0]
                        density_matrix = density_matrix.reshape(
                            2 ** self.n_system_qubits, 2 ** self.n_system_qubits, order='F')
                        density_matrix = qubit_indices.system(density_matrix)
                        self.T[subscript][i, j] = self.T[subscript][j, i] = [
                            (state_vectors_exact[:, k].conj() @ density_matrix @ state_vectors_exact[:, k]).real
                            for k in range(self.n_states[subscript])]

                    if self.verbose:
                        print(f'T[{subscript}][{i}, {j}] = {self.T[subscript][i, j]}')

        # Unpack T values to N values based on Eq. (18) of Kosugi and Matsushita 2021.
        for subscript in ['n']:
            for i in range(2 * self.n_orbitals):
                for j in range(i + 1, 2 * self.n_orbitals):
                    self.N[subscript][i, j] = self.N[subscript][j, i] = \
                        np.exp(-1j * np.pi / 4) * (self.T[subscript + 'p'][i, j] - self.T[subscript + 'm'][i, j]) \
                        + np.exp(1j * np.pi / 4) * (self.T[subscript + 'p'][j, i] - self.T[subscript + 'm'][j, i])
                    
                    if self.verbose:
                        print(f"N[{subscript}][{i}, {j}] =", self.N[subscript][i, j])

        h5file.close()


    def response_function(
        self, omegas: Sequence[float], eta: float = 0.0, save_data: bool = True
    ) -> np.ndarray:
        """Returns the charge-charge response function at given frequencies.

        Args:
            omegas: The frequencies at which the response function is calculated.
            eta: The imaginary part, i.e. broadening factor.
            save: Whether to save the response function to file.

        Returns:
            (Optional) The charge-charge response function for orbital i and orbital j.
        """
        for label in ["00", "11", "01", "10"]:
            i = int(label[0])
            j = int(label[1])
            chis = []
            for omega in omegas:
                for lam in [1, 2, 3]:  # [1, 2, 3] is hardcoded
                    chi = np.sum(
                        self.N[2 * i : 2 * (i + 1), 2 * j : 2 * (j + 1), lam]
                    ) / (omega + 1j * eta - (self.energies['s'][lam] - self.energies['gs']))
                    chi += np.sum(
                        self.N[2 * i : 2 * (i + 1), 2 * j : 2 * (j + 1), lam]
                    ).conjugate() / (
                        -omega - 1j * eta - (self.energies['s'][lam] - self.energies['gs'])
                    )
                chis.append(chi)
            chis = np.array(chis)
            if save_data:
                if not os.path.exists("data"):
                    os.makedirs("data")
                np.savetxt(
                    f"data/{self.datfname}_chi{label}.dat",
                    np.vstack((omegas, chis.real, chis.imag)).T,
                )
            else:
                return chis
