"""
======================================================
Response Function (:mod:`fd_greens.response_function`)
======================================================
"""

import os
from itertools import product
from typing import Sequence, Optional, Mapping

import h5py
import numpy as np

from .molecular_hamiltonian import MolecularHamiltonian
from .qubit_indices import QubitIndices
from .parameters import ErrorMitigationParameters, get_method_indices_pairs, REVERSE_QUBIT_ORDER
from .general_utils import project_density_matrix, purify_density_matrix, reverse_qubit_order, two_qubit_state_tomography

np.set_printoptions(precision=6, suppress=True)

class ResponseFunction:
    """Frequency-domain charge-charge response function."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        fname: str = 'lih',
        suffix: str = '',
        method: str = 'exact',
        verbose: bool = True
    ) -> None:
        """Initializes a ``ResponseFunction`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
            fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
            method: The method for extracting transition amplitudes.
            verbose: Whether to print out observable values.
        """
        # Input attributes.
        self.hamiltonian = hamiltonian
        self.fname = fname 
        self.h5fname = fname + '.h5'
        self.suffix = suffix
        self.method = method
        self.verbose = verbose

        self.parameters = ErrorMitigationParameters()
        self.parameters.write(fname)

        # Load energies and state vectors from HDF5 file.
        with h5py.File(self.h5fname, 'r') as h5file:
            self.energies = {'gs': h5file['gs/energy'][()], 'n': h5file['es/energies'][:]}
            self.state_vectors = {'n': h5file['es/states'][:]}

        # Derived attributes.
        method_indices_pairs = get_method_indices_pairs('')
        self.n_states = {'n': self.state_vectors['n'].shape[1]}
        self.n_orbitals = len(hamiltonian.active_indices)
        self.orbital_labels = list(product(range(self.n_orbitals), ['u', 'd']))
        self.n_system_qubits = 2 * self.n_orbitals - method_indices_pairs.n_tapered
        self.qubit_indices_dict = QubitIndices.get_excited_qubit_indices_dict(
            2 * self.n_orbitals, method_indices_pairs)

        # Initialize array quantities N and T.
        self.N = {subscript: np.zeros((2 * self.n_orbitals, 2 * self.n_orbitals, self.n_states[subscript]),
                    dtype=complex) for subscript in ['n']}
        self.T = {subscript: np.zeros((2 * self.n_orbitals, 2 * self.n_orbitals, self.n_states[subscript[0]]), 
                    dtype=complex) for subscript in ['np', 'nm']}

    def _process_diagonal(self) -> None:
        """Processes diagonal transition amplitudes results."""
        h5file = h5py.File(self.h5fname, 'r+')
        for i in range(2 * self.n_orbitals):
            m, s = self.orbital_labels[i]
            circuit_label = f'circ{m}{s}'

            for subscript in ['n']:
                qubit_indices = self.qubit_indices_dict[subscript]
                state_vectors_exact = self.state_vectors[subscript]

                if self.method == 'exact':
                    state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f'psi{self.suffix}']
                    if REVERSE_QUBIT_ORDER:
                        state_vector = reverse_qubit_order(state_vector)
                    state_vector = qubit_indices(state_vector)
                    h5file[f'psi{self.suffix}/{subscript}{i}'] = state_vector

                    self.N[subscript][i, i] = np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                elif self.method == 'tomo':
                    tomography_labels = [''.join(x) for x in product('xyz', repeat=2)]

                    array_all = []
                    for tomography_label in tomography_labels:
                        array_raw = h5file[f"{circuit_label}/{tomography_label}"].attrs[f"counts{self.suffix}"]
                        repetitions = np.sum(array_raw)
                        ancilla_index = int(qubit_indices.ancilla.str[0], 2)
                        if REVERSE_QUBIT_ORDER:
                            array_raw = reverse_qubit_order(array_raw)
                            array_label =  array_raw[ancilla_index :: 2]  / repetitions
                        else:
                            array_label =  array_raw[ancilla_index * 4 : (ancilla_index + 1) * 4] / repetitions
                        array_all += list(array_label)
                    
                    # Slice the density matrix and save to HDF5 file.
                    density_matrix = two_qubit_state_tomography(array_all)
                    density_matrix = qubit_indices.system(density_matrix)

                    # Normalize and optionally project or purify the density matrix.
                    trace = np.trace(density_matrix)
                    density_matrix /= trace
                    if self.parameters.PROJECT_DENSITY_MATRICES:
                        density_matrix = project_density_matrix(density_matrix)
                    if self.parameters.PURIFY_DENSITY_MATRICES:
                        density_matrix = purify_density_matrix(density_matrix)

                    h5file[f'rho{self.suffix}/{subscript}{i}'] = density_matrix

                    self.N[subscript][i, i] = [
                        trace * (state_vectors_exact[:, j].conj() @ density_matrix @ state_vectors_exact[:, j]).real
                        for j in range(self.n_states[subscript])]
                
                if self.verbose:
                    print(f'N[{subscript}][{i}, {i}] = {self.N[subscript][i, i]}')

        h5file.close()

    def _process_off_diagonal(self) -> None:
        """Processes off-diagonal transition amplitude results."""
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
                        if REVERSE_QUBIT_ORDER:
                            state_vector = reverse_qubit_order(state_vector)
                        state_vector = qubit_indices(state_vector)
                        h5file[f'psi{self.suffix}/{subscript}{i}{j}'] = state_vector

                        self.T[subscript][i, j] = self.T[subscript][j, i] = np.abs(
                            state_vectors_exact.conj().T @ state_vector) ** 2

                    elif self.method == 'tomo':
                        tomography_labels = [''.join(x) for x in product('xyz', repeat=2)]

                        array_all = []
                        for tomography_label in tomography_labels:
                            array_raw = h5file[f"{circuit_label}/{tomography_label}"].attrs[f"counts{self.suffix}"]
                            repetitions = np.sum(array_raw)
                            ancilla_index = int(qubit_indices.ancilla.str[0], 2)
                            if REVERSE_QUBIT_ORDER:
                                array_raw = reverse_qubit_order(array_raw)
                                array_label = array_raw[ancilla_index :: 4]  / repetitions
                            else:
                                array_label = array_raw[ancilla_index * 4 : (ancilla_index + 1) * 4] / repetitions
                            array_all += list(array_label)

                        # Obtain the density matrix through tomography and optionally modify it.
                        density_matrix = two_qubit_state_tomography(array_all)
                        density_matrix = qubit_indices.system(density_matrix)
                        trace = np.trace(density_matrix)
                        density_matrix /= trace
                        if self.parameters.PROJECT_DENSITY_MATRICES:
                            density_matrix = project_density_matrix(density_matrix)
                        if self.parameters.PURIFY_DENSITY_MATRICES:
                            density_matrix = purify_density_matrix(density_matrix)
                        h5file[f'rho{self.suffix}/{subscript}{i}{j}'] = density_matrix

                        self.T[subscript][i, j] = self.T[subscript][j, i] = [
                            trace * (state_vectors_exact[:, k].conj() @ density_matrix @ state_vectors_exact[:, k]).real
                            for k in range(self.n_states[subscript[0]])]

                    if self.verbose:
                        print(f'T[{subscript}][{i}, {j}] = {self.T[subscript][i, j]}')

        # Unpack T values to N values according to Eq. (18) of Kosugi and Matsushita 2021.
        for subscript in ['n']:
            for i in range(2 * self.n_orbitals):
                for j in range(i + 1, 2 * self.n_orbitals):
                    self.N[subscript][i, j] = self.N[subscript][j, i] = \
                        np.exp(-1j * np.pi / 4) * (self.T[subscript + 'p'][i, j] - self.T[subscript + 'm'][i, j]) \
                        + np.exp(1j * np.pi / 4) * (self.T[subscript + 'p'][j, i] - self.T[subscript + 'm'][j, i])
                    
                    if self.verbose:
                        print(f"N[{subscript}][{i}, {j}] =", self.N[subscript][i, j])

        h5file.close()

    def process(self) -> None:
        """Processes both diagonal and off-diagonal results and saves data to file."""
        # Delete the group names if they already exist.
        with h5py.File(self.h5fname, 'r+') as h5file:
            for group_name in [f'psi{self.suffix}', f'rho{self.suffix}', f'amp{self.suffix}']:
                if group_name in h5file:
                    del h5file[group_name]

        # Call the private functions to process results.
        self._process_diagonal()
        self._process_off_diagonal()

        # Sum N over spins.
        self.N_summed = {subscript: self.N[subscript].reshape(
            (self.n_orbitals, 2, self.n_orbitals, 2, self.n_states[subscript])).sum((1, 3)) 
            for subscript in ['n']}

        # Save N, N_summed and T to file.
        with h5py.File(self.h5fname, 'r+') as h5file:
            for subscript, array in self.N.items():
                h5file[f'amp{self.suffix}/N_{subscript}'] = array
            for subscript, array in self.N_summed.items():
                h5file[f'amp{self.suffix}/N_summed_{subscript}'] = array
            for subscript, array in self.T.items():
                h5file[f'amp{self.suffix}/T_{subscript}'] = array


    def response_function(
        self, omegas: Sequence[float], eta: float = 0.0, save_data: bool = True
    ) -> Optional[Mapping[str, np.ndarray]]:
        """Returns the charge-charge response function at given frequencies.

        Args:
            omegas: The frequencies at which the response function is calculated.
            eta: The broadening factor.
            save_data: Whether to save the response function to file.

        Returns:
            chis_all (Optional): A dictionary from orbital strings to the corresponding charge-charge response functions.
        """
        # Sum over spins in N.
        N_summed = self.N['n'].reshape((self.n_orbitals, 2, self.n_orbitals, 2, self.n_states['n'])).sum((1, 3))

        chis_all = dict()
        for i in range(self.n_orbitals):
            for j in range(self.n_orbitals):  
                chis = []

                for omega in omegas:
                    chi = np.sum(N_summed[i, j] / (omega + 1j * eta - self.energies['n'] + self.energies['gs']))
                    chi += np.sum(N_summed[i, j].conj() / (-omega - 1j * eta - self.energies['n'] + self.energies['gs']))
                    chis.append(chi)
                chis = np.array(chis)

                if save_data:
                    if not os.path.exists("data"):
                        os.makedirs("data")
                    np.savetxt(
                        f"data/{self.fname}{self.suffix}_chi{i}{j}.dat",
                        np.vstack((omegas, chis.real, chis.imag)).T)
                else:
                    chis_all[f'{i}{j}'] = chis

        if not save_data:
            return chis_all
