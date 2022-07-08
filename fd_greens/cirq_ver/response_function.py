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
from .general_utils import project_density_matrix, purify_density_matrix, quantum_state_tomography, reverse_qubit_order
from .helpers import save_to_hdf5

np.set_printoptions(precision=6, suppress=True)

class ResponseFunction:
    """Frequency-domain charge-charge response function."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        fname: str = 'lih_resp_expt',
        suffix: str = '',
        method: str = 'exact',
        verbose: bool = True,
        fname_exact: Optional[str] = None
    ) -> None:
        """Initializes a ``ResponseFunction`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
            fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
            method: The method for extracting transition amplitudes.
            verbose: Whether to print out observable values.
            fname_exact: The exact HDF5 file name, if USE_EXACT_TRACES Is set to True.
        """
        assert method in ["exact", "tomo", "alltomo"]

        # Input attributes.
        self.hamiltonian = hamiltonian
        self.fname = fname
        self.suffix = suffix
        self.method = method
        self.verbose = verbose
        self.fname_exact = fname_exact

        # Load error mitigation parameters.
        self.parameters = ErrorMitigationParameters()
        self.parameters.write(fname)
        if method == 'tomo' and self.parameters.USE_EXACT_TRACES:
            assert fname_exact is not None

        # Load energies and state vectors from HDF5 file.
        with h5py.File(self.fname + '.h5', 'r') as h5file:
            self.energies = {'gs': h5file['gs/energy'][()], 'n': h5file['es/energies'][:]}
            self.state_vectors = {'n': h5file['es/states'][:]}

        # Derived attributes.
        method_indices_pairs = get_method_indices_pairs('')
        self.n_states = {'n': self.state_vectors['n'].shape[1]}
        self.n_spatial_orbitals = len(hamiltonian.active_indices)
        self.n_spin_orbitals = 2 * self.n_spatial_orbitals
        self.orbital_labels = list(product(range(self.n_spatial_orbitals), ['u', 'd']))
        self.n_system_qubits = self.n_spin_orbitals - method_indices_pairs.n_tapered
        self.qubit_indices_dict = QubitIndices.get_excited_qubit_indices_dict(
            self.n_spin_orbitals, method_indices_pairs)

        # Initialize array quantities N and T.
        self.subscripts_diagonal = ['n']
        self.subscripts_off_diagonal = ['np', 'nm']
        self.N = dict()
        self.T = dict()
        for subscript in self.subscripts_diagonal:
            self.N[subscript] = np.zeros(
                (self.n_spin_orbitals, self.n_spin_orbitals, self.n_states[subscript]), dtype=complex)
        for subscript in self.subscripts_off_diagonal:
            self.T[subscript] = np.zeros(
                (self.n_spin_orbitals, self.n_spin_orbitals, self.n_states[subscript[0]]), dtype=complex)

    def _process_diagonal(self) -> None:
        """Processes diagonal transition amplitudes results."""
        h5file = h5py.File(self.fname + '.h5', 'r+')
        for i in range(self.n_spin_orbitals):
            m, s = self.orbital_labels[i]
            circuit_label = f'circ{m}{s}'

            for subscript in self.subscripts_diagonal:
                qubit_indices = self.qubit_indices_dict[subscript]
                state_vectors_exact = self.state_vectors[subscript]

                # Define dataset names for convenience.
                trace_dsetname = f"trace{self.suffix}/{subscript}{i}"
                psi_dsetname = f"psi{self.suffix}/{subscript}{i}"
                rho_dsetname = f"rho{self.suffix}/{subscript}{i}"

                if self.method == 'exact':
                    state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f'psi{self.suffix}']
                    if REVERSE_QUBIT_ORDER:
                        state_vector = reverse_qubit_order(state_vector)
                    state_vector = qubit_indices(state_vector)

                    save_to_hdf5(h5file, trace_dsetname, np.linalg.norm(state_vector) ** 2)
                    save_to_hdf5(h5file, psi_dsetname, state_vector)

                    self.N[subscript][i, i] = np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                else:
                    if self.method == 'tomo':
                        # Tomograph to obtain the density matrix.
                        density_matrix = quantum_state_tomography(
                            h5file, n_qubits=self.n_system_qubits, circuit_label=circuit_label, 
                            suffix=self.suffix, ancilla_index=int(qubit_indices.ancilla.str[0], 2))
                        
                        # Modify the density matrix due to physical constraints.
                        trace = np.trace(density_matrix).real
                        density_matrix /= trace
                        if self.parameters.PROJECT_DENSITY_MATRICES:
                            density_matrix = project_density_matrix(density_matrix)
                        if self.parameters.PURIFY_DENSITY_MATRICES:
                            density_matrix = purify_density_matrix(density_matrix)
                        density_matrix = qubit_indices.system(density_matrix)

                    elif self.method == 'alltomo':
                        # Tomography to obtain the density matrix.
                        density_matrix = quantum_state_tomography(
                            h5file, n_qubits=self.n_system_qubits + 1, 
                            circuit_label=circuit_label, suffix=self.suffix)
                        
                        # Modify the density matrix due to physical constraints.
                        if self.parameters.PROJECT_DENSITY_MATRICES:
                            density_matrix = project_density_matrix(density_matrix)
                        if self.parameters.PURIFY_DENSITY_MATRICES:
                            density_matrix = purify_density_matrix(density_matrix)
                        density_matrix = qubit_indices(density_matrix)
                        trace = np.trace(density_matrix).real
                        density_matrix /= trace

                    # Save the density matrix and its trace to HDF5 file.
                    save_to_hdf5(h5file, trace_dsetname, trace)
                    save_to_hdf5(h5file, rho_dsetname, density_matrix)

                    if self.parameters.USE_EXACT_TRACES:
                        with h5py.File(self.fname_exact + '.h5', 'r') as h5file_exact:
                            trace = h5file_exact[trace_dsetname][()]

                    N_element = []
                    for j in range(self.n_states[subscript]):
                        N_element.append(trace * (
                            state_vectors_exact[:, j].conj()
                            @ density_matrix
                            @ state_vectors_exact[:, j]
                        ).real)
                    self.N[subscript][i, i] = N_element
                
                if self.verbose:
                    print(f'N[{subscript}][{i}, {i}] = {self.N[subscript][i, i]}')

        h5file.close()

    def _process_off_diagonal(self) -> None:
        """Processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.fname + '.h5', "r+")

        for i in range(self.n_spin_orbitals):
            m, s = self.orbital_labels[i]
            for j in range(i + 1, self.n_spin_orbitals):
                m_, s_ = self.orbital_labels[j]
                circuit_label = f'circ{m}{s}{m_}{s_}'

                for subscript in self.subscripts_off_diagonal:
                    qubit_indices = self.qubit_indices_dict[subscript]
                    state_vectors_exact = self.state_vectors[subscript[0]]

                    # Define dataset names for convenience.
                    trace_dsetname = f"trace{self.suffix}/{subscript}{i}{j}"
                    psi_dsetname = f"psi{self.suffix}/{subscript}{i}{j}"
                    rho_dsetname = f"rho{self.suffix}/{subscript}{i}{j}"

                    if self.method == 'exact':
                        state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f'psi{self.suffix}']
                        if REVERSE_QUBIT_ORDER:
                            state_vector = reverse_qubit_order(state_vector)
                        state_vector = qubit_indices(state_vector)

                        save_to_hdf5(h5file, trace_dsetname, np.linalg.norm(state_vector) ** 2)
                        save_to_hdf5(h5file, psi_dsetname, state_vector)

                        self.T[subscript][i, j] = self.T[subscript][j, i] = np.abs(
                            state_vectors_exact.conj().T @ state_vector) ** 2

                    else:
                        if self.method == 'tomo':
                            # Tomograph the density matrix.
                            density_matrix = quantum_state_tomography(
                                h5file, n_qubits=self.n_system_qubits, circuit_label=circuit_label,
                                suffix=self.suffix, ancilla_index=int(qubit_indices.ancilla.str[0], 2))
                            
                            # Normalize then optionally modify the density matrix.
                            trace = np.trace(density_matrix).real
                            density_matrix /= trace
                            if self.parameters.PROJECT_DENSITY_MATRICES:
                                density_matrix = project_density_matrix(density_matrix)
                            if self.parameters.PURIFY_DENSITY_MATRICES:
                                density_matrix = purify_density_matrix(density_matrix)
                            density_matrix = qubit_indices.system(density_matrix)

                        elif self.method == 'alltomo':
                            # Tomography to get the density matrix.
                            density_matrix = quantum_state_tomography(
                                h5file, n_qubits=self.n_system_qubits + 2, 
                                circuit_label=circuit_label, suffix=self.suffix)

                            # Optionally modify then normalize the density matrix.
                            if self.parameters.PROJECT_DENSITY_MATRICES:
                                density_matrix = project_density_matrix(density_matrix)
                            if self.parameters.PURIFY_DENSITY_MATRICES:
                                density_matrix = purify_density_matrix(density_matrix)
                            density_matrix = qubit_indices(density_matrix)
                            trace = np.trace(density_matrix).real
                            density_matrix /= trace

                        # Save the density matrix and its trace to HDF5 file.
                        save_to_hdf5(h5file, trace_dsetname, trace)
                        save_to_hdf5(h5file, rho_dsetname, density_matrix)

                        if self.parameters.USE_EXACT_TRACES:
                            with h5py.File(self.fname_exact + '.h5', 'r') as h5file_exact:
                                trace = h5file_exact[trace_dsetname][()]
                        
                        T_element = []
                        for k in range(self.n_states[subscript[0]]):
                            T_element.append(trace * (
                                state_vectors_exact[:, k].conj() 
                                @ density_matrix 
                                @ state_vectors_exact[:, k]
                            ).real)
                        self.T[subscript][i, j] = self.T[subscript][j, i] = T_element

                    if self.verbose:
                        print(f'T[{subscript}][{i}, {j}] = {self.T[subscript][i, j]}')

        # Unpack T values to N values according to Eq. (18) of Kosugi and Matsushita 2021.
        for subscript in ['n']:
            for i in range(self.n_spin_orbitals):
                for j in range(i + 1, self.n_spin_orbitals):
                    self.N[subscript][i, j] = self.N[subscript][j, i] = \
                        np.exp(-1j * np.pi / 4) * (self.T[subscript + 'p'][i, j] - self.T[subscript + 'm'][i, j]) \
                        + np.exp(1j * np.pi / 4) * (self.T[subscript + 'p'][j, i] - self.T[subscript + 'm'][j, i])
                    
                    if self.verbose:
                        print(f"N[{subscript}][{i}, {j}] =", self.N[subscript][i, j])

        h5file.close()

    def process(self) -> None:
        """Processes both diagonal and off-diagonal results and saves data to file."""
        # Call the private functions to process results.
        self._process_diagonal()
        self._process_off_diagonal()

        # Sum N over spins.
        self.N_summed = dict()
        for subscript in self.subscripts_diagonal:
            self.N_summed[subscript] = self.N[subscript].reshape(
                (self.n_spatial_orbitals, 2, self.n_spatial_orbitals, 2, self.n_states[subscript])
            ).sum((1, 3))

        # Save N, N_summed and T to HDF5 file.
        with h5py.File(self.fname + '.h5', 'r+') as h5file:
            for subscript in self.subscripts_diagonal:
                save_to_hdf5(h5file, f"amp{self.suffix}/N_{subscript}", self.N[subscript])
                save_to_hdf5(h5file, f"amp{self.suffix}/N_summed_{subscript}", self.N_summed[subscript])
            for subscript in self.subscripts_off_diagonal:
                save_to_hdf5(h5file, f"amp{self.suffix}/T_{subscript}", self.T[subscript])

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
        chis_all = dict()
        for i in range(self.n_spatial_orbitals):
            for j in range(self.n_spatial_orbitals):  
                chis = []

                for omega in omegas:
                    chi = np.sum(self.N_summed['n'][i, j]
                        / (omega + 1j * eta - self.energies['n'] + self.energies['gs'])) \
                        + np.sum(self.N_summed['n'][i, j].conj()
                        / (-omega - 1j * eta - self.energies['n'] + self.energies['gs']))
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
