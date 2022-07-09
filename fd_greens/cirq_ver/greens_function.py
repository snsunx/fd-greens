"""
===================================================
Green's Function (:mod:`fd_greens.greens_function`)
===================================================
"""

from typing import Sequence, Optional

import h5py
import numpy as np

from .molecular_hamiltonian import MolecularHamiltonian
from .qubit_indices import QubitIndices
from .parameters import REVERSE_QUBIT_ORDER, ErrorMitigationParameters, MethodIndicesPairs
from .general_utils import (
    project_density_matrix,
    purify_density_matrix,
    quantum_state_tomography,
    reverse_qubit_order
)
from .helpers import save_data_to_file, save_to_hdf5

np.set_printoptions(precision=6)

class GreensFunction:
    """Frequency-domain Green's function."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        fname: str = "lih",
        suffix: str = "",
        spin: str = "d",
        method: str = "exact",
        verbose: bool = True,
        fname_exact: Optional[str] = None,
    ) -> None:
        """Initializes a ``GreensFunction`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
            fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
            spin: Spin of the second quantized operators. Either ``'u'`` or ``'d'``.
            method: The method used for calculating the transition amplitudes. Either ``'exact'`` or ``'tomo'``.
            verbose: Whether to print out transition amplitude values.
            fname_exact: The exact HDF5 file name, if USE_EXACT_TRACES is set tuo True.
        """
        assert spin in ['u', 'd']
        assert method in ["exact", "tomo", "alltomo"]
        
        # Input attributes.
        self.hamiltonian = hamiltonian
        self.fname = fname
        self.h5fname = fname + ".h5"
        self.suffix = suffix
        self.spin = spin
        self.method = method
        self.verbose = verbose
        self.fname_exact = fname_exact

        # Load error mitigation parameters.
        self.parameters = ErrorMitigationParameters()
        self.parameters.write(fname)
        if "tomo" in method and self.parameters.USE_EXACT_TRACES:
            assert fname_exact is not None

        # Load energies and state vectors from HDF5 file.
        with h5py.File(self.fname + '.h5', "r") as h5file:
            self.energies = {"gs": h5file["gs/energy"][()], 
                             "e": h5file["es/energies_e"][:],
                             "h": h5file["es/energies_h"][:]}
            self.state_vectors = {"e": h5file["es/states_e"][:], 
                                  "h": h5file["es/states_h"][:]}

        # Derived attributes.
        method_indices_pairs = MethodIndicesPairs.get_pairs(spin)
        self.n_states = {subscript: self.state_vectors[subscript].shape[1] for subscript in ['e', 'h']}
        self.n_spatial_orbitals = len(self.hamiltonian.active_indices)
        self.n_system_qubits = 2 * self.n_spatial_orbitals - method_indices_pairs.n_tapered

        self.qubit_indices_dict = QubitIndices.get_eh_qubit_indices_dict(
            2 * self.n_spatial_orbitals, spin, method_indices_pairs)

        # Initialize array quantities B, D and G.
        self.subscripts_diagonal = ["e", "h"]
        self.subscripts_off_diagonal = ["ep", "em", "hp", "hm"]
        self.B = dict()
        self.D = dict()
        self.G = dict()
        for subscript in self.subscripts_diagonal:
            self.B[subscript] = np.zeros(
                (self.n_spatial_orbitals, self.n_spatial_orbitals, self.n_states[subscript]), dtype=complex)
            self.G[subscript] = np.zeros((self.n_spatial_orbitals, self.n_spatial_orbitals), dtype=complex)
        for subscript in self.subscripts_off_diagonal:
            self.D[subscript] = np.zeros(
                (self.n_spatial_orbitals, self.n_spatial_orbitals, self.n_states[subscript[0]]), dtype=complex)

    def _process_diagonal_results(self) -> None:
        """Processes diagonal transition amplitude results."""
        h5file = h5py.File(self.fname + '.h5', "r+")

        for m in range(self.n_spatial_orbitals):
            circuit_label = f"circ{m}{self.spin}"

            for subscript in self.subscripts_diagonal:
                qubit_indices = self.qubit_indices_dict[subscript]
                state_vectors_exact = self.state_vectors[subscript]

                # Define dataset names for convenience.
                trace_dsetname = f"trace{self.suffix}/{subscript}{m}{self.spin}"
                psi_dsetname = f"psi{self.suffix}/{subscript}{m}{self.spin}"
                rho_dsetname = f"rho{self.suffix}/{subscript}{m}{self.spin}"

                if self.method == 'exact':
                    state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f'psi{self.suffix}']
                    if REVERSE_QUBIT_ORDER:
                        state_vector = reverse_qubit_order(state_vector)
                    state_vector = qubit_indices(state_vector)

                    save_to_hdf5(h5file, trace_dsetname, np.linalg.norm(state_vector) ** 2)
                    save_to_hdf5(h5file, psi_dsetname, state_vector)

                    self.B[subscript][m, m] = np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                else:
                    if self.method == 'tomo':
                        # Tomograph the density matrix.
                        density_matrix = quantum_state_tomography(
                            h5file, 
                            n_qubits=self.n_system_qubits,
                            circuit_label=circuit_label,
                            suffix=self.suffix, 
                            ancilla_index = int(qubit_indices.ancilla.str[0], 2),
                            reverse=REVERSE_QUBIT_ORDER)

                        # Optionally project or purify the density matrix.
                        trace = np.trace(density_matrix).real
                        density_matrix /= trace
                        if self.parameters.PROJECT_DENSITY_MATRICES:
                            density_matrix = project_density_matrix(density_matrix)
                        if self.parameters.PURIFY_DENSITY_MATRICES:
                            density_matrix = purify_density_matrix(density_matrix)
                        density_matrix = qubit_indices.system(density_matrix)
                
                    elif self.method == 'alltomo':
                        density_matrix = quantum_state_tomography(
                            h5file, 
                            n_qubits=self.n_system_qubits + 1,
                            circuit_label=circuit_label,
                            suffix=self.suffix, 
                            reverse=REVERSE_QUBIT_ORDER)

                        if self.parameters.PROJECT_DENSITY_MATRICES:
                            density_matrix = project_density_matrix(density_matrix)
                        if self.parameters.PURIFY_DENSITY_MATRICES:
                            density_matrix = purify_density_matrix(density_matrix)
                        density_matrix = qubit_indices(density_matrix)
                        trace = np.trace(density_matrix).real
                        density_matrix /= trace

                    save_to_hdf5(h5file, trace_dsetname, trace)
                    save_to_hdf5(h5file, rho_dsetname, density_matrix)

                    if self.parameters.USE_EXACT_TRACES:
                        with h5py.File(self.fname_exact + '.h5', 'r') as h5file_exact:
                            trace = h5file_exact[trace_dsetname][()]

                    B_element = []
                    for k in range(self.n_states[subscript]):
                        B_element.append(trace * (
                            state_vectors_exact[:, k].conj()
                            @ density_matrix
                            @ state_vectors_exact[:, k]
                        ).real)
                    self.B[subscript][m, m] = B_element

                if self.verbose:
                    print(f"B[{subscript}][{m}, {m}] = {self.B[subscript][m, m]}")
                    
        h5file.close()

    def _process_off_diagonal_results(self) -> None:
        """Processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.fname + '.h5', "r+")

        for m in range(self.n_spatial_orbitals):
            for n in range(m + 1, self.n_spatial_orbitals):
                circuit_label = f"circ{m}{n}{self.spin}"

                for subscript in self.subscripts_off_diagonal:
                    qubit_indices = self.qubit_indices_dict[subscript]
                    state_vectors_exact = self.state_vectors[subscript[0]]

                    trace_dsetname = f"trace{self.suffix}/{subscript}{m}{n}{self.spin}"
                    psi_dsetname = f"psi{self.suffix}/{subscript}{m}{n}{self.spin}"
                    rho_dsetname = f"rho{self.suffix}/{subscript}{m}{n}{self.spin}"

                    if self.method == 'exact':
                        state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f"psi{self.suffix}"]
                        if REVERSE_QUBIT_ORDER:
                            state_vector = reverse_qubit_order(state_vector)
                        state_vector = qubit_indices(state_vector)
                    
                        save_to_hdf5(h5file, trace_dsetname, np.linalg.norm(state_vector) ** 2)
                        save_to_hdf5(h5file, psi_dsetname, state_vector)

                        self.D[subscript][m, n] = self.D[subscript][n, m] = \
                            np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                    else:
                        if self.method == "tomo":
                            # Tomograph the density matrix.
                            density_matrix = quantum_state_tomography(
                                h5file,
                                n_qubits=self.n_system_qubits,
                                circuit_label=circuit_label,
                                suffix=self.suffix,
                                ancilla_index=int(qubit_indices.ancilla.str[0], 2),
                                reverse=REVERSE_QUBIT_ORDER)

                            # Optionally project or purify the density matrix
                            trace = np.trace(density_matrix)
                            density_matrix /= trace
                            if self.parameters.PROJECT_DENSITY_MATRICES:
                                density_matrix = project_density_matrix(density_matrix)
                            if self.parameters.PURIFY_DENSITY_MATRICES:
                                density_matrix = purify_density_matrix(density_matrix)
                            density_matrix = qubit_indices.system(density_matrix)

                        elif self.method == "alltomo":
                            density_matrix = quantum_state_tomography(
                                h5file,
                                n_qubits=self.n_system_qubits + 2,
                                circuit_label=circuit_label,
                                suffix=self.suffix,
                                reverse=REVERSE_QUBIT_ORDER)

                            if self.parameters.PROJECT_DENSITY_MATRICES:
                                density_matrix = project_density_matrix(density_matrix)
                            if self.parameters.PURIFY_DENSITY_MATRICES:
                                density_matrix = purify_density_matrix(density_matrix)
                            density_matrix = qubit_indices(density_matrix)
                            trace = np.trace(density_matrix).real
                            density_matrix /= trace

                    save_to_hdf5(h5file, trace_dsetname, trace)
                    save_to_hdf5(h5file, rho_dsetname, density_matrix)

                    if self.parameters.USE_EXACT_TRACES:
                        with h5py.File(self.fname_exact + '.h5', 'r') as h5file_exact:
                            trace = h5file_exact[trace_dsetname][()]
                    
                    D_element = []
                    for k in range(self.n_states[subscript[0]]):
                        D_element.append(trace * (
                            state_vectors_exact[:, k].conj()
                            @ density_matrix
                            @ state_vectors_exact[:, k]
                        ).real)
                    self.D[subscript][m, n] = self.D[subscript][n, m] = D_element

                    if self.verbose:
                        print(f"D[{subscript}][{m}, {n}] =", self.D[subscript][m, n])

        # Unpack D values to B values according to Eq. (18) of Kosugi and Matsushita 2020.
        for m in range(self.n_spatial_orbitals):
            for n in range(m + 1, self.n_spatial_orbitals):
                for subscript in self.subscripts_diagonal:
                    self.B[subscript][m, n] = self.B[subscript][n, m] = \
                        np.exp(-1j * np.pi / 4) * (self.D[subscript + "p"][m, n] - self.D[subscript + "m"][m, n]) \
                        + np.exp(1j * np.pi / 4) * (self.D[subscript + "p"][n, m] - self.D[subscript + "m"][n, m])
                    
                    if self.verbose:
                        print(f"B[{subscript}][{m}, {n}] =", self.B[subscript][m, n])

        h5file.close()

    def process(self):
        """Processes both diagonal and off-diagonal results and saves data to file."""
        # Call the private functions to process results.
        self._process_diagonal_results()
        self._process_off_diagonal_results()

        # Save B and D to HDF5 file.
        with h5py.File(self.fname + '.h5', 'r+') as h5file:
            for subscript, array in self.B.items():
                save_to_hdf5(h5file, f"amp{self.suffix}/B_{subscript}", array)
            for subscript, array in self.D.items():
                save_to_hdf5(h5file, f"amp{self.suffix}/D_{subscript}", array)

    def mean_field_greens_function(self, omega: float, eta: float = 0.0) -> np.ndarray:
        """Returns the mean-field Green's function.
        
        Args:
            omega: The frequency at which the mean-field Green's function is calculated.
            eta: The broadening factor.

        Returns:
            G0: The mean-field Green's function.
        """
        orbital_energies = self.hamiltonian.orbital_energies[self.hamiltonian.active_indices]

        G0 = np.zeros((self.n_spatial_orbitals, self.n_spatial_orbitals), dtype=complex)
        for i in range(self.n_spatial_orbitals):
            G0[i, i] = 1 / (omega + 1j * eta - orbital_energies[i])
        return G0

    def greens_function(self, omega: float, eta: float = 0.0) -> np.ndarray:
        """Returns the the Green's function at given frequency and broadening.

        Args:
            omega: The frequency at which the Green's function is calculated.
            eta: The broadening factor.
        
        Returns:
            G: The Green's function.
        """
        for m in range(self.n_spatial_orbitals):
            for n in range(self.n_spatial_orbitals):
                # 2 is for both up and down spins.
                self.G["e"][m, n] = 2 * np.sum(self.B["e"][m, n]
                    / (omega + 1j * eta + self.energies["gs"] - self.energies["e"]))
                self.G["h"][m, n] = 2 * np.sum(self.B["h"][m, n]
                    / (omega + 1j * eta - self.energies["gs"] + self.energies["h"]))
        G = self.G["e"] + self.G["h"]
        return G

    def spectral_function(self, omegas: Sequence[float], eta: float = 0.0) -> None:
        """Returns the spectral function at given frequencies.

        Args:
            omegas: The frequencies at which the spectral function is calculated.
            eta: The broadening factor.
            save_data: Whether to save the spectral function to file.
        
        Returns:
            As: The spectral function numpy array.
        """
        As = []
        for omega in omegas:
            G = self.greens_function(omega, eta)
            A = -1 / np.pi * np.imag(np.trace(G))
            As.append(A)
        As = np.array(As)

        save_data_to_file("data", f"{self.fname}{self.suffix}_A", np.vstack(omegas, As).T)

    def self_energy(self, omegas: Sequence[float], eta: float = 0.0) -> None:
        """Returns the trace of self-energy at given frequencies.

        Args:
            omegas: The frequencies at which the self-energy is calculated.
            eta: The broadening factor.
            save_data: Whether to save the self-energy to file.

        Returns:
            TrSigmas: Trace of the self-energy.
        """
        TrSigmas = []
        for omega in omegas:
            G0 = self.mean_field_greens_function(omega, eta)
            G = self.greens_function(omega, eta)

            Sigma = np.linalg.pinv(G0) - np.linalg.pinv(G)
            TrSigmas.append(np.trace(Sigma))
        TrSigmas = np.array(TrSigmas)

        save_data_to_file(
            "data", f"{self.fname}{self.suffix}_TrSigma",
            np.vstack((omegas, TrSigmas.real, TrSigmas.imag)).T)
