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
from .parameters import ErrorMitigationParameters, Z2TransformInstructions
from .general_utils import (
    project_density_matrix,
    purify_density_matrix,
    quantum_state_tomography,
)
from .helpers import save_data_to_file, save_to_hdf5

np.set_printoptions(precision=6)

class GreensFunction:
    """Frequency-domain Green's function."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        h5fname: str = "lih",
        suffix: str = "",
        method: str = "exact",
        verbose: bool = True,
        h5fname_exact: Optional[str] = None,
    ) -> None:
        """Initializes a ``GreensFunction`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
            h5fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
            method: The method used for calculating the transition amplitudes.
            verbose: Whether to print out transition amplitude values.
            h5fname_exact: The exact HDF5 file name, if USE_EXACT_TRACES is set to True.
        """
        assert method in ["exact", "tomo", "alltomo"]
        
        # Input attributes.
        self.hamiltonian = hamiltonian
        self.h5fname = h5fname
        self.suffix = suffix
        self.method = method
        self.verbose = verbose
        self.h5fname_exact = h5fname_exact

        self.subscripts_diagonal = ["e", "h"]
        self.subscripts_off_diagonal = ["ep", "em", "hp", "hm"]

        # Load error mitigation parameters.
        self.mitigation_params = ErrorMitigationParameters()
        self.mitigation_params.write(h5fname)
        if "tomo" in method and self.mitigation_params.USE_EXACT_TRACES:
            assert h5fname_exact is not None

        # Load energies and state vectors from HDF5 file.
        self.energies = dict()
        self.state_vectors = dict()
        self.n_states = dict()
        self.qubit_indices = dict()
        h5file = h5py.File(h5fname + ".h5", "r")

        for spin in ["u"]: #, "d"]:
            print(h5file)
            print(h5file.keys())
            self.energies["gs" + spin] = h5file[f"gs{spin}/energy"][()]
            # XXX: Same for "u" and "d".
            instructions = Z2TransformInstructions.get_instructions(spin)
            self.n_spatial_orbitals = len(self.hamiltonian.active_indices)
            self.n_system_qubits = 2 * self.n_spatial_orbitals - instructions.n_tapered
            self.qubit_indices[spin] = QubitIndices.get_eh_qubit_indices_dict(
                2 * self.n_spatial_orbitals, spin, instructions)

            for s in self.subscripts_diagonal:
                print("h5file =", h5file)
                self.energies[s + spin] = h5file[f"es{spin}/energies_{s}"][:]
                self.state_vectors[s + spin] = h5file[f"es{spin}/states_{s}"][:]
                self.n_states[s + spin] = self.state_vectors[s + spin].shape[1]

        h5file.close()

        # Initialize array quantities B, D and G.
        B_zeros_array = np.zeros(
            (self.n_spatial_orbitals, self.n_spatial_orbitals, self.n_states["eu"]), dtype=complex)
        D_zeros_array = np.zeros(
            (self.n_spatial_orbitals, self.n_spatial_orbitals, self.n_states["eu"]), dtype=complex)
        self.B = {"u": dict(), "d": dict()}
        self.D = {"u": dict(), "d": dict()}
        for spin in ["u"]: #, "d"]:
            for subscript in self.subscripts_diagonal:
                self.B[spin][subscript] = B_zeros_array.copy()
            for subscript in self.subscripts_off_diagonal:
                self.D[spin][subscript] = D_zeros_array.copy()

    def _process_diagonal_results(self, spin: str) -> None:
        """Processes diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname + ".h5", "r+")

        for m in range(self.n_spatial_orbitals):
            circuit_label = f"circ{m}{spin}"

            for s in self.subscripts_diagonal:
                qubit_indices = self.qubit_indices[spin][s]
                state_vectors_exact = self.state_vectors[s + spin]

                # Define dataset names for convenience.
                trace_dsetname = f"trace{spin}{self.suffix}/{s}{m}"
                psi_dsetname = f"psi{spin}{self.suffix}/{s}{m}"
                rho_dsetname = f"rho{spin}{self.suffix}/{s}{m}"
                print(f"{rho_dsetname = }")

                if self.method == 'exact':
                    state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f'psi{self.suffix}']
                    state_vector = qubit_indices(state_vector)

                    save_to_hdf5(h5file, trace_dsetname, np.linalg.norm(state_vector) ** 2)
                    save_to_hdf5(h5file, psi_dsetname, state_vector)

                    self.B[spin][s][m, m] = np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                else:
                    if self.method == 'tomo':
                        # Tomograph the density matrix.
                        density_matrix = quantum_state_tomography(
                            h5file, 
                            n_qubits=self.n_system_qubits,
                            circuit_label=circuit_label,
                            suffix=self.suffix, 
                            ancilla_index = int(qubit_indices.ancilla.str[0], 2))

                        # Optionally project or purify the density matrix.
                        trace = np.trace(density_matrix).real
                        density_matrix /= trace
                        if self.mitigation_params.PROJECT_DENSITY_MATRICES:
                            density_matrix = project_density_matrix(density_matrix)
                        if self.mitigation_params.PURIFY_DENSITY_MATRICES:
                            density_matrix = purify_density_matrix(density_matrix)
                        density_matrix = qubit_indices.system(density_matrix)
                
                    elif self.method == 'alltomo':
                        density_matrix = quantum_state_tomography(
                            h5file, 
                            n_qubits=self.n_system_qubits + 1,
                            circuit_label=circuit_label,
                            suffix=self.suffix)

                        if self.mitigation_params.PROJECT_DENSITY_MATRICES:
                            density_matrix = project_density_matrix(density_matrix)
                        if self.mitigation_params.PURIFY_DENSITY_MATRICES:
                            density_matrix = purify_density_matrix(density_matrix)
                        density_matrix = qubit_indices(density_matrix)
                        trace = np.trace(density_matrix).real
                        density_matrix /= trace

                    save_to_hdf5(h5file, trace_dsetname, trace)
                    save_to_hdf5(h5file, rho_dsetname, density_matrix)

                    if self.mitigation_params.USE_EXACT_TRACES:
                        with h5py.File(self.h5fname_exact + '.h5', 'r') as h5file_exact:
                            trace = h5file_exact[trace_dsetname][()]

                    B_element = []
                    for k in range(self.n_states[s + spin]):
                        B_element.append(trace * (
                            state_vectors_exact[:, k].conj()
                            @ density_matrix
                            @ state_vectors_exact[:, k]
                        ).real)
                    self.B[spin][s][m, m] = B_element

                if self.verbose:
                    print(f"# B[{spin}][{s}][{m}, {m}] = {self.B[spin][s][m, m]}")
                    
        h5file.close()

    def _process_off_diagonal_results(self, spin: str) -> None:
        """Processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname + ".h5", "r+")

        for m in range(self.n_spatial_orbitals):
            for n in range(m + 1, self.n_spatial_orbitals):
                circuit_label = f"circ{m}{n}{spin}"

                for s in self.subscripts_off_diagonal:
                    qubit_indices = self.qubit_indices[spin][s]
                    state_vectors_exact = self.state_vectors[s[0] + spin]

                    trace_dsetname = f"trace{spin}{self.suffix}/{s}{m}{n}"
                    psi_dsetname = f"psi{spin}{self.suffix}/{s}{m}{n}"
                    rho_dsetname = f"rho{spin}{self.suffix}/{s}{m}{n}"

                    if self.method == 'exact':
                        state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f"psi{self.suffix}"]
                        state_vector = qubit_indices(state_vector)
                    
                        save_to_hdf5(h5file, trace_dsetname, np.linalg.norm(state_vector) ** 2)
                        save_to_hdf5(h5file, psi_dsetname, state_vector)

                        self.D[spin][s][m, n] = self.D[spin][s][n, m] = \
                            np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                    else:
                        if self.method == "tomo":
                            # Tomograph the density matrix.
                            density_matrix = quantum_state_tomography(
                                h5file,
                                n_qubits=self.n_system_qubits,
                                circuit_label=circuit_label,
                                suffix=self.suffix,
                                ancilla_index=int(qubit_indices.ancilla.str[0], 2))

                            # Optionally project or purify the density matrix.
                            trace = np.trace(density_matrix).real
                            density_matrix /= trace
                            if self.mitigation_params.PROJECT_DENSITY_MATRICES:
                                density_matrix = project_density_matrix(density_matrix)
                            if self.mitigation_params.PURIFY_DENSITY_MATRICES:
                                density_matrix = purify_density_matrix(density_matrix)
                            density_matrix = qubit_indices.system(density_matrix)

                        elif self.method == "alltomo":
                            density_matrix = quantum_state_tomography(
                                h5file,
                                n_qubits=self.n_system_qubits + 2,
                                circuit_label=circuit_label,
                                suffix=self.suffix)

                            if self.mitigation_params.PROJECT_DENSITY_MATRICES:
                                density_matrix = project_density_matrix(density_matrix)
                            if self.mitigation_params.PURIFY_DENSITY_MATRICES:
                                density_matrix = purify_density_matrix(density_matrix)
                            density_matrix = qubit_indices(density_matrix)
                            trace = np.trace(density_matrix).real
                            density_matrix /= trace

                        save_to_hdf5(h5file, trace_dsetname, trace)
                        save_to_hdf5(h5file, rho_dsetname, density_matrix)

                        if self.mitigation_params.USE_EXACT_TRACES:
                            with h5py.File(self.h5fname_exact + '.h5', 'r') as h5file_exact:
                                trace = h5file_exact[trace_dsetname][()]
                    
                        D_element = []
                        for k in range(self.n_states[s[0] + spin]):
                            D_element.append(trace * (
                                state_vectors_exact[:, k].conj()
                                @ density_matrix
                                @ state_vectors_exact[:, k]).real)
                        self.D[spin][s][m, n] = self.D[spin][s][n, m] = D_element

                    if self.verbose:
                        print(f"# D[{spin}][{s}][{m}, {n}] =", self.D[spin][s][m, n])

        # Unpack D values to B values according to Eq. (18) of Kosugi and Matsushita 2020.
        for m in range(self.n_spatial_orbitals):
            for n in range(m + 1, self.n_spatial_orbitals):
                for s in self.subscripts_diagonal:
                    self.B[spin][s][m, n] = self.B[spin][s][n, m] = \
                        np.exp(-1j * np.pi / 4) * (self.D[spin][s + "p"][m, n] - self.D[spin][s + "m"][m, n]) \
                        + np.exp(1j * np.pi / 4) * (self.D[spin][s + "p"][n, m] - self.D[spin][s + "m"][n, m])
                    
                    if self.verbose:
                        print(f"# B[{spin}][{s}][{m}, {n}] =", self.B[spin][s][m, n])

        h5file.close()

    def process(self):
        """Processes both diagonal and off-diagonal results and saves data to file."""        
        h5file = h5py.File(self.h5fname + ".h5", "r+")
        for spin in ["u"]: # , "d"]:
            self._process_diagonal_results(spin)
            self._process_off_diagonal_results(spin)
            for s, array in self.B[spin].items():
                save_to_hdf5(h5file, f"amp{spin}{self.suffix}/B_{s}", array)
            for s, array in self.D[spin].items():
                save_to_hdf5(h5file, f"amp{spin}{self.suffix}/D_{s}", array)
        h5file.close()

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

    def greens_function(self, omega: float, eta: float = 0.5) -> np.ndarray:
        """Returns the the Green's function at given frequency and broadening.

        Args:
            omega: The frequency at which the Green's function is calculated.
            eta: The broadening factor.
        
        Returns:
            G: The Green's function.
        """
        G_e = np.zeros((self.n_spatial_orbitals, self.n_spatial_orbitals), dtype=complex)
        G_h = np.zeros((self.n_spatial_orbitals, self.n_spatial_orbitals), dtype=complex)
        for spin in ["u"]: # , "d"]:
            for m in range(self.n_spatial_orbitals):
                for n in range(self.n_spatial_orbitals):
                    G_e[m, n] += np.sum(self.B[spin]["e"][m, n]
                        / (omega + 1j * eta + self.energies["gs" + spin] - self.energies["e" + spin]))
                    G_h[m, n] += np.sum(self.B[spin]["h"][m, n]
                        / (omega + 1j * eta - self.energies["gs" + spin] + self.energies["h" + spin]))
        G = G_e + G_h
        return G

    def spectral_function(
        self,
        omegas: Sequence[float],
        eta: float = 0.0,
        dirname: str = "data/obs",
        datfname: Optional[str] = None
    ) -> None:
        """Returns the spectral function at given frequencies.

        Args:
            omegas: The frequencies at which the spectral function is calculated.
            eta: The broadening factor.
            save_data: Whether to save the spectral function to file.
            dirname: Name of the directory of the data file.
            datfname: Name of the data file.
        
        Returns:
            As: The spectral function numpy array.
        """
        if datfname is None:
            datfname = f"{self.h5fname}{self.suffix}_A"
        
        As = []
        for omega in omegas:
            G = self.greens_function(omega, eta)
            A = -1 / np.pi * np.imag(np.trace(G))
            As.append(A)
        As = np.array(As)

        save_data_to_file(dirname, datfname, np.vstack((omegas, As)).T)

    def self_energy(
        self,
        omegas: Sequence[float],
        eta: float = 0.0,
        dirname: str = "data/obs",
        datfname: Optional[str] = None
    ) -> None:
        """Returns the trace of self-energy at given frequencies.

        Args:
            omegas: The frequencies at which the self-energy is calculated.
            eta: The broadening factor.
            save_data: Whether to save the self-energy to file.
            dirname: Name of the directory of the data file.
            datfname: Name of the data file.

        Returns:
            TrSigmas: Trace of the self-energy.
        """
        if datfname is None:
            datfname = f"{self.h5fname}{self.suffix}_TrSigma"
        
        TrSigmas = []
        for omega in omegas:
            G0 = self.mean_field_greens_function(omega, eta)
            G = self.greens_function(omega, eta)
            Sigma = np.linalg.pinv(G0) - np.linalg.pinv(G)
            TrSigmas.append(np.trace(Sigma))
        TrSigmas = np.array(TrSigmas)

        save_data_to_file(dirname, datfname, np.vstack((omegas, TrSigmas.real, TrSigmas.imag)).T)
