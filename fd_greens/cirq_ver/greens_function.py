"""
===================================================
Green's Function (:mod:`fd_greens.greens_function`)
===================================================
"""

import os
from typing import Sequence, Optional
from itertools import product

import h5py
import numpy as np

from .parameters import method_indices_pairs
from .molecular_hamiltonian import MolecularHamiltonian
from .qubit_indices import QubitIndices
from .parameters import basis_matrix
from .utilities import reverse_qubit_order


np.set_printoptions(precision=6)

class GreensFunction:
    """Frequency-domain Green's function."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        h5fname: str = "lih",
        suffix: str = "",
        spin: str = "d",
        method: str = "exact",
        verbose: bool = True,
    ) -> None:
        """Initializes a ``GreensFunction`` object.
        
        Args:
            h5fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
        """
        self.hamiltonian = hamiltonian
        self.h5fname = h5fname + ".h5"
        self.datfname = h5fname + suffix
        self.suffix = suffix
        self.spin = spin
        self.method = method
        self.verbose = verbose

        # Load energies and state vectors from HDF5 file.
        with h5py.File(h5fname + ".h5", "r") as h5file:
            self.energies = {"gs": h5file["gs/energy"][()], 
                             "e": h5file["es/energies_e"][:],
                             "h": h5file["es/energies_h"][:]}
            self.state_vectors = {"e": h5file["es/states_e"][:], 
                                  "h": h5file["es/states_h"][:]}

        # Initialize array quantities B, D and G.
        self.n_states = {subscript: self.state_vectors[subscript].shape[1] for subscript in ['e', 'h']}
        self.n_orbitals = len(self.hamiltonian.active_indices)
        self.n_system_qubits = 2 * self.n_orbitals - len(dict(method_indices_pairs)['taper'])
        # self.qubit_indices_dict = get_qubit_indices_dict(2 * self.n_orbitals, spin, method_indices_pairs)

        self.qubit_indices_dict = QubitIndices.get_eh_qubit_indices_dict(
            2 * self.n_orbitals, spin, method_indices_pairs)
        self.B = {subscript: np.zeros((self.n_orbitals, self.n_orbitals, self.n_states[subscript]), dtype=complex)
                  for subscript in ["e", "h"]}
        self.D = {subscript: np.zeros((self.n_orbitals, self.n_orbitals, self.n_states[subscript[0]]), dtype=complex)
                  for subscript in ["ep", "em", "hp", "hm"]}
        self.G = {subscript: np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
                  for subscript in ["e", "h"]}
        
        self._process_diagonal_results()
        self._process_off_diagonal_results()

    def _process_diagonal_results(self) -> None:
        """Post-processes diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, "r+")

        for m in range(self.n_orbitals):
            circuit_label = f"circ{m}{self.spin}"

            for subscript in ['e', 'h']:
                qubit_indices = self.qubit_indices_dict[subscript]
                state_vectors_exact = self.state_vectors[subscript]

                if self.method == 'exact':
                    state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f'psi{self.suffix}']
                    state_vector = reverse_qubit_order(state_vector) # XXX
                    state_vector = qubit_indices(state_vector)

                    self.B[subscript][m, m] = np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                elif self.method == 'tomo':
                    tomography_labels = [''.join(x) for x in product('xyz', repeat=2)]

                    array = []
                    for tomography_label in tomography_labels:
                        array = h5file[f"{circuit_label}/{tomography_label}"].attrs[f"counts{self.suffix}"]
                        array = reverse_qubit_order(array) # XXX
                        start_index = int(qubit_indices.ancilla.str[0], 2)
                        array_label = array[start_index :: 2] / np.sum(array)
                        array += list(array_label)

                    density_matrix = np.linalg.lstsq(basis_matrix, array)[0]
                    density_matrix = density_matrix.reshape(
                        2 ** self.n_system_qubits, 2 ** self.n_system_qubits, order='F')
                    density_matrix = qubit_indices.system(density_matrix)

                    self.B[subscript][m, m] = [
                        (state_vectors_exact[:, i].conj() @ density_matrix @ state_vectors_exact[:, i]).real
                        for i in range(self.n_states[subscript])]

                if self.verbose:
                    print(f"B[{subscript}][{m}, {m}] = {self.B[subscript][m, m]}")
                    
        h5file.close()

    def _process_off_diagonal_results(self) -> None:
        """Post-processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, "r+")

        for m in range(self.n_orbitals):
            for n in range(m + 1, self.n_orbitals):
                circuit_label = f"circ{m}{n}{self.spin}"

                for subscript in ['ep', 'em', 'hp', 'hm']:
                    qubit_indices = self.qubit_indices_dict[subscript]
                    state_vectors_exact = self.state_vectors[subscript[0]]

                    if self.method == 'exact':
                        state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f"psi{self.suffix}"]
                        state_vector = reverse_qubit_order(state_vector)
                        state_vector = qubit_indices(state_vector)

                        self.D[subscript][m, n] = self.D[subscript][n, m] = \
                            np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                    elif self.method == 'tomo':
                        tomography_labels = [''.join(x) for x in product('xyz', repeat=2)]

                        array = []
                        for tomography_label in tomography_labels:
                            array = h5file[f"{circuit_label}/{tomography_label}"].attrs[f"counts{self.suffix}"]
                            array = reverse_qubit_order(array) # XXX
                            start_index = int(qubit_indices.ancilla.str[0], 2)
                            array_label = array[start_index :: 2 ** 2]  / np.sum(array)
                            array += list(array_label)

                        density_matrix = np.linalg.lstsq(basis_matrix, array)[0]
                        density_matrix = density_matrix.reshape(
                            2 ** self.n_system_qubits, 2 ** self.n_system_qubits, order='F')
                        density_matrix = qubit_indices.system(density_matrix)
                        
                        self.D[subscript][m, n] = self.D[subscript][n, m] = [
                            (state_vectors_exact[:, i].conj() @ density_matrix @ state_vectors_exact[:, i]).real
                            for i in range(self.n_states[subscript[0]])]

                    if self.verbose:
                        print(f"D[{subscript}][{m}, {n}] =", self.D[subscript][m, n])

        # Unpack D values to B values according to Eq. (18) of Kosugi and Matsushita 2020.
        for m in range(self.n_orbitals):
            for n in range(m + 1, self.n_orbitals):
                for subscript in ["e", "h"]:
                    self.B[subscript][m, n] = self.B[subscript][n, m] = \
                        np.exp(-1j * np.pi / 4) * (self.D[subscript + "p"][m, n] - self.D[subscript + "m"][m, n]) \
                        + np.exp(1j * np.pi / 4) * (self.D[subscript + "p"][n, m] - self.D[subscript + "m"][n, m])
                    if self.verbose:
                        print(f"B[{subscript}][{m}, {n}] =", self.B[subscript][m, n])

        h5file.close()

    @property
    def density_matrix(self) -> np.ndarray:
        """The density matrix obtained from the transition amplitudes."""
        rho = np.sum(self.B["h"], axis=2)
        return rho

    def greens_function(self, omega: float, eta: float = 0.0) -> np.ndarray:
        """Returns the the Green's function at given frequency and broadening.

        Args:
            omega: The frequency at which the Green's function is calculated.
            eta: The broadening factor.
        
        Returns:
            G_array: The Green's function at given frequency and broadening.
        """
        for m in range(self.n_orbitals):
            for n in range(self.n_orbitals):
                self.G["e"][m, n] = 2 * np.sum(self.B["e"][m, n]
                    / (omega + 1j * eta + self.energies["gs"] - self.energies["e"]))
                self.G["h"][m, n] = 2 * np.sum(self.B["h"][m, n]
                    / (omega + 1j * eta - self.energies["gs"] + self.energies["h"]))
        G_array = self.G["e"] + self.G["h"]
        return G_array

    def spectral_function(
        self, omegas: Sequence[float], eta: float = 0.0, save_data: bool = True
    ) -> Optional[np.ndarray]:
        """Returns the spectral function at certain frequencies.

        Args:
            omegas: The frequencies at which the spectral function is calculated.
            eta: The broadening factor.
            save_data: Whether to save the spectral function to file.
        
        Returns:
            As: The spectral function numpy array.
        """
        A_arrays = []
        for omega in omegas:
            G_array = self.greens_function(omega, eta)
            A_array = -1 / np.pi * np.imag(np.trace(G_array))
            A_arrays.append(A_array)
        A_arrays = np.array(A_arrays)

        if save_data:
            if not os.path.exists("data"):
                os.makedirs("data")
            np.savetxt("data/" + self.datfname + "_A.dat", np.vstack((omegas, A_arrays)).T)
        else:
            return A_arrays

    def self_energy(
        self, omegas: Sequence[float], eta: float = 0.0, save_data: bool = True
    ) -> Optional[np.ndarray]:
        """Returns the trace of self-energy at frequency omega.

        Args:
            omegas: The frequencies at which the spectral function is calculated.
            eta: The imaginary part, i.e. broadening factor.
            save_data: Whether to save the spectral function to file.

        Returns:
            TrSigmas: Trace of the self-energy.
        """
        e_orb = self.hamiltonian.molecule.orbital_energies[self.hamiltonian.active_indices]

        TrSigmas = []
        for omega in omegas:
            G = self.greens_function(omega + 1j * eta)
            G_HF = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
            for i in range(self.n_orbitals):
                G_HF[i, i] = 1 / (omega - e_orb[i])

            Sigma = np.linalg.pinv(G_HF) - np.linalg.pinv(G)
            TrSigmas.append(np.trace(Sigma))
        TrSigmas = np.array(TrSigmas)

        if save_data:
            if not os.path.exists("data"):
                os.makedirs("data")
            np.savetxt(
                "data/" + self.datfname + "_TrS.dat",
                np.vstack((omegas, TrSigmas.real, TrSigmas.imag)).T,
            )
        else:
            return TrSigmas
