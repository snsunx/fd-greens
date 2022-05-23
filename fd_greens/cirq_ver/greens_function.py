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

from .molecular_hamiltonian import MolecularHamiltonian
from .qubit_indices import QubitIndices
from .parameters import basis_matrix, REVERSE_QUBIT_ORDER, get_method_indices_pairs
from .utilities import reverse_qubit_order


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
    ) -> None:
        """Initializes a ``GreensFunction`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
            fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
            spin: Spin of the second quantized operators. Either ``'u'`` or ``'d'``.
            method: The method used for calculating the transition amplitudes. Either ``'exact'`` or ``'tomo'``.
            verbose: Whether to print out transition amplitude values.
        """
        assert spin in ['u', 'd']
        assert method in ['exact', 'tomo']
        
        # Input attributes.
        self.hamiltonian = hamiltonian
        self.fname = fname
        self.h5fname = fname + ".h5"
        self.suffix = suffix
        self.spin = spin
        self.method = method
        self.verbose = verbose

        # Load energies and state vectors from HDF5 file.
        with h5py.File(self.h5fname, "r") as h5file:
            self.energies = {"gs": h5file["gs/energy"][()], 
                             "e": h5file["es/energies_e"][:],
                             "h": h5file["es/energies_h"][:]}
            self.state_vectors = {"e": h5file["es/states_e"][:], 
                                  "h": h5file["es/states_h"][:]}

        # Derived attributes.
        method_indices_pairs = get_method_indices_pairs(spin)
        self.n_states = {subscript: self.state_vectors[subscript].shape[1] for subscript in ['e', 'h']}
        self.n_orbitals = len(self.hamiltonian.active_indices)
        # self.n_system_qubits = 2 * self.n_orbitals - len(dict(method_indices_pairs[spin])['taper'])
        self.n_system_qubits = 2 * self.n_orbitals - method_indices_pairs.n_tapered

        self.qubit_indices_dict = QubitIndices.get_eh_qubit_indices_dict(
            2 * self.n_orbitals, spin, method_indices_pairs)

        # Initialize array quantities B, D and G.
        self.B = {subscript: np.zeros((self.n_orbitals, self.n_orbitals, self.n_states[subscript]), dtype=complex)
                  for subscript in ["e", "h"]}
        self.D = {subscript: np.zeros((self.n_orbitals, self.n_orbitals, self.n_states[subscript[0]]), dtype=complex)
                  for subscript in ["ep", "em", "hp", "hm"]}
        self.G = {subscript: np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
                  for subscript in ["e", "h"]}
        
        self._process_diagonal_results()
        self._process_off_diagonal_results()

    def _process_diagonal_results(self) -> None:
        """Processes diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, "r+")

        for m in range(self.n_orbitals):
            circuit_label = f"circ{m}{self.spin}"

            for subscript in ['e', 'h']:
                qubit_indices = self.qubit_indices_dict[subscript]
                state_vectors_exact = self.state_vectors[subscript]

                if self.method == 'exact':
                    state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f'psi{self.suffix}']
                    if REVERSE_QUBIT_ORDER:
                        state_vector = reverse_qubit_order(state_vector)
                    state_vector = qubit_indices(state_vector)

                    self.B[subscript][m, m] = np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                elif self.method == 'tomo':
                    tomography_labels = [''.join(x) for x in product('xyz', repeat=2)]

                    array_all = []
                    for tomography_label in tomography_labels:
                        array = h5file[f"{circuit_label}/{tomography_label}"].attrs[f"counts{self.suffix}"]
                        if REVERSE_QUBIT_ORDER:
                            array = reverse_qubit_order(array)
                            start_index = int(qubit_indices.ancilla.str[0], 2)
                            array_label = array[start_index :: 2] / np.sum(array)
                            # print(list(range(start_index, len(array), 2)))
                        else:
                            indices = [''.join(l) for l in product('01', repeat=2)]
                            # print(qubit_indices.ancilla.str[0])
                            indices = [qubit_indices.ancilla.str[0] + x for x in indices]
                            indices = [int(x, 2) for x in indices]
                            print(indices)
                            array_label = array[indices] / np.sum(array)
                        array_all += list(array_label)

                    density_matrix = np.linalg.lstsq(basis_matrix, array_all)[0]
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
        """Processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, "r+")

        for m in range(self.n_orbitals):
            for n in range(m + 1, self.n_orbitals):
                circuit_label = f"circ{m}{n}{self.spin}"

                for subscript in ['ep', 'em', 'hp', 'hm']:
                    qubit_indices = self.qubit_indices_dict[subscript]
                    state_vectors_exact = self.state_vectors[subscript[0]]

                    if self.method == 'exact':
                        state_vector = h5file[f'{circuit_label}/transpiled'].attrs[f"psi{self.suffix}"]
                        if REVERSE_QUBIT_ORDER:
                            state_vector = reverse_qubit_order(state_vector)
                        state_vector = qubit_indices(state_vector)

                        self.D[subscript][m, n] = self.D[subscript][n, m] = \
                            np.abs(state_vectors_exact.conj().T @ state_vector) ** 2

                    elif self.method == 'tomo':
                        tomography_labels = [''.join(x) for x in product('xyz', repeat=2)]

                        array_all = []
                        for tomography_label in tomography_labels:
                            array = h5file[f"{circuit_label}/{tomography_label}"].attrs[f"counts{self.suffix}"]
                            if REVERSE_QUBIT_ORDER:
                                array = reverse_qubit_order(array)
                                start_index = int(qubit_indices.ancilla.str[0], 2)
                                array_label = array[start_index :: 2 ** 2]  / np.sum(array)
                                # print(list(range(start_index, len(array), 2)))
                            else:
                                indices = [''.join(l) for l in product('01', repeat=2)]
                                indices = [qubit_indices.ancilla.str[0] + x for x in indices]
                                indices = [int(x, 2) for x in indices]
                                array_label = array[indices] / np.sum(array)
                            array_all += list(array_label)

                        density_matrix = np.linalg.lstsq(basis_matrix, array_all)[0]
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

    def mean_field_greens_function(self, omega: float, eta: float = 0.0) -> np.ndarray:
        """Returns the mean-field Green's function.
        
        Args:
            omega: The frequency at which the mean-field Green's function is calculated.
            eta: The broadening factor.

        Returns:
            G0: The mean-field Green's function.
        """
        orbital_energies = self.hamiltonian.orbital_energies[self.hamiltonian.active_indices]

        G0 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        for i in range(self.n_orbitals):
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
        for m in range(self.n_orbitals):
            for n in range(self.n_orbitals):
                # 2 is for both up and down spins.
                self.G["e"][m, n] = 2 * np.sum(self.B["e"][m, n]
                    / (omega + 1j * eta + self.energies["gs"] - self.energies["e"]))
                self.G["h"][m, n] = 2 * np.sum(self.B["h"][m, n]
                    / (omega + 1j * eta - self.energies["gs"] + self.energies["h"]))
        G = self.G["e"] + self.G["h"]
        return G

    def spectral_function(
        self, omegas: Sequence[float], eta: float = 0.0, save_data: bool = True
    ) -> Optional[np.ndarray]:
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

        if save_data:
            if not os.path.exists("data"):
                os.makedirs("data")
            np.savetxt(f"data/{self.fname}{self.suffix}_A.dat", np.vstack((omegas, As)).T)
        else:
            return As

    def self_energy(
        self, omegas: Sequence[float], eta: float = 0.0, save_data: bool = True
    ) -> Optional[np.ndarray]:
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

        if save_data:
            if not os.path.exists("data"):
                os.makedirs("data")
            np.savetxt(
                f"data/{self.fname}{self.suffix}_TrSigma.dat", 
                np.vstack((omegas, TrSigmas.real, TrSigmas.imag)).T)
        else:
            return TrSigmas
