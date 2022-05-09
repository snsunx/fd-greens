"""
========================================================
Green's Function (:mod:`fd_greens.main.greens_function`)
========================================================
"""

import os
from typing import Union, Sequence, Optional
from scipy.special import binom

import h5py
import numpy as np
from itertools import product

from .parameters import method_indices_pairs
from .molecular_hamiltonian import MolecularHamiltonian
from .qubit_indices import get_qubit_indices_dict
from ..utils import get_overlap, basis_matrix


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

        # TODO: Get this 4 from the Hamiltonian.
        self.qubit_indices_dict = get_qubit_indices_dict(4, spin, method_indices_pairs)

        # TODO: Put e_orb in self_energy.
        e_orb = np.diag(self.hamiltonian.molecule.orbital_energies)
        act_inds = self.hamiltonian.act_inds
        self.e_orb = e_orb[act_inds][:, act_inds]

        # TODO: Think about how to handle this.
        self.n_sys = 2

        # Load energies and state vectors from HDF5 file.
        h5file = h5py.File(h5fname + ".h5", "r")
        self.energies = {"gs": h5file["gs/energy"][()], 
                         "e": h5file["es/energies_e"][:], "h": h5file["es/energies_h"][:]}
        self.state_vectors = {"e": h5file["es/states_e"][:], "h": h5file["es/states_h"][:]}
        h5file.close()

        # Initialize array quantities B, D and G.
        self.n_states = {"e": self.state_vectors["e"].shape[1], "h": self.state_vectors["h"].shape[1]}
        self.n_orbitals = len(self.hamiltonian.active_indices)
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
            circuit_label = f"circ{m}{self.spin}/transpiled"
            if self.method == "exact":
                state_vector = h5file[circuit_label].attrs[f"psi{self.suffix}"]
                for subscript in ["e", "h"]:
                    # XXX: I don't thik the ::-1 is correct, but it matches the qiskit implementation.
                    state_vector_subscript = self.qubit_indices_dict[subscript](state_vector)[::-1]

                    # Obtain the B matrix elements by computing the overlaps.
                    self.B[subscript][m, m] = \
                        np.abs(self.state_vectors[subscript].conj().T @ state_vector_subscript) ** 2
                    if self.verbose:
                        print(f"B[{subscript}][{m}, {m}] = {self.B[subscript][m, m]}")

            elif self.method == "tomo":
                for subscript in ["e", "h"]:
                    # Stack counts_arr over all tomography labels together. The procedure is to
                    # first extract the raw counts_arr, slice the counts array to the specific
                    # label, and then stack the counts_arr_label to counts_arr_key.
                    tomography_labels = ["".join(x) for x in product("xyz", repeat=2)]
                    counts_arr_key = np.array([])
                    qubit_indices = self.qubit_indices_dict[subscript]

                    for tomography_label in tomography_labels:
                        counts_arr = h5file[
                            f"{circuit_label}/{tomography_label}"
                        ].attrs[f"counts{self.suffix}"]
                        start_index = int(
                            "".join([str(i) for i in qubit_indices.ancilla])[::-1], 2
                        )
                        counts_arr_label = counts_arr[start_index :: 2]
                        counts_arr_label = counts_arr_label / np.sum(counts_arr)
                        counts_arr_key = np.hstack((counts_arr_key, counts_arr_label))

                    # Obtain the density matrix from tomography. Slice the density matrix
                    # based on whether we are considering 'e' or 'h' on the system qubits.
                    density_matrix = np.linalg.lstsq(basis_matrix, counts_arr_key)[0]
                    density_matrix = density_matrix.reshape(
                        2 ** self.n_sys, 2 ** self.n_sys, order="F"
                    )
                    density_matrix = qubit_indices.system(density_matrix)

                    # Obtain the B matrix elements by computing the overlaps between
                    # the density matrix and the states from EHStatesSolver.
                    self.B[subscript][m, m] = [
                        get_overlap(self.states[subscript][:, i], density_matrix)
                        for i in range(self.n_states[subscript])
                    ]
                    if self.verbose:
                        print(f"B[{subscript}][{m}, {m}] = {self.B[subscript][m, m]}")

        h5file.close()

    def _process_off_diagonal_results(self) -> None:
        """Post-processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, "r+")

        for m in range(self.n_orbitals):
            for n in range(m + 1, self.n_orbitals):
                circuit_label = f"circ{m}{n}{self.spin}/transpiled"
                if self.method == "exact":
                    state_vector = h5file[circuit_label].attrs[f"psi{self.suffix}"]
                    for subscript in ["ep", "em", "hp", "hm"]:
                        state_vector_subscript = self.qubit_indices_dict[subscript](state_vector)[::-1]

                        # Obtain the D matrix elements by computing the overlaps.
                        self.D[subscript][m, n] = self.D[subscript][n, m] = \
                            np.abs(self.state_vectors[subscript[0]].conj().T @ state_vector_subscript) ** 2
                        if self.verbose:
                            print(f"D[{subscript}][{m}, {n}] =", self.D[subscript][m, n])

                elif self.method == "tomo":
                    for key, qind in zip(self.keys_off_diag, self.qinds_anc_off_diag):
                        # Stack counts_arr over all tomography labels together. The procedure is to
                        # first extract the raw counts_arr, slice the counts array to the specific
                        # label, and then stack the counts_arr_label to counts_arr_key.
                        counts_arr_key = np.array([])
                        for tomo_label in self.tomo_labels:
                            counts_arr = h5file[f"circ01{self.spin}/{tomo_label}"].attrs[
                                f"counts{self.suffix}"
                            ]
                            start = int("".join([str(i) for i in qind])[::-1], 2)
                            counts_arr_label = counts_arr[
                                start :: 2 ** 2
                            ]  # 4 is because 2 ** 2
                            counts_arr_label = counts_arr_label / np.sum(counts_arr)
                            counts_arr_key = np.hstack((counts_arr_key, counts_arr_label))

                        # Obtain the density matrix from tomography. Slice the density matrix
                        # based on whether we are considering 'e' or 'h' on the system qubits.
                        rho = np.linalg.lstsq(basis_matrix, counts_arr_key)[0]
                        rho = rho.reshape(2 ** self.n_sys, 2 ** self.n_sys, order="F")
                        rho = self.qinds_sys[key[0]](rho)

                        # Obtain the D matrix elements by computing the overlaps between
                        # the density matrix and the states from EHStatesSolver.
                        self.D[subscript][m, n] = self.D[subscript][n, m] = [
                            get_overlap(self.states[key[0]][:, i], rho)
                            for i in range(self.n_states[key[0]])
                        ]
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
        TrSigmas = []
        for omega in omegas:
            G = self.greens_function(omega + 1j * eta)
            G_HF = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
            for i in range(self.n_orbitals):
                G_HF[i, i] = 1 / (omega - self.e_orb[i, i])

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
