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

from fd_greens.cirq_ver.main.qubit_indices import get_qubit_indices_dict

from .molecular_hamiltonian import MolecularHamiltonian
from ..utils import write_hdf5, get_overlap, basis_matrix


class GreensFunction:
    """A class to calculate frequency-domain Green's function."""

    def __init__(
        self,
        h: MolecularHamiltonian,
        h5fname: str = "lih",
        suffix: str = "",
        spin: str = "d",
        method: str = "exact",
        verbose: bool = True,
    ) -> None:
        """Initializes a GreensFunction object.
        
        Args:
            h5fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
        """
        self.datfname = h5fname + suffix
        self.h = h
        h5file = h5py.File(h5fname + ".h5", "r")
        self.h5file = h5file
        self.h5fname = h5fname + ".h5"
        self.energy_gs = h5file["gs/energy"]
        self.energies_es = dict()
        self.energies_es["e"] = h5file["es/energies_e"]
        self.energies_es["h"] = h5file["es/energies_h"]
        self.verbose = verbose
        self.method = method
        self.suffix = suffix
        self.spin = spin

        self.qubit_indices_dict = get_qubit_indices_dict(spin)

        e_orb = np.diag(self.h.molecule.orbital_energies)
        act_inds = self.h.act_inds
        self.e_orb = e_orb[act_inds][:, act_inds]

        # Hardcoded problem parameters.
        self.n_anc_diag = 1
        self.n_anc_off_diag = 2
        self.n_sys = 2

        # Number of spatial orbitals and (N±1)-electron states.
        self.states = {"e": h5file["es/states_e"][:], "h": h5file["es/states_h"][:]}

        self.n_states = {"e": self.states["e"].shape[1], "h": self.states["h"].shape[1]}
        self.n_elec = self.h.molecule.n_electrons
        self.n_orb = len(self.h.act_inds)
        self.n_occ = self.n_elec // 2 - len(self.h.occ_inds)
        self.n_vir = self.n_orb - self.n_occ
        self.n_e = int(binom(2 * self.n_orb, 2 * self.n_occ + 1)) // 2
        self.n_h = int(binom(2 * self.n_orb, 2 * self.n_occ - 1)) // 2

        self.B = {
            subscript: np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
            for subscript in ["e", "h"]
        }
        self.D = {
            subscript: np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
            for subscript in ["ep", "em", "hp", "hm"]
        }
        self.h = h

        self.n_orb = self.B["e"].shape[0]

        h5file.close()

        # self._process_diagonal_results()

    def _process_diagonal_results(self) -> None:
        """Post-processes diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, "r+")

        for m in range(self.n_orb):  # 0, 1
            circuit_label = f"circ{m}{self.spin}/transpiled"
            if self.method == "exact":
                state_vector = h5file[circuit_label].attrs[f"psi{self.suffix}"]
                for subscript in ["e", "h"]:
                    state_vector_subscript = self.qubit_indices_dict[subscript](
                        state_vector
                    )

                    # Obtain the B matrix elements by computing the overlaps.
                    # TODO: Can change this to get_overlap.
                    self.B[subscript][m, m] = (
                        np.abs(self.states[subscript].conj().T @ state_vector_subscript)
                        ** 2
                    )
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
                        counts_arr_label = counts_arr[
                            start_index :: 2 ** self.n_anc_diag
                        ]
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

    '''
    def _process_off_diagonal_results(self) -> None:
        """Post-processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, "r+")

        if self.method == "exact":
            psi = h5file[f"circ01{self.spin}/transpiled"].attrs[f"psi{self.suffix}"]
            # print("psi norm^2 =", np.linalg.norm(psi) ** 2)
            l = []
            for i, x in enumerate(psi):
                if abs(x) > 1e-8:
                    l.append(f"{int(bin(i)[2:]):04}")
            # print('nonzero indices =', l)

            for key in self.keys_off_diag:
                qinds = self.qinds_tot_off_diag[key]
                # print("qinds =", qinds)
                psi_key = qinds(psi)
                # print("psi_key norm^2 =", np.linalg.norm(psi_key) ** 2)

                # Obtain the D matrix elements by computing the overlaps.
                self.D[key][0, 1] = self.D[key][1, 0] = (
                    abs(self.states[key[0]].conj().T @ psi_key) ** 2
                )
                if self.verbose:
                    print(f"D[{key}][0, 1] =", self.D[key][0, 1])

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
                        start :: 2 ** self.n_anc_off_diag
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
                self.D[key][0, 1] = self.D[key][1, 0] = [
                    get_overlap(self.states[key[0]][:, i], rho)
                    for i in range(self.n_states[key[0]])
                ]
                if self.verbose:
                    print(f"D[{key}][0, 1] =", self.D[key][0, 1])

        # Unpack D values to B values based on Eq. (18) of Kosugi and Matsushita 2020.
        for key in self.keys_diag:
            self.B[key][0, 1] = self.B[key][1, 0] = np.exp(-1j * np.pi / 4) * (
                self.D[key + "p"][0, 1] - self.D[key + "m"][0, 1]
            ) + np.exp(1j * np.pi / 4) * (
                self.D[key + "p"][1, 0] - self.D[key + "m"][1, 0]
            )
            if self.verbose:
                print(f"B[{key}][0, 1] =", self.B[key][0, 1])

        h5file.close()
    '''

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
            G: The Green's function at given frequency and broadening.
        """
        G_dict = {
            subscript: np.zeros((self.n_orb, self.n_orb), dtype=complex)
            for subscript in ["e", "h"]
        }

        for m in range(self.n_orb):
            for n in range(self.n_orb):
                G_dict["e"][m, n] = 2 * np.sum(
                    self.B["e"][m, n]
                    / (omega + 1j * eta + self.energy_gs - self.energies_es["e"])
                )
                G_dict["h"][m, n] = 2 * np.sum(
                    self.B["h"][m, n]
                    / (omega + 1j * eta - self.energy_gs + self.energies_es["h"])
                )
        G_array = G_dict["e"] + G_dict["h"]
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
            np.savetxt(
                "data/" + self.datfname + "_A.dat", np.vstack((omegas, A_arrays)).T
            )
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
            G_HF = np.zeros((self.n_orb, self.n_orb), dtype=complex)
            for i in range(self.n_orb):
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
