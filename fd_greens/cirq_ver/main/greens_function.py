"""
========================================================
Green's Function (:mod:`fd_greens.main.greens_function`)
========================================================
"""

import os
from typing import Union, Sequence, Optional

import h5py
import numpy as np


class GreensFunction:
    """A class to calculate frequency-domain Green's function."""

    def __init__(self, h5fname: str = "lih", suffix: str = "") -> None:
        """Initializes a GreensFunction object.
        
        Args:
            h5fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
        """
        self.datfname = h5fname + suffix

        h5file = h5py.File(h5fname + ".h5", "r")
        self.energy_gs = h5file["gs/energy"]
        self.energies_e = h5file["es/energies_e"]
        self.energies_h = h5file["es/energies_h"]
        self.B_e = h5file[f"amp/B_e{suffix}"]
        self.B_h = h5file[f"amp/B_h{suffix}"]
        self.n_orb = self.B_e.shape[0]
        self.e_orb = h5file[f"amp/e_orb{suffix}"]

    def _process_diagonal_results(self) -> None:
        """Post-processes diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, "r+")

        for m in range(self.n_orb):  # 0, 1
            if self.method == "exact":
                psi = h5file[f"circ{m}{self.spin}/transpiled"].attrs[
                    f"psi{self.suffix}"
                ]
                for key in self.keys_diag:
                    # print('key =', key)
                    psi[abs(psi) < 1e-8] = 0.0
                    # print('psi =', psi)
                    psi_key = self.qinds_tot_diag[key](psi)
                    # print('psi_key =', psi_key)
                    # print('states[key] =', self.states[key])

                    # Obtain the B matrix elements by computing the overlaps.
                    self.B[key][m, m] = np.abs(self.states[key].conj().T @ psi_key) ** 2
                    if self.verbose:
                        print(f"B[{key}][{m}, {m}] = {self.B[key][m, m]}")

            elif self.method == "tomo":
                for key, qind in zip(self.keys_diag, self.qinds_anc_diag):
                    # Stack counts_arr over all tomography labels together. The procedure is to
                    # first extract the raw counts_arr, slice the counts array to the specific
                    # label, and then stack the counts_arr_label to counts_arr_key.
                    counts_arr_key = np.array([])
                    for tomo_label in self.tomo_labels:
                        counts_arr = h5file[f"circ{m}{self.spin}/{tomo_label}"].attrs[
                            f"counts{self.suffix}"
                        ]
                        start = int("".join([str(i) for i in qind])[::-1], 2)
                        counts_arr_label = counts_arr[start :: 2 ** self.n_anc_diag]
                        counts_arr_label = counts_arr_label / np.sum(counts_arr)
                        counts_arr_key = np.hstack((counts_arr_key, counts_arr_label))

                    # Obtain the density matrix from tomography. Slice the density matrix
                    # based on whether we are considering 'e' or 'h' on the system qubits.
                    rho = np.linalg.lstsq(basis_matrix, counts_arr_key)[0]
                    rho = rho.reshape(2 ** self.n_sys, 2 ** self.n_sys, order="F")
                    rho = self.qinds_sys[key](rho)

                    # Obtain the B matrix elements by computing the overlaps between
                    # the density matrix and the states from EHStatesSolver.
                    self.B[key][m, m] = [
                        get_overlap(self.states[key][:, i], rho)
                        for i in range(self.n_states[key])
                    ]
                    if self.verbose:
                        print(f"B[{key}][{m}, {m}] = {self.B[key][m, m]}")

        h5file.close()

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

    def _save_data(self) -> None:
        """Saves transition amplitudes data to HDF5 file."""
        h5file = h5py.File(self.h5fname, "r+")

        # Saves B_e and B_h.
        write_hdf5(h5file, "amp", f"B_e{self.suffix}", self.B["e"])
        write_hdf5(h5file, "amp", f"B_h{self.suffix}", self.B["h"])

        # Saves orbital energies for calculating the self-energy.
        e_orb = np.diag(self.h.molecule.orbital_energies)
        act_inds = self.h.act_inds
        write_hdf5(h5file, "amp", f"e_orb{self.suffix}", e_orb[act_inds][:, act_inds])

        h5file.close()
        
    @property
    def density_matrix(self) -> np.ndarray:
        """The density matrix obtained from the transition amplitudes."""
        rho = np.sum(self.B_h, axis=2)
        return rho

    def greens_function(self, omega: float, eta: float = 0.0) -> np.ndarray:
        """Returns the the Green's function at frequency omega.

        Args:
            omega: The frequency at which the Green's function is calculated.
            eta: The imaginary part, i.e. broadening factor.
        
        Returns:
            G: The Green's function numpy array.
        """
        # Green's function arrays
        G_e = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        G_h = np.zeros((self.n_orb, self.n_orb), dtype=complex)

        for m in range(self.n_orb):
            for n in range(self.n_orb):
                G_e[m, n] = 2 * np.sum(
                    self.B_e[m, n]
                    / (omega + 1j * eta + self.energy_gs - self.energies_e)
                )
                G_h[m, n] = 2 * np.sum(
                    self.B_h[m, n]
                    / (omega + 1j * eta - self.energy_gs + self.energies_h)
                )
        G = G_e + G_h
        return G

    def spectral_function(
        self, omegas: Sequence[float], eta: float = 0.0, save_data: bool = True
    ) -> Optional[np.ndarray]:
        """Returns the spectral function at certain frequencies.

        Args:
            omegas: The frequencies at which the spectral function is calculated.
            eta: The imaginary part, i.e. broadening factor.
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
            np.savetxt("data/" + self.datfname + "_A.dat", np.vstack((omegas, As)).T)
        else:
            return As

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
