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
            if not os.path.exists('data'):
                os.makedirs('data')
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
            if not os.path.exists('data'):
                os.makedirs('data')
            np.savetxt(
                "data/" + self.datfname + "_TrS.dat",
                np.vstack((omegas, TrSigmas.real, TrSigmas.imag)).T,
            )
        else:
            return TrSigmas
