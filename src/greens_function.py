"""Green's function module."""

from collections import defaultdict
from typing import Union, Sequence, Optional

import h5py
import numpy as np

class GreensFunction:
    """A class to calculate frequency-domain Green's function."""
    
    def __init__(self, fname: str = 'lih') -> None:
        """Initializes a GreensFunction object.
        
        Args:
            fname: The h5py file name.
        """
        self.fname = fname
        h5file = h5py.File(fname + '.h5', 'r')
        self.energy_gs = h5file['gs/energy']
        self.energies_e = h5file['eh/energies_e']
        self.energies_h = h5file['eh/energies_h']
        self.B_e = h5file['amp/B_e']
        self.B_h = h5file['amp/B_h']
        self.n_orb = self.B_e.shape[0]
        self.e_orb = h5file['amp/e_orb']

        self.n_orb = 2
        self.n_e = 2
        self.n_h = 2

        # Transition amplitude arrays.
        assert self.n_e == self.n_h # XXX
        self.B = defaultdict(lambda: np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex))
        self.D = defaultdict(lambda: np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex))

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
            The Green's function numpy array.
        """

        # Green's function arrays
        G_e = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        G_h = np.zeros((self.n_orb, self.n_orb), dtype=complex)

        for m in range(self.n_orb):
            for n in range(self.n_orb):
                G_e[m, n] = 2 * np.sum(self.B_e[m, n] / (omega + 1j * eta + self.energy_gs - self.energies_e))
                G_h[m, n] = 2 * np.sum(self.B_h[m, n] / (omega + 1j * eta - self.energy_gs + self.energies_h))
        G = G_e + G_h
        return G

    def spectral_function(self,
                          omegas: Sequence[float], 
                          eta: float = 0.0,
                          save: bool = True) -> Optional[np.ndarray]:
        """Returns the spectral function at frequency omega.

        Args:
            omegas: The frequencies at which the spectral function is calculated.
            eta: The imaginary part, i.e. broadening factor.
            save: Whether to save the spectral function to file.
        
        Returns:
            (Optional) The spectral function numpy array.
        """
        As = []
        for omega in omegas:
            G = self.greens_function(omega, eta)
            A = -1 / np.pi * np.imag(np.trace(G))
            As.append(A)
        As = np.array(As)
        
        if save:
            np.savetxt('data/' + self.fname + '_A.dat', np.vstack((omegas, As)).T)
        else:
            return As

    def self_energy(self, 
                    omegas: Sequence[float],
                    eta: float = 0.0,
                    save: bool = True) -> Optional[np.ndarray]:
        """Returns the self-energy at frequency omega.

        Args:
            omegas: The frequencies at which the spectral function is calculated.
            eta: The imaginary part, i.e. broadening factor.
            save: Whether to save the spectral function to file.

        Returns:
            (Optional) Trace of the self-energy.
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

        if save:
            np.savetxt('data/' + self.fname + '_TrSigma.dat', 
                       np.vstack((omegas, TrSigmas.real, TrSigmas.imag)).T)
        else:
            return TrSigmas
