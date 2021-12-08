"""Green's function module."""

from typing import Union, Sequence

import h5py
import numpy as np
from ground_state_solvers import GroundStateSolver
from number_states_solvers import EHStatesSolver
from amplitudes_solvers import EHAmplitudesSolver

class GreensFunction:
    """A class to calculate frequency-domain Green's function."""
    
    def __init__(self, fname: str
                 # gs_solver: GroundStateSolver,
                 # es_solver: EHStatesSolver,
                 # amp_solver: EHAmplitudesSolver
                ) -> None:
        """Initializes a GreensFunction object.
        
        Args:
            fname: The h5py file name.
            gs_solver: The ground state solver.
            es_solver: The N+/-1 electron states solver.
            amp_solver: The transition amplitudes solver.
        """
        # Ground state and N+/-1 electron states energies
        # self.energy_gs = gs_solver.energy
        # self.energies_e = es_solver.energies_e
        # self.energies_h = es_solver.energies_h

        # Transition amplitudes
        # self.B_e = amp_solver.B_e
        # self.B_h = amp_solver.B_h
        # self.n_orb = amp_solver.n_orb

        # Orbital energies
        # h = amp_solver.h
        # e_orb = np.diag(h.molecule.orbital_energies)
        # self.e_orb = e_orb[h.act_inds][:, h.act_inds]

        self.fname = fname
        f = h5py.File(fname + '.h5py', 'r')
        self.energy_gs = f['energy_gs']
        self.energies_e = f['energies_e']
        self.energies_h = f['energies_h']
        self.B_e = f['B_e']
        self.B_h = f['B_h']
        self.n_orb = self.B_e.shape[0]
        self.e_orb = f['e_orb']

    @property
    def density_matrix(self) -> np.ndarray:
        """The density matrix obtained from the transition amplitudes."""
        rho = np.sum(self.B_h, axis=2)
        return rho

    def greens_function(self, omega: Union[float, complex]) -> np.ndarray:
        """Returns the the Green's function at frequency omega.

        Args:
            omega: The frequency at which the Green's function is calculated.

        Returns:
            The Green's function numpy array.
        """

        # Green's function arrays
        G_e = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        G_h = np.zeros((self.n_orb, self.n_orb), dtype=complex)

        for m in range(self.n_orb):
            for n in range(self.n_orb):
                G_e[m, n] = 2 * np.sum(self.B_e[m, n] / (omega + self.energy_gs - self.energies_e))
                G_h[m, n] = 2 * np.sum(self.B_h[m, n] / (omega - self.energy_gs + self.energies_h))
        G = G_e + G_h
        return G

    def spectral_function(self,
                          omegas: Sequence[float], 
                          eta: float = 0.0,
                          save: bool = True) -> np.ndarray:
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
            G = self.greens_function(omega + 1j * eta)
            A = -1 / np.pi * np.imag(np.trace(G))
            As.append(A)
        
        if save:
            np.savetxt('data/' + self.fname + '_A.dat', np.vstack((omegas, As)).T)
        else:
            return As

    def self_energy(self, 
                    omegas: Sequence[float],
                    eta: float = 0.0,
                    save: bool = True) -> np.ndarray:
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
