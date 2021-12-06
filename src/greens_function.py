"""A new GreensFunction class that takes solvers as inputs."""

from typing import Union

import numpy as np
from vqe import GroundStateSolver
from number_state_solvers import EHStatesSolver
from eh_amplitudes_solver import EHAmplitudesSolver

class GreensFunction:
    """A class to calculate frequency-domain Green's function."""
    
    def __init__(self, 
                 gs_solver: GroundStateSolver,
                 eh_solver: EHStatesSolver,
                 amp_solver: EHAmplitudesSolver
                ) -> None:
        """Initializes a GreensFunction object.
        
        Args:
            gs_solver: The ground state solver.
            eh_solver: The N+/-1 electron states solver.
            amp_solver: The transition amplitudes solver.
        """
        # Ground state and N+/-1 electron states energies
        self.energy_gs = gs_solver.energy
        self.energies_e = eh_solver.energies_e
        self.energies_h = eh_solver.energies_h

        # Transition amplitudes
        self.B_e = amp_solver.B_e
        self.B_h = amp_solver.B_h
        self.n_orb = amp_solver.n_orb

        # Orbital energies
        h = amp_solver.h
        e_orb = np.diag(h.molecule.orbital_energies)
        self.e_orb = e_orb[h.act_inds][:, h.act_inds]

    @property
    def density_matrix(self) -> np.ndarray:
        """The density matrix obtained from the transition amplitudes"""
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

    def spectral_function(self, omega: Union[float, complex]) -> np.ndarray:
        """Returns the spectral function at frequency omega.

        Args:
            The frequency at which the spectral function is calculated.
        
        Returns:
            The spectral function numpy array.
        """
        G = self.greens_function(omega)
        A = -1 / np.pi * np.imag(np.trace(G))
        return A

    def self_energy(self, omega: Union[float, complex]) -> np.ndarray:
        """Returns the self-energy at frequency omega.

        Args:
            The frequency at which the self-energy is calculated.

        Returns:
            The self-energy numpy array.
        """
        G = self.greens_function(omega)
        # print(np.linalg.norm(G.real))
        # print(np.linalg.norm(G.imag))

        G_HF = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        for i in range(self.n_orb):
            G_HF[i, i] = 1 / (omega - self.e_orb[i, i])

        Sigma = np.linalg.pinv(G_HF) - np.linalg.pinv(G)
        return Sigma