"""Response function module."""

from typing import Union

import numpy as np

import params
from ground_state_solvers import GroundStateSolver
from number_states_solvers import ExcitedStatesSolver
from amplitudes_solvers import ExcitedAmplitudesSolver

class ResponseFunction:
    """A class to calculate frequency-domain Green's function."""
    
    def __init__(self, 
                 gs_solver: GroundStateSolver,
                 es_solver: ExcitedStatesSolver,
                 amp_solver: ExcitedAmplitudesSolver
                ) -> None:
        """Initializes a ResponseFunction object.
        
        Args:
            gs_solver: The ground state solver.
            es_solver: The excited states solver.
            amp_solver: The transition amplitudes solver.
        """
        # Ground state and excited states energies
        self.energy_gs = gs_solver.energy
        self.energies_s = es_solver.energies_s
        self.energies_t = es_solver.energies_t
        self.energies_exc = np.hstack((self.energies_s, self.energies_t))

        # print(self.energies_exc.shape)

        # Transition amplitudes
        self.L = amp_solver.L
        self.n_states = amp_solver.n_states

        # print(self.L.shape)

    def response_function(self, omega: Union[float, complex], i: int, j: int) -> np.ndarray:
        """Returns the charge-charge response function at frequency omega.

        Args:
            omega: The frequency at which the response function is calculated.
            i: Row orbital index.
            j: Column orbital index.

        Returns:
            The charge-charge response function for orbital i and orbital j.
        """

        chi = 0
        for lam in range(self.n_states):
            chi += self.L[i, j, lam] / (omega - (self.energies_exc[lam] - self.energy_gs))
            chi += self.L[i, j, lam].conjugate() / (-omega - (self.energies_exc[lam] - self.energy_gs))

        return chi

    def cross_section(self, omega: Union[float, complex]) -> complex:
        """Returns the photo-absorption cross section.
        
        Args:
            The frequency at which the response function is calculated.

        Returns:
            The photo-absorption cross section.
        """
        alpha = 0
        for i in range(2): # XXX: 2 is hardcoded.
            alpha += -self.response_function(omega, i, i)

        sigma = 4 * np.pi / params.c * omega * alpha.imag
        return sigma
