from typing import Union

import numpy as np
from vqe import GroundStateSolver
from number_state_solvers import ExcitedStatesSolver
from excited_amplitudes_solver import ExcitedAmplitudesSolver

class ResponseFunction:
    """A class to calculate frequency-domain Green's function."""
    
    def __init__(self, 
                 gs_solver: GroundStateSolver,
                 exc_solver: ExcitedStatesSolver,
                 amp_solver: ExcitedAmplitudesSolver
                ) -> None:
        """Initializes a GreensFunction object.
        
        Args:
            gs_solver: The ground state solver.
            states_solver: The excited states solver.
            amp_solver: The transition amplitudes solver.
        """
        # Ground state and excited states energies
        self.energy_gs = gs_solver.energy
        self.energies_exc = exc_solver.energies_exc

        # Transition amplitudes
        self.L = amp_solver.L
        self.n_states = amp_solver.n_states

    def charge_charge_response(self, omega: Union[float, complex]) -> np.ndarray:
        """Returns the the Green's function at frequency omega.

        Args:
            The frequency at which the response function is calculated.

        Returns:
            The charge-charge response function numpy array.
        """

        R = 0
        for lam in range(self.n_states):
            R += self.L[lam] / (omega - (self.energies_exc[lam] - self.energy_gs))
            R += self.L[lam].conjugate() / (-omega - (self.energies_exc[lam] - self.energy_gs))

        return R
