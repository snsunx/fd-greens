"""Response function module."""

from typing import Sequence, Union

import h5py
import numpy as np
import params
# from ground_state_solvers import GroundStateSolver
# from number_states_solvers import ExcitedStatesSolver
# from amplitudes_solvers import ExcitedAmplitudesSolver

class ResponseFunction:
    """A class to calculate frequency-domain Green's function."""
    
    def __init__(self, fname: str
                 # gs_solver: GroundStateSolver,
                 # es_solver: ExcitedStatesSolver,
                 # amp_solver: ExcitedAmplitudesSolver
                ) -> None:
        """Initializes a ResponseFunction object.
        
        Args:
            gs_solver: The ground state solver.
            es_solver: The excited states solver.
            amp_solver: The transition amplitudes solver.
        """
        # Ground state and excited states energies
        # self.energy_gs = gs_solver.energy
        # self.energies_s = es_solver.energies_s
        # self.energies_t = es_solver.energies_t
        # self.energies_exc = np.hstack((self.energies_s, self.energies_t))

        # Transition amplitudes
        # self.L = amp_solver.L
        # self.n_states = amp_solver.n_states

        self.fname = fname
        f = h5py.File(fname + '.h5py', 'r')
        self.energy_gs = f['energy_gs']
        self.energies_s = f['energies_s']
        self.energies_t = f['energies_t']
        self.energies_exc = np.hstack((self.energies_s, self.energies_t))
        self.L = f['L']
        self.n_states = f['n_states']

    def response_function(self, omegas: Sequence[float], 
                          i: int,
                          j: int,
                          eta: float = 0.0,
                          save: bool = True) -> np.ndarray:
        """Returns the charge-charge response function at frequency omega.

        Args:
            omegas: The frequencies at which the response function is calculated.
            i: Row orbital index.
            j: Column orbital index.
            eta: The imaginary part, i.e. broadening factor.
            save: Whether to save the response function to file.

        Returns:
            (Optional) The charge-charge response function for orbital i and orbital j.
        """

        chis = []
        for omega in omegas:
            chi = 0
            for lam in [1, 2, 3]: # XXX: Hardcoded
                chi += self.L[i, j, lam] / (omega + 1j * eta - (self.energies_exc[lam] - self.energy_gs))
                chi += self.L[i, j, lam].conjugate() / (-omega - 1j * eta - (self.energies_exc[lam] - self.energy_gs))
            chis.append(chi)
        chis = np.array(chis)

        if save:
            np.savetxt('data/' + self.fname + '_chi.dat', np.vstack((omegas, chis.real, chis.imag)).T)
        else:
            return chi

    def cross_section(self, 
                      omegas: Sequence[float], 
                      eta: float = 0.0, 
                      save: bool = True) -> complex:
        """Returns the photo-absorption cross section.
        
        Args:
            omegas: The frequencies at which the response function is calculated.
            eta: The imaginary part, i.e. broadening factor.
            save: Whether to save the photo-absorption cross section to file.

        Returns:
            (Optional) The photo-absorption cross section.
        """
        sigmas = []
        for omega in omegas:
            alpha = 0
            for i in range(2): # XXX: Hardcoded.
                alpha += -self.response_function([omega + 1j * eta], i, i, save=False)

            sigma = 4 * np.pi / params.c * omega * alpha.imag
            sigmas.append(sigma)
        sigmas = np.array(sigmas)

        if save:
            np.savetxt('data/' + self.fname + '_sigma.dat', np.vstack((omegas, sigmas.real, sigmas.imag)).T)
        else:
            return sigma
