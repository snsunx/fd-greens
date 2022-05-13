"""
======================================================
Response Function (:mod:`fd_greens.response_function`)
======================================================
"""

import os
from typing import Sequence

import h5py
import numpy as np


class ResponseFunction:
    """A class to calculate frequency-domain charge-charge response function."""

    def __init__(self, h5fname: str = "lih", suffix: str = "") -> None:
        """Initializes a ResponseFunction object.
        
        Args:
            h5fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
        """
        self.datfname = h5fname + suffix
        h5file = h5py.File(h5fname + ".h5", "r")
        self.energy_gs = h5file["gs/energy"]
        self.energies_s = h5file["es/energies_s"]
        # self.energies_t = h5file['es/energies_t']
        # self.energies_exc = np.hstack((self.energies_s, self.energies_t))
        self.N = h5file[f"amp/N{suffix}"]

    def response_function(
        self, omegas: Sequence[float], eta: float = 0.0, save_data: bool = True
    ) -> np.ndarray:
        """Returns the charge-charge response function at given frequencies.

        Args:
            omegas: The frequencies at which the response function is calculated.
            eta: The imaginary part, i.e. broadening factor.
            save: Whether to save the response function to file.

        Returns:
            (Optional) The charge-charge response function for orbital i and orbital j.
        """
        for label in ["00", "11", "01", "10"]:
            i = int(label[0])
            j = int(label[1])
            chis = []
            for omega in omegas:
                for lam in [1, 2, 3]:  # [1, 2, 3] is hardcoded
                    chi = np.sum(
                        self.N[2 * i : 2 * (i + 1), 2 * j : 2 * (j + 1), lam]
                    ) / (omega + 1j * eta - (self.energies_s[lam] - self.energy_gs))
                    chi += np.sum(
                        self.N[2 * i : 2 * (i + 1), 2 * j : 2 * (j + 1), lam]
                    ).conjugate() / (
                        -omega - 1j * eta - (self.energies_s[lam] - self.energy_gs)
                    )
                chis.append(chi)
            chis = np.array(chis)
            if save_data:
                if not os.path.exists("data"):
                    os.makedirs("data")
                np.savetxt(
                    f"data/{self.datfname}_chi{label}.dat",
                    np.vstack((omegas, chis.real, chis.imag)).T,
                )
            else:
                return chis
