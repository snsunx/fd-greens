"""
========================================
Helpers (:mod:`fd_greens.utils.helpers`)
========================================
"""

from typing import Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

from ..main.molecular_hamiltonian import MolecularHamiltonian

def print_information(amp_solver) -> None:

    print("----- Printing out physical quantities -----")
    print(f"Number of electrons is {amp_solver.n_elec}")
    print(f"Number of orbitals is {amp_solver.n_orb}")
    print(f"Number of occupied orbitals is {amp_solver.n_occ}")
    print(f"Number of virtual orbitals is {amp_solver.n_vir}")
    print(f"Number of (N+1)-electron states is {amp_solver.n_e}")
    print(f"Number of (N-1)-electron states is {amp_solver.n_h}")
    print("--------------------------------------------")