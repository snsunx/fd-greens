"""
========================================
Helpers (:mod:`fd_greens.utils.helpers`)
========================================
"""

from typing import Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

from ..main.molecular_hamiltonian import MolecularHamiltonian


def get_lih_hamiltonian(bond_distance: float) -> MolecularHamiltonian:
    """Returns the HOMO-LUMO LiH Hamiltonian with bond length r.
    
    Args:
        bond_distance: The bond length of the molecule in Angstrom.
    
    Returns:
        hamiltonian: The molecular Hamiltonian.
    """
    hamiltonian = MolecularHamiltonian(
        [["Li", (0, 0, 0)], ["H", (0, 0, bond_distance)]], 
        "sto3g", 
        occupied_indices=[0],
        active_indices=[1, 2]
    )
    return hamiltonian

def print_information(amp_solver) -> None:

    print("----- Printing out physical quantities -----")
    print(f"Number of electrons is {amp_solver.n_elec}")
    print(f"Number of orbitals is {amp_solver.n_orb}")
    print(f"Number of occupied orbitals is {amp_solver.n_occ}")
    print(f"Number of virtual orbitals is {amp_solver.n_vir}")
    print(f"Number of (N+1)-electron states is {amp_solver.n_e}")
    print(f"Number of (N-1)-electron states is {amp_solver.n_h}")
    print("--------------------------------------------")