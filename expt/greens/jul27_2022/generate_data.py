"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')

import numpy as np

from fd_greens import GreensFunction, HARTREE_TO_EV, get_alkali_hydride_hamiltonian


def main():
    omegas = np.arange(-32, 32, 0.1)
    eta = 0.02 * HARTREE_TO_EV
    for fname in sys.argv[1:]:
        if fname[:2] == 'na':
            hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.7)
        else:
            hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)
        if 'exact' in fname:
            greens = GreensFunction(hamiltonian, fname=fname, spin='u', method='exact')
        else:
            molecule_name = fname.split('_')[0]
            if 'alltomo' in fname:
                greens = GreensFunction(hamiltonian, fname=fname, spin='u', method='alltomo', fname_exact=f"{molecule_name}_greens_exact")
            else:
                greens = GreensFunction(hamiltonian, fname=fname, spin='u', method='tomo', fname_exact=f"{molecule_name}_greens_exact")
        greens.process()
        greens.spectral_function(omegas, eta)
        greens.self_energy(omegas, eta)

if __name__ == '__main__':
    main()
