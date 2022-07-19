import sys
sys.path.append('../../..')

import numpy as np

from fd_greens import GreensFunction, get_alkali_hydride_hamiltonian, HARTREE_TO_EV

def main():
    print("Start generating data.")

    omegas = np.arange(-20, 20, 0.01)
    eta = 0.02 * HARTREE_TO_EV
    for spin in ['u']:
        for fname in ['nah_greens_exact', 'nah_greens_tomo', 'nah_greens_tomo2q', 'kh_greens_exact', 'kh_greens_tomo', 'kh_greens_tomo2q']:
            if fname[:2] == 'na':
                hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.7)
            else:
                hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)
            if 'exact' in fname:
                greens = GreensFunction(hamiltonian, fname=fname, method='exact', spin=spin)
            else:
                if 'alltomo' in fname:
                    greens = GreensFunction(hamiltonian, fname=fname, method='alltomo', spin=spin, fname_exact='lih_greens_exact')
                else:
                    greens = GreensFunction(hamiltonian, fname=fname, method='tomo', spin=spin, fname_exact='lih_greens_exact')
            greens.process()
            greens.spectral_function(omegas, eta)
            greens.self_energy(omegas, eta)

    print("Finished generating data.")

if __name__ == '__main__':
    main()
