import sys
sys.path.append('../../..')

import numpy as np

from fd_greens import GreensFunction, get_lih_hamiltonian, HARTREE_TO_EV

def main():
    print("Start generating data.")
    
    hamiltonian = get_lih_hamiltonian(3.0)
    omegas = np.arange(-20, 20, 0.01)
    eta = 0.02 * HARTREE_TO_EV
    for spin in ['u', 'd']:
        for fname in ['lih_3A_expt1']:
            if 'exact' in fname:
                greens = GreensFunction(hamiltonian, fname=fname, method='exact', spin=spin)
                greens.process()
                greens.spectral_function(omegas, eta)
                greens.self_energy(omegas, eta)
            else:
                greens = GreensFunction(hamiltonian, fname=fname, method='tomo', spin=spin)
                greens.process()
                greens.spectral_function(omegas, eta)
                greens.self_energy(omegas, eta)

                greens = GreensFunction(hamiltonian, fname=fname, method='tomo', spin=spin, suffix='_miti')
                greens.process()
                greens.spectral_function(omegas, eta)
                greens.self_energy(omegas, eta)

    print("Finished generating data.")

if __name__ == '__main__':
    main()
