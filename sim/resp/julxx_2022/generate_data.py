"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')

import numpy as np

from fd_greens import ResponseFunction, get_h2_hamiltonian, HARTREE_TO_EV


def main():
    hamiltonian = get_h2_hamiltonian(1.6)
    omegas = np.arange(-20, 20, 0.01)
    eta = 0.02 * HARTREE_TO_EV
    for fname in ['lih_resp_exact', 'lih_resp_noisy']:
        if 'exact' in fname:
            resp = ResponseFunction(hamiltonian, fname=fname, method='exact')
        else:
            if 'alltomo' in fname:
                resp = ResponseFunction(hamiltonian, fname=fname, method='alltomo', fname_exact='lih_resp_exact')
            else:
                resp = ResponseFunction(hamiltonian, fname=fname, method='tomo', fname_exact='lih_resp_exact')
        resp.process()
        resp.response_function(omegas, eta)

        # resp = ResponseFunction(hamiltonian, fname=fname, method='tomo', suffix='_miti')
        # resp.process()
        # resp.response_function(omegas, eta)

if __name__ == '__main__':
    main()
