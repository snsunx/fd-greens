"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')

import numpy as np

from fd_greens import ResponseFunction, get_nah_hamiltonian, HARTREE_TO_EV


def main():
    hamiltonian = get_nah_hamiltonian(3.7)
    omegas = np.arange(-32, 32, 0.05)
    eta = 0.02 * HARTREE_TO_EV
    for fname in ['nah_resp_exact', 'nah_resp_tomo_raw', 'nah_resp_tomo2q_raw']:
        if 'exact' in fname:
            resp = ResponseFunction(hamiltonian, fname=fname, method='exact')
        elif 'tomo' in fname:
            if 'alltomo' in fname:
                resp = ResponseFunction(hamiltonian, fname=fname, method='alltomo', fname_exact='nah_resp_exact')
            else:
                resp = ResponseFunction(hamiltonian, fname=fname, method='tomo', fname_exact='nah_resp_exact')
        resp.process()
        resp.response_function(omegas, eta)

        # resp = ResponseFunction(hamiltonian, fname=fname, method='tomo', suffix='_miti')
        # resp.process()
        # resp.response_function(omegas, eta)

if __name__ == '__main__':
    main()
