"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')

import numpy as np

from fd_greens import ResponseFunction, get_lih_hamiltonian, HARTREE_TO_EV


def main():
    hamiltonian = get_lih_hamiltonian(3.0)
    omegas = np.arange(-20, 20, 0.01)
    eta = 0.02 * HARTREE_TO_EV
    for fname in ['lih_resp_exact', 'lih_resp_tomo_pur', 'lih_resp_tomo2q_pur', 'lih_resp_alltomo_pur', 'lih_resp_alltomo2q_pur']:
        if 'exact' in fname:
            resp = ResponseFunction(hamiltonian, fname=fname, method='exact')
        elif 'tomo' in fname:
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
