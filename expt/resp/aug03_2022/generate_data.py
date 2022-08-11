"""Process Jun 2022 results and save to HDF5 files."""

import sys

sys.path.append('../../..')

import argparse

import numpy as np

from fd_greens import generate_response_function, generate_fidelity_matrix, generate_trace_matrix


# def main():
#     omegas = np.arange(-32, 32, 0.1)
#     eta = 0.02 * HARTREE_TO_EV
#     # for fname in ['nah_resp_exact', 'nah_resp_tomo', 'nah_resp_tomo2q', 'kh_resp_exact', 'kh_resp_tomo', 'kh_resp_tomo2q']:
#     for fname in sys.argv[1:]:
#         if fname[:2] == 'na':
#             hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.7)
#         else:
#             hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)
#         if 'exact' in fname:
#             resp = ResponseFunction(hamiltonian, fname=fname, method='exact')
#         else:
#             molecule_name = fname.split('_')[0]
#             if 'alltomo' in fname:
#                 resp = ResponseFunction(hamiltonian, fname=fname, method='alltomo', fname_exact=f'{molecule_name}_resp_exact')
#             else:
#                 resp = ResponseFunction(hamiltonian, fname=fname, method='tomo', fname_exact=f'{molecule_name}_resp_exact')
#         resp.process()
#         resp.response_function(omegas, eta)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observable", nargs="+", type=str, dest="observable_fnames", default=None)
    parser.add_argument("-f", "--fidelity-matrix", nargs=2, type=str, dest="fidelity_matrix_fnames", default=None)
    args = parser.parse_args()

    if args.observable_fnames is not None:
        generate_response_function(args.observable_fnames)

    if args.fidelity_matrix_fnames is not None:
        generate_fidelity_matrix(*args.fidelity_matrix_fnames, 'n')
        

if __name__ == '__main__':
    main()
