"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')
from typing import Sequence

import argparse

import h5py
import numpy as np

from fd_greens import ResponseFunction, get_nah_hamiltonian, HARTREE_TO_EV, get_fidelity

def generate_observables(h5fnames: Sequence[str]) -> None:
    hamiltonian = get_nah_hamiltonian(3.7)
    omegas = np.arange(-32, 32, 0.1)
    eta = 0.02 * HARTREE_TO_EV
    for fname in h5fnames:
        if 'exact' in fname:
            resp = ResponseFunction(hamiltonian, fname=fname, method='exact')
        else:
            if 'alltomo' in fname:
                resp = ResponseFunction(hamiltonian, fname=fname, method='alltomo', fname_exact='lih_resp_exact')
            else:
                resp = ResponseFunction(hamiltonian, fname=fname, method='tomo', fname_exact='lih_resp_exact')
        resp.process()
        resp.response_function(omegas, eta)

def generate_fidelity_trajectory(h5fnames: Sequence[str]) -> None:
    h5file0 = h5py.File(h5fnames[0] + ".h5", "r")
    h5file1 = h5py.File(h5fnames[1] + ".h5", "r")

    if "psi" in h5file0:
        group0 = h5file0["psi"]
    elif "rho" in h5file0:
        group0 = h5file0["rho"]
    if "psi" in h5file1:
        group1 = h5file1["psi"]
    elif "rho" in h5file1:
        group1 = h5file1["rho"]

    d = dict()
    for key in group0.keys():
        state0 = group0[key]
        state1 = group1[key]
        # print(f"{state0.shape =}")
        # print(f"{state1.shape =}")

        fidelity = get_fidelity(state0, state1)
        # print(f"fidelity = {fidelity}")
        d.update({int(key): fidelity})
    d = dict(sorted(d.items()))

    with open("fidtraj.dat", 'w') as f:
        for key, value in d.items():
            f.write(f"{key:2d} {value:.8f}\n")

    h5file0.close()
    h5file1.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observable", nargs="+", type=str, default=None)
    parser.add_argument("-f", "--fidelity", nargs=2, type=str, default=None)
    args = parser.parse_args()

    if args.observable is not None:
        generate_observables(args.observable)

    if args.fidelity is not None:
        generate_fidelity_trajectory(args.fidelity)

if __name__ == '__main__':
    main()
