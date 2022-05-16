import sys

sys.path.append("../../")

import warnings

warnings.filterwarnings("ignore")

import numpy as np

from fd_greens.main import EHAmplitudesSolver, GreensFunction
from fd_greens.main.params import HARTREE_TO_EV
from fd_greens.utils import (
    get_lih_hamiltonian,
    initialize_hdf5,
    process_berkeley_results,
    get_tomography_labels,
)


def main_process_results():
    circ_labels = ["0d", "1d", "01d"]
    tomo_labels = get_tomography_labels(2)

    for h5fname in h5fnames:
        for circ_label in circ_labels:
            for tomo_label in tomo_labels:
                process_berkeley_results(h5fname, circ_label, tomo_label, counts_name)


def main_generate_data(**run_kwargs):
    for h5fname in h5fnames:
        amp_solver = EHAmplitudesSolver(
            h, h5fname=h5fname, suffix=suffix, anc=[0, 1], spin=spin
        )
        amp_solver.run(build=False, execute=False, **run_kwargs)

        greens_func = GreensFunction(h5fname=h5fname, suffix=suffix)
        greens_func.spectral_function(omegas, eta)
        greens_func.self_energy(omegas, eta)


def main():
    global h, h5fname, h5fnames, spin, counts_name, suffix, omegas, eta
    h = get_lih_hamiltonian(3.0)
    method = "tomo"
    h5fname = "lih_3A_run2"
    h5fnames = ["lih_1p6A", "lih_1p6A_run2", "lih_3A_run2"]
    spin = "d"
    suffix = "_d_exp_proc"
    counts_name = "counts_d"
    omegas = np.arange(-20, 20, 0.01)
    eta = 0.02 * HARTREE_TO_EV

    # initialize_hdf5(h5fname)
    main_process_results()
    main_generate_data(method=method)


if __name__ == "__main__":
    main()
