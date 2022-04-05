import sys

sys.path.append("../../")

import numpy as np

from fd_greens.main import ExcitedAmplitudesSolver, ResponseFunction
from fd_greens.main.params import HARTREE_TO_EV
from fd_greens.utils import (
    get_tomography_labels,
    process_berkeley_results,
    get_lih_hamiltonian,
)


def main_process_results():
    h5fnames = ["lih", "lih_run2"]
    circ_labels = [
        "0u",
        "0d",
        "1u",
        "1d",
        "0u0d",
        "0u1u",
        "0u1d",
        "0d1u", 
        "0d1d",
        "1u1d",
    ]
    tomo_labels = get_tomography_labels(2)
    # counts_name = "counts_noisy"

    for h5fname in h5fnames:
        for circ_label in circ_labels:
            for tomo_label in tomo_labels:
                process_berkeley_results(h5fname, circ_label, tomo_label, counts_name)


def main_generate_data(**run_kwargs):
    # from fd_greens.utils import initialize_hdf5
    # initialize_hdf5(calc="resp") # Do I need this?

    amp_solver = ExcitedAmplitudesSolver(
        h, h5fname=h5fname, method=method, suffix=suffix
    )
    amp_solver.run(build=False, execute=False, **run_kwargs)

    resp_func = ResponseFunction(h5fname=h5fname, suffix=suffix)
    resp_func.response_function(omegas, eta)


def main():
    global h, h5fname, method, counts_name, suffix, omegas, eta
    h = get_lih_hamiltonian(3.0)
    method = "tomo"
    h5fname = "lih"
    _suffix = "_noisy"
    counts_name = "counts" + _suffix
    suffix = _suffix + "_exp_proc"
    omegas = np.arange(-30, 30, 0.1)
    eta = 0.02 * HARTREE_TO_EV

    main_process_results()
    main_generate_data(method=method)


if __name__ == "__main__":
    main()
