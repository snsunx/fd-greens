import sys

sys.path.append("../../")

from fd_greens.utils import get_tomography_labels, process_berkeley_results

if __name__ == "__main__":
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
    counts_name = "counts_noisy"

    for h5fname in h5fnames:
        for circ_label in circ_labels:
            for tomo_label in tomo_labels:
                process_berkeley_results(h5fname, circ_label, tomo_label, counts_name)
