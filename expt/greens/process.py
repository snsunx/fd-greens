import sys

sys.path.append("../../")

from fd_greens.utils import get_tomography_labels, process_berkeley_results


if __name__ == "__main__":
    h5fnames = ["lih_1p6A", "lih_1p6A_run2", "lih_3A_run2"]
    circ_labels = ["0d", "1d", "01d"]
    tomo_labels = get_tomography_labels(2)
    counts_name = "counts_d"

    for h5fname in h5fnames:
        for circ_label in circ_labels:
            for tomo_label in tomo_labels:
                process_berkeley_results(h5fname, circ_label, tomo_label, counts_name)
