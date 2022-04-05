import sys

sys.path.append("../../")

from itertools import product

from fd_greens.utils import (
    plot_counts,
    compute_tvd,
    get_circuit_depth,
    get_n_2q_gates,
    get_n_3q_gates,
)


def main_counts_greens():
    # h5fnames = ['lih_1p6A', 'lih_1p6A_run2', 'lih_3A_run2']
    h5fnames = ["../greens/lih_1p6A_run2"]
    circ_labels = ["0d", "1d", "01d"]
    counts_name = "counts_d"
    tomo_labels = ["".join(x) for x in product("xyz", repeat=2)]

    for h5fname in h5fnames:
        for circ_label in circ_labels:
            for tomo_label in tomo_labels:
                plot_counts(h5fname, counts_name, circ_label, tomo_label)


def main_counts_resp():
    h5fnames = ["../resp/lih", "../resp/lih_run2"]
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
    counts_name = "counts_noisy"
    tomo_labels = ["".join(x) for x in product("xyz", repeat=2)]

    for h5fname in h5fnames:
        for circ_label in circ_labels:
            for tomo_label in tomo_labels:
                plot_counts(h5fname, counts_name, circ_label, tomo_label)


def main_tvd_greens():
    # h5fnames = ['lih_1p6A', 'lih_1p6A_run2', 'lih_3A_run2']
    h5fnames = ["../greens/lih_1p6A_run2"]
    circ_labels = ["0d", "1d", "01d"]
    counts_name = "counts_d"

    print(
        f"File name     | Circuit label | Circuit depth | # 2q gates | # 3q gates | TVD    "
    )
    print(
        f"================================================================================="
    )
    for h5fname in h5fnames:
        for circ_label in circ_labels:
            depth = get_circuit_depth(h5fname, circ_label)
            count_2q = get_n_2q_gates(h5fname, circ_label)
            count_3q = get_n_3q_gates(h5fname, circ_label)
            tvd = compute_tvd(h5fname, circ_label, counts_name)
            print(
                f"{h5fname:13} | {circ_label:13} | {depth:13} | {count_2q:10} | {count_3q:10} | {tvd:6.4f}"
            )


def main_tvd_resp():
    h5fnames = ["../resp/lih", "../resp/lih_run2"]
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
    counts_name = "counts_noisy"

    print(
        f"File name     | Circuit label | Circuit depth | # 2q gates | # 3q gates | TVD    "
    )
    print(
        f"================================================================================="
    )
    for h5fname in h5fnames:
        for circ_label in circ_labels:
            depth = get_circuit_depth(h5fname, circ_label)
            count_2q = get_n_2q_gates(h5fname, circ_label)
            count_3q = get_n_3q_gates(h5fname, circ_label)
            tvd = compute_tvd(h5fname, circ_label, counts_name)
            print(
                f"{h5fname:13} | {circ_label:13} | {depth:13} | {count_2q:10} | {count_3q:10} | {tvd:6.4f}"
            )

def main():
    main_tvd_resp()

if __name__ == "__main__":
    main()
