import sys
sys.path.append('../../')
from itertools import product
from fd_greens.utils import plot_counts, compute_tvd

def main_counts():
    # h5fnames = ['lih_1p6A']
    h5fnames = ['lih', 'lih_run2']
    # circ_labels = ['0d', '1d', '01d']
    circ_labels = ['0u', '0d', '1u', '1d', '0u0d', '0u1u', '0u1d', '0d1u', '0d1d', '1u1d']
    counts_name = 'counts_noisy'

    for h5fname in h5fnames:
        for circ_label in circ_labels:
            # n_qubits = len([x for x in circ_label if x in ['0', '1']]) + 2
            tomo_labels = [''.join(x) for x in product('xyz', repeat=2)]
            tvd = compute_tvd(h5fname, circ_label, counts_name)
            print(f'{h5fname} {circ_label} {tvd:.4f}')
            # for tomo_label in tomo_labels:
            #     plot_counts(h5fname, counts_name, circ_label, tomo_label)

if __name__ == '__main__':
    main_counts()
