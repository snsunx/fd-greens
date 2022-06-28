import sys
sys.path.append('../../..')

import pickle

from fd_greens import get_circuit_labels, get_tomography_labels, plot_fidelity_by_depth


def main():
    pkl_data = pickle.load(open('greens_3A_run0616_0.pkl', 'rb'))

    for circ_label in get_circuit_labels(2, 'greens', 'd'):
        for tomo_label in get_tomography_labels(2):
            if 'results' in pkl_data[f'{circ_label}{tomo_label}_by_depth'].keys():
                print(f"Plotting fidelity by depth of {circ_label}/{tomo_label}.")
                plot_fidelity_by_depth(
                    'lih_3A_sim', 'greens_3A_run0616_0', 
                    f'{circ_label}/{tomo_label}', f'{circ_label}{tomo_label}',
                    4, dirname='figs/fid_by_depth', 
                    figname=f'fid_by_depth_{circ_label}{tomo_label}', 
                    mark_itoffoli=True)

if __name__ == '__main__':
    main()