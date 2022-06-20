import sys
sys.path.append('../../..')

import pickle

from fd_greens.cirq_ver.helpers import get_circuit_labels, get_tomography_labels, plot_fidelity_by_depth

if __name__ == '__main__':
    pkl_data = pickle.load(open('greens_3A_run0616_0.pkl', 'rb'))

    # for run_id in [0, 1, 2, 3]:
    #     plot_fidelity_by_depth('lih_3A_sim', f'greens_3A_run0613_{run_id}', 'circ01d/zz', 'circ01d', 4, figname=f'fid_by_depth_{run_id}')

    for circ_label in get_circuit_labels(2, 'greens', 'd'):
        for tomo_label in get_tomography_labels(2):
            if 'results' in pkl_data[f'{circ_label}{tomo_label}_by_depth'].keys():
                print(f'{circ_label}/{tomo_label}')
                plot_fidelity_by_depth('lih_3A_sim', 'greens_3A_run0616_0', f'{circ_label}/{tomo_label}', f'{circ_label}{tomo_label}', 4, figname=f'fid_by_depth_0616_{circ_label}{tomo_label}', mark_itoffoli=True)
