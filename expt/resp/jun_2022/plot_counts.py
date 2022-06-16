"""Plot simulated and experimental bitstring counts."""

import sys
sys.path.append('../../..')

from itertools import product

from fd_greens.cirq_ver.helpers import plot_counts, get_circuit_labels

def plot(run_id):
	for circ_name in get_circuit_labels(2, 'resp'):
		for tomo_name in [''.join(x) for x in product('xyz', repeat=2)]:
			plot_counts(
				'lih_resp_sim', f'lih_3A_expt{run_id}', f'{circ_name}/{tomo_name}',
				f'{circ_name}/{tomo_name}', 'counts', 'counts', dirname=f'figs{run_id}')


if __name__ == '__main__':
	for run_id in [0, 1, 2, 3]:
		plot(run_id)
