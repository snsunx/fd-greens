"""Plot simulated and experimental bitstring counts."""

import sys
sys.path.append('../../..')

from itertools import product

from fd_greens.cirq_ver.helpers import plot_counts

def plot(run_id):
	for circ_name in ['circ0u', 'circ0d', 'circ1u', 'circ1d', 'circ01u', 'circ01d']:
		for tomo_name in [''.join(x) for x in product('xyz', repeat=2)]:
			plot_counts(
				'lih_3A_sim', f'lih_3A_expt{run_id}', f'{circ_name}/{tomo_name}',
				f'{circ_name}/{tomo_name}', 'counts', 'counts_miti', dirname=f'figs{run_id}_miti')


if __name__ == '__main__':
	plot(0)
	plot(1)
	plot(2)
	plot(3)