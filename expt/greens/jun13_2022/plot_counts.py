"""Plot simulated and experimental bitstring counts."""

import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import plot_bitstring_counts
from fd_greens.cirq_ver.helpers import get_circuit_labels, get_tomography_labels

def plot(run_id):
	for circ_name in get_circuit_labels(2, mode='greens', spin='ud'):
		for tomo_name in get_tomography_labels(2):
			print(f"Plotting counts of {circ_name}/{tomo_name}.")
			plot_bitstring_counts(
				'lih_3A_sim', f'lih_3A_expt{run_id}', f'{circ_name}/{tomo_name}',
				f'{circ_name}/{tomo_name}', 'counts', 'counts_miti', dirname=f'figs{run_id}_miti')


if __name__ == '__main__':
	plot(0)
	# plot(1)
	# plot(2)
	# plot(3)