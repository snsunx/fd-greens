"""Plot simulated and experimental bitstring counts."""

import sys
sys.path.append('../../..')

from fd_greens import get_circuit_labels, get_tomography_labels, plot_bitstring_counts

def main():
	for circ_name in get_circuit_labels(2, spin='u') + get_circuit_labels(2, spin='d'):
		for tomo_name in get_tomography_labels(2):
			plot_bitstring_counts('lih_3A_sim', 'lih_3A_expt', 
						f'{circ_name}/{tomo_name}', f'{circ_name}/{tomo_name}',
						'counts', 'counts', dirname='figs/counts')

if __name__ == '__main__':
	main()
