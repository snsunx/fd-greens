import sys
sys.path.append('../../..')

from fd_greens import get_tomography_labels, plot_counts, get_circuit_labels

def main():
	for circ_name in get_circuit_labels(2, 'resp'):
		for tomo_name in get_tomography_labels(2):
			print(f"Plotting bitstring counts of {circ_name}/{tomo_name}")
			plot_counts(
				'lih_resp_tomo', f'lih_3A_expt', f'{circ_name}/{tomo_name}',
				f'{circ_name}/{tomo_name}', 'counts', 'counts', dirname=f'figs/counts')

if __name__ == '__main__':
	main()
