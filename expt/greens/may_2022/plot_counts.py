import sys
sys.path.append('../../..')

from itertools import product

from fd_greens.cirq_ver.helpers import plot_counts

for circ_name in ['circ0u', 'circ0d', 'circ1u', 'circ1d', 'circ01u', 'circ01d']:
	for tomo_name in [''.join(x) for x in product('xyz', repeat=2)]:
		plot_counts(
			'lih_3A_sim_new', 'lih_3A_expt_0527', f'{circ_name}/{tomo_name}',
			f'{circ_name}/{tomo_name}', 'counts', 'counts', dirname='figs_0527')
