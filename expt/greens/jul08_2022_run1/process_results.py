import sys

sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import process_all_bitstring_counts

def main():
    h5fname_exact = 'lih_greens_exact'
    pklfname = 'greens_0708_run1'

    for h5fname_expt in ['lih_greens_tomo_raw', 'lih_greens_tomo_pur', 'lih_greens_tomo_trace']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, pkldsetname='base')
    
    for h5fname_expt in ['lih_greens_tomo2q_raw', 'lih_greens_tomo2q_pur', 'lih_greens_tomo2q_trace']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, pkldsetname='2q')

    for h5fname_expt in ['lih_greens_alltomo_raw', 'lih_greens_alltomo_pur', 'lih_greens_alltomo_trace']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, pkldsetname='base_alltomo')

    for h5fname_expt in ['lih_greens_alltomo2q_raw', 'lih_greens_alltomo2q_pur', 'lih_greens_alltomo2q_trace']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, pkldsetname='2q_alltomo')

if __name__ == '__main__':
    main()
