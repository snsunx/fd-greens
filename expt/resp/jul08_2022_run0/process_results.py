import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import process_all_bitstring_counts


def main():
    h5fname_exact = 'lih_resp_exact'
    pklfname = 'resp_0708_run0'
    # npyfname = 'response_greens_0701_0'
    
    for h5fname_expt in ['lih_resp_tomo_raw', 'lih_resp_tomo_pur', 'lih_resp_tomo_trace']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, mode='resp', pkldsetname='base')

    for h5fname_expt in ['lih_resp_tomo2q_raw', 'lih_resp_tomo2q_pur', 'lih_resp_tomo2q_trace']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, mode='resp', pkldsetname='2q')

    for h5fname_expt in ['lih_resp_alltomo_raw', 'lih_resp_alltomo_pur', 'lih_resp_alltomo_trace']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, mode='resp', pkldsetname='base_alltomo')

    for h5fname_expt in ['lih_resp_alltomo2q_raw', 'lih_resp_alltomo2q_pur', 'lih_resp_alltomo2q_trace']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, mode='resp', pkldsetname='2q_alltomo')

if __name__ == '__main__':
    main()
