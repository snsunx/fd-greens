import sys
sys.path.append('../../..')
import argparse

from fd_greens.cirq_ver.helpers import process_all_bitstring_counts

def main():
    pklfname = 'resp_0720_run0'

    for h5fname_expt in ['nah_resp_tomo_raw', 'nah_resp_tomo_pur', 'nah_resp_tomo_trace']:
        process_all_bitstring_counts(h5fname_expt, 'nah_resp_exact', pklfname, pkldsetname='base_nah', mode='resp')
    
    for h5fname_expt in ['nah_resp_tomo2q_raw', 'nah_resp_tomo2q_pur', 'nah_resp_tomo2q_trace']:
        process_all_bitstring_counts(h5fname_expt, 'nah_resp_exact', pklfname, pkldsetname='2q_nah', mode='resp')

    for h5fname_expt in ['kh_resp_tomo_raw', 'kh_resp_tomo_pur', 'kh_resp_tomo_trace']:
        process_all_bitstring_counts(h5fname_expt, 'kh_resp_exact', pklfname, pkldsetname='base_kh', mode='resp')

    for h5fname_expt in ['kh_resp_tomo2q_raw', 'kh_resp_tomo2q_pur', 'kh_resp_tomo2q_trace']:
        process_all_bitstring_counts(h5fname_expt, 'kh_resp_exact', pklfname, pkldsetname='2q_kh', mode='resp')

if __name__ == '__main__':
    main()
