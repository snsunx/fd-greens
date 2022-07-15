import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import process_all_bitstring_counts


def main():
    h5fname_exact = 'nah_resp_exact'
    pklfname = 'resp_0714_nah_run0'
    # npyfname = 'response_greens_0701_0'
    
    for h5fname_expt in ['nah_resp_tomo_raw', 'nah_resp_tomo_pur', 'nah_resp_tomo_trace']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, mode='resp', pkldsetname='base')

    for h5fname_expt in ['nah_resp_tomo2q_raw', 'nah_resp_tomo2q_pur', 'nah_resp_tomo2q_trace']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, mode='resp', pkldsetname='2q')

if __name__ == '__main__':
    main()
