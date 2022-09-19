import sys
sys.path.append('../../..')
import argparse

from fd_greens.cirq_ver import process_all_bitstring_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5fname_expt")
    parser.add_argument("h5fname_exact")
    parser.add_argument("pklfname")
    parser.add_argument("--npyfname")
    args = parser.parse_args()

    process_all_bitstring_counts(
        args.h5fname_expt,
        args.h5fname_exact,
        args.pklfname,
        calculation_mode="greens",
        npyfname=args.npyfname
    )

    # for h5fname_expt in ['nah_greens_tomo_raw', 'nah_greens_tomo_pur', 'nah_greens_tomo_trace']:
    #     process_all_bitstring_counts(h5fname_expt, 'nah_greens_exact', pklfname, pkldsetname='base_nah')
    
    # for h5fname_expt in ['nah_greens_tomo2q_raw', 'nah_greens_tomo2q_pur', 'nah_greens_tomo2q_trace']:
    #     process_all_bitstring_counts(h5fname_expt, 'nah_greens_exact', pklfname, pkldsetname='2q_nah')

    # for h5fname_expt in ['kh_greens_tomo_raw', 'kh_greens_tomo_pur', 'kh_greens_tomo_trace']:
    #     process_all_bitstring_counts(h5fname_expt, 'kh_greens_exact', pklfname, pkldsetname='base_kh')

    # for h5fname_expt in ['kh_greens_tomo2q_raw', 'kh_greens_tomo2q_pur', 'kh_greens_tomo2q_trace']:
    #     process_all_bitstring_counts(h5fname_expt, 'kh_greens_exact', pklfname, pkldsetname='2q_kh')

if __name__ == '__main__':
    main()
