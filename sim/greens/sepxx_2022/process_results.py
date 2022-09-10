import sys
sys.path.append('../../..')
import argparse

from fd_greens.cirq_ver import process_all_bitstring_counts

def main():
    # pklfname = 'greens_0720_run0'

    parser = argparse.ArgumentParser()
    parser.add_argument("h5fnames", nargs=2)
    parser.add_argument("--pkldsetname")
    args = parser.parse_args()

    process_all_bitstring_counts(args.h5fnames[0], args.h5fnames[1], "greens_0720_run0", pkldsetname=args.pkldsetname)

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
