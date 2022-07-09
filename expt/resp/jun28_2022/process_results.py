import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import process_all_bitstring_counts


def main():
    h5fname_exact = 'lih_resp_exact'
    pklfname = 'resp_3A_run0628_0'
    npyfname = 'response_greens_0628_1'
    for h5fname_expt in ['lih_resp_expt']:
        process_all_bitstring_counts(
            h5fname_expt, 
            h5fname_exact,
            pklfname,
            pkldsetname='full', 
            mode='resp',
            npyfname=npyfname
        )

if __name__ == '__main__':
    main()
