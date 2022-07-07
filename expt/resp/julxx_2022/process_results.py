import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import process_all_bitstring_counts


def main():
    h5fname_exact = 'lih_resp_exact'
    pklfname = 'resp_0701_run0'
    npyfname = 'response_greens_0701_0'
    for h5fname_expt in ['lih_resp_expt', 'lih_resp_pur']:
        process_all_bitstring_counts(
            h5fname_expt, 
            h5fname_exact,
            pklfname,
            pkldsetname='base', 
            mode='resp',
            npyfname=npyfname
        )

    for h5fname_expt in ['lih_resp_expt2q', 'lih_resp_pur2q']:
        process_all_bitstring_counts(
            h5fname_expt,
            h5fname_exact,
            pklfname,
            pkldsetname='2q',
            mode='resp',
            npyfname=npyfname
        )

if __name__ == '__main__':
    main()
