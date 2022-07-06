import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import display_fidelities

def main():
    # display_fidelities('lih_resp_exact', 'lih_resp_expt', 'n', figname='fid_expt')
    # display_fidelities('lih_resp_exact', 'lih_resp_expt2q', 'n', figname='fid_expt2q')
    display_fidelities('lih_resp_exact', 'lih_resp_pur', 'n', suffix='_miti', figname='fid_pur_miti')
    display_fidelities('lih_resp_exact', 'lih_resp_pur2q', 'n', suffix='_miti', figname='fid_pur2q_miti')

if __name__ == '__main__':
    main()
