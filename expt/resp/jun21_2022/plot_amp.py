import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import display_fidelities

def main():
    display_fidelities('lih_resp_exact', 'lih_resp_expt', 'n', figname='fid_expt')
    display_fidelities('lih_resp_exact', 'lih_resp_proj', 'n', figname='fid_proj')
    display_fidelities('lih_resp_exact', 'lih_resp_pur', 'n', figname='fid_pur')

if __name__ == '__main__':
    main()
