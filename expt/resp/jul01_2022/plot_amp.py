import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import display_fidelities, display_traces

def main():
    # display_fidelities('lih_resp_exact', 'lih_resp_expt', 'n', figname='fid_expt')
    # display_fidelities('lih_resp_exact', 'lih_resp_expt2q', 'n', figname='fid_expt2q')
    # display_fidelities('lih_resp_exact', 'lih_resp_pur', 'n', suffix='_miti', figname='fid_pur_miti')
    # display_fidelities('lih_resp_exact', 'lih_resp_pur2q', 'n', suffix='_miti', figname='fid_pur2q_miti')
    display_traces('lih_resp_exact', 'n', figname='trace_exact')
    display_traces('lih_resp_pur', 'n', figname='trace_pur')
    display_traces('lih_resp_pur2q', 'n', figname='trace_pur2q')

if __name__ == '__main__':
    main()
