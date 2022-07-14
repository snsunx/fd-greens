import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import display_fidelities, display_traces

def main():
    display_fidelities('lih_resp_exact', 'lih_resp_tomo_pur', 'n', figname='fid_tomo_pur')
    display_fidelities('lih_resp_exact', 'lih_resp_tomo2q_pur', 'n', figname='fid_tomo2q_pur')
    display_fidelities('lih_resp_exact', 'lih_resp_alltomo_pur', 'n', figname='fid_alltomo_pur')
    display_fidelities('lih_resp_exact', 'lih_resp_alltomo2q_pur', 'n', figname='fid_alltomo2q_pur')
    display_traces('lih_resp_exact', 'n', figname='trace_exact')
    display_traces('lih_resp_tomo_pur', 'n', figname='trace_pur')
    display_traces('lih_resp_tomo2q_pur', 'n', figname='trace_pur2q')

if __name__ == '__main__':
    main()
