import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import display_fidelities, display_traces

def main():
    display_fidelities('nah_resp_exact', 'nah_resp_tomo_raw', 'n', figname='fid_tomo_raw')
    display_fidelities('nah_resp_exact', 'nah_resp_tomo2q_raw', 'n', figname='fid_tomo2q_raw')
    # display_fidelities('nah_resp_exact', 'nah_resp_alltomo_pur', 'n', figname='fid_alltomo_pur')
    # display_fidelities('nah_resp_exact', 'nah_resp_alltomo2q_pur', 'n', figname='fid_alltomo2q_pur')
    # display_traces('nah_resp_exact', 'n', figname='trace_exact')
    # display_traces('nah_resp_tomo_pur', 'n', figname='trace_pur')
    # display_traces('nah_resp_tomo2q_pur', 'n', figname='trace_pur2q')

if __name__ == '__main__':
    main()
