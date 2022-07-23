import sys
sys.path.append('../../..')
import argparse

from fd_greens.cirq_ver.plot_utils import display_fidelities, display_traces

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exact', type=str)
    parser.add_argument('expt', type=str)
    parser.add_argument('-n', '--figname', type=str, dest='figname')
    args = parser.parse_args()

    if args.figname is None:
        figname = 'fid_' + args.expt
    else:
        figname = 'fid_' + args.figname

    display_fidelities(args.exact, args.expt, subscript='n', figname=figname)
    display_traces(args.exact, 'n', figname='trace_' + args.exact)
    display_traces(args.expt, 'n', figname='trace_' + args.expt)

    # display_fidelities('lih_resp_exact', 'lih_resp_tomo2q_pur', 'n', figname='fid_tomo2q_pur')
    # display_traces('lih_resp_exact', 'n', figname='trace_exact')
    # display_traces('lih_resp_tomo_pur', 'n', figname='trace_pur')
    # display_traces('lih_resp_tomo2q_pur', 'n', figname='trace_pur2q')

if __name__ == '__main__':
    main()
