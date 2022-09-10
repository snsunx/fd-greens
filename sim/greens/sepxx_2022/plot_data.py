import sys
sys.path.append('../../..')
import argparse

from fd_greens import plot_spectral_function, plot_trace_self_energy

def main():
    print("Start plotting data.")

    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', type=str, nargs='+')
    parser.add_argument('-n', type=str, dest='figname')
    args = parser.parse_args()

    # fnames = ['nah_greens_exact', 'nah_greens_tomo_pur', 'nah_greens_tomo2q_pur']
    suffixes = ['', '', '']
    labels = ['Exact', 'iToffoli', 'CZ']

    plot_spectral_function(args.fnames, suffixes, labels=labels, text="legend", dirname="figs/data", figname=args.figname + '_A')
    plot_trace_self_energy(args.fnames, suffixes, labels=labels, text="legend", dirname="figs/data", figname=args.figname + '_TrSigma')

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
