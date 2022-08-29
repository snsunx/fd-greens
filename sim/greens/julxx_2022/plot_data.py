import sys
sys.path.append('../../..')

from fd_greens import plot_spectral_function, plot_trace_self_energy

def main():
    print("Start plotting data.")

    fnames = ['nah_greens_exact', 'nah_greens_tomo', 'nah_greens_tomo2q']
    suffixes = ['', '', '']
    labels = ['Exact', 'Tomo', 'Tomo 2Q']
    plot_spectral_function(fnames, suffixes, labels=labels, text="legend", dirname="figs/data", figname="nah_A")
    plot_trace_self_energy(fnames, suffixes, labels=labels, text="legend", dirname="figs/data", figname="nah_TrSigma")

    fnames = ['kh_greens_exact', 'kh_greens_tomo', 'kh_greens_tomo2q']
    suffixes = ['', '', '']
    labels = ['Exact', 'Tomo', 'Tomo 2Q']
    plot_spectral_function(fnames, suffixes, labels=labels, text="legend", dirname="figs/data", figname="kh_A")
    plot_trace_self_energy(fnames, suffixes, labels=labels, text="legend", dirname="figs/data", figname="kh_TrSigma")

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
