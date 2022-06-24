import sys
sys.path.append('../../..')

from fd_greens import plot_spectral_function, plot_trace_self_energy

def main():
    print("Start plotting data.")

    fnames = ['lih_3A_exact', 'lih_3A_expt1', 'lih_3A_expt1']
    suffixes = ['', '', '_miti']
    labels = ['Exact', 'Expt', 'Expt (miti)']

    plot_spectral_function(fnames, suffixes, labels=labels, text="legend", dirname="figs/data")
    plot_trace_self_energy(fnames, suffixes, labels=labels, text="legend", dirname="figs/data")

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
