import sys
sys.path.append('../../..')

from fd_greens import plot_spectral_function, plot_trace_self_energy

def main():
    print("Start plotting data.")

    fnames = ['lih_greens_exact', 'lih_greens_tomo_trace', 'lih_greens_tomo2q_trace']
    suffixes = ['', '', '']
    labels = ['Exact', 'Base', '2Q']

    plot_spectral_function(fnames, suffixes, labels=labels, text="legend", dirname="figs/data")
    plot_trace_self_energy(fnames, suffixes, labels=labels, text="legend", dirname="figs/data")

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
