import sys
sys.path.append('..')

from fd_greens import plot_spectral_function, plot_trace_self_energy

def main():
    print("Start plotting data.")

    datdirname = "jul08_2022"
    fnames = ['lih_greens_exact', 'lih_greens_noisy', 'lih_greens_tomo_raw', 'lih_greens_tomo_pur']
    suffixes = ['', '', '', '']
    labels = ['Exact', 'Noisy Sim', "Raw", "Purified"]

    plot_spectral_function(fnames, suffixes, datdirname=datdirname, labels=labels, text="legend", dirname="figs/data")
    plot_trace_self_energy(fnames, suffixes, datdirname=datdirname, labels=labels, text="legend", dirname="figs/data")

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
