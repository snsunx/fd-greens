import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 21,
    'figure.subplot.left': 0.09,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.96,
    'figure.subplot.bottom': 0.17,
    'figure.subplot.wspace': 0.0,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'lines.markeredgewidth': 2.5
})

np.set_printoptions(formatter={'float': '{: 6.3f}'.format})


def plot_spectral_function(
    fig: plt.Figure,
    ax: plt.Axes,
    datfname_exact: str,
    datfname_expt: str,
    panel_name: str,
) -> None:
    """Plots the spectral function in Fig. 3.
    
    Args:
        fig: The Figure object.
        ax: The Axes object.
        datfname_exact: The exact data file name.
        datfname_expt: The experimental data file name.
        panel_name:  The text to be put in the panel label, e.g. "A".
    """
    # Load and plot the exact data
    omegas, As = np.loadtxt(datfname_exact + ".dat").T
    ax.plot(omegas, As, color='k', label="Exact")
    peaks, _ = find_peaks(As)
    amps_exact = As[peaks]

    # Load and plot the experimental data
    omegas, As = np.loadtxt(datfname_expt + ".dat").T
    ax.plot(omegas, As, ls='--', lw=3, color='xkcd:medium blue', marker='x', markevery=0.12, label="Expt.")
    peaks, _ = find_peaks(As)
    amps_expt = As[peaks]

    # Set axis labels and legends
    ax.text(0.92, 0.9, r"\textbf{" + panel_name + "}", transform=ax.transAxes)
    ax.set_xlabel("$\omega$ (eV)")
    if panel_name == 'A':
        ax.set_ylabel("$A$ (eV$^{-1}$)")
        ax.legend(loc='center', bbox_to_anchor=(0.23, 0.87, 0.0, 0.0), frameon=False)
    ax.set_xticks(np.arange(-30, 30 + 1e-8, 10))
    ax.set_yticks(np.arange(0.0, 0.5 + 1e-8, 0.1))

    # Print out information about amplitudes
    print("Exact peaks (eV^-1):       ", amps_exact)
    print("Expt. peaks (eV^-1):       ", amps_expt)
    print("Percentage difference (%): ", (amps_expt - amps_exact) / amps_exact)


def main():
    print("> Start plotting data.")

    # Data file names
    dirname = "sep13_2022"
    datfname_nah_exact = f"../../expt/greens/{dirname}/data/obs/nah_greens_exact_A"
    datfname_nah_expt  = f"../../expt/greens/{dirname}/data/obs/nah_greens_tomo_expt_A"
    datfname_kh_exact  = f"../../expt/greens/{dirname}/data/obs/kh_greens_exact_A"
    datfname_kh_expt   = f"../../expt/greens/{dirname}/data/obs/kh_greens_tomo_expt_A"

    # Create the figure and axis objects
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Plot the figures
    plot_spectral_function(fig, axes[0], datfname_nah_exact, datfname_nah_expt, panel_name='A')
    plot_spectral_function(fig, axes[1], datfname_kh_exact, datfname_kh_expt, panel_name='B')
    fig.savefig("fig3_spectral_function.png", dpi=300)

    print("> Finished plotting data.")


if __name__ == '__main__':
    main()
