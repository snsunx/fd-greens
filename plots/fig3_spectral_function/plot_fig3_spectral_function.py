import sys
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 23,
    'figure.subplot.left': 0.09,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.96,
    'figure.subplot.bottom': 0.17,
    'figure.subplot.wspace': 0.0,
    'lines.linewidth': 2,
    'lines.markersize': 8
})


def plot_A(
    ax: plt.Axes,
    # mol_name: str,
    datfname_exact: str,
    datfname_expt: str, 
    panel_name: str,
    # include_xlabel: bool = True,
    include_ylabel: bool = True,
    include_legend: bool = True,
) -> None:
    """Plots the spectral function in Fig. 2.
    
    Args:
        ax: The Axes object.
        mol_name: The molecule name. "nah" or "kh".
        panel_name:  The text to be put in the panel label, e.g. "(a)".
        include_xlabel: Whether to include x-axis label.
        include_ylabel: Whether to include y-axis label.
    """
    global fig

    omegas, As = np.loadtxt(datfname_exact + ".dat").T
    ax.plot(omegas, As, color='k', label="Exact")

    omegas, As = np.loadtxt(datfname_expt + ".dat").T
    ax.plot(omegas, As, ls='--', lw=3, color='xkcd:medium blue', marker='x', markevery=0.12, label="Expt.")

    ax.text(0.92, 0.9, r"\textbf{" + panel_name + "}", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    if include_ylabel:
        ax.set_ylabel("$A$ (eV$^{-1}$)")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    if include_legend:
        ax.legend(loc='center', bbox_to_anchor=(0.23, 0.87, 0.0, 0.0), frameon=False)


def main():
    print("Start plotting data.")
    global fig

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    plot_A(
        axes[0],
        f'../../expt/greens/jul20_2022/data/nah_greens_exact_A',
        f'../../expt/greens/jul20_2022/data/nah_greens_tomo_pur_A',
        'A',
        include_legend=True
    )
    plot_A(
        axes[1],
        f'../../expt/greens/jul20_2022/data/kh_greens_exact_A',
        f'../../expt/greens/jul20_2022/data/kh_greens_tomo_pur_A',
        'B',
        include_ylabel=False,
        include_legend=False
    )

    fig.savefig("fig3_spectral_function.png", dpi=300)

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
