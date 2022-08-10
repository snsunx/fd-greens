import sys
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 22,
    'figure.subplot.left': 0.1,
    'figure.subplot.right': 0.96,
    'figure.subplot.top': 0.87,
    'figure.subplot.bottom': 0.14,
    'lines.linewidth': 1.5,
    'lines.markersize': 8
})


def plot_A(
    ax: plt.Axes,
    mol_name: str,
    panel_name: str,
    include_xlabel: bool = True,
    include_ylabel: bool = True
) -> None:
    """Plots the spectral function in Fig. 2.
    
    Args:
        ax: The Axes object.
        mol_name: The molecule name. "nah" or "kh".
        panel_name:  The text to be put in the panel label, e.g. "(a)".
        include_xlabel: Whether to include x-axis label.
        include_ylabel: Whether to include y-axis label.
    """
    assert mol_name in ['nah', 'kh']

    global fig

    omegas, As = np.loadtxt(f'../../expt/greens/jul20_2022/data/{mol_name}_greens_exact_A.dat').T
    ax.plot(omegas, As, color='k', label="Exact")

    omegas, As = np.loadtxt(f'../../expt/greens/jul20_2022/data/{mol_name}_greens_tomo_pur_A.dat').T
    ax.plot(omegas, As, ls='--', lw=3, marker='x', markevery=0.12, label="Hardware")

    # omegas, As = np.loadtxt(f'../../expt/greens/jul20_2022/data/{mol_name}_greens_tomo2q_pur_A.dat').T
    # ax.plot(omegas, As, ls='--', lw=3, marker='x', markevery=0.12, label="CZ")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    if include_xlabel:
        ax.set_xlabel("$\omega$ (eV)")
    if include_ylabel:
        ax.set_ylabel("$A$ (eV$^{-1}$)")

    ax.legend(ncol=3, loc='center', bbox_to_anchor=(0.5, 0.94, 0.0, 0.0), bbox_transform=fig.transFigure)


def plot_TrSigma(
    ax: plt.axes,
    mol_name: str,
    panel_name: str,
    mode: str,
    include_xlabel: bool = True,
    include_ylabel: bool = False
) -> None:
    """Plots trace of the self-energy in Fig. 2.
    
    Args:
        ax: The Axes object.
        mol_name: The molecule name. "nah" or "kh".
        panel_name:  The text to be put in the panel label, e.g. "(a)".
        mode: Component of the trace of self-energy, "real" or "imag".
        include_xlabel: Whether to include x-axis label.
        include_ylabel: Whether to include y-axis label.
    """
    assert mol_name in ['nah', 'kh']
    assert mode in ['real', 'imag']

    omegas, reals, imags = np.loadtxt('../../expt/greens/jul20_2022/data/nah_greens_exact_TrSigma.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt('../../expt/greens/jul20_2022/data/nah_greens_tomo_pur_TrSigma.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt('../../expt/greens/jul20_2022/data/nah_greens_tomo2q_pur_TrSigma.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    if include_xlabel:
        ax.set_xlabel("$\omega$ (eV)")
    if include_ylabel:
        ylabel_string = mode[0].upper() + mode[1] + " Tr$\Sigma$ (eV)"
        ax.set_ylabel(ylabel_string)


def main():
    print("Start plotting data.")
    global fig

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex='col', sharey='row')

    plot_A(axes[0], 'nah', '(a) NaH', include_xlabel=True, include_ylabel=True)
    plot_A(axes[1], 'kh', '(b) KH', include_xlabel=True, include_ylabel=False)

    # plot_TrSigma(axes[1, 0], 'nah', '(c)', 'real', include_xlabel=False, include_ylabel=True)
    # plot_TrSigma(axes[1, 1], 'kh', '(d)', 'real', include_xlabel=False, include_ylabel=False)

    # plot_nah_TrSigma(ax[0, 1], 'imag')
    # plot_nah_chi00(ax[0, 2])

    # plot_kh_A(ax[1, 0])
    # plot_kh_TrSigma(ax[1, 1], 'imag')
    # plot_kh_chi00(ax[1, 2])

    fig.savefig(f"{sys.argv[0][5:-3]}.png", dpi=300)

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
