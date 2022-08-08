import numpy as np
from typing import Sequence

import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 20,
    'figure.subplot.left': 0.1,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.93,
    'figure.subplot.bottom': 0.065,
    'lines.linewidth': 2
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
        ax: The axes object.
        mol_name: The molecule name. "nah" or "kh".
        panel_name:  The text to be put in the panel label, e.g. "(a)".
        include_xlabel: Whether to include x-axis label.
        include_ylabel: Whether to include y-axis label.
    """
    assert mol_name in ['nah', 'kh']

    global fig

    omegas, As = np.loadtxt(
        f'../../expt/greens/jul20_2022/data/{mol_name}_greens_exact_A.dat').T
    ax.plot(omegas, As, color='k', label="Exact")

    omegas, As = np.loadtxt(
        f'../../expt/greens/jul20_2022/data/{mol_name}_greens_tomo_pur_A.dat').T
    ax.plot(omegas, As, ls='--', lw=3, label="iToffoli")

    omegas, As = np.loadtxt(
        f'../../expt/greens/jul20_2022/data/{mol_name}_greens_tomo2q_pur_A.dat').T
    ax.plot(omegas, As, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    if include_xlabel:
        ax.set_xlabel("$\omega$ (eV)")
    if include_ylabel:
        ax.set_ylabel("$A$ (eV$^{-1}$)")

    ax.legend(ncol=3, loc='center', bbox_to_anchor=(
        0.5, 0.965, 0.0, 0.0), bbox_transform=fig.transFigure)


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
        ax: The axes object.
        mol_name: The molecule name. "nah" or "kh".
        panel_name:  The text to be put in the panel label, e.g. "(a)".
        mode: Component of the trace of self-energy, "real" or "imag".
        include_xlabel: Whether to include x-axis label.
        include_ylabel: Whether to include y-axis label.
    """
    assert mol_name in ['nah', 'kh']
    assert mode in ['real', 'imag']

    omegas, reals, imags = np.loadtxt(
        '../../expt/greens/jul20_2022/data/nah_greens_exact_TrSigma.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt(
        '../../expt/greens/jul20_2022/data/nah_greens_tomo_pur_TrSigma.dat').T

    if mode == 'real':
        ax.plot(omegas, reals, ls='--', lw=3, label="iToffoli")
    else:
        ax.plot(omegas, imags, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt(
        '../../expt/greens/jul20_2022/data/nah_greens_tomo2q_pur_TrSigma.dat').T
    if mode == 'real':
        ax.plot(omegas, reals, ls='--', lw=3, label="CZ")
    else:
        ax.plot(omegas, imags, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    if include_xlabel:
        ax.set_xlabel("$\omega$ (eV)")
    if include_ylabel:
        ylabel_string = mode[0].upper() + mode[1] + " Tr$\Sigma$ (eV)"
        ax.set_ylabel(ylabel_string)


def plot_chi(
    ax: plt.Axes,
    mol_name: str,
    panel_name: str,
    component: str,
    mode: str,
    include_xlabel: bool = True,
    include_ylabel: bool = True
) -> None:
    """Plots the charge-charge response function in Fig. 2.
    
    Args:
        ax: The axes object.
        mol_name: The molecule name. "nah" or "kh".
        panel_name:  The text to be put in the panel label, e.g. "(a)".
        component: Component of the observable, "00", "01", "10" or "11".
        mode: Real or imaginary part of the observable, "real" or "imag".
        include_xlabel: Whether to include x-axis label.
        include_ylabel: Whether to include y-axis label.
    """
    assert mol_name in ['nah', 'kh']
    assert component in ['00', '01']
    assert mode in ['real', 'imag']

    global fig

    omegas, reals, imags = np.loadtxt(
        f'../../expt/resp/jul20_2022/data/{mol_name}_resp_exact_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt(
        f'../../expt/resp/jul20_2022/data/{mol_name}_resp_tomo_pur_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt(
        f'../../expt/resp/jul20_2022/data/{mol_name}_resp_tomo2q_pur_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    if include_xlabel:
        ax.set_xlabel("$\omega$ (eV)")
    if include_ylabel:
        ylabel_string = mode[0].upper() + mode[1] + "$\chi_{" + component + "}$ (eV$^{-1}$)"
        ax.set_ylabel(ylabel_string)


def main():
    print("Start plotting data.")
    global fig

    fig, axes = plt.subplots(3, 2, figsize=(10, 14), sharex='col', sharey='row')

    plot_A(axes[0, 0], 'nah', '(a) NaH', include_xlabel=False, include_ylabel=True)
    plot_A(axes[0, 1], 'kh', '(b) KH', include_xlabel=False, include_ylabel=False)

    # plot_TrSigma(axes[1, 0], 'nah', '(c)', 'real', include_xlabel=False, include_ylabel=True)
    # plot_TrSigma(axes[1, 1], 'kh', '(d)', 'real', include_xlabel=False, include_ylabel=False)

    plot_chi(axes[1, 0], 'nah', '(c) NaH', '00', 'imag', include_xlabel=False, include_ylabel=True)
    plot_chi(axes[1, 1], 'kh', '(d) KH', '00', 'imag', include_xlabel=True, include_ylabel=False)

    plot_chi(axes[2, 0], 'nah', '(e) NaH', '01', 'imag', include_xlabel=True, include_ylabel=True)
    plot_chi(axes[2, 1], 'kh', '(f) KH', '01', 'imag', include_xlabel=True, include_ylabel=False)

    # plot_nah_TrSigma(ax[0, 1], 'imag')
    # plot_nah_chi00(ax[0, 2])

    # plot_kh_A(ax[1, 0])
    # plot_kh_TrSigma(ax[1, 1], 'imag')
    # plot_kh_chi00(ax[1, 2])

    fig.savefig(f"fig2_itoffoli_vs_cz.png", dpi=300)

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
