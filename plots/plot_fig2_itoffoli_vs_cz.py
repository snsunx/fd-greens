import numpy as np
from typing import Sequence

import matplotlib.pyplot as plt

LINESTYLES_A = {}

plt.rcParams.update({
	'text.usetex': True,
	'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 21,
    'figure.subplot.left': 0.05,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.93,
    'figure.subplot.bottom': 0.065,
    'lines.linewidth': 2
})

def plot_nah_A(ax: plt.Axes) -> None:
    global fig

    omegas, As = np.loadtxt('../sim/greens/jul18_2022/data/nah_greens_exact_A.dat').T
    ax.plot(omegas, As, color='k', label="Exact")

    omegas, As = np.loadtxt('../sim/greens/jul18_2022/data/nah_greens_tomo_A.dat').T
    ax.plot(omegas, As, ls='--', lw=3, label="iToffoli")

    omegas, As = np.loadtxt('../sim/greens/jul18_2022/data/nah_greens_tomo2q_A.dat').T
    ax.plot(omegas, As, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, "(a)", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("$A$ (eV$^{-1}$)")

    ax.legend(ncol=3, loc='center', bbox_to_anchor=(0.5, 0.965, 0.0, 0.0), bbox_transform=fig.transFigure)


def plot_nah_TrSigma(ax: plt.axes, mode: str = 'real') -> None:
    assert mode in ['real', 'imag']

    omegas, reals, imags = np.loadtxt('../sim/greens/jul18_2022/data/nah_greens_exact_TrSigma.dat').T
    if mode == 'real':
        ax.plot(omegas, reals, color='k', label="Exact")
    else:
        ax.plot(omegas, imags, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt('../sim/greens/jul18_2022/data/nah_greens_tomo_TrSigma.dat').T
    if mode == 'real':
        ax.plot(omegas, reals, ls='--', lw=3, label="iToffoli")
    else:
        ax.plot(omegas, imags, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt('../sim/greens/jul18_2022/data/nah_greens_tomo2q_TrSigma.dat').T
    if mode == 'real':
        ax.plot(omegas, reals, ls='--', lw=3, label="CZ")
    else:
        ax.plot(omegas, imags, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, "(c)", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Tr$\Sigma$ (eV)")

def plot_kh_A(ax: plt.Axes) -> None:
    omegas, As = np.loadtxt('../sim/greens/jul18_2022/data/kh_greens_exact_A.dat').T
    ax.plot(omegas, As, color='k', label="Exact")

    omegas, As = np.loadtxt('../sim/greens/jul18_2022/data/kh_greens_tomo_A.dat').T
    ax.plot(omegas, As, ls='--', lw=3, label="iToffoli")

    omegas, As = np.loadtxt('../sim/greens/jul18_2022/data/kh_greens_tomo2q_A.dat').T
    ax.plot(omegas, As, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, "(b)", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("$A$ (eV$^{-1}$)")
    # ax.legend()

def plot_kh_TrSigma(ax: plt.axes, mode: str = 'real') -> None:
    omegas, reals, imags = np.loadtxt('../sim/greens/jul18_2022/data/kh_greens_exact_TrSigma.dat').T
    if mode == 'real':
        ax.plot(omegas, reals, color='k', label="Exact")
    else:
        ax.plot(omegas, imags, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt('../sim/greens/jul18_2022/data/kh_greens_tomo_TrSigma.dat').T
    if mode == 'real':
        ax.plot(omegas, reals, ls='--', lw=3, label="iToffoli")
    else:
        ax.plot(omegas, imags, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt('../sim/greens/jul18_2022/data/kh_greens_tomo2q_TrSigma.dat').T
    if mode == 'real':
        ax.plot(omegas, reals, ls='--', lw=3, label="CZ")
    else:
        ax.plot(omegas, imags, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, "(d)", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Tr$\Sigma$ (eV)")

def plot_kh_imag_TrSigma(ax: plt.axes) -> None:
    omegas, _, imag = np.loadtxt('../sim/greens/jul18_2022/data/kh_greens_exact_TrSigma.dat').T
    ax.plot(omegas, imag, color='k', label="Exact")

    omegas, _, imag = np.loadtxt('../sim/greens/jul18_2022/data/kh_greens_tomo_TrSigma.dat').T
    ax.plot(omegas, imag, ls='--', lw=3, label="iToffoli")

    omegas, _, imag = np.loadtxt('../sim/greens/jul18_2022/data/kh_greens_tomo2q_TrSigma.dat').T
    ax.plot(omegas, imag, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.04, "(f)", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Im Tr$\Sigma$ (eV)")
    # ax.legend()

def plot_nah_chi00(ax: plt.Axes) -> None:
    global fig

    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/nah_resp_exact_chi00.dat').T
    ax.plot(omegas, imags, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/nah_resp_tomo_chi00.dat').T
    ax.plot(omegas, imags, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/nah_resp_tomo2q_chi00.dat').T
    ax.plot(omegas, imags, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, "(a)", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Re $\chi_{00}$ (eV$^{-1}$)")
    # ax.legend()

    ax.legend(ncol=3, loc='center', bbox_to_anchor=(0.5, 0.965, 0.0, 0.0), bbox_transform=fig.transFigure)
    # ax.add_artist(legend_exact)

    # ax.legend()

def plot_nah_chi01(ax: plt.axes) -> None:
    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/nah_resp_exact_chi01.dat').T
    ax.plot(omegas, imags, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/nah_resp_tomo_chi01.dat').T
    ax.plot(omegas, imags, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/nah_resp_tomo2q_chi01.dat').T
    ax.plot(omegas, imags, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, "(c)", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Re $\chi_{01}$ (eV)")
    # ax.legend()

def plot_kh_chi00(ax: plt.Axes) -> None:
    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/kh_resp_exact_chi00.dat').T
    ax.plot(omegas, imags, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/kh_resp_tomo_chi00.dat').T
    ax.plot(omegas, imags, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/kh_resp_tomo2q_chi00.dat').T
    ax.plot(omegas, imags, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, "(c)", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Re $\chi_{00}$ (eV$^{-1}$)")
    # ax.legend()

def plot_kh_chi01(ax: plt.Axes) -> None:
    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/kh_resp_exact_chi01.dat').T
    ax.plot(omegas, imags, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/kh_resp_tomo_chi01.dat').T
    ax.plot(omegas, imags, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt('../sim/resp/jul18_2022/data/kh_resp_tomo2q_chi01.dat').T
    ax.plot(omegas, imags, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, "(d)", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Re $\chi_{01}$ (eV$^{-1}$)")
    # ax.legend()

def main():
    print("Start plotting data.")
    global fig

    fig, ax = plt.subplots(2, 3, figsize=(20, 11))

    plot_nah_A(ax[0, 0])
    plot_nah_TrSigma(ax[0, 1], 'imag')
    plot_nah_chi00(ax[0, 2])

    plot_kh_A(ax[1, 0])
    plot_kh_TrSigma(ax[1, 1], 'imag')
    plot_kh_chi00(ax[1, 2])

    fig.savefig(f"figs/fig2_itoffoli_vs_cz.png", dpi=300)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
