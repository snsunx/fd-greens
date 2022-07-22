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
    'figure.subplot.left': 0.08,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.93,
    'figure.subplot.bottom': 0.065,
    'lines.linewidth': 2
})

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

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    plot_nah_chi00(axes[0, 0])
    plot_nah_chi01(axes[0, 1])

    plot_kh_chi00(axes[1, 0])
    plot_kh_chi01(axes[1, 1])

    fig.savefig(f"figs/fig3_resp.png", dpi=300)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
