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

def plot_A(ax: plt.Axes, mol_name: str, panel_name: str) -> None:
    assert mol_name in ['nah', 'kh']

    global fig

    omegas, As = np.loadtxt(f'../expt/greens/jul20_2022/data/{mol_name}_greens_exact_A.dat').T
    ax.plot(omegas, As, color='k', label="Exact")

    omegas, As = np.loadtxt(f'../expt/greens/jul20_2022/data/{mol_name}_greens_tomo_pur_A.dat').T
    ax.plot(omegas, As, ls='--', lw=3, label="iToffoli")

    omegas, As = np.loadtxt(f'../expt/greens/jul20_2022/data/{mol_name}_greens_tomo2q_pur_A.dat').T
    ax.plot(omegas, As, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("$A$ (eV$^{-1}$)")

    ax.legend(ncol=3, loc='center', bbox_to_anchor=(0.5, 0.965, 0.0, 0.0), bbox_transform=fig.transFigure)


def plot_TrSigma(ax: plt.axes, mol_name: str, panel_name: str, mode: str) -> None:
    assert mol_name in ['nah', 'kh']
    assert mode in ['real', 'imag']

    omegas, reals, imags = np.loadtxt('../expt/greens/jul20_2022/data/nah_greens_exact_TrSigma.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt('../expt/greens/jul20_2022/data/nah_greens_tomo_pur_TrSigma.dat').T

    if mode == 'real':
        ax.plot(omegas, reals, ls='--', lw=3, label="iToffoli")
    else:
        ax.plot(omegas, imags, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt('../expt/greens/jul20_2022/data/nah_greens_tomo2q_pur_TrSigma.dat').T
    if mode == 'real':
        ax.plot(omegas, reals, ls='--', lw=3, label="CZ")
    else:
        ax.plot(omegas, imags, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Tr$\Sigma$ (eV)")

def plot_chi(ax: plt.Axes, mol_name: str, panel_name: str, component: str, mode: str) -> None:
    assert mol_name in ['nah', 'kh']
    assert component in ['00', '01']
    assert mode in ['real', 'imag']

    global fig

    omegas, reals, imags = np.loadtxt(f'../expt/resp/jul20_2022/data/{mol_name}_resp_exact_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt(f'../expt/resp/jul20_2022/data/{mol_name}_resp_tomo_pur_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="iToffoli")

    omegas, reals, imags = np.loadtxt(f'../expt/resp/jul20_2022/data/{mol_name}_resp_tomo2q_pur_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="CZ")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    if mode == 'real':
        ax.set_ylabel("Re $\chi_{" + component + "}$ (eV$^{-1}$)")
    else:
        ax.set_ylabel("Im $\chi_{" + component + "}$ (eV$^{-1}$)")
    
def main():
    print("Start plotting data.")
    global fig

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    plot_A(axes[0, 0], 'nah', '(a)')
    plot_A(axes[1, 0], 'kh', '(b)')

    # plot_TrSigma(axes[0, 1], 'nah', '(c)', 'real')
    # plot_TrSigma(axes[1, 1], 'kh', '(d)', 'real')

    plot_chi(axes[0, 1], 'nah', '(c)', '00', 'imag')
    plot_chi(axes[1, 1], 'kh', '(d)', '00', 'imag')

    plot_chi(axes[0, 2], 'nah', '(e)', '01', 'imag')
    plot_chi(axes[1, 2], 'kh', '(f)', '01', 'imag')

    # plot_nah_TrSigma(ax[0, 1], 'imag')
    # plot_nah_chi00(ax[0, 2])

    # plot_kh_A(ax[1, 0])
    # plot_kh_TrSigma(ax[1, 1], 'imag')
    # plot_kh_chi00(ax[1, 2])

    fig.savefig(f"figs/fig2_itoffoli_vs_cz.png", dpi=300)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
