import sys
import numpy as np
from typing import Sequence

import matplotlib.pyplot as plt

plt.rcParams.update({
	'text.usetex': True,
	'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 22,
    'figure.subplot.left': 0.08,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.9,
    'figure.subplot.bottom': 0.05,
    'lines.linewidth': 2
})

def plot_chi(
    ax: plt.Axes,
    mol_name: str,
    panel_name: str,
    component: str,
    mode: str, 
    include_ylabel: bool = True,
    include_legend: bool = False
) -> None:
    omegas, reals, imags = np.loadtxt(f'../../expt/resp/jul20_2022/data/{mol_name}_resp_exact_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt(f'../../expt/resp/jul20_2022/data/{mol_name}_resp_tomo_raw_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="Raw")

    omegas, reals, imags = np.loadtxt(f'../../expt/resp/jul20_2022/data/{mol_name}_resp_tomo_pur_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="Purified")

    omegas, reals, imags = np.loadtxt(f'../../expt/resp/jul20_2022/data/{mol_name}_resp_tomo2q_trace_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="Purified + \n trace corrected")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    if include_ylabel:
        ylabel_string = mode[0].upper() + mode[1] + " $\chi_{" + component + "}$ (eV$^{-1}$)"
        ax.set_ylabel(ylabel_string)

    if include_legend:
        ax.legend(loc='center', bbox_to_anchor=(1.0, 0.0, 0.55, 1.0))


def main():
    print("Start plotting data.")

    fig = plt.figure(figsize=(19, 6))
    ax0 = fig.add_axes([0.07, 0.13, 0.34, 0.8])
    ax1 = fig.add_axes([0.48, 0.13, 0.34, 0.8])

    plot_chi(ax0, 'nah', '(a) NaH', '00', 'imag', include_legend=False)
    plot_chi(ax1, 'kh', '(b) KH', '00', 'imag', include_legend=True)

    fig.savefig(f"{sys.argv[0][5:-3]}.png", dpi=200)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
