import sys
import numpy as np
from typing import Sequence

import matplotlib.pyplot as plt

plt.rcParams.update({
	'text.usetex': True,
	'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 20,
    'figure.subplot.left': 0.08,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.9,
    'figure.subplot.bottom': 0.05,
    'lines.linewidth': 3
})

def plot_chi(
    ax: plt.Axes,
    datfnames: Sequence[str],
    labels: Sequence[str],
    use_real_part: bool = False,
    include_legend: bool = False,
    panel_name: str = '(a)',
) -> None:

    for datfname, label in zip(datfnames, labels):
        omegas, reals, imags = np.loadtxt(datfname).T
        obs = reals if use_real_part else imags
        ls = '-' if label == "Exact" else '--'
        marker = 'x' if "RC" in label else ''
        ax.plot(omegas, obs, ls=ls, marker=marker, markevery=30, label=label)

    # omegas, reals, imags = np.loadtxt(f'../../expt/resp/jul20_2022/data/obs/nah_resp_tomo_raw_chi{component}.dat').T
    # obs = reals if use_real_part else imags
    # ax.plot(omegas, obs, ls='--', lw=3, label="Raw")

    # omegas, reals, imags = np.loadtxt(f'../../expt/resp/jul20_2022/data/obs/nah_resp_tomo_pur_chi{component}.dat').T
    # obs = reals if use_real_part else imags
    # ax.plot(omegas, obs, ls='--', lw=3, label="Purified")

    # omegas, reals, imags = np.loadtxt(f'../../expt/resp/jul20_2022/data/obs/nah_resp_tomo2q_trace_chi{component}.dat').T
    # obs = reals if use_real_part == 'real' else imags
    # ax.plot(omegas, obs, ls='--', lw=3, label="Purified + \n trace corrected")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    component = "00"
    prefix = "Re" if use_real_part else "Im"
    ylabel_string = prefix + " $\chi_{" + component + "}$ (eV$^{-1}$)"
    ax.set_ylabel(ylabel_string)
    ax.legend()

    # if include_legend:
    #     ax.legend(loc='center', bbox_to_anchor=(1.0, 0.0, 0.55, 1.0))


def main():
    print("Start plotting data.")

    # f'../../expt/resp/jul20_2022/data/obs/nah_resp_exact_chi00.dat')

    fig = plt.figure(figsize=(22, 6.5))
    ax_a = fig.add_axes([0.055, 0.13, 0.27, 0.83])
    ax_b = fig.add_axes([0.385, 0.13, 0.27, 0.83])
    ax_c = fig.add_axes([0.715, 0.13, 0.27, 0.83])

    plot_chi(
        ax_a, 
        [f'../../expt/resp/jul20_2022/data/obs/nah_resp_exact_chi00.dat',
         f'../../expt/resp/jul20_2022/data/obs/nah_resp_tomo_pur_chi00.dat',
         f'../../expt/resp/aug03_2022/data/obs/nah_resp_tomo_rc_pur_chi00.dat'],
        ["Exact", "No RC", "RC"],
        use_real_part=False,
        panel_name="(a)"
    )
    plot_chi(
        ax_b,
        [f'../../expt/resp/jul20_2022/data/obs/nah_resp_exact_chi00.dat',
         f'../../expt/resp/jul20_2022/data/obs/nah_resp_tomo_pur_chi00.dat',
         f'../../expt/resp/jul20_2022/data/obs/nah_resp_tomo2q_pur_chi00.dat'],
        ["Exact", "iToffoli", "CZ"],
        use_real_part=False,
        panel_name="(b)"
    )
    plot_chi(
        ax_c, 
        [f'../../expt/resp/jul20_2022/data/obs/nah_resp_exact_chi00.dat',
         f'../../expt/resp/jul20_2022/data/obs/nah_resp_tomo_raw_chi00.dat',
         f'../../expt/resp/jul20_2022/data/obs/nah_resp_tomo_pur_chi00.dat',
         f'../../expt/resp/jul20_2022/data/obs/nah_resp_tomo_trace_chi00.dat'],
        ["Exact", "Raw", "Purified", "Purified + \nTrace Corr."],
        use_real_part=False,
        panel_name="(c)"
    )
    

    fig.savefig(f"{sys.argv[0][5:-3]}.png", dpi=200)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
