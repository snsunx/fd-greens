from typing import Optional

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 23,
    'figure.subplot.left': 0.1,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.94,
    'figure.subplot.bottom': 0.07,
    'figure.subplot.hspace': 0.2,
    'figure.subplot.wspace': 0.28,
    'lines.linewidth': 3,
    'lines.markersize': 11,
    'lines.markeredgewidth': 3,
})


def plot_response_function(
    fig: plt.Figure,
    ax: plt.Axes,
    datfname_exact: str,
    datfname_itof: str,
    datfname_cz: str,
    mol_name: str,
    panel_name: str = 'B',
) -> None:
    """Plots the response function by comparing iToffoli vs CZ decompositions."""
    # Print header
    if panel_name == 'A':
        print("=============== Panel A: chi00 without RC ===============")
    elif panel_name == 'B':
        print("=============== Panel B: chi00 with RC    ===============")
    elif panel_name == 'C':
        print("=============== Panel C: chi01 without RC ===============")
    elif panel_name == 'D':
        print("=============== Panel D: chi01 with RC    ===============")
    
    # Plot the exact data
    omegas, _, obs_exact = np.loadtxt(datfname_exact + ".dat").T
    ax.plot(omegas, obs_exact, color='k', label="Exact")

    # Find peak heights from exact data
    peaks_exact, _ = find_peaks(obs_exact, height=0.01)
    if panel_name == 'C':
        peaks_exact = peaks_exact[::-1]
    amps_exact = obs_exact[peaks_exact]

    # Plot the iToffoli data
    omegas, _, obs_itof = np.loadtxt(datfname_itof + ".dat").T
    ax.plot(
        omegas, obs_itof, ls='--', marker='+',
        ms=plt.rcParams['lines.markersize'] + 2,
        markevery=0.12, color='xkcd:medium blue',
        label="iToffoli")
    
    # Find peak heights from iToffoli data
    peaks_itof, _ = find_peaks(obs_itof, height=0.01)
    amps_itof = obs_itof[peaks_itof]
    deviation_itof = (amps_itof - amps_exact) / amps_exact * 100

    # Plot the CZ data
    omegas, _, obs_cz = np.loadtxt(datfname_cz + ".dat").T
    ax.plot(omegas, obs_cz, ls='--', marker='x', markevery=0.12, color='xkcd:pinkish', label="CZ")

    # Find peak heights from CZ data
    peaks_cz, _ = find_peaks(obs_cz, height=0.01)
    amps_cz = obs_cz[peaks_cz]
    deviation_cz = (amps_cz - amps_exact) / amps_exact * 100

    # Print out information about peak heights
    print(f'Peak locations and amplitudes in exact:    {peaks_exact} {np.array_str(amps_exact, precision=4)}')
    print(f'Peak locations and amplitudes in iToffoli: {peaks_itof} {np.array_str(amps_itof, precision=4)}')
    print(f"Peak locations and amplitudes in CZ:       {peaks_cz} {np.array_str(amps_cz, precision=4)}")
    print(f'Peak height deviation in iToffoli (%):     {np.array_str(deviation_itof, precision=2)}')
    print(f'Peak height deviation in CZ (%):           {np.array_str(deviation_cz, precision=2)}')

    # Set panel name and axis labels
    ax.text(0.92, 0.9, r"\textbf{" + panel_name + "}", transform=ax.transAxes)
    ax.set_xlabel("$\omega$ (eV)")
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ylabel_string = "Im $\chi_{" + datfname_exact[-2:] + "}$ (eV$^{-1}$)"
    ax.set_ylabel(ylabel_string)
    if mol_name == 'nah':
        if panel_name == 'A':
            ax.set_yticks(np.arange(-0.3, 0.3 + 1e-8, 0.1))
        elif panel_name == 'C':
            ax.set_yticks(np.arange(-0.2, 0.2 + 1e-8, 0.1))
        elif panel_name in ['B', 'D']:
            ax.set_yticks(np.arange(-0.15, 0.15 + 1e-8, 0.05))
    elif mol_name == 'kh':
        ax.set_yticks(np.arange(-0.3, 0.3 + 1e-8, 0.1))
        

    # Set legend
    if panel_name == "A":
        ax.legend(loc='center', bbox_to_anchor=(0.25, 0.25, 0.0, 0.0), frameon=False, fontsize=22)
        ax.text(0.5, 1.04, "Without RC", ha='center', transform=ax.transAxes)
    elif panel_name == "B":
        ax.text(0.5, 1.04, "With RC", ha='center', transform=ax.transAxes)

    # handles1, labels1 = ax.get_legend_handles_labels()

def main():
    print("> Start plotting data.")

    # NOTE: mol_name should be set to 'nah' or 'kh'
    mol_name = 'kh'

    # Figure parameters
    nrows = 2
    ncols = 2
    width = 13
    height = 12

    # Data file names
    datfname_exact_00 = f'../../expt/resp/sep13_2022_{mol_name}/data/obs/{mol_name}_resp_exact_chi00'
    datfname_exact_01 = f'../../expt/resp/sep13_2022_{mol_name}/data/obs/{mol_name}_resp_exact_chi01'
    datfname_itof_a   = f'../../expt/resp/sep13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_pur_chi00'
    datfname_cz_a     = f'../../expt/resp/sep13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo2q_pur_chi00'
    datfname_itof_b   = f'../../expt/resp/sep13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_rc_pur_chi00'
    datfname_cz_b     = f'../../expt/resp/sep13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo2q_rc_pur_chi00'
    datfname_itof_c   = f'../../expt/resp/sep13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_pur_chi01'
    datfname_cz_c     = f'../../expt/resp/sep13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo2q_pur_chi01'
    datfname_itof_d   = f'../../expt/resp/sep13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_rc_pur_chi01'
    datfname_cz_d     = f'../../expt/resp/sep13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo2q_rc_pur_chi01'

    # Create the figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]

    # Plot the figures
    plot_response_function(fig, ax_a, datfname_exact_00, datfname_itof_a, datfname_cz_a, mol_name, panel_name='A')
    plot_response_function(fig, ax_b, datfname_exact_00, datfname_itof_b, datfname_cz_b, mol_name, panel_name='B')
    plot_response_function(fig, ax_c, datfname_exact_01, datfname_itof_c, datfname_cz_c, mol_name, panel_name='C')
    plot_response_function(fig, ax_d, datfname_exact_01, datfname_itof_d, datfname_cz_d, mol_name, panel_name='D')

    # Save the figure
    if mol_name == 'nah':
        fig.savefig("fig6_response_function.png", dpi=300)
    elif mol_name == 'kh':
        fig.savefig("figs3_response_function_kh.png", dpi=300)

    print("> Finished plotting data.")


if __name__ == '__main__':
    main()
