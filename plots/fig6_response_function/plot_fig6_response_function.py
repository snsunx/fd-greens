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
    'figure.subplot.left': 0.12,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.94,
    'figure.subplot.bottom': 0.07,
    'figure.subplot.hspace': 0.2,
    'figure.subplot.wspace': 0.28,
    'lines.linewidth': 3,
    'lines.markersize': 11,
    'lines.markeredgewidth': 3,
})

def rms_error(predictions, targets):
    """RMS error between experimental and exact data."""
    return np.sqrt(((predictions - targets) ** 2).mean()) * 1000

def plot_chi_itoffoli_vs_cz(
    fig: plt.Figure,
    ax: plt.Axes,
    datfname_exact: str,
    datfname_itof: str,
    datfname_cz: str,
    use_real_part: bool = False,
    panel_name: str = 'B',
    include_legend: bool = False,
    include_xlabel: bool = True,
    include_ylabel: bool = True,
) -> None:
    """Plots the response function by comparing iToffoli vs CZ decompositions."""
    # Print header
    if panel_name == 'A':
        print(f"========== Panel A: chi00 without RC ==========")
    elif panel_name == 'B':
        print("========== Panel B: chi00 with RC ===========")
    elif panel_name == 'C':
        print("========== Panel C: chi01 without RC ===========")
    elif panel_name == 'D':
        print("========== Panel D: chi01 with RC ==========")
    
    # Load response function from data file
    omegas, reals, imags = np.loadtxt(datfname_exact + ".dat").T
    obs_exact = reals if use_real_part else imags
    peaks, _ = find_peaks(obs_exact, height=0.01)
    amps_exact = obs_exact[peaks]
    ax.plot(omegas, obs_exact, color='k', label="Exact")

    # Plot the iToffoli data
    omegas, reals, imags = np.loadtxt(datfname_itof + ".dat").T
    obs = reals if use_real_part else imags
    ax.plot(
        omegas, obs, ls='--', marker='+', 
        ms=plt.rcParams['lines.markersize'] + 2, 
        markevery=0.12, color='xkcd:medium blue',
        label="iToffoli")

    # Calculate and print out the RMS error
    # print(f'rms_error (itof) = {rms_error(obs_exact, obs):.1f} meV')

    # Calculate and print out the peak height deviation
    peaks_itof, _ = find_peaks(obs, height=0.01)
    amps_itof = obs[peaks_itof]
    deviation_itof = (amps_itof - amps_exact) / amps_exact * 100
    print(f'peak height deviation (itof) = {np.array_str(deviation_itof, precision=2)} %')


    # Plot the CZ data
    omegas, reals, imags = np.loadtxt(datfname_cz + ".dat").T
    obs = reals if use_real_part else imags
    ax.plot(omegas, obs, ls='--', marker='x', markevery=0.12, color='xkcd:pinkish', label="CZ")
    
    # Calculate and print out the RMS error
    # print(f'rms_error (cz) = {rms_error(obs_exact, obs):.1f} meV')

    # Calculate and print out the peak height deviation
    peaks_cz, _ = find_peaks(obs, height=0.01)
    amps_cz = obs[peaks_cz]
    deviation_cz = (amps_cz - amps_exact) / amps_exact * 100
    print(amps_cz, amps_exact)
    print(f'peak height deviation (cz) = {np.array_str(deviation_cz, precision=2)} %')



    # try:
    #     print("peaks_itof      ", peaks_itof)
    #     print("peaks_cz        ", peaks_cz)
    #     print("amps_itof      ", amps_itof)
    #     print("amps_cz        ", amps_cz)
    #     percentage_itof = (amps_itof - amps_exact) / amps_exact * 100
    #     percentage_cz = (amps_cz - amps_exact) / amps_exact * 100
    #     print("deviation itof", np.array_str(percentage_itof, precision=2), "%")
    #     print("deviation cz  ", np.array_str(percentage_cz, precision=2), "%")
    # except:
    #     print("Failed to calculate RMS error")

    ax.text(0.92, 0.9, r"\textbf{" + panel_name + "}", transform=ax.transAxes)

    if include_xlabel:
        ax.set_xlabel("$\omega$ (eV)")
        ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    if panel_name in ["A", "C"]:
        ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    else:
        ax.set_yticks([-0.1, 0.0, 0.1])
    
    if include_ylabel:
        prefix = "Re" if use_real_part else "Im"
        ylabel_string = prefix + " $\chi_{" + datfname_exact[-2:] + "}$ (eV$^{-1}$)"
        ax.set_ylabel(ylabel_string)

    if include_legend:
        ax.legend(loc='center', bbox_to_anchor=(0.25, 0.25, 0.0, 0.0), frameon=False, fontsize=22)
    
    if panel_name == "A":
        ax.text(0.5, 1.04, "Without RC", ha='center', transform=ax.transAxes)
    elif panel_name == "B":
        ax.text(0.5, 1.04, "With RC", ha='center', transform=ax.transAxes)

    # handles1, labels1 = ax.get_legend_handles_labels()
    
def main():
    print("> Start plotting data.")

    # NOTE: mol_name should be set to 'nah' or 'kh'
    mol_name = 'nah'

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
    plot_chi_itoffoli_vs_cz(fig, ax_a, datfname_exact_00, datfname_itof_a, datfname_cz_a, panel_name='A', include_legend=True)
    plot_chi_itoffoli_vs_cz(fig, ax_b, datfname_exact_00, datfname_itof_b, datfname_cz_b, panel_name='B')
    plot_chi_itoffoli_vs_cz(fig, ax_c, datfname_exact_01, datfname_itof_c, datfname_cz_c, panel_name='C')
    plot_chi_itoffoli_vs_cz(fig, ax_d, datfname_exact_01, datfname_itof_d, datfname_cz_d, panel_name='D')

    # Save the figure
    if mol_name == 'nah':
        fig.savefig(f"fig6_response_function.png", dpi=300)
    elif mol_name == 'kh':
        fig.savefig(f"figs3_response_function_kh.png", dpi=300)    

    print("> Finished plotting data.")


if __name__ == '__main__':
    main()
