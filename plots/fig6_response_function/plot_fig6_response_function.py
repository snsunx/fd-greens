from xml.etree.ElementInclude import include
import numpy as np
from typing import Optional

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
    'lines.markeredgewidth': 2,
})


def plot_chi_itoffoli_vs_cz(
    ax: plt.Axes,
    datfname_exact: str,
    datfname_pur: Optional[str] = None,
    datfname_2q_pur: Optional[str] = None,
    datfname_pur_rc: Optional[str] = None,
    datfname_2q_pur_rc: Optional[str] = None,
    use_real_part: bool = False,
    panel_name: str = 'B',
    include_legend: bool = False,
    include_xlabel: bool = True,
    include_ylabel: bool = True,
) -> None:
    """Plots the response function by comparing iToffoli vs CZ decompositions."""
    global handles1, labels1

    component = datfname_exact[-2:]
    print("component =", component)

    omegas, reals, imags = np.loadtxt(datfname_exact + ".dat").T
    obs = reals if use_real_part else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    if datfname_pur is not None:
        omegas, reals, imags = np.loadtxt(datfname_pur + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', marker='x', markevery=0.15, color='xkcd:medium blue', label="iToffoli")

    if datfname_2q_pur is not None:
        omegas, reals, imags = np.loadtxt(datfname_2q_pur + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', marker='x', markevery=0.15, color='xkcd:pinkish', label="CZ")

    if datfname_pur_rc is not None:
        omegas, reals, imags = np.loadtxt(datfname_pur_rc + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', marker='x', markevery=0.12, color='xkcd:medium blue', label="iToffoli")

    if datfname_2q_pur_rc is not None:
        omegas, reals, imags = np.loadtxt(datfname_2q_pur_rc + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', marker='x', markevery=0.12, color='xkcd:pinkish', label="CZ")

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
        ax.text(0.45, 1.04, "RC", transform=ax.transAxes)
    elif panel_name == "B":
        ax.text(0.45, 1.04, "No RC", transform=ax.transAxes)

    handles1, labels1 = ax.get_legend_handles_labels()



def plot_chi_purified_vs_trace_corr(
    ax: plt.Axes,
    datfname_exact: str,
    datfname_pur: Optional[str] = None,
    datfname_trace: Optional[str] = None,
    datfname_pur_rc: Optional[str] = None,
    datfname_trace_rc: Optional[str] = None,
    use_real_part: bool = False,
    include_legend: bool = True,
    include_xlabel: bool = True,
    include_ylabel: bool = True,
    panel_name: str = '(a)',
    component: str ='00', 
) -> None:    
    """Plots the response function by comparing purified and trace corrected results."""
    global handles2, labels2

    component = datfname_exact[-2:]
    print("component =", component)
    
    omegas, reals, imags = np.loadtxt(datfname_exact + ".dat").T
    obs = reals if use_real_part else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    if datfname_pur is not None:
        omegas, reals, imags = np.loadtxt(datfname_pur + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', marker='+', markevery=0.15, color='xkcd:medium blue', label="Purified w/o RC")

    if datfname_trace is not None:
        omegas, reals, imags = np.loadtxt(datfname_trace + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', marker='+', markevery=0.15, color='xkcd:grass green', label="Trace-corrected\nw/o RC")

    if datfname_pur_rc is not None:
        omegas, reals, imags = np.loadtxt(datfname_pur_rc + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', marker='x', markevery=0.12, color='xkcd:medium blue', label="Purified w/ RC")

    if datfname_trace_rc is not None:
        omegas, reals, imags = np.loadtxt(datfname_trace_rc + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', marker='x', markevery=0.12, color='xkcd:grass green', label="Trace-corrected\nw/ RC")

    ax.text(0.92, 0.9, r"\textbf{" + panel_name + "}", transform=ax.transAxes)

    if include_xlabel:
        ax.set_xlabel("$\omega$ (eV)")
    if panel_name == "E":
        ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    else:
        ax.set_yticks([-0.1, 0.0, 0.1])
    if include_ylabel:
        prefix = "Re" if use_real_part else "Im"
        ylabel_string = prefix + " $\chi_{" + datfname_exact[-2:] + "}$ (eV$^{-1}$)"
        ax.set_ylabel(ylabel_string)

    # if include_legend:
    #     ax.legend(loc='center', bbox_to_anchor=(0.75, 0.25, 0.0, 0.0), frameon=False, fontsize=19)
    
    handles2, labels2 = ax.get_legend_handles_labels()

def place_legend() -> None:
    global fig, handles1, labels1, handles2, labels2

    handles = [handles1[0], handles1[2], handles1[1]]
    labels = ["Exact", "CZ", "iToffoli"]

    plt.legend(
        ncol=3,
        handles=handles,
        labels=labels,
        frameon=False,
        bbox_to_anchor=(0.52, 0.96, 0.0, 0.0),
        loc='center',
        bbox_transform=fig.transFigure,
        columnspacing=7
    )
    
def main():
    print("Start plotting data.")

    global fig
    mol_name = "nah"

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    plot_chi_itoffoli_vs_cz(
        axes[0, 0],
        f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_exact_chi00',
        datfname_pur=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_pur_chi00',
        datfname_2q_pur=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo2q_pur_chi00',
        use_real_part=False,
        panel_name="A",
        include_legend=True
    )

    plot_chi_itoffoli_vs_cz(
        axes[0, 1],
        f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_exact_chi00',
        datfname_pur_rc=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_rc_pur_chi00',
        datfname_2q_pur_rc=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo2q_rc_pur_chi00',
        use_real_part=False,
        panel_name="B",
        # include_ylabel=False
    )

    plot_chi_itoffoli_vs_cz(
        axes[1, 0],
        f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_exact_chi01',
        datfname_pur=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_pur_chi01',
        datfname_2q_pur=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo2q_pur_chi01',
        use_real_part=False,
        panel_name="C",
    )

    plot_chi_itoffoli_vs_cz(
        axes[1, 1],
        f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_exact_chi01',
        datfname_pur_rc=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_rc_pur_chi01',
        datfname_2q_pur_rc=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo2q_rc_pur_chi01',
        use_real_part=False,
        panel_name="D",
        # include_ylabel=False
    )

    # plot_chi_purified_vs_trace_corr(
    #     axes[2, 0],
    #     f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_exact_chi01',
    #     datfname_pur=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_pur_chi01',
    #     datfname_trace=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_trace_chi01',
    #     use_real_part=False,
    #     panel_name="E",
    # )

    # plot_chi_purified_vs_trace_corr(
    #     axes[2, 1],
    #     f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_exact_chi01',
    #     datfname_pur_rc=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_rc_pur_chi01',
    #     datfname_trace_rc=f'../../expt/resp/aug13_2022_{mol_name}/data/obs/{mol_name}_resp_tomo_rc_trace_chi01',
    #     use_real_part=False,
    #     panel_name="F",
    #     include_ylabel=False
    # )

    # axes[0, 0].text(0.9, 0.9, "RC", transform=axes[0, 0].transAxes)
    # axes[0, 1].text(0.6, 0.6, "No RC", transform=axes[0, 1].transAxes)
    
    # place_legend()

    fig.savefig(f"fig6_response_function.png", dpi=300)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
