import numpy as np
from typing import Optional

import matplotlib.pyplot as plt

plt.rcParams.update({
	'text.usetex': True,
	'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 20,
    'figure.subplot.left': 0.07,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.95,
    'figure.subplot.bottom': 0.06,
    'figure.subplot.wspace': 0.24,
    'lines.linewidth': 3,
    'lines.markersize': 10,
    'lines.markeredgewidth': 1.5,
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
    include_legend: bool = True,
) -> None:
    """Plots the response function by comparing iToffoli vs CZ decompositions."""
    omegas, reals, imags = np.loadtxt(datfname_exact + ".dat").T
    obs = reals if use_real_part else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    if datfname_pur is not None:
        omegas, reals, imags = np.loadtxt(datfname_pur + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='-.', color='xkcd:medium blue', label="iToffoli w/o RC")

    if datfname_2q_pur is not None:
        omegas, reals, imags = np.loadtxt(datfname_2q_pur + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='-.', color='xkcd:pinkish', label="CZ w/o RC")

    if datfname_pur_rc is not None:
        omegas, reals, imags = np.loadtxt(datfname_pur_rc + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', color='xkcd:medium blue', label="iToffoli w/ RC")

    if datfname_2q_pur_rc is not None:
        omegas, reals, imags = np.loadtxt(datfname_2q_pur_rc + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', color='xkcd:pinkish', label="CZ w/ RC")

    ax.text(-0.13, 1.00, r"\textbf{" + panel_name + "}", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    # ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    if panel_name == "D":
        ax.set_yticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    prefix = "Re" if use_real_part else "Im"
    ylabel_string = prefix + " $\chi_{" + datfname_exact[-2:] + "}$ (eV$^{-1}$)"
    ax.set_ylabel(ylabel_string)
    if include_legend:
        ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.04, 0.5, 0.5), frameon=False, fontsize=18)

def plot_chi_purified_vs_trace_corr(
    ax: plt.Axes,
    datfname_exact: str,
    datfname_pur: Optional[str] = None,
    datfname_trace: Optional[str] = None,
    datfname_pur_rc: Optional[str] = None,
    datfname_trace_rc: Optional[str] = None,
    use_real_part: bool = False,
    include_legend: bool = True,
    panel_name: str = '(a)',
    component: str ='00', 
) -> None:
    """Plots the response function by comparing purified and trace corrected results."""
    omegas, reals, imags = np.loadtxt(datfname_exact + ".dat").T
    obs = reals if use_real_part else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    if datfname_pur is not None:
        omegas, reals, imags = np.loadtxt(datfname_pur + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='-.', color='xkcd:medium blue', label="Purified w/o RC")

    if datfname_trace is not None:
        omegas, reals, imags = np.loadtxt(datfname_trace + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='-.', color='xkcd:grass green', label="Trace-corrected\nw/o RC")

    if datfname_pur_rc is not None:
        omegas, reals, imags = np.loadtxt(datfname_pur_rc + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', color='xkcd:medium blue', label="Purified w/ RC")

    if datfname_trace_rc is not None:
        omegas, reals, imags = np.loadtxt(datfname_trace_rc + ".dat").T
        obs = reals if use_real_part else imags
        ax.plot(omegas, obs, ls='--', color='xkcd:grass green', label="Trace-corrected\nw/ RC")

    ax.text(-0.13, 1.00, r"\textbf{" + panel_name + "}", transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    # ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    prefix = "Re" if use_real_part else "Im"
    ylabel_string = prefix + " $\chi_{" + datfname_exact[-2:] + "}$ (eV$^{-1}$)"
    ax.set_ylabel(ylabel_string)
    if include_legend:
        ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.52, 0.5, 0.5), frameon=False, fontsize=18)
    
def main():
    print("Start plotting data.")

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    plot_chi_itoffoli_vs_cz(
        axes[0, 0],
        f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_exact_chi00',
        datfname_pur=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo_pur_chi00',
        datfname_2q_pur=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo2q_pur_chi00',
        use_real_part=False,
        panel_name="A",
    )

    plot_chi_itoffoli_vs_cz(
        axes[1, 0],
        f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_exact_chi00',
        datfname_pur_rc=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo_rc_pur_chi00',
        datfname_2q_pur_rc=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo2q_rc_pur_chi00',
        use_real_part=False,
        panel_name="B",
    )

    plot_chi_itoffoli_vs_cz(
        axes[0, 1],
        f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_exact_chi01',
        datfname_pur=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo_pur_chi01',
        datfname_2q_pur=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo2q_pur_chi01',
        use_real_part=False,
        panel_name="C",
    )

    plot_chi_itoffoli_vs_cz(
        axes[1, 1],
        f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_exact_chi01',
        datfname_pur_rc=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo_rc_pur_chi01',
        datfname_2q_pur_rc=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo2q_rc_pur_chi01',
        use_real_part=False,
        panel_name="D",
    )

    plot_chi_purified_vs_trace_corr(
        axes[0, 2],
        f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_exact_chi01',
        datfname_pur=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo_pur_chi01',
        datfname_trace=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo_trace_chi01',
        use_real_part=False,
        panel_name="E",
    )

    plot_chi_purified_vs_trace_corr(
        axes[1, 2],
        f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_exact_chi01',
        datfname_pur_rc=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo_rc_pur_chi01',
        datfname_trace_rc=f'../../expt/resp/aug13_2022_nah/data/obs/nah_resp_tomo_rc_trace_chi01',
        use_real_part=False,
        panel_name="F",
    )
    
    fig.savefig(f"fig6_response_function.png", dpi=200)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
