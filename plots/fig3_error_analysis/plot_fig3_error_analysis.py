import numpy as np
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

def plot_purity(ax: plt.Axes, mol_name: str, panel_name: str) -> None:
    purities = np.loadtxt(f'../expt/resp/jul20_2022/data/purity_{mol_name}_resp_tomo_raw.dat')

    dim = purities.shape[0]

    ax.imshow(purities, vmin=0.0, vmax=1.0)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            if purities[i, j] > 0.5:
                color = 'black'
            else:
                color = 'white'
            ax.text(j, i, f"{purities[i, j]:.2f}", color=color, ha='center', va='center')

    ax.text(-0.9, -0.2, panel_name)

def plot_fidelity(ax: plt.Axes, mol_name: str, panel_name: str):
    fid_raw = np.loadtxt(f'../expt/resp/jul20_2022/data/fid_{mol_name}_resp_tomo_raw.dat')
    fid_pur = np.loadtxt(f'../expt/resp/jul20_2022/data/fid_{mol_name}_resp_tomo_pur.dat')

    dim = fid_raw.shape[0]

    ax.imshow(fid_raw, vmin=0.0, vmax=1.0)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            color = 'black' if fid_raw[i, j] > 0.5 else 'white'
            ax.text(j, i, f"{fid_raw[i, j]:.2f}\n({fid_pur[i, j]:.2f})", color=color, ha='center', va='center')

    ax.text(-0.9, -0.2, panel_name)

def plot_trace(ax: plt.Axes, mol_name: str, panel_name: str) -> None:
    traces_exact = np.loadtxt(f'../expt/resp/jul20_2022/data/trace_{mol_name}_resp_exact.dat')
    traces_expt = np.loadtxt(f'../expt/resp/jul20_2022/data/trace_{mol_name}_resp_tomo_pur.dat')

    dim = traces_exact.shape[0]

    ax.imshow(traces_expt, vmin=0.0, vmax=1.0)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            if traces_expt[i, j] > 0.5:
                color = 'black'
            else:
                color = 'white'
            ax.text(j, i, f"{traces_expt[i, j]:.2f}\n({traces_exact[i, j]:.2f})", color=color, ha='center', va='center')

    ax.text(-0.9, -0.2, panel_name)


def plot_chi(ax: plt.Axes, mol_name: str, panel_name: str, component: str, mode: str, include_ylabel: bool = True) -> None:
    global fig

    omegas, reals, imags = np.loadtxt(f'../expt/resp/jul20_2022/data/{mol_name}_resp_exact_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, color='k', label="Exact")

    omegas, reals, imags = np.loadtxt(f'../expt/resp/jul20_2022/data/{mol_name}_resp_tomo_raw_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="Raw")

    omegas, reals, imags = np.loadtxt(f'../expt/resp/jul20_2022/data/{mol_name}_resp_tomo_pur_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="Purified")

    omegas, reals, imags = np.loadtxt(f'../expt/resp/jul20_2022/data/{mol_name}_resp_tomo2q_trace_chi{component}.dat').T
    obs = reals if mode == 'real' else imags
    ax.plot(omegas, obs, ls='--', lw=3, label="Trace corrected")

    ax.text(0.03, 0.92, panel_name, transform=ax.transAxes)

    ax.set_xlabel("$\omega$ (eV)")
    if include_ylabel:
        ylabel_string = mode[0].upper() + mode[1] + "Re $\chi_{" + component + "}$ (eV$^{-1}$)"
        ax.set_ylabel(ylabel_string)
    # ax.legend()


def main():
    print("Start plotting data.")
    global fig

    # fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    fig = plt.figure(figsize=(20, 11))
    grid = GridSpec(2, 6, figure=fig)

    ax_a = fig.add_subplot(grid[0, :2])
    ax_b = fig.add_subplot(grid[0, 2:4])
    ax_c = fig.add_subplot(grid[0, 4:])

    ax_d = fig.add_subplot(grid[1, :3])
    ax_e = fig.add_subplot(grid[1, 3:])
    
    plot_purity(ax_a, 'nah', '(a)')
    plot_fidelity(ax_b, 'nah', '(b)')
    plot_trace(ax_c, 'nah', '(c)')
    plot_chi(ax_d, 'nah', '(d)', '00', 'imag')
    plot_chi(ax_e, 'nah', '(e)', '01', 'imag', include_ylabel=False)

    fig.savefig(f"figs/fig3_error_analysis.png", dpi=100)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
