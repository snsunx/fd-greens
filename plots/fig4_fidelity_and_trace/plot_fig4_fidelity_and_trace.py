import sys
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
	'text.usetex': True,
	'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 20,
    'figure.subplot.left': 0.08,
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.93,
    'figure.subplot.bottom': 0.065,
    'lines.linewidth': 2
})


def plot_fidelity(ax: plt.Axes, ax_sample: plt.Axes, ax_cbar: plt.Axes, mol_name: str) -> None:
    """Plots the fidelity of experimental states in Fig. 3.
    
    Args:
        fig: The figure object.
        ax: The axis object of the purity matrix.
        ax_sample: The axis object for the sample tile.
        ax_cbar: The axis object for the coloar bar.
        mol_name: Name of the molecule, "nah" or "kh".
    """
    global fig

    # Load fidelities from experimental files.
    fid_raw = np.loadtxt(f'../../expt/resp/jul20_2022/data/fid_{mol_name}_resp_tomo_raw.dat')
    fid_pur = np.loadtxt(f'../../expt/resp/jul20_2022/data/fid_{mol_name}_resp_tomo_pur.dat')

    dim = fid_raw.shape[0]
    cmap = plt.get_cmap("viridis")

    # Display the fidelities of raw and purified results.
    im = ax.imshow(fid_raw, vmin=0.0, vmax=1.0, cmap=cmap)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            color = 'black' if fid_raw[i, j] > 0.5 else 'white'
            ax.text(j, i, f"{fid_raw[i, j]:.2f}\n({fid_pur[i, j]:.2f})", color=color, ha='center', va='center')

    # Set ticks and tick labels.
    ax.tick_params(length=0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])
    ax.xaxis.tick_top()
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])

    # Set the panel name and observable name.
    ax.text(0.04, 0.97, '(a)', transform=fig.transFigure)
    ax.text(0.29, 0.94, "Fidelity:", transform=fig.transFigure)

    # Display the sample tile.
    ax_sample.imshow([[0.85]], vmin=0.0, vmax=1.0, cmap=cmap)
    ax_sample.set_xticks([])
    ax_sample.set_yticks([])
    ax_sample.text(0, 0, "Raw\n(Pur.)", color='k', ha='center', va='center')

    # Display the color bar.
    fig.colorbar(im, cax=ax_cbar)

def plot_trace(ax: plt.Axes, ax_sample: plt.Axes, ax_cbar: plt.Axes, mol_name: str) -> None:
    """Plots the trace in Fig. 3.

    Args:
        ax: The axis object of the purity matrix.
        ax_sample: The axis object for the sample tile.
        ax_cbar: The axis object for the coloar bar.
        mol_name: Name of the molecule, "nah" or "kh".
    """
    global fig

    # Load traces from exact and experimental files.
    traces_exact = np.loadtxt(f'../../expt/resp/jul20_2022/data/trace_{mol_name}_resp_exact.dat')
    traces_expt = np.loadtxt(f'../../expt/resp/jul20_2022/data/trace_{mol_name}_resp_tomo_pur.dat')
    traces_diff = np.abs(traces_exact - traces_expt)
    trace_diff_max = np.max(traces_diff)
    trace_diff_min = np.min(traces_diff)

    dim = traces_exact.shape[0]
    cmap = plt.get_cmap("BuPu")

    # Display the traces of exact and experimental results.
    im = ax.imshow(traces_diff, vmin=trace_diff_min, vmax=trace_diff_max, cmap=cmap)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            color = 'k' if traces_diff[i, j] < trace_diff_max / 2 else 'w'
            ax.text(j, i, f"{traces_expt[i, j]:.2f}\n({traces_exact[i, j]:.2f})", color=color, ha='center', va='center')

    # Set ticks and tick labels.
    ax.tick_params(length=0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])
    ax.xaxis.tick_top()
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])

    # Set the panel name and observable name.
    ax.text(0.04, 0.47, '(b)', transform=fig.transFigure)
    ax.text(0.32, 0.45, "Trace: ", transform=fig.transFigure)

    # Display the sample tile.
    ax_sample.imshow([[trace_diff_max * 0.25]], vmin=trace_diff_min, vmax=trace_diff_max, cmap=cmap)
    ax_sample.set_xticks([])
    ax_sample.set_yticks([])
    ax_sample.text(0, 0, "Expt.\n(Exact)", color='k', ha='center', va='center')

    # Display the color bar.
    fig.colorbar(im, cax=ax_cbar)

def main():
    print("Start plotting data.")
    global fig

    fig = plt.figure(figsize=(6, 12))
    ax0 = fig.add_axes([0.03, 0.53, 0.85, 0.35])
    ax0_cbar = fig.add_axes([0.85, 0.53, 0.05, 0.35])
    ax0_sample = fig.add_axes([0.47, 0.91, 0.154, 0.077])
    ax1 = fig.add_axes([0.03, 0.03, 0.85, 0.35])
    ax1_cbar = fig.add_axes([0.85, 0.03, 0.05, 0.35])
    ax1_sample = fig.add_axes([0.47, 0.42, 0.154, 0.077])

    plot_fidelity(ax0, ax0_sample, ax0_cbar, 'nah')
    plot_trace(ax1, ax1_sample, ax1_cbar, 'nah')

    fig.savefig(f"{sys.argv[0][5:-3]}.png", dpi=200)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
