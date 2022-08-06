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

def plot_trace(fig: plt.Figure, ax: plt.Axes, ax_sample: plt.Axes, ax_cbar: plt.Axes, mol_name: str) -> None:
    """Plots the trace in Fig. 3.

    Args:
        fig: The figure object.
        ax: The axis object of the purity matrix.
        ax_sample: The axis object for the sample tile.
        ax_cbar: The axis object for the coloar bar.
        mol_name: Name of the molecule, "nah" or "kh".
    """
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

    # Set the ticks to empty.
    ax.set_xticks([])
    ax.set_yticks([])

    # Set the panel name and observable name.
    ax.text(0.03, 0.92, '(c)', transform=fig.transFigure)
    ax.text(0.33, 0.87, "Trace: ", transform=fig.transFigure)

    # Display the sample tile.
    ax_sample.imshow([[trace_diff_max * 0.25]], vmin=trace_diff_min, vmax=trace_diff_max, cmap=cmap)
    ax_sample.set_xticks([])
    ax_sample.set_yticks([])
    ax_sample.text(0, 0, "Expt. \n (Exact)", color='k', ha='center', va='center')

    # Display the color bar.
    fig.colorbar(im, cax=ax_cbar)


def main():
    print("Start plotting data.")

    # Create Figure and Axes objects.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.03, 0.03, 0.85, 0.7])
    ax_cbar = fig.add_axes([0.85, 0.03, 0.05, 0.7])
    ax_sample = fig.add_axes([0.47, 0.81, 0.16, 0.16])
    
    plot_trace(fig, ax, ax_sample, ax_cbar, 'nah')

    fig.savefig(f"fig3c_trace.png", dpi=200)

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
