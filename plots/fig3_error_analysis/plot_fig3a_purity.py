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


def plot_purity(fig: plt.Figure, ax: plt.Axes, ax_sample: plt.Axes, ax_cbar: plt.Axes, mol_name: str) -> None:
    """Plots the purity of experimental states in Fig. 3.
    
    Args:
        fig: The figure object.
        ax: The axis object of the purity matrix.
        ax_sample: The axis object for the sample tile.
        ax_cbar: The axis object for the coloar bar.
        mol_name: Name of the molecule, "nah" or "kh".
    """
    purities = np.loadtxt(f'../../expt/resp/jul20_2022/data/purity_{mol_name}_resp_tomo_raw.dat')

    dim = purities.shape[0]
    cmap = plt.get_cmap("winter")

    im = ax.imshow(purities, vmin=0.0, vmax=1.0, cmap=cmap)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            color = 'k' if purities[i, j] > 0.5 else 'w'
            ax.text(j, i, f"{purities[i, j]:.2f}", color=color, ha='center', va='center')

    # Set ticks and tick labels.
    ax.tick_params(length=0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])
    ax.xaxis.tick_top()
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])

    # Set panel label and content name.
    ax.text(0.03, 0.92, '(a)', transform=fig.transFigure)
    ax.text(0.33, 0.88, "Purity:", transform=fig.transFigure)

    # Display the sample tile.
    ax_sample.imshow([[0.85]], vmin=0.0, vmax=1.0, cmap=cmap)
    ax_sample.set_xticks([])
    ax_sample.set_yticks([])
    ax_sample.text(0, 0, "Raw", color='k', ha='center', va='center')

    # Display the color bar.
    fig.colorbar(im, cax=ax_cbar)


def main():
    print("Start plotting data.")

    # Create the figure and axis objects.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.03, 0.03, 0.85, 0.7])
    ax_cbar = fig.add_axes([0.85, 0.03, 0.05, 0.7])
    ax_sample = fig.add_axes([0.47, 0.82, 0.15, 0.15])
    
    plot_purity(fig, ax, ax_sample, ax_cbar, 'nah')

    fig.savefig(f"fig3a_purity.png", dpi=200)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
