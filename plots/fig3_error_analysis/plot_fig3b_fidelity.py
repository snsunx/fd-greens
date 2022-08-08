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


def plot_fidelity(fig: plt.Figure, ax: plt.Axes, ax_sample: plt.Axes, ax_cbar: plt.Axes, mol_name: str) -> None:
    """Plots the fidelity of experimental states in Fig. 3.
    
    Args:
        fig: The figure object.
        ax: The axis object of the purity matrix.
        ax_sample: The axis object for the sample tile.
        ax_cbar: The axis object for the coloar bar.
        mol_name: Name of the molecule, "nah" or "kh".
    """
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

    # Set the ticks to empty.
    ax.set_xticks([])
    ax.set_yticks([])

    # Set the panel name and observable name.
    ax.text(0.03, 0.92, '(b)', transform=fig.transFigure)
    ax.text(0.29, 0.87, "Fidelity:", transform=fig.transFigure)

    # Display the sample tile.
    ax_sample.imshow([[0.85]], vmin=0.0, vmax=1.0, cmap=cmap)
    ax_sample.set_xticks([])
    ax_sample.set_yticks([])
    ax_sample.text(0, 0, "Raw \n (Pur.)", color='k', ha='center', va='center')

    # Display the color bar.
    fig.colorbar(im, cax=ax_cbar)

def main():
    print("Start plotting data.")
    global fig

    # fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.03, 0.03, 0.85, 0.7])
    ax_cbar = fig.add_axes([0.85, 0.03, 0.05, 0.7])
    ax_sample = fig.add_axes([0.47, 0.81, 0.15, 0.15])
    
    plot_fidelity(fig, ax, ax_sample, ax_cbar, 'nah')

    fig.savefig(f"fig3b_fidelity.png", dpi=200)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
