import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
	'text.usetex': True,
	'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 25,
    'figure.subplot.left': 0.02,
    'figure.subplot.right': 0.97,
    'figure.subplot.top': 0.93,
    'figure.subplot.bottom': 0.065,
    'lines.linewidth': 2
})


def plot_fidelity_matrix(
    fig: plt.Figure,
    ax: plt.Axes,
    datfname: str,
    panel_name: str = '(a)',
) -> None:
    """Plot the system-qubit state fidelity matrix.
    
    Args:
        fig: The Figure object to plot on.
        ax: The Axes object to plot on.
        datfname: The data file name.
        panel_name: Name of the panel.
    """
    global height, width
    print("=" * 20 + " " + panel_name + " " + "=" * 20)

    bbox = ax.get_position()

    # Load fidelity matrix from data file
    fid_mat = np.loadtxt(datfname) * 100
    dim = fid_mat.shape[0]

    # Calculate the average diagonal and off-diagonal fidelities
    fid_diag_avg = np.mean(np.diag(fid_mat))
    fid_off_diag_avg = (np.sum(fid_mat) - np.trace(fid_mat)) / dim / (dim - 1)
    print(f"Average diagonal fidelity     = {fid_diag_avg:.1f}%")
    print(f"Average off-diagonal fidelity = {fid_off_diag_avg:.1f}%")

    # Display the fidelity matrix with color map
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(fid_mat, vmin=0, vmax=100, cmap=cmap)

    # Add numerical value of fidelity to each panel
    for i in range(dim): 
        for j in range(dim):
            # Here x is the column index and y is the row index
            color = 'black' if fid_mat[i, j] > 50 else 'white'
            ax.text(j, i, f"{fid_mat[i, j]:.1f}", color=color, ha='center', va='center')

    # Set tick positions and labels
    tick_pos = [0, 1, 2, 3]
    tick_labels = [r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$']
    ax.tick_params(length=0)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.xaxis.tick_top()
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels)

    # Set the panel name and observable name
    ax.text(-0.1, 1.05, r"$\textbf{" + panel_name + "}$", transform=ax.transAxes)

    # Display the color bar
    ax_cbar = fig.add_axes([bbox.xmax - 0.02, bbox.ymin, 0.025, panel_height], transform=ax.transAxes)
    fig.text(0.5, 1.03, '\%', ha='center', transform=ax_cbar.transAxes)
    fig.colorbar(im, cax=ax_cbar)

    # Add labels for the panels
    rc_text_h = 0.5
    rc_text_v = 1.18
    pur_text_h = -0.26
    pur_text_v = 0.5
    if panel_name == "A":
        ax.text(rc_text_h, rc_text_v, "Without RC", ha='center', va='center', transform=ax.transAxes)
        ax.text(pur_text_h, pur_text_v, "Raw", ha='center', va='center', transform=ax.transAxes)
    elif panel_name == "B":
        ax.text(rc_text_h, rc_text_v, "With RC", ha='center', va='center', transform=ax.transAxes)
    if panel_name == "C":
        ax.text(pur_text_h, pur_text_v, "Purified", ha='center', va='center', transform=ax.transAxes)

if __name__ == '__main__':
    print("> Start plotting data.")

    # Set molecule name
    mol_name = "nah"

    # Figure parameters
    fig_width = 12
    fig_height = 9.5
    fig_width_height_ratio = fig_width / fig_height 

    # Panel parameters
    panel_width = 0.48
    panel_height = panel_width / fig_width_height_ratio
    panel_width = panel_height
    panel_hstart = 0.1
    panel_hsep = 0.46
    panel_vstart = 0.035
    panel_vsep = 0.465

    # Create the figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_a = fig.add_axes([panel_hstart,              panel_vstart + panel_vsep, panel_width, panel_height])
    ax_b = fig.add_axes([panel_hstart + panel_hsep, panel_vstart + panel_vsep, panel_width, panel_height])
    ax_c = fig.add_axes([panel_hstart,              panel_vstart,              panel_width, panel_height])
    ax_d = fig.add_axes([panel_hstart + panel_hsep, panel_vstart,              panel_width, panel_height])

    # Data file names
    datfname_a = f'../../expt/resp/sep13_2022_{mol_name}/data/mat/fid_mat_{mol_name}_resp_tomo_raw.dat'
    datfname_b = f'../../expt/resp/sep13_2022_{mol_name}/data/mat/fid_mat_{mol_name}_resp_tomo_rc_raw.dat'
    datfname_c = f'../../expt/resp/sep13_2022_{mol_name}/data/mat/fid_mat_{mol_name}_resp_tomo_pur.dat'
    datfname_d = f'../../expt/resp/sep13_2022_{mol_name}/data/mat/fid_mat_{mol_name}_resp_tomo_rc_pur.dat'

    # Plot the figures
    plot_fidelity_matrix(fig, ax_a, datfname_a, panel_name='A')
    plot_fidelity_matrix(fig, ax_b, datfname_b, panel_name='B')
    plot_fidelity_matrix(fig, ax_c, datfname_c, panel_name='C')
    plot_fidelity_matrix(fig, ax_d, datfname_d, panel_name='D')

    # Save figure to file
    if mol_name == 'nah':
        fig.savefig(f"fig5_fidelity_matrix.png", dpi=200)
    elif mol_name == 'kh':
        fig.savefig(f"fig5_fidelity_matrix_kh.png", dpi=200)

    print("> Finished plotting data.")
