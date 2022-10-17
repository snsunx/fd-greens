import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
	'text.usetex': True,
	'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 24,
    'figure.subplot.left': 0.02,
    'figure.subplot.right': 0.97,
    'figure.subplot.top': 0.93,
    'figure.subplot.bottom': 0.065,
    'lines.linewidth': 2
})


def plot_fidelity_matrix(
    ax: plt.Axes,
    # datfname_raw: str,
    datfname_pur: str,
    panel_name: str = '(a)',
) -> None:
    """Plots the fidelity matrix  in Fig. 3."""
    global fig, height, width

    bbox = ax.get_position()

    # Load fidelities from experimental files.
    # fid_raw = np.loadtxt(datfname_raw)
    fid_pur = np.loadtxt(datfname_pur)

    dim = fid_pur.shape[0]
    cmap = plt.get_cmap("viridis")

    # Display the fidelities of raw and purified results.
    im = ax.imshow(fid_pur, vmin=0.0, vmax=1.0, cmap=cmap)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            color = 'black' if fid_pur[i, j] > 0.5 else 'white'
            ax.text(j, i, f"{fid_pur[i, j]:.2f}", color=color, ha='center', va='center')

    # Set ticks and tick labels.
    ax.tick_params(length=0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])
    ax.xaxis.tick_top()
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])

    # Set the panel name and observable name.
    ax.text(-0.1, 1.05, r"$\textbf{" + panel_name + "}$", transform=ax.transAxes)
    # ax.text(0.24, 1.19, "Fidelity:", transform=ax.transAxes)

    # Display the sample tile.
    # ax_sample = fig.add_axes([(bbox.xmin + bbox.xmax) / 2 - 0.025, bbox.ymax + 0.07, 0.135 / 2, 0.135 / 2])
    # ax_sample.imshow([[0.85]], vmin=0.0, vmax=1.0, cmap=cmap)
    # ax_sample.set_xticks([])
    # ax_sample.set_yticks([])
    # ax_sample.text(0, 0, "Pur.\n(Raw)", color='k', ha='center', va='center')

    # Display the color bar.
    ax_cbar = fig.add_axes([bbox.xmax + 0.02, bbox.ymin, 0.025, width * 12 / height], transform=ax.transAxes)
    fig.colorbar(im, cax=ax_cbar)

    if panel_name == "A":
        ax.text(0.5, 1.16, "Without RC", ha='center', transform=ax.transAxes)
    elif panel_name == "B":
        ax.text(0.5, 1.16, "With RC", ha='center', transform=ax.transAxes)

def main():
    print("Start plotting data.")
    global fig, height, width

    mol_name = "kh"
    height = 11
    width = 0.35

    # fig = plt.figure(figsize=(12, 13))
    fig = plt.figure(figsize=(12, height))
    ax_a = fig.add_axes([0.05, 0.5, width, width * 12 / height])
    ax_b = fig.add_axes([0.55, 0.5, width, width * 12 / height])
    ax_c = fig.add_axes([0.05, 0.02, width, width * 12 / height])
    ax_d = fig.add_axes([0.55, 0.02, width, width * 12 / height])

    plot_fidelity_matrix(
        ax_a,
        f'../../expt/resp/sep13_2022_{mol_name}/data/mat/fid_mat_{mol_name}_resp_tomo_raw.dat',
        panel_name='A'
    )

    plot_fidelity_matrix(
        ax_b,
        f'../../expt/resp/sep13_2022_{mol_name}/data/mat/fid_mat_{mol_name}_resp_tomo_rc_raw.dat',
        panel_name='B'
    )

    plot_fidelity_matrix(
        ax_c,
        f'../../expt/resp/sep13_2022_{mol_name}/data/mat/fid_mat_{mol_name}_resp_tomo_pur.dat',
        panel_name='C'
    )

    plot_fidelity_matrix(
        ax_d,
        f'../../expt/resp/sep13_2022_{mol_name}/data/mat/fid_mat_{mol_name}_resp_tomo_rc_pur.dat',
        panel_name='D'
    )

    # plot_trace_matrix(
    #     ax_c,
    #     f'../../expt/resp/sep13_2022_{mol_name}/data/mat/trace_mat_{mol_name}_resp_exact.dat',
    #     f'../../expt/resp/sep13_2022_{mol_name}/data/mat/trace_mat_{mol_name}_resp_tomo_raw.dat',
    #     panel_name='C'
    # )

    # plot_trace_matrix(
    #     ax_d,
    #     f'../../expt/resp/sep13_2022_{mol_name}/data/mat/trace_mat_{mol_name}_resp_exact.dat',
    #     f'../../expt/resp/sep13_2022_{mol_name}/data/mat/trace_mat_{mol_name}_resp_tomo_rc_raw.dat',
    #     panel_name='D'
    # )

    if mol_name == 'nah':
        fig.savefig(f"fig5_fidelity_matrix.png", dpi=200)
    elif mol_name == 'kh':
        fig.savefig(f"fig5_fidelity_matrix_kh.png", dpi=200)

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
