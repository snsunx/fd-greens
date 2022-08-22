import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
	'text.usetex': True,
	'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 23,
    'figure.subplot.left': 0.08,
    'figure.subplot.right': 0.97,
    'figure.subplot.top': 0.93,
    'figure.subplot.bottom': 0.065,
    'lines.linewidth': 2
})


def plot_fidelity_matrix(
    ax: plt.Axes,
    datfname_raw: str,
    datfname_pur: str,
    panel_name: str = '(a)',
) -> None:
    """Plots the fidelity matrix  in Fig. 3."""
    global fig

    bbox = ax.get_position()

    # Load fidelities from experimental files.
    fid_raw = np.loadtxt(datfname_raw)
    fid_pur = np.loadtxt(datfname_pur)

    dim = fid_raw.shape[0]
    cmap = plt.get_cmap("viridis")

    # Display the fidelities of raw and purified results.
    im = ax.imshow(fid_pur, vmin=0.5, vmax=1.0, cmap=cmap)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            color = 'black' if fid_pur[i, j] > 0.75 else 'white'
            ax.text(j, i, f"{fid_pur[i, j]:.2f}\n({fid_raw[i, j]:.2f})", color=color, ha='center', va='center')

    # Set ticks and tick labels.
    ax.tick_params(length=0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])
    ax.xaxis.tick_top()
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])

    # Set the panel name and observable name.
    ax.text(-0.05, 1.2, r"$\textbf{" + panel_name + "}$", transform=ax.transAxes)
    ax.text(0.24, 1.17, "Fidelity:", transform=ax.transAxes)

    # Display the sample tile.
    ax_sample = fig.add_axes([(bbox.xmin + bbox.xmax) / 2 + 0.005, bbox.ymax + 0.035, 0.075, 0.075])
    ax_sample.imshow([[0.85]], vmin=0.0, vmax=1.0, cmap=cmap)
    ax_sample.set_xticks([])
    ax_sample.set_yticks([])
    ax_sample.text(0, 0, "Pur.\n(Raw)", color='k', ha='center', va='center')

    # Display the color bar.
    ax_cbar = fig.add_axes([bbox.xmax + 0.02, bbox.ymin, 0.025, 0.35], transform=ax.transAxes)
    fig.colorbar(im, cax=ax_cbar)
    ax.text(1.05, 1.04, "Pur.", transform=ax.transAxes)

def plot_trace_matrix(
    ax: plt.Axes, 
    datfname_exact: str,
    datfname_expt: str,
    panel_name: str = '(c)'
) -> None:
    """Plots the trace matrix in Fig. 4."""
    global fig

    bbox = ax.get_position()
    # Load traces from exact and experimental files.
    traces_exact = np.loadtxt(datfname_exact)
    traces_expt = np.loadtxt(datfname_expt)
    traces_diff = np.abs(traces_exact - traces_expt)
    trace_diff_max = np.max(traces_diff)
    trace_diff_min = np.min(traces_diff)

    print("trace_diff_max = ", trace_diff_max)

    dim = traces_exact.shape[0]
    cmap = plt.get_cmap("BuPu")

    # Display the traces of exact and experimental results.
    im = ax.imshow(traces_diff, vmin=0.0, vmax=0.1, cmap=cmap)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            color = 'k' if traces_diff[i, j] < trace_diff_max / 2 else 'w'
            ax.text(
                j, i, f"{traces_expt[i, j]:.2f}\n({traces_exact[i, j]:.2f})",
                color=color, ha='center', va='center'
            )

    # Set ticks and tick labels.
    ax.tick_params(length=0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])
    ax.xaxis.tick_top()
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([r'$0\uparrow$', r'$0\downarrow$', r'$1\uparrow$', r'$1\downarrow$'])

    # Set the panel name and observable name.
    ax.text(-0.05, 1.2, r"$\textbf{" + panel_name + "}$", transform=ax.transAxes)
    ax.text(0.2, 1.16, "Trace\n(ancilla prob.):", transform=ax.transAxes, fontsize=20)

    # Display the sample tile.
    ax_sample = fig.add_axes([(bbox.xmin + bbox.xmax) / 2 + 0.045, bbox.ymax + 0.035, 0.075, 0.075])
    ax_sample.imshow([[trace_diff_max * 0.25]], vmin=trace_diff_min, vmax=trace_diff_max, cmap=cmap)
    ax_sample.set_xticks([])
    ax_sample.set_yticks([])
    ax_sample.text(0, 0, "Expt.\n(Exact)", color='k', ha='center', va='center', fontsize=20)

    # Display the color bar.
    ax_cbar = fig.add_axes([bbox.xmax + 0.02, bbox.ymin, 0.025, 0.35], transform=ax.transAxes)
    fig.colorbar(im, cax=ax_cbar)
    ax.text(0.945, 1.04, r"$|$Exact$-$Expt.$|$", transform=ax.transAxes, fontsize=17)

def main():
    print("Start plotting data.")
    global fig

    mol_name = "nah"

    fig = plt.figure(figsize=(12, 12))
    ax_a = fig.add_axes([0.05, 0.52, 0.35, 0.35])
    ax_b = fig.add_axes([0.55, 0.52, 0.35, 0.35])
    ax_c = fig.add_axes([0.05, 0.02, 0.35, 0.35])
    ax_d = fig.add_axes([0.55, 0.02, 0.35, 0.35])

    plot_fidelity_matrix(
        ax_a,
        f'../../expt/resp/aug13_2022_{mol_name}/data/mat/fid_mat_resp_tomo_raw.dat',
        f'../../expt/resp/aug13_2022_{mol_name}/data/mat/fid_mat_resp_tomo_pur.dat',
        panel_name='A'
    )
    plot_fidelity_matrix(
        ax_b,
        f'../../expt/resp/aug13_2022_{mol_name}/data/mat/fid_mat_resp_tomo_rc_raw.dat',
        f'../../expt/resp/aug13_2022_{mol_name}/data/mat/fid_mat_resp_tomo_rc_pur.dat',
        panel_name='B'
    )
    plot_trace_matrix(
        ax_c,
        f'../../expt/resp/aug13_2022_{mol_name}/data/mat/trace_mat_resp_exact.dat',
        f'../../expt/resp/aug13_2022_{mol_name}/data/mat/trace_mat_resp_tomo_raw.dat',
        panel_name='C'
    )
    plot_trace_matrix(
        ax_d,
        f'../../expt/resp/aug13_2022_{mol_name}/data/mat/trace_mat_resp_exact.dat',
        f'../../expt/resp/aug13_2022_{mol_name}/data/mat/trace_mat_resp_tomo_rc_raw.dat',
        panel_name='D'
    )

    fig.savefig(f"fig5_fidelity_and_trace.png", dpi=200)    

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
