import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 22,
    'figure.subplot.left': 0.12,
    'figure.subplot.right': 0.95,
    'figure.subplot.top': 0.95,
    'figure.subplot.bottom': 0.13,
    'figure.subplot.wspace': 0.4,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})


def plot_fidelity_by_depth(
    ax: plt.Axes,
    datfname_3q: str,
    datfname_2q: str,
    is_inset: bool = False,
    # include_legend: bool = True,
    # panel_name: str = "A"
) -> None:
    """Plot the fidelity of the circuit by circuit depth."""
    depths_3q, nq_gates_3q, fidelities_3q = np.loadtxt(datfname_3q).T
    depths_2q, _, fidelities_2q = np.loadtxt(datfname_2q).T
    effective_depths_3q = np.arange(1, len(depths_3q) + 1)
    effective_depths_2q = np.arange(1, len(depths_2q) + 1)

    locations_itoffoli = np.array([i for i in range(len(depths_3q)) if nq_gates_3q[i - 1] == 3])
    fidelities_itoffoli = [fidelities_3q[n] for n in locations_itoffoli]
    # print(f"{locations_itoffoli = }")

    lw = 2.7 if not is_inset else 2
    ms = 8 if not is_inset else 6
    ms_itoffoli = 16 if not is_inset else 12
    ax.plot(effective_depths_3q, fidelities_3q, 
            color='xkcd:medium blue', lw=lw, marker='.', ms=ms, label="iToffoli")
    ax.plot(effective_depths_2q, fidelities_2q, 
            color='xkcd:orange yellow', lw=lw, marker='.', ms=ms, label="CZ")
    ax.plot(locations_itoffoli + 1, fidelities_itoffoli, ls='',
            color='xkcd:pinkish red', marker='x', ms=ms_itoffoli, mew=3.0)

    # ax.text(0.92, 0.9, r"\textbf{" + panel_name + "}", transform=ax.transAxes)
    
    if is_inset:
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlabel("Circuit depth", fontsize=16)
        ax.set_ylabel("Fidelity", fontsize=16)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.text(0.635, 0.9, "Classical simulation", transform=ax.transAxes, 
                fontsize=16, ha='center', va='center')
    else:
        # if include_legend:
        ax.legend(loc='center', bbox_to_anchor=(0.17, 0.12, 0.0, 0.0), frameon=False)
        ax.set_xlabel("Circuit depth")
        # if panel_name == "A":
        ax.set_ylabel("Fidelity")
        ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax_inset = fig.add_axes([0.57, 0.585, 0.36, 0.34])

    plot_fidelity_by_depth(
        ax,
        "../../expt/resp/sep13_2022_traj/data/traj/fid_vs_depth_nah_resp_circ0u1u.dat",
        "../../expt/resp/sep13_2022_traj/data/traj/fid_vs_depth_nah_resp_circ0u1u2q.dat"
    )

    plot_fidelity_by_depth(
        ax_inset,
        "../../sim/resp/sep13_2022/data/traj/fid_vs_depth_nah_resp_circ0u1u_n0814.dat",
        "../../sim/resp/sep13_2022/data/traj/fid_vs_depth_nah_resp_circ0u1u2q_n0814.dat",
        is_inset=True
    )

    fig.savefig("fig4_fidelity_by_depth.png", dpi=200)

# def main() -> None:
#     fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharey=True)

#     plot_fidelity_by_depth(
#         axes[0],
#         "../../expt/resp/sep13_2022_traj/data/traj/fid_vs_depth_nah_resp_circ0u1u.dat",
#         "../../expt/resp/sep13_2022_traj/data/traj/fid_vs_depth_nah_resp_circ0u1u2q.dat",
#         panel_name="A"
#     )

#     plot_fidelity_by_depth(
#         axes[1],
#         "../../sim/resp/sep13_2022/data/traj/fid_vs_depth_nah_resp_circ0u1u_n0814.dat",
#         "../../sim/resp/sep13_2022/data/traj/fid_vs_depth_nah_resp_circ0u1u2q_n0814.dat",
#         include_legend=False,
#         panel_name="B"
#     )

#     fig.savefig("fig4_fidelity_by_depth.png", dpi=200)

if __name__ == '__main__':
    main()
