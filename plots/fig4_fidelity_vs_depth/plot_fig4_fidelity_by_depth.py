import sys
sys.path.append('../..')

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 20,
    'figure.subplot.left': 0.14,
    'figure.subplot.right': 0.95,
    'figure.subplot.top': 0.93,
    'figure.subplot.bottom': 0.12,
    'figure.subplot.wspace': 0.4,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

def plot_fidelity_by_depth(ax: plt.Axes, datfname_3q: str, datfname_2q: str, is_inset: bool = False) -> None:
    depths_3q, nq_gates_3q, fidelities_3q = np.loadtxt(datfname_3q)[1:].T
    depths_2q, _, fidelities_2q = np.loadtxt(datfname_2q)[1:].T

    locations_itoffoli = np.array([i for i in range(len(depths_3q)) if nq_gates_3q[i] == 3])
    fidelities_itoffoli = [fidelities_3q[n] for n in locations_itoffoli]

    ax.plot(np.arange(len(depths_3q)), fidelities_3q, marker='.', label="iToffoli")
    ax.plot(np.arange(len(depths_2q)), fidelities_2q, marker='x', label="CZ")
    ax.plot(locations_itoffoli, fidelities_itoffoli, ls='', color='r', marker='x', ms=10, mew=2.0)
    
    if is_inset:
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlabel("Circuit depth", fontsize=16)
        ax.set_ylabel("Fidelity", fontsize=16)
    else:
        ax.legend(loc='lower left', frameon=False)
        ax.set_xlabel("Circuit depth")
        ax.set_ylabel("Fidelity")
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax_inset = fig.add_axes([0.57, 0.56, 0.36, 0.34])

    plot_fidelity_by_depth(
        ax, 
        "../../expt/resp/aug13_2022_traj/data/traj/fid_vs_depth_kh_resp_circ0u1d_expt.dat",
        "../../expt/resp/aug13_2022_traj/data/traj/fid_vs_depth_kh_resp_circ0u1d2q_expt.dat"
    )

    plot_fidelity_by_depth(
        ax_inset, 
        "../../sim/resp/aug13_2022/data/traj/fid_vs_depth_kh_resp_circ0u1d_n0814.dat",
        "../../sim/resp/aug13_2022/data/traj/fid_vs_depth_kh_resp_circ0u1d2q_n0814.dat",
        is_inset=True
    )

    fig.savefig("fig4_fidelity_by_depth.png", dpi=200)

if __name__ == '__main__':
    main()
