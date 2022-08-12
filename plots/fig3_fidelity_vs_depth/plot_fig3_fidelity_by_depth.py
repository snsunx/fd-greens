import sys
sys.path.append('../..')

import pickle
import json

import numpy as np
import h5py
import cirq

import matplotlib.pyplot as plt

from fd_greens.cirq_ver.circuit_string_converter import CircuitStringConverter
from fd_greens.cirq_ver.general_utils import get_non_z_locations, histogram_to_array

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
    'lines.linewidth': 2,
    'lines.markersize': 8
})

def plot_fidelity_by_depth(ax: plt.Axes) -> None:

    _, locations_nq, fidelities = np.loadtxt('../../sim/resp/augxx_2022/fidtraj.dat').T
    depth = len(locations_nq)

    depths = np.arange(depth)
    locations_3q = [i for i in range(depth) if locations_nq[i] == 3]
    fidelities_3q = [fidelities[n] for n in locations_3q]

    x_depths = [i + 1 for i in depths]
    x_locations_3q = [i + 1 for i in locations_3q]

    ax.plot(x_depths, fidelities, marker='.')
    ax.plot(x_locations_3q, fidelities_3q, ls='', color='r', marker='x', ms=10, mew=2.0)
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Fidelity")


def main():

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_fidelity_by_depth(ax)

    fig.savefig(f"{sys.argv[0][5:-3]}.png", dpi=200)

if __name__ == '__main__':
    main()
