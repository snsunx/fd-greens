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
    'lines.linewidth': 2
})

def plot_fid_by_depth(ax: plt.Axes) -> None:

    depths, fidelities = np.loadtxt('../../sim/resp/augxx_2022/fidtraj.dat').T

    
    ax.plot(fidelities, marker='.')
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Fidelity")


def main():

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_fid_by_depth(ax)

    fig.savefig(f"{sys.argv[0][5:-3]}.png", dpi=200)

if __name__ == '__main__':
    main()
