"""
==================================
Helpers (:mod:`fd_greens.helpers`)
==================================
"""

import os
from typing import Sequence, Optional
from itertools import product

import cirq
import h5py
import numpy as np
import matplotlib.pyplot as plt


def initialize_hdf5(
    fname: str = 'lih',
    mode: str = 'greens',
    # spin: str = '',
    n_orbitals: int = 2,
    overwrite: bool = True,
    create_datasets: bool = False
) -> None:
    """Initializes an HDF5 file.
    
    Args:
        fname: The HDF5 file name.
        mode: Calculation mode. Either ``'greens'`` or ``'resp'``.
        spin: Spin of the second-quantized operators.
        n_orbitals: Number of orbitals. Defaults to 2.
        overwrite: Whether to overwrite groups if they are found in the HDF5 file.
    """
    assert mode in ['greens', 'resp']
    
    h5fname = fname + '.h5'
    if os.path.exists(h5fname):
        h5file = h5py.File(h5fname, 'r+')
    else:
        h5file = h5py.File(h5fname, 'w')

    orbital_labels = list(product(range(n_orbitals), ['u', 'd']))
    orbital_labels = [f'{x[0]}{x[1]}' for x in orbital_labels]

    group_names = ['gs', 'es', 'amp']
    for i in range(len(orbital_labels)):
        group_names.append(f'circ{orbital_labels[i]}')
        for j in range(i + 1, len(orbital_labels)):
            if mode == 'resp' or orbital_labels[i][-1] == orbital_labels[j][-1]:
                 group_names.append(f'circ{orbital_labels[i]}{orbital_labels[j]}')

    for group_name in group_names:
        if group_name in h5file.keys():
            if overwrite:
                del h5file[group_name]
                h5file.create_group(group_name)
        else:
            h5file.create_group(group_name)
        
        if create_datasets and group_name not in ['gs', 'es', 'amp']:
            tomography_labels = [''.join(x) for x in product('xyz', repeat=2)]
            for tomography_label in tomography_labels:
                h5file.create_dataset(f'{group_name}/{tomography_label}', data='')
    
    h5file.close()

def plot_spectral_function(
    h5fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Sequence[str] = None,
    annotations: Optional[Sequence[dict]] = None,
    linestyles: Optional[Sequence[dict]] = None,
    figname: str = "A",
    text: Optional[str] = None,
    n_curves: Optional[int] = None,
) -> None:
    """Plots the spectral function.
    
    Args:
        h5fnames: Names of the HDF5 files from which the curves are generated.
        suffixes: Suffixes of the curves.
        labels: Legend labels of the curves.
        annotations: Annotation options to be passed into ax.text.
        linestyles: The linestyles of the curves to be passed into ax.plot.
        figname: The name of the figure to be saved.
        text: Whether to add labels by legend, annotation or none.
        n_curves: Number of curves.
    """
    assert text in [None, "legend", "annotation"]

    if n_curves is None:
        n_curves = max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves

    plt.clf()
    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
    for i in range(n_curves):
        omegas, As = np.loadtxt(f"data/{h5fnames[i]}{suffixes[i]}_A.dat").T
        ax.plot(omegas, As, label=labels[i], **linestyles[i])
    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("$A$ (eV$^{-1}$)")
    if text == "legend":
        ax.legend()
    elif text == "annotation":
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)

    if not os.path.exists("figs"):
        os.makedirs("figs")
    fig.savefig(f"figs/{figname}.png", dpi=300, bbox_inches="tight")

plot_A = plot_spectral_function


def plot_trace_self_energy(
    h5fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    annotations: Optional[Sequence[str]] = None,
    figname: str = "TrS",
    linestyles: Optional[Sequence[dict]] = None,
    text: str = None,
    n_curves: Optional[int] = None,
) -> None:
    """Plots the trace of the self-energy.
    
    Args:
        h5fnames: Names of the HDF5 files from which the curves are generated.
        suffixes: Suffixes of the curves.
        labels: Legend labels of the curves.
        annotations: Annotation options to be passed into ax.text.
        linestyles: The linestyles of the curves to be passed into ax.plot.
        figname: The name of the figure to be saved.
        text: Whether to add labels by legend, annotation or none.
        n_curves: Number of curves.
    """
    assert text in [None, "legend", "annotation"]

    if n_curves is None:
        n_curves = 2 * max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves

    plt.clf()
    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
    for i in range(n_curves // 2):
        omegas, real, imag = np.loadtxt(f"data/{h5fnames[i]}{suffixes[i]}_TrS.dat").T
        ax.plot(omegas, real, label=labels[i] + " (real)", **linestyles[2 * i])
        ax.plot(omegas, imag, label=labels[i] + " (imag)", **linestyles[2 * i + 1])
    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Tr$\Sigma$ (eV)")
    if text == "legend":
        ax.legend()
    elif text == "annotation":
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)

    if not os.path.exists("figs"):
        os.makedirs("figs")
    fig.savefig(f"figs/{figname}.png", dpi=300, bbox_inches="tight")

plot_TrS = plot_trace_self_energy


def plot_response_function(
    h5fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    annotations: Optional[Sequence[str]] = None,
    figname: str = "chi",
    circ_label: str = "00",
    linestyles: Sequence[dict] = None,
    text: Optional[str] = None,
    n_curves: Optional[int] = None,
) -> None:
    """Plots the charge-charge response function.

    Args:
        h5fnames: Names of the HDF5 files from which the curves are generated.
        suffixes: Suffixes of the curves.
        labels: Legend labels of the curves.
        annotations: Annotation options to be passed into ax.text.
        linestyles: The linestyles of the curves to be passed into ax.plot.
        figname: The name of the figure to be saved.
        circ_label: The circuit label.
        text: Whether to add labels by legend, annotation or none.
        n_curves: Number of curves.
    """
    if n_curves is None:
        n_curves = 2 * max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves

    plt.clf()
    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
    for i in range(n_curves // 2):
        omegas, real, imag = np.loadtxt(
            f"data/{h5fnames[i]}{suffixes[i]}_chi{circ_label}.dat"
        ).T
        ax.plot(omegas, real, label=labels[i] + " (real)", **linestyles[2 * i])
        ax.plot(omegas, imag, label=labels[i] + " (imag)", **linestyles[2 * i + 1])
    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("$\chi_{" + circ_label + "}$ (eV$^{-1}$)")
    if text == "legend":
        ax.legend()
    elif text == "annotation":
        # for kwargs in annotations:
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)

    if not os.path.exists("figs"):
        os.makedirs("figs")
    fig.savefig(f"figs/{figname}{circ_label}.png", dpi=300, bbox_inches="tight")

plot_chi = plot_response_function


def plot_counts(
    fname_sim: str, 
    fname_expt: str,
    dset_name_sim: str,
    dset_name_expt: str,
    counts_name_sim: str = '',
    counts_name_expt: str = '', 
    width: float = 0.5
) -> None:
    """Plots the simulation and experimental bitstring counts.
    
    Args:
        h5fname: The HDF5 file name.
        counts_name: Name of the bitstring counts.
        circ_label: The circuit label.
        tomo_label: The tomography label.
        width: The width of the bars in the bar chart.
    """
    h5file_sim = h5py.File(fname_sim + '.h5', 'r')
    h5file_expt = h5py.File(fname_expt + '.h5', 'r')
    array_sim = h5file_sim[dset_name_sim].attrs[counts_name_sim][:]
    array_expt = h5file_expt[dset_name_expt].attrs[counts_name_expt][:]

    # Compute total variational distance.
    assert abs(np.sum(array_sim) - 1) < 1e-8
    assert abs(np.sum(array_expt) - 1) < 1e-8
    distance = np.sum(np.abs(array_sim - array_expt)) / 2

    n_qubits = int(np.log2(len(array_sim)))
    x = np.arange(2 ** n_qubits)
    tick_labels = ["".join(x) for x in product("01", repeat=n_qubits)]

    plt.clf()

    fig, ax = plt.subplots(figsize=(len(array_sim) * 0.6 + 2, len(array_sim) * 0.5))
    ax.bar(x - width / 3, array_sim, width / 1.5, label="Sim")
    ax.bar(x + width / 3, array_expt, width / 1.5, label="Expt")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Ratio")
    ax.set_title(f"Total Variational Distance: {distance:.4f}")
    ax.legend()

    if not os.path.exists("figs"):
        os.makedirs("figs")
    fig.savefig(f"figs/counts_{dset_name_sim.replace('/', '_')}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    h5file_sim.close()
    h5file_expt.close()

def print_circuit(circuit: cirq.Circuit) -> None:
    """Prints out a circuit 10 elements at a time.
    
    Args:
        circuit: The circuit to be printed.
    """
    if len(circuit) < 10:
        print(circuit)
    else:
        for i in range(len(circuit) // 10 + 1):
            print(circuit[i * 10: min((i + 1) * 10, len(circuit) - 1)], '\n')
            print('-' * 120, '\n')