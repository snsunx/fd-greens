"""
==================================
Helpers (:mod:`fd_greens.helpers`)
==================================
"""

import os
from typing import Sequence, List, Optional
from itertools import product

import cirq
import h5py
import numpy as np
import matplotlib.pyplot as plt


def get_circuit_labels(n_orbitals: int, mode: str = 'greens', spin: str = '') -> List[str]:
    """Returns the circuit labels of a """
    assert mode in ['greens', 'resp']
    if mode == 'greens':
        assert spin in ['u', 'd']
    else:
        assert spin == ''

    orbital_labels = [str(i) for i in range(n_orbitals)]

    if mode == 'greens':
        orbital_labels = [str(i) for i in range(n_orbitals)]
    elif mode == 'resp':
        orbital_labels = list(product(range(n_orbitals), ['u', 'd']))
        orbital_labels = [f'{x[0]}{x[1]}' for x in orbital_labels]

    circuit_labels = []
    for i in range(len(orbital_labels)):
        circuit_labels.append(f'circ{orbital_labels[i]}{spin}')
        for j in range(i + 1, len(orbital_labels)):
            circuit_labels.append(f'circ{orbital_labels[i]}{orbital_labels[j]}{spin}')
    
    return circuit_labels

def initialize_hdf5(
    fname: str = 'lih',
    mode: str = 'greens',
    spin: str = '',
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

    if mode == 'greens':
        orbital_labels = [str(i) for i in range(n_orbitals)]
    elif mode == 'resp':
        orbital_labels = list(product(range(n_orbitals), ['u', 'd']))
        orbital_labels = [f'{x[0]}{x[1]}' for x in orbital_labels]

    # TODO: group_names should be constructed from get_circuit_labels.
    group_names = ['gs', 'es', 'amp']
    for i in range(len(orbital_labels)):
        group_names.append(f'circ{orbital_labels[i]}{spin}')
        for j in range(i + 1, len(orbital_labels)):
            group_names.append(f'circ{orbital_labels[i]}{orbital_labels[j]}{spin}')

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
                print(f'Creating {group_name}/{tomography_label}')
                h5file.create_dataset(f'{group_name}/{tomography_label}', data='')
    
    h5file.close()

def plot_spectral_function(
    fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Sequence[str] = None,
    annotations: Optional[Sequence[dict]] = None,
    linestyles: Optional[Sequence[dict]] = None,
    dirname: str = "figs",
    figname: str = "A",
    text: Optional[str] = None,
    n_curves: Optional[int] = None,
) -> None:
    """Plots the spectral function.
    
    Args:
        fnames: Names of the HDF5 files from which the curves are generated.
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
        n_curves = max(len(fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [{}] * n_curves

    plt.clf()
    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(fnames, suffixes, labels, linestyles):
    for i in range(n_curves):
        omegas, As = np.loadtxt(f"data/{fnames[i]}{suffixes[i]}_A.dat").T
        ax.plot(omegas, As, label=labels[i], **linestyles[i])
    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("$A$ (eV$^{-1}$)")
    if text == "legend":
        ax.legend()
    elif text == "annotation":
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(f"{dirname}/{figname}.png", dpi=300, bbox_inches="tight")

plot_A = plot_spectral_function


def plot_trace_self_energy(
    fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    annotations: Optional[Sequence[str]] = None,
    dirname: str = "figs",
    figname: str = "TrSigma",
    linestyles: Optional[Sequence[dict]] = None,
    text: str = None,
    n_curves: Optional[int] = None,
) -> None:
    """Plots the trace of the self-energy.
    
    Args:
        fnames: Names of the HDF5 files from which the curves are generated.
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
        n_curves = 2 * max(len(fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [{}] * n_curves

    plt.clf()
    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(fnames, suffixes, labels, linestyles):
    for i in range(n_curves // 2):
        omegas, real, imag = np.loadtxt(f"data/{fnames[i]}{suffixes[i]}_TrSigma.dat").T
        ax.plot(omegas, real, label=labels[i] + " (real)", **linestyles[2 * i])
        ax.plot(omegas, imag, label=labels[i] + " (imag)", **linestyles[2 * i + 1])
    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Tr$\Sigma$ (eV)")
    if text == "legend":
        ax.legend()
    elif text == "annotation":
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(f"{dirname}/{figname}.png", dpi=300, bbox_inches="tight")

plot_TrS = plot_trace_self_energy


def plot_response_function(
    fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    annotations: Optional[Sequence[str]] = None,
    dirname: str = "figs",
    figname: str = "chi",
    circ_label: str = "00",
    linestyles: Sequence[dict] = None,
    text: Optional[str] = None,
    n_curves: Optional[int] = None,
) -> None:
    """Plots the charge-charge response function.

    Args:
        fnames: Names of the HDF5 files from which the curves are generated.
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
        n_curves = 2 * max(len(fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [{}] * n_curves

    # for h5fname, suffix, label, linestyle in zip(fnames, suffixes, labels, linestyles):
    for circ_label in ['00', '01', '10', '11']:
        plt.clf()
        fig, ax = plt.subplots()
        for i in range(n_curves // 2):
            omegas, real, imag = np.loadtxt(f"data/{fnames[i]}{suffixes[i]}_chi{circ_label}.dat").T
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

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fig.savefig(f"{dirname}/{figname}{circ_label}.png", dpi=300, bbox_inches="tight")

plot_chi = plot_response_function


def plot_counts(
    fname_sim: str, 
    fname_expt: str,
    dset_name_sim: str,
    dset_name_expt: str,
    counts_name_sim: str = '',
    counts_name_expt: str = '', 
    width: float = 0.5,
    dirname: str = 'figs'
) -> None:
    """Plots the simulation and experimental bitstring counts along with total variational distance.
    
    Args:
        fname_sim: Name of the simulation HDF5 file.
        fname_expt: Name of the experimental HDF5 file.
        dset_name_sim: Name of the simulation dataset.
        dset_name_expt: Name of the experimental dataset.
        counts_name_sim: Name of the simulation bitstring counts.
        counts_name_expt: Name fo the experimental bitstring counts.
        width: Width of bars in the bar chart.
    """
    # print(fname_sim)
    # h5file_sim = h5py.File(fname_sim + '.h5', 'r')
    # h5file_expt = h5py.File(fname_expt + '.h5', 'r')
    with h5py.File(fname_sim + '.h5', 'r') as h5file_sim:
        array_sim = h5file_sim[dset_name_sim].attrs[counts_name_sim][:]
    with h5py.File(fname_expt + '.h5', 'r') as h5file_expt:
        array_expt = h5file_expt[dset_name_expt].attrs[counts_name_expt][:]

    # Compute total variational distance.
    assert abs(np.sum(array_sim) - 1) < 1e-8
    assert abs(np.sum(array_expt) - 1) < 1e-8
    distance = np.sum(np.abs(array_sim - array_expt)) / 2

    # Create x tick locations and labels.
    n_qubits = int(np.log2(len(array_sim)))
    x = np.arange(2 ** n_qubits)
    tick_labels = ["".join(y) for y in product("01", repeat=n_qubits)]

    plt.clf()
    fig, ax = plt.subplots(figsize=(len(array_sim) * 0.6 + 2, 4))
    ax.bar(x - width / 3, array_sim, width / 1.5, label="Sim")
    ax.bar(x + width / 3, array_expt, width / 1.5, label="Expt")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Ratio")
    ax.set_title(f"Fidelity (1 - TVD): {1 - distance:.4f}")
    ax.legend()

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.set_facecolor('white')
    fig.savefig(f"{dirname}/counts_{dset_name_sim.replace('/', '_')}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


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