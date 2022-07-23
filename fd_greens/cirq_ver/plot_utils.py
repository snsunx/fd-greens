"""
============================================
Plot Utilities (:mod:`fd_greens.plot_utils`)
============================================
"""

import os
from typing import Sequence, Optional
from itertools import product
import warnings

import pickle
import json
import cirq
import h5py
import numpy as np
import matplotlib.pyplot as plt

from .circuit_string_converter import CircuitStringConverter
from .general_utils import get_fidelity, get_non_z_locations, histogram_to_array
from .helpers import process_bitstring_counts


LINESTYLES_A = [
    {},
    {"ls": "--", "marker": "", "markevery": 30},
    {"ls": "--", "marker": "", "markevery": 30},
    {"ls": "--", "marker": "", "markevery": 30}
]
LINESTYLES_TRSIGMA = [
    {"color": "xkcd:red"},
    {"color": "xkcd:blue"},
    {"ls": "--", "marker": "", "markevery": 100, "color": "xkcd:orange"},
    {"ls": "--", "marker": "", "markevery": 100, "color": "xkcd:teal"},
    {"ls": "--", "marker": "", "markevery": 100, "color": "xkcd:purple"},
    {"ls": "--", "marker": "", "markevery": 100, "color": "xkcd:navy"},
    {"ls": "--", "marker": "", "markevery": 100, "color": "xkcd:grass green"},
    {"ls": "--", "marker": "", "markevery": 100, "color": "xkcd:magenta"},
    {"ls": "--", "marker": "", "markevery": 100, "color": "xkcd:teal"},
    {"ls": "--", "marker": "", "markevery": 100, "color": "xkcd:orange"}
]
LINESTYLES_CHI = LINESTYLES_TRSIGMA
FIGURE_DPI = 250


def plot_spectral_function(
    fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Sequence[str] = None,
    annotations: Optional[Sequence[dict]] = None,
    linestyles: Sequence[dict] = LINESTYLES_A,
    datdirname: str = "data",
    dirname: str = "figs/data",
    figname: str = "A",
    text: Optional[str] = None,
    n_curves: Optional[int] = None
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
        omegas, As = np.loadtxt(f"{datdirname}/{fnames[i]}{suffixes[i]}_A.dat").T
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
    fig.savefig(f"{dirname}/{figname}.png", dpi=FIGURE_DPI, bbox_inches="tight")

plot_A = plot_spectral_function


def plot_trace_self_energy(
    fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    annotations: Optional[Sequence[str]] = None,
    datdirname: str = "data",
    dirname: str = "figs/data",
    figname: str = "TrSigma",
    linestyles: Sequence[dict] = LINESTYLES_TRSIGMA,
    text: str = None,
    n_curves: Optional[int] = None,
    separate_real_imag: bool = True
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
    if separate_real_imag:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(n_curves // 2):
            omegas, real, imag = np.loadtxt(f"{datdirname}/{fnames[i]}{suffixes[i]}_TrSigma.dat").T

            axes[0].plot(omegas, real, label=labels[i], **linestyles[2 * i])
            axes[0].set_xlabel("$\omega$ (eV)")
            axes[0].set_ylabel("Re Tr$\Sigma$ (eV)")

            axes[1].plot(omegas, imag, label=labels[i], **linestyles[2 * i + 1])
            axes[1].set_xlabel("$\omega$ (eV)")
            axes[1].set_ylabel("Im Tr$\Sigma$ (eV)")
            if text == "legend":
                axes[0].legend()
                axes[1].legend()
            elif text == "annotation":
                pass
    else:
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
    fig.savefig(f"{dirname}/{figname}.png", dpi=FIGURE_DPI, bbox_inches="tight")

plot_TrS = plot_trace_self_energy


def plot_response_function(
    fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    annotations: Optional[Sequence[str]] = None,
    datdirname: str = "data",
    dirname: str = "figs/data",
    figname: str = "chi",
    circ_label: str = "00",
    linestyles: Sequence[dict] = LINESTYLES_CHI,
    text: Optional[str] = None,
    n_curves: Optional[int] = None,
    separate_real_imag: bool = True,
    scaling_factor: float = 1.0
) -> None:
    """Plots the charge-charge response function.

    Args:
        fnames: Names of the HDF5 files from which the curves are generated.
        suffixes: Suffixes of the curves.
        labels: Legend labels of the curves.
        annotations: Annotation options to be passed into ax.text.
        dirname: Directory name.
        figname: Figure name.
        linestyles: The linestyles of the curves to be passed into ax.plot.
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
    if scaling_factor != 1.0:
        warnings.warn(f"Using a scaling factor of {scaling_factor} when plotting response function.")

    # for h5fname, suffix, label, linestyle in zip(fnames, suffixes, labels, linestyles):
    for circ_label in ['00', '01', '10', '11']:
        plt.clf()
        if separate_real_imag:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            for i in range(n_curves // 2):
                omegas, real, imag = np.loadtxt(f"{datdirname}/{fnames[i]}{suffixes[i]}_chi{circ_label}.dat").T
                if 'exact' not in fnames[i]:
                    real *= scaling_factor
                    imag *= scaling_factor

                axes[0].plot(omegas, real, label=labels[i], **linestyles[2 * i])
                axes[0].set_xlabel("$\omega$ (eV)")
                axes[0].set_ylabel("Re $\chi_{" + circ_label + "}$ (eV$^{-1}$)")
                
                axes[1].plot(omegas, imag, label=labels[i], **linestyles[2 * i + 1])
                axes[1].set_xlabel("$\omega$ (eV)")
                axes[1].set_ylabel("Im $\chi_{" + circ_label + "}$ (eV$^{-1}$)")

            if text == "legend":
                axes[0].legend()
                axes[1].legend()

        else:
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
        fig.savefig(f"{dirname}/{figname}{circ_label}.png", dpi=FIGURE_DPI, bbox_inches="tight")

plot_chi = plot_response_function


def plot_bitstring_counts(
    fname_sim: str, 
    fname_expt: str,
    dset_name_sim: str,
    dset_name_expt: str,
    counts_name_sim: str = '',
    counts_name_expt: str = '', 
    width: float = 0.5,
    dirname: str = 'figs/counts',
    figname: str = 'counts'
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
    fig.savefig(f"{dirname}/{figname}_{dset_name_sim.replace('/', '_')}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

plot_counts = plot_bitstring_counts

def plot_fidelity_by_depth(
    fname_sim: str,
    fname_expt: str,
    circ_name_sim: str,
    circ_name_expt: str,
    n_qubits: int,
    pad_zero: Optional[str] = None,
    dirname: str = 'figs/fid_by_depth',
    figname: str = 'fid_by_depth',
    repetitions: int = 5000,
    mark_itoffoli: bool = False,
) -> None:
    """Plots the fidelity between simulated and experimental circuits vs circuit depth."""
    # If pad_zero not given, can deduce from n_qubits. 
    # For 3-qubit circuits, assume the experiments are run on Q4, Q5, Q6.
    if pad_zero is None:
        if n_qubits == 3:
            pad_zero = 'end'
        elif n_qubits == 4:
            pad_zero = ''

    # Create the qubits and circuit string converter.
    qubits = cirq.LineQubit.range(n_qubits)
    converter = CircuitStringConverter(qubits)

    # TODO: The part below need to be written better.
    # Load simulation circuit.
    with h5py.File(fname_sim + '.h5', 'r') as h5file:
        qtrl_strings = json.loads(h5file[circ_name_sim][()])
    circuit_sim = converter.convert_strings_to_circuit(qtrl_strings)
    non_z_locations_sim = get_non_z_locations(circuit_sim)
    # print(f'{len(circuit_sim) = }')

    # Load experimental circuit and data.
    pkl_data = pickle.load(open(fname_expt + '.pkl', 'rb'))
    qtrl_strings_expt = pkl_data[circ_name_expt + '_by_depth']['circs'][-1]
    results_expt = pkl_data[circ_name_expt + '_by_depth']['results']
    non_z_locations_expt = get_non_z_locations(qtrl_strings_expt)
    # non_z_locations_expt_comp = get_non_z_locations(combine_simultaneous_cz(qtrl_strings_expt))
    # print(f'{len(non_z_locations_expt) = }')
    # print(f'{len(non_z_locations_expt_comp) = }')

    if len(non_z_locations_sim) != len(non_z_locations_expt):
        warnings.warn("Circuit depth don't match!")
    
    fidelities = []
    locations_itoffoli = []
    for i, (i_sim, i_expt) in enumerate(zip(non_z_locations_sim, non_z_locations_expt)):
        # If mark_toffoli set to True, obtain the locations of iToffoli gates in natural running indices.
        # print('i = ', i, i_sim, i_expt)
        assert qtrl_strings[i_expt] == qtrl_strings_expt[i_expt]
        if mark_itoffoli:
            moment = converter.convert_strings_to_circuit([qtrl_strings_expt[i_expt]]).moments[0]
            if len(moment) == 1 and moment.operations[0].gate.num_qubits() == 3:
                locations_itoffoli.append(i)

        # Compute the simulated bitstring counts.
        circuit_moment = converter.convert_strings_to_circuit(qtrl_strings[:i_expt + 1]) \
             + [cirq.measure(q) for q in qubits]
        result_moment = cirq.Simulator().run(circuit_moment, repetitions=repetitions)
        histogram = result_moment.multi_measurement_histogram(keys=[str(i) for i in range(n_qubits)])
        array_sim = histogram_to_array(histogram)

        # Obtain the array of the experimental bitstring counts.
        array_expt = process_bitstring_counts(results_expt[i_expt], pad_zero=pad_zero)
        
        # Compute the fidelity (1 - TVD).
        fidelity = 1 - np.sum(np.abs(array_sim - array_expt)) / 2
        # print(f'{fidelity = }')
        fidelities.append(fidelity)

    
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(fidelities, marker='.')
    if mark_itoffoli:
        ax.plot(fidelities, color='r', ls='', marker='x', ms=10, mew=2, markevery=locations_itoffoli)
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Fidelity (1 - TVD)")

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(f'{dirname}/{figname}.png', dpi=FIGURE_DPI, bbox_inches='tight')

def display_fidelities(
    fname_exact: str,
    fname_expt: str,
    subscript: str,
    suffix: str = '',
    spin: str = '',
    dirname: str = 'figs/amp',
    figname: str = 'fid'
) -> None:
    """Displays fidelities between exact and experimental results.
    
    Args:
        fname_exact: The file name of the exact calculation.
        fname_expt: The file name of the experimental calculation.
        subscript: The subscript of the quantities. ``'e'``, ``'h'`` or ``'n'``.
        spin: The spin of the states.
        dirname: The directory name.
        figname: The figure name.
    """
    assert spin in ['u', 'd', '']
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    h5file_exact = h5py.File(fname_exact + '.h5', 'r')
    h5file_expt = h5py.File(fname_expt + '.h5', 'r')

    # XXX: 2 and 4 are hardcoded.
    if spin in ['u', 'd']:
        dim = 2
    else:
        dim = 4
    
    purities = np.zeros((dim, dim))
    fidelities = np.zeros((dim, dim))

    for i in range(dim):
        state_exact = h5file_exact[f'psi/{subscript}{i}{spin}'][:]
        state_expt = h5file_expt[f'rho{suffix}/{subscript}{i}{spin}'][:]
        purities[i, i] = np.trace(state_expt @ state_expt).real
        fidelities[i, i] = get_fidelity(state_exact, state_expt)

        for j in range(i + 1, dim):
            # p is filled on the upper half triangle.
            state_exact = h5file_exact[f'psi/{subscript}p{i}{j}{spin}'][:]
            state_expt = h5file_expt[f'rho{suffix}/{subscript}p{i}{j}{spin}'][:]
            purities[i, j] = np.trace(state_expt @ state_expt).real
            fidelities[i, j] = get_fidelity(state_exact, state_expt)

            # m is filled on the lower half triangle.
            state_exact = h5file_exact[f'psi/{subscript}m{i}{j}{spin}'][:]
            state_expt = h5file_expt[f'rho{suffix}/{subscript}m{i}{j}{spin}'][:]
            purities[j, i] = np.trace(state_expt @ state_expt).real
            fidelities[j, i] = get_fidelity(state_exact, state_expt)

    np.savetxt(f"data/purity_{figname[4:]}.dat", purities)
    np.savetxt(f"data/{figname}.dat", fidelities)

    fig, ax = plt.subplots()
    im = ax.imshow(fidelities, vmin=0.0, vmax=1.0)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            if fidelities[i, j] > 0.4:
                ax.text(j, i, f"{fidelities[i, j]:.2f}", color='black', ha='center', va='center')
            else:
                ax.text(j, i, f"{fidelities[i, j]:.2f}", color='white', ha='center', va='center')
    fig.colorbar(im, ticks=np.arange(0.0, 1.0 + 1e-8, 0.2))
    fig.savefig(f'{dirname}/{figname}.png', dpi=FIGURE_DPI, bbox_inches='tight')

    h5file_exact.close()
    h5file_expt.close()

def display_traces(
    fname: str,
    subscript: str,
    suffix: str = '',
    spin: str = '',
    dirname: str = 'figs/amp',
    figname: str = 'trace'
) -> None:
    """Displays fidelities between exact and experimental results.
    
    Args:
        fname_exact: The file name of the exact calculation.
        fname_expt: The file name of the experimental calculation.
        subscript: The subscript of the quantities. ``'e'``, ``'h'`` or ``'n'``.
        spin: The spin of the states.
        dirname: The directory name.
        figname: The figure name.
    """
    assert spin in ['u', 'd', '']

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    h5file = h5py.File(fname+ '.h5', 'r')

    # XXX: 2 and 4 are hardcoded.
    if spin in ['u', 'd']:
        dim = 2
    else:
        dim = 4
    
    traces = np.zeros((dim, dim))
    for i in range(dim):
        traces[i, i] = h5file[f"trace{suffix}/{subscript}{i}{spin}"][()]
        for j in range(i + 1, dim):
            # p (m) is filled on the upper (lower) half triangle.
            traces[i, j] = h5file[f"trace{suffix}/{subscript}p{i}{j}{spin}"][()]
            traces[j, i] = h5file[f"trace{suffix}/{subscript}m{i}{j}{spin}"][()]

    np.savetxt(f"data/{figname}.dat", traces)

    fig, ax = plt.subplots()
    im = ax.imshow(traces, vmin=0.0, vmax=1.0)
    for i in range(dim):
        for j in range(dim):
            # Here x is the column index and y is the row index.
            if traces[i, j] > 0.5:
                ax.text(j, i, f"{traces[i, j]:.2f}", color='black', ha='center', va='center')
            else:
                ax.text(j, i, f"{traces[i, j]:.2f}", color='white', ha='center', va='center')
    fig.colorbar(im, ticks=np.arange(0.0, 1.0 + 1e-8, 0.2))
    fig.savefig(f'{dirname}/{figname}.png', dpi=FIGURE_DPI, bbox_inches='tight')

    h5file.close()
    