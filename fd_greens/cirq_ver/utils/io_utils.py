"""
===============================================
I/O Utilities (:mod:`fd_greens.utils.io_utils`)
===============================================
"""
import os
from typing import Any

import h5py
import numpy as np
from qiskit import QuantumCircuit


def initialize_hdf5(fname: str = "lih", calc: str = "greens") -> None:
    """Creates the HDF5 file and group names if they do not exist.
    
    The group created in the HDF5 file are:
    gs (for ground-state calculation), 
    eh (for (N+/-1)-state calculation)),
    amp (for transition amplitude calculation),
    circ0 (the (0, 0)-element circuit),
    circ1 (the (1, 1)-element circuit),
    circ01 (The (0, 1)-element circuit).
    
    Args:
        fname: The HDF5 file name.
        calc: Calculation mode. Either ``'greens'`` or ``'resp'``.
    """
    assert calc in ["greens", "resp"]

    fname += ".h5"
    if os.path.exists(fname):
        h5file = h5py.File(fname, "r+")
    else:
        h5file = h5py.File(fname, "w")

    if calc == "greens":
        grpnames = [
            "gs",
            "es",
            "amp",
            "circ0u",
            "circ1u",
            "circ01u",
            "circ0d",
            "circ1d",
            "circ01d",
        ]
    else:
        grpnames = [
            "gs",
            "es",
            "amp",
            "circ0u",
            "circ1u",
            "circ0d",
            "circ1d",
            "circ0u0d",
            "circ0u1u",
            "circ0u1d",
            "circ0d1u",
            "circ0d1d",
            "circ1u1d",
        ]
    for grpname in grpnames:
        if grpname not in h5file.keys():
            h5file.create_group(grpname)
    h5file.close()


def write_hdf5(
    h5file: h5py.File, grpname: str, dsetname: str, data: Any, overwrite: bool = True
) -> None:
    """Writes a data object to a dataset in an HDF5 file.
    
    Args:
        h5file: The HDF5 file.
        grpname: The name of the group to save the data under.
        dsetname: The name of the dataset to save the data to.
        data: The data to be saved.
        overwrite: Whether the dataset is overwritten.
    """
    if overwrite:
        if dsetname in h5file[grpname].keys():
            del h5file[f"{grpname}/{dsetname}"]
    try:
        h5file[f"{grpname}/{dsetname}"] = data
    except:
        pass


# circuit_to_qasm_str = lambda circ: convert_circuit_to_string(circ, "qasm")

# TODO: This function is not necessary. Can remove.
def save_circuit_figure(circ: QuantumCircuit, suffix: str) -> None:
    """Saves the circuit figure under the directory ``figs``.
    
    Args:
        circ: The circuit to be saved.
        suffix: The suffix to be appended to the figure name.
    """
    if not os.path.exists("figs"):
        os.makedirs("figs")
    fig = circ.draw("mpl")
    fig.savefig(f"figs/circ{suffix}.png", bbox_inches="tight")
