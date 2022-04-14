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

from .general_utils import circuit_equal


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


def circuit_to_qasm_str(circ: QuantumCircuit) -> str:
    """Converts a circuit to a QASM string.
    
    This function is required to transpile circuits that contain C0C0iX and CCZ gates. 
    The ``QuantumCircuit.qasm()`` method in Qiskit does not implement these gates.
    
    Args:
        circ: The circuit to be transformed to a QASM string.
    
    Returns:
        qasm_str: The QASM string of the circuit.
    """
    # The header of the QASM string.
    qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

    # Define 3q gates c0c0ix and ccz, since these are not defined in the standard library.
    qasm_str += (
        "gate c0c0ix p0,p1,p2 {x p0; x p1; ccx p0,p1,p2; cp(pi/2) p0,p1; x p0; x p1;}\n"
    )
    qasm_str += "gate ccz p0,p1,p2 {h p2; ccx p0,p1,p2; h p2;}\n"

    # Define quantum and classical registers.
    if len(circ.qregs) > 0:
        n_qubits = len(circ.qregs[0])
        qasm_str += f"qreg q[{n_qubits}];\n"
    if len(circ.cregs) > 0:
        n_clbits = len(circ.cregs[0])
        qasm_str += f"creg c[{n_clbits}];\n"

    for inst, qargs, cargs in circ.data:
        # Build instruction string, quantum register string and
        # optionally classical register string.
        if len(inst.params) > 0 and not isinstance(inst.params[0], np.ndarray):
            # 1q or 2q gate with parameters
            params_str = ",".join([str(x) for x in inst.params])
            inst_str = f"{inst.name}({params_str})"
        else:  # 1q, 2q gate without parameters, 3q gate, measure or barrier
            inst_str = inst.name
        qargs_inds = [q._index for q in qargs]
        qargs_str = ",".join([f"q[{i}]" for i in qargs_inds])
        if cargs != []:
            cargs_inds = [c._index for c in cargs]
            cargs_str = ",".join([f"c[{i}]" for i in cargs_inds])

        if inst.name in [
            "rz",
            "rx",
            "ry",
            "h",
            "x",
            "p",
            "u3",
            "cz",
            "swap",
            "cp",
            "barrier",
            "c0c0ix",
            "ccz",
        ]:
            qasm_str += f"{inst_str} {qargs_str};\n"
        elif inst.name == "measure":
            qasm_str += f"{inst_str} {qargs_str} -> {cargs_str};\n"
        else:
            raise TypeError(
                f"Instruction {inst.name} cannot be converted to QASM string."
            )

    # Temporary check statement.
    circ_new = QuantumCircuit.from_qasm_str(qasm_str)
    assert circuit_equal(circ, circ_new)
    return qasm_str


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
