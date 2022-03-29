"""
===============================================
I/O Utilities (:mod:`fd_greens.utils.io_utils`)
===============================================
"""
import os
import h5py
from typing import Any
import numpy as np

from qiskit import QuantumCircuit
from .general_utils import get_unitary


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
        calc: Calculation mode. Either 'greens' or 'resp'.
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
    """Converts a circuit to QASM string.
    
    This function generates QASM string of the C0iXC0 gate and is required to transpile circuits
    that contain this gate, as the QASM string generation function in Qiskit does not implement 
    this gate.
    
    Args:
        circ: The circuit to be transformed to a QASM string.
    
    Returns:
        qasm_str: The QASM string corresponding to the circuit.
    """
    # print(set([x[0].name for x in circ.data]))
    # The header of the QASM string.
    qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    qasm_str += (
        "gate ccix p0,p1,p2 {x p0; x p1; ccx p0,p1,p2; cp(pi/2) p0, p1; x p0; x p1;}\n"
    )
    # qasm_str += 'gate ccix p0,p1,p2 {x p0; x p1; ccx p0,p1,p2; x p0; x p1;}\n'
    if len(circ.qregs) > 0:
        n_qubits = len(circ.qregs[0])
        qasm_str += f"qreg q[{n_qubits}];\n"
    if len(circ.cregs) > 0:
        n_clbits = len(circ.cregs[0])
        qasm_str += f"creg c[{n_clbits}];\n"

    for inst, qargs, cargs in circ.data:
        if inst.name == "rz":
            qasm_str += f"{inst.name}({inst.params[0]}) q[{qargs[0]._index}];\n"
        elif inst.name == "rx":
            qasm_str += f"{inst.name}({inst.params[0]}) q[{qargs[0]._index}];\n"
        elif inst.name == "p":
            qasm_str += f"{inst.name}({inst.params[0]}) q[{qargs[0]._index}];\n"
        elif inst.name == "cz":
            qasm_str += f"{inst.name} q[{qargs[0]._index}],q[{qargs[1]._index}];\n"
        elif inst.name == "cp":
            qasm_str += f"{inst.name}({inst.params[0]}) q[{qargs[0]._index}],q[{qargs[1]._index}];\n"
        elif inst.name == "unitary":
            assert [q._index for q in qargs] == [0, 2, 1]
            qasm_str += f"ccix q[0],q[2],q[1];\n"
        elif inst.name == "swap":
            qasm_str += f"swap q[{qargs[0]._index}],q[{qargs[1]._index}];\n"
        elif inst.name == "measure":
            qasm_str += f"{inst.name} q[{qargs[0]._index}] -> c[{cargs[0]._index}];\n"

    # Temporary check statement.
    uni = get_unitary(circ)
    circ_new = QuantumCircuit.from_qasm_str(qasm_str)
    uni_new = get_unitary(circ_new)
    assert np.allclose(uni, uni_new)
    return qasm_str


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
