"""
===============================================================
Postprocessing Utilities (:mod:`fd_greens.utils.process_utils`)
===============================================================
"""

from fd_greens.utils.circuit_utils import QuantumCircuitLike
import h5py
from typing import Optional
from itertools import product
import numpy as np
from qiskit import QuantumCircuit
from .general_utils import reverse_qubit_order

bitstrings = ["".join(x) for x in product("012", repeat=3)]


def take_01(
    arr: np.ndarray, circ_label: Optional[str] = None, reverse: bool = True
) -> np.ndarray:
    """Takes the bitstrings with 0 and 1 in a bitstring counts and reverse the qubit order.
    
    Args:
        arr: The bitstring counts array.
        circ_label: The circuit label.
    
    Returns:
        arr_new: The processed bitstring counts array.
    """
    # Extract the number of qubits and construct the bitstrings. For certain circuit labels,
    # permute the bitstring elements according to the SWAP gates at the end.
    n_qubits = int(np.log(arr.shape[0]) / np.log(3))
    bitstrings = ["".join(x) for x in product("012", repeat=n_qubits)]
    if circ_label in ["0u", "1u"]:
        print("bitstring swapped")
        bitstrings = ["".join([x[i] for i in [0, 2, 1]]) for x in bitstrings]
    if circ_label in ["0u1u", "0d1u"]:
        print("bitstring swapped")
        bitstrings = ["".join([x[i] for i in [0, 1, 3, 2]]) for x in bitstrings]

    # Construct the new array by taking only the bitstrings that don't contain 2.
    arr_new = []
    for i, x in enumerate(bitstrings):
        if "2" not in x:
            arr_new.append(arr[i])
    arr_new = np.array(arr_new)
    if reverse:
        arr_new = reverse_qubit_order(arr_new)
    return arr_new


def process_berkeley_results(
    h5fname: str, circ_label: str, tomo_label: str, counts_name: str
) -> None:
    """Processes the Berkeley hardware results and saves the processed bitstring counts 
    in the HDF5 file.
    
    Args:
        h5fname: Name of the HDF5 file.
        circ_label: Label of the quantum circuit, e.g. '0u', '0d', '01d'.
        tomo_label: The tomography label, e.g. 'xx', 'xy', 'xz'.
        counts_name: Name of the bitstring counts.
    """
    h5file = h5py.File(h5fname + ".h5", "r+")

    # Obtain the experimental bitstring counts with suffix _exp.
    dset = h5file[f"circ{circ_label}/{tomo_label}"]
    counts_exp = dset.attrs[f"{counts_name}_exp"]

    # Process the results and saves to a new bitstring counts attributes with suffix _exp_proc.
    counts_exp_proc = take_01(counts_exp, circ_label=circ_label)
    dset.attrs[f"{counts_name}_exp_proc"] = counts_exp_proc


def compute_tvd(h5fname: str, circ_label: str, counts_name: str) -> float:
    """Calculates the average total variational distance (TVD) of a circuit.
    
    Args:
        h5fname: The HDF5 file name.
        circ_label: The circuit label.
        counts_name: Name of the bitstring counts.
    """
    h5file = h5py.File(h5fname + ".h5", "r")
    tomo_labels = ["".join(x) for x in product("xyz", repeat=2)]

    tvds = []
    for tomo_label in tomo_labels:
        dset = h5file[f"circ{circ_label}/{tomo_label}"]
        counts = dset.attrs[counts_name]
        counts_norm = counts / np.sum(counts)
        counts_exp = dset.attrs[f"{counts_name}_exp_proc"]
        counts_exp_norm = counts_exp / np.sum(counts_exp)

        tvd = np.sum(np.abs(counts_norm - counts_exp_norm)) / 2
        tvds.append(tvd)

    tvd_avg = np.average(tvds)
    return tvd_avg
