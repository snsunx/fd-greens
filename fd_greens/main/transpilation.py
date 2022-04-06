"""
===========================================================
Circuit Transpilation (:mod:`fd_greens.main.transpilation`)
===========================================================
"""

from typing import Mapping, Tuple, Optional, Iterable, Union, Sequence, List
from itertools import combinations

import numpy as np
from permutation import Permutation

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Qubit, Clbit, Barrier
from qiskit.extensions import (
    XGate,
    HGate,
    RXGate,
    RZGate,
    CPhaseGate,
    CZGate,
    SwapGate,
    UGate,
)

from .params import C0C0iXGate
from ..utils import (
    save_circuit_figure,
    create_circuit_from_inst_tups,
    remove_instructions,
    get_registers_in_inst_tups,
    circuit_equal,
)

QubitLike = Union[int, Qubit]
ClbitLike = Union[int, Clbit]
InstructionTuple = Tuple[Instruction, List[QubitLike], Optional[List[ClbitLike]]]


def transpile_into_berkeley_gates(
    circ: QuantumCircuit, circ_label: str, savefig: bool = True
) -> QuantumCircuit:
    """Transpiles the circuit into native gates on the Berkeley device.

    This is the main function in this module and calls the other functions as subroutines.
    The basis gates are assumed to be :math:`X_{\pi/2}`, virtual Z, CS, CZ, and C0iXC0.
    The transpilation procedure depends on the circuit label. This function consists of
    the following steps:

    1. Permute and add SWAP gates (``transpile_by_permutation``)
    
    2. Transpile three-qubit gates (``transpile_3q_gates``)
    
    3. Transpile two-qubit gates (``transpile_2q_gates``)
    
    4. Transpile single-qubit gates (``transpile_1q_gates``)
    
    Args:
        circ: The circuit to be transpiled to native gates on the Berkeley device.
        circ_label: The circuit label, e.g. ``'0u'``, ``'0d'`` etc.
        savefig: Whether to save the circuit figures.
    
    Returns:
        circ_new: The circuit after transpilation to the native gates on the Berkeley device.
    """
    print(f"Transpiling circ{circ_label} into native gates on Berkeley device")
    if savefig:
        save_circuit_figure(circ, circ_label + "_untranspiled")

    circ_new = remove_instructions(circ, ["barrier"])
    circ_new = transpile_by_permutation(circ_new, circ_label, savefig=savefig)
    if len(circ.qregs[0]) == 4:
        circ_new = transpile_3q_gates(circ_new, circ_label, savefig=savefig)
    circ_new = transpile_2q_gates(circ_new, circ_label, savefig=savefig)
    # if circ_label[0] != 'r':
    circ_new = transpile_1q_gates(circ_new, circ_label, savefig=savefig)

    if savefig:
        save_circuit_figure(circ_new, circ_label + "_transpiled")
    return circ_new


def transpile_by_permutation(
    circ: QuantumCircuit, circ_label: str, savefig: bool = False
) -> QuantumCircuit:
    """Performs transpilation by permutation based on the circuit label.

    This function is a wrapper of ``permute_qubits``, which inserts SWAP gates at specific 
    start and end positions, which are predetermined based on the circuit label. The goal
    is to permute qubits so that all multi-qubit gates act on adjacent qubits.
    
    Args:
        circ: The circuit to be transpiled by permutation.
        circ_label: The circuit label.
        savefig: Whether to save the circuit figure after transpilation.
    
    Returns:
        circ_new: The new circuit after transpilation by permutation.
    """
    if circ_label in ["0u", "r0u", "r1u"]:
        circ = permute_qubits(circ, [1, 2])
    elif circ_label in ["0d"]:
        circ = permute_qubits(circ, [1, 2], end=17)
    elif circ_label in ["r0u0d"]:
        circ = permute_qubits(circ, [2, 3], end=18)
    elif circ_label in ["r0u1d"]:
        circ = permute_qubits(circ, [2, 3], end=-4)
    elif circ_label in ["r0u1u"]:
        circ = permute_qubits(circ, [2, 3])
    elif circ_label in ["r0d1u"]:
        circ = permute_qubits(circ, [2, 3], start=-4)
    elif circ_label in ["r1u1d"]:
        circ = permute_qubits(circ, [2, 3], end=16)
    elif circ_label in ["01u"]:
        # TODO: Finish this part.
        """
        circ_new = combine_1q_gates(circ_new)
        circ_new = combine_2q_gates(circ_new)
        circ_new = permute_qubits(circ_new, [2, 3], start=-5)
        circ_new = convert_1q_to_xpi2(circ_new)
        circ_new = combine_1q_gates(circ_new)
        """
    elif circ_label in ["01d"]:
        circ = permute_qubits(circ, [2, 3], end=20)

    if savefig:
        save_circuit_figure(circ, circ_label + "_permuted")
    return circ


def transpile_3q_gates(
    circ: QuantumCircuit, circ_label: str, savefig: bool = False
) -> QuantumCircuit:
    """Performs three-qubit gate transpilation based on the circuit label.

    This function is a wrapper of ``convert_ccz_to_cixc``, which transpiles the CCZ gates
    in four-qubit circuits into C0iXC0 along with additional single- and two-qubit gates.
    
    Args:
        circ: The circuit on which CCZ gates need to be transpiled.
        circ_label: The circuit label.
        savefig: Whether to save the circuit figure after transpilation.
    
    Returns:
        circ_new: The new circuit after transpiling the CCZ gates.
    """
    assert len(circ.qregs[0]) == 4
    circ = convert_ccz_to_cixc(circ)
    circ = combine_1q_gates(circ)
    circ = combine_2q_gates(circ)
    if savefig:
        save_circuit_figure(circ, circ_label + "_after3q")
    return circ


def transpile_2q_gates(
    circ: QuantumCircuit, circ_label: str, savefig: bool = False
) -> QuantumCircuit:
    """Performs two-qubit gate transpilation based on the circuit label.

    Args:
        circ: The circuit on which two-qubit gates need to be transpiled.
        circ_label: The circuit label.
        savefig: Whether to save the circuit figure after transpilation.
    
    Returns:
        circ_new: The new circuit after transpiling the two-qubit gates.
    """
    reverse = True if circ_label == "0d" else False
    if circ_label in ["01d"]:
        circ = transpile_subcircuit(circ, {(0, 1): ["swap", "czxcz"]})
    if circ_label in ["r0u0d", "r0d1d", "r0u1u", "r0u1d", "r0d1u", "r1u1d"]:
        circ = transpile_subcircuit(circ, {(0, 1): ["swap", "czxcz"]})
    circ = convert_swap_to_cz(circ, reverse=reverse)
    circ = combine_1q_gates(circ)
    circ = combine_2q_gates(circ)
    if savefig:
        save_circuit_figure(circ, circ_label + "_after2q")
    return circ


def transpile_1q_gates(
    circ: QuantumCircuit, circ_label: str, savefig: bool = False
) -> QuantumCircuit:
    """Performs single-qubit gate transpilation based on the circuit label.
    
    Args:
        circ: The circuit on which single-qubit gates need to be transpiled.
        circ_label: The circuit label.
        savefig: Whether to save the circuit figure after transpilation.
    
    Returns:
        circ_new: The new circuit after transpiling the single-qubit gates.
    """
    circ = convert_1q_to_xpi2(circ)
    qubit_trans = dict()
    for i in range(len(circ.qregs[0])):
        qubit_trans[(i,)] = [
            "xzpi2x",
            "combz",
            "xzpix",
            "combz",
            "3xpi2",
            "combz",
            "xzpi2x",
            "combz",
        ]
    circ = transpile_subcircuit(circ, qubit_trans)
    if savefig:
        save_circuit_figure(circ, circ_label + "_after1q")
    return circ


def permute_qubits(
    circ: QuantumCircuit,
    swap_inds: Sequence[int],
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> QuantumCircuit:
    """Permutes qubits in a circuit and inserts SWAP gates.
    
    Args:
        circ: The circuit to be transpiled.
        swap_inds: The qubit indices to permute.
        start: The start index of qubit permutation.
        end: The end index of qubit permutation.

    Returns:
        circ_new: The new circuit after transpilation.
    """
    # Initialize the start and end indices. If not given, initialize them to 0
    # and length of the circuit. If either index is passed in as an integer -n,
    # it corresponds to length of the circuit - n.
    if start is None:
        start = 0
    if end is None:
        end = len(circ)
    if start < 0:
        start += len(circ)
    if end < 0:
        end += len(circ)
    inst_tups = circ.data.copy()

    def conjugate(gate_inds, swap_inds):
        # Conjugates gate indices as for elements in a symmetric group.
        # Note that indices in the permutation package starts from 1, so we
        # need to add 1, permute, and subtract by 1 in the gate indices.
        perm = Permutation.cycle(*[i + 1 for i in swap_inds])
        gate_inds = [i + 1 for i in gate_inds]
        gate_inds_new = [perm(i) for i in gate_inds]
        gate_inds_new = sorted([i - 1 for i in gate_inds_new])
        return gate_inds_new

    # Permute the qubit indices based on a permutation cycle constructed from `swap_inds`.
    inst_tups_new = []
    for i, (inst, qargs, cargs) in enumerate(inst_tups):
        if i >= start and i < end:
            qargs = sorted([q._index for q in qargs])
            qargs = conjugate(qargs, swap_inds)
        inst_tups_new.append((inst, qargs, cargs))

    # Insert SWAP gates at the start and end indices. The SWAP gate at the start index
    # is only inserted when it is not 0, since otherwise it acts on the initial all 0
    # state and has no effect.
    inst_tups_new.insert(end, (SwapGate(), swap_inds, []))
    if start != 0:
        inst_tups_new.insert(start, (SwapGate(), swap_inds, []))

    # Create the new circuit and check it is equivalent to the original circuit.
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new


def convert_ccz_to_cixc(circ: QuantumCircuit) -> QuantumCircuit:
    r"""Converts CCZ gates to C0iXC0 gates along with other gates.

    Each CCZ gate in the original circuit is decomposed into :math:`\text{CiZC}\ \text{CS}^\dagger(q_0,q_2)`.
    The :math:`\text{CS}^\dagger(q_0,q_2)` gate is then decomposed into :math:`\text{SWAP}(q_0,q_1)
    \text{CS}^\dagger(q_1,q_2) \text{SWAP}(q_0,q_1)`, and only one of the SWAP gates is further decomposed into
    the form in arXiv:2111.04572 depending on whether it is an even (the one after is decomposed) or odd occurrence
    (the one in front is decomposed). The latter decomposition is for further simplification of the gates
    by ``simplify_swap_gates`` and ``simplify_czxcz``.
    
    Args:
        circ: The circuit on which CCZ gates are to be called to C0iXC0 gates.
    
    Returns:
        circ_new: The circuit after CCZ gates are called to C0iXC0 gates.
    """
    inst_tups = circ.data.copy()
    inst_tups_new = []

    count = 0
    for inst, qargs, cargs in inst_tups:
        if inst.name == "ccz":
            # CiZC instruction tuples.
            cizc_inst_tups = [
                (XGate(), [qargs[0]], []),
                (HGate(), [qargs[1]], []),
                (XGate(), [qargs[2]], []),
                (C0C0iXGate, [qargs[0], qargs[2], qargs[1]], []),
                (XGate(), [qargs[0]], []),
                (HGate(), [qargs[1]], []),
                (XGate(), [qargs[2]], []),
            ]

            # Instruction tuples for a SWAP gate absent a CZ.
            swapmcz_inst_tups = [
                (RXGate(np.pi / 2), [qargs[0]], []),
                (RXGate(np.pi / 2), [qargs[1]], []),
                (CZGate(), [qargs[0], qargs[1]], []),
                (RXGate(np.pi / 2), [qargs[0]], []),
                (RXGate(np.pi / 2), [qargs[1]], []),
                (CZGate(), [qargs[0], qargs[1]], []),
                (RXGate(np.pi / 2), [qargs[0]], []),
                (RXGate(np.pi / 2), [qargs[1]], []),
            ]

            # CSdag(q1,q2) CZ(q0,q1) SWAP(q0,q1) instruction tuples.
            csdagczswap_inst_tups = [
                (CPhaseGate(-np.pi / 2), [qargs[1], qargs[2]], []),
                (CZGate(), [qargs[0], qargs[1]], []),
                (SwapGate(), [qargs[0], qargs[1]], []),
            ]

            # For even occurrences of the 3q gate, decompose the first SWAP gate
            # and move CZ gate after CSdag.
            if count % 2 == 0:
                inst_tups_new += cizc_inst_tups
                inst_tups_new += swapmcz_inst_tups
                inst_tups_new += csdagczswap_inst_tups
            # For odd occurrences of the 3q gate, decompose the second SWAP gate
            # and move CZ gate in front of CSdag.
            else:
                inst_tups_new += csdagczswap_inst_tups[::-1]
                inst_tups_new += swapmcz_inst_tups
                inst_tups_new += cizc_inst_tups

            count += 1

        else:
            inst_tups_new.append((inst, qargs, cargs))

    # Check the transpiled circuit is equivalent to the original circuit.
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new


def convert_swap_to_cz(
    circ: QuantumCircuit,
    q_pairs: Sequence[Tuple[int, int]] = None,
    reverse: bool = False,
) -> QuantumCircuit:
    r"""Converts SWAP gates to CZ and :math:`X_{\pi/2}` gates except for the ones at the end.
    
    Args:
        circ: The circuit on which the SWAP gates need to be transpiled
        q_pairs: The qubit pairs on which the SWAP gates are converted.
        reverse: Whether the :math:`X_{\pi/2}(q_0)X_{\pi/2}(q_1) \text{CZ}` sequence
            is applied in the reversed way.

    Returns:
        circ_new: The new circuit after converting the SWAP gates.
    """
    inst_tups = circ.data.copy()
    inst_tups_new = []

    # Iterate from the end of the circuit, do not start converting SWAP gates to CZ gates
    # unless after encountering a non-SWAP gate. This is because SWAP gates at the end can
    # be kept track of classically.
    convert_swap = False
    for inst, qargs, cargs in inst_tups[::-1]:
        if inst.name == "swap":  # SWAP gate
            q_pair = tuple([x._index for x in qargs])
            q_pair_in = True
            if q_pairs is not None:
                q_pair_in = q_pair in q_pairs
            if convert_swap and q_pair_in:  # SWAP gate not at the end, convert
                swap_inst_tups = [
                    (RXGate(np.pi / 2), [qargs[0]], []),
                    (RXGate(np.pi / 2), [qargs[1]], []),
                    (CZGate(), [qargs[0], qargs[1]], []),
                ] * 3
                if reverse:
                    inst_tups_new = list(reversed(swap_inst_tups)) + inst_tups_new
                else:
                    inst_tups_new = swap_inst_tups + inst_tups_new
            else:  # SWAP gate at the end, do not convert
                inst_tups_new.insert(0, (inst, qargs, cargs))
        else:  # Non-SWAP gate, set convert_swap to True
            inst_tups_new.insert(0, (inst, qargs, cargs))
            convert_swap = True

    # Create the transpiled circuit and check it is equivalent to the original circuit.
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new


def convert_1q_to_xpi2(circ: QuantumCircuit) -> QuantumCircuit:
    r"""Converts single-qubit gates :math:`H`, :math:`X` and :math:`Y_{\theta}` to :math:`X_{\pi/2}`
    and virtual :math:`Z` gates.

    The conversion rules used in this function are:
    
    .. math::

        H &\to Z_{\pi/2} X_{\pi/2} Z_{\pi/2}
        
        X &\to X_{\pi/2} X_{\pi/2}

        Y_\theta &\to X_{\pi/2} Z_{\pi-\theta} X_{\pi/2} Z_{-\pi}

        U_3(\theta, \phi, \lambda) &\to Z_\phi X_{\pi/2} Z_{\pi-\theta} X_{\pi/2} Z_{\lambda-\pi}
    
    Args:
        circ: The circuit on which single-qubit gates need to be transpiled.

    Returns:
        circ_new: The new circuit after transpiling hte single-qubit gates.
    """
    inst_tups = circ.data.copy()
    inst_tups_new = []
    for inst, qargs, cargs in inst_tups:
        if inst.name == "h":
            inst_tups_new += [
                (RZGate(np.pi / 2), qargs, []),
                (RXGate(np.pi / 2), qargs, []),
                (RZGate(np.pi / 2), qargs, []),
            ]
        elif inst.name == "x":
            inst_tups_new += [
                (RXGate(np.pi / 2), qargs, []),
                (RXGate(np.pi / 2), qargs, []),
            ]
        elif inst.name == "ry":
            theta = inst.params[0]
            inst_tups_new += [
                (RZGate(-np.pi), qargs, []),
                (RXGate(np.pi / 2), qargs, []),
                (RZGate((np.pi - theta) % (2 * np.pi)), qargs, []),
                (RXGate(np.pi / 2), qargs, []),
            ]
        elif inst.name == "u3":
            theta, phi, lam = inst.params
            inst_tups_new += [
                (RZGate(lam - np.pi), qargs, []),
                (RXGate(np.pi / 2), qargs, []),
                (RZGate(np.pi - theta), qargs, []),
                (RXGate(np.pi / 2), qargs, []),
                (RZGate(phi), qargs, []),
            ]
        else:
            inst_tups_new.append((inst, qargs, cargs))

    # Create the transpiled circuit and check it is equivalent to the original circuit.
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new


def combine_1q_gates(
    circ: QuantumCircuit, qubits: Optional[Iterable[int]] = None
) -> QuantumCircuit:
    r"""Combines certain single-qubit gates to identities on given qubits.

    The combination rules used in this function are:

    .. math:: 

        H\ H &\to I
    
        X\ X &\to I
    
        X_{\theta} X_{-\theta} &\to I
    
    Args:
        circ: The circuit on which 1q gates are to be combined.
        qubits: Qubit indices on which 1q gates are to be combined.

    Returns:
        circ_new: The circuit after certain 1q gates are combined.
    """
    inst_tups = circ.data.copy()
    if qubits is None:
        qreg, _ = get_registers_in_inst_tups(inst_tups)
        qubits = range(len(qreg))

    # for i, (inst, qargs, cargs) in enumerate(inst_tups[:10]):
    #     print(i, inst.name, qargs, cargs)

    del_inds = []
    for q in qubits:
        # Initialize i_old and inst_old. i_old is the index of the previous 1q gate,
        # and inst_old, which is set to UGate(0, 0, 0), is a sentinel that should not appear
        # in the circuit.
        i_old = None
        inst_old = UGate(0, 0, 0)
        for i, (inst, qargs, _) in enumerate(inst_tups):
            if q in [x._index for x in qargs]:
                if len(qargs) == 1:  # 1q gate
                    # if verbose and inst.name == 'rx' and qargs[0]._index == 1:
                    #     print(i, 'inst params', inst.params[0], len(qargs))
                    if inst.name == inst_old.name:
                        # if inst.name == 'rx': print(q, inst.params[0], inst_old.params[0])
                        # Encountering a 1q gate the same as the previous gate. The two gates are
                        # deleted when they are H gates, X gates, or Rx(\theta) and Rx(-\theta).
                        if inst.name in ["h", "x"] or (
                            inst.name == "rx"
                            and abs(inst.params[0] + inst_old.params[0]) < 1e-8
                        ):
                            del_inds += [i_old, i]
                            i_old = None
                            inst_old = UGate(0, 0, 0)
                    else:
                        # Encountering a 1q gate not the same as the previous gate.
                        # Update i_old and inst_old and continue the search.
                        i_old = i
                        inst_old = inst.copy()
                else:
                    # Encountering a 2q gate. Start over the search by resetting i_old and inst_old.
                    i_old = None
                    inst_old = UGate(0, 0, 0)

    # Create the new instruction tuples, which do not contain gates with indices in del_inds.
    inst_tups_new = [inst_tups[i] for i in range(len(inst_tups)) if i not in del_inds]

    # Create the transpiled circuit and check it is equivalent to the original circuit.
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new


def combine_2q_gates(
    circ: QuantumCircuit, qubit_pairs: Optional[Iterable[Tuple[int, int]]] = None
) -> QuantumCircuit:
    r"""Combines certain two-qubit gates into identities on given qubit pairs.

    The combination rules used in this function are:

    .. math::

        \text{CZ}\ \text{CZ} &\to I

        \text{SWAP}\ \text{SWAP} &\to I
    
    Args:
        circ: The circuit on which certain two-qubit gates need to be combined.
        qubit_pairs: An iterable of the qubit pairs on which two-qubit gates are combined.
    
    Returns:
        circ_new: The new circuit after certain two-qubit gates are combined.
    """
    inst_tups = circ.data.copy()
    if qubit_pairs is None:
        qreg, _ = get_registers_in_inst_tups(inst_tups)
        qubits = range(len(qreg))
        qubit_pairs = list(combinations(qubits, 2))

    del_inds = []
    count = 0
    for q_pair in qubit_pairs:
        i_old = None
        inst_old = UGate(0, 0, 0)
        for i, (inst, qargs, _) in enumerate(inst_tups):
            qarg_inds = [x._index for x in qargs]
            if tuple(q_pair) == tuple(qarg_inds):
                if (
                    (
                        inst.name == inst_old.name == "cp"
                        and inst.params[0] == inst_old.params[0] == np.pi
                    )
                    or inst.name == inst_old.name == "cz"
                    or inst.name == inst_old.name == "swap"
                ):
                    del_inds += [i_old, i]
                    i_old = None
                    inst_old = UGate(0, 0, 0)
                else:
                    i_old = i
                    inst_old = inst.copy()
            else:
                # Encountering a gate that is not a CS gate. Start over the search by
                # resetting i_old and inst_old to the initial values.
                i_old = None
                inst_old = UGate(0, 0, 0)

    # Include only the gates that are not in del_inds. Insert CZ gates
    # at the insert_inds and create the new circuit.
    inst_tups_new = [inst_tups[i] for i in range(len(inst_tups)) if i not in del_inds]
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new


def transpile_subcircuit(
    circ: QuantumCircuit, qubits_trans_map: Mapping[Sequence[int], Sequence[str]]
) -> QuantumCircuit:
    """Transpiles a circuit by transpiling the subcircuits.
    
    This function is carried out by:
    
    1. Split the circuit into subcircuits
    
    2. Apply the simplification functions
    
    3. Merge into the main circuit.
    
    Args:
        circ: The circuit to be transpiled.
        qubits_trans_map: A dictionary with qubit indices as the keys and transpilation
            function strings as the values.

    Returns:
        circ_new: The circuit after transpiling on the subcircuits.
    """
    # print('circ\n', circ)
    inst_tups = circ.data.copy()
    for qubits, trans_strs in qubits_trans_map.items():
        # Split the inst_tups of the main circuit into a subcircuit and a main circuit.
        # Also record the barrier locations which will be used in reconstructing the circuit.
        inst_tups_sub, inst_tups_main, barr_loc = split_subcircuit(inst_tups, qubits)
        # print(create_circuit_from_inst_tups(inst_tups_sub))
        # print(inst_tups_main)
        # inst_tups_main = [inst_tup for inst_tup in inst_tups_main if inst_tup[0] is not None]
        # print(create_circuit_from_inst_tups([inst_tup for inst_tup in inst_tups_main if inst_tup[0] is not None]))

        for trans_str in trans_strs:
            trans_func = transpilation_dict[trans_str]
            inst_tups_sub = trans_func(inst_tups_sub)
        inst_tups = merge_subcircuit(inst_tups_sub, inst_tups_main, barr_loc, qubits)
        # print(create_circuit_from_inst_tups(inst_tups.copy()))

    # print('circ\n', circ)
    circ_new = create_circuit_from_inst_tups(inst_tups)
    # print('circ_new\n', circ_new)
    assert circuit_equal(circ, circ_new)
    return circ_new


def split_subcircuit(
    inst_tups: Sequence[InstructionTuple], qubits_subcirc: Sequence[int]
) -> Tuple[List[InstructionTuple], List[InstructionTuple], List[int]]:
    """Splits a subcircuit from a circuit.

    This function is called as a subroutine of ``transpile_subcircuit``.
    
    Args:
        inst_tups: The circuit that need to be split in instruction tuple form.
        qubits_subcirc: The qubit indices of the subcircuit.

    Returns:
        inst_tups_sub: The subcircuit instruction tuples.
        inst_tups_main: The main circuit instruction tuples.
        barr_loc: The main-circuit locations of the barriers in the subcircuits.
    """
    n_qubits = len(qubits_subcirc)
    map_qubits = lambda qubits_gate: [qubits_subcirc.index(q) for q in qubits_gate]

    # inst_tups = circ.data.copy()
    barr_loc = []
    inst_tups_sub = []
    inst_tups_main = []

    for i, (inst, qargs, cargs) in enumerate(inst_tups):
        if isinstance(qargs[0], int):
            qubits_gate = qargs
        else:
            qubits_gate = [q._index for q in qargs]

        # if qargs_ind == qubits:
        if set(qubits_gate).issubset(set(qubits_subcirc)):
            # Gate qubits are a subset of subcircuit qubits. Append the instruction tuple to the
            # subcircuit. Append (None, None, None) to the main circuit, which will be handled
            # in the merge function.
            inst_tups_sub.append((inst, map_qubits(qubits_gate), cargs))
            inst_tups_main.append((None, None, None))
            # inst_tups_main += [(None, None, None)]
        elif set(qubits_gate).intersection(set(qubits_subcirc)):
            # Gate qubits intersect subcircuit qubits. Insert a barrier in the circuit and
            # record the barrier location. Append the instruction tuple to the main circuit.
            barr_loc.append(i)
            inst_tups_sub.append((Barrier(n_qubits), range(n_qubits), []))
            inst_tups_main.append((inst, qargs, cargs))
        else:
            # Gate qubits do not overlap with subcircuit qubits. Only append the instruction tuple
            # to the main circuit.
            inst_tups_main.append((inst, qargs, cargs))

    return inst_tups_sub, inst_tups_main, barr_loc


def merge_subcircuit(
    inst_tups_sub: Sequence[InstructionTuple],
    inst_tups_main: Sequence[InstructionTuple],
    barr_loc: Sequence[int],
    qubits: Sequence[int],
) -> QuantumCircuit:
    """Merges a subcircuit into the main circuit.
    
    This function is called as a subroutine of ``transpile_subcircuit``.

    Args:
        inst_tups_sub: Instruction tuples of the subcircuit.
        inst_tups_main: Instruction tuples of the main circuit.
        barr_loc: Locations of the barriers in the main circuit.
        qubits: Qubit indices of the subcircuit.
    
    Returns:
        inst_tups: The instruction tuples after merging the subcircuit and the main circuit.
    """
    map_qubits = lambda x: [qubits[i] for i in x]
    inst_tups = []

    barr_inds = []
    inst_tups_sub_new = []
    barr_ind = -1
    for inst, qargs, cargs in inst_tups_sub:
        if inst.name != "barrier":
            inst_tups_sub_new.append((inst, qargs, cargs))
            barr_inds.append(barr_ind)
        else:
            barr_ind = barr_loc.pop(0)

    for i, (inst, qargs, cargs) in enumerate(inst_tups_main):
        if inst is not None:
            # If the instruction is not one of the subcircuit instructions,
            # just append it to inst_tups.
            inst_tups.append((inst, qargs, cargs))
        else:
            # Look for instruction in inst_tups_sub.
            # print('barr_inds =', barr_inds)
            # barr_ind = barr_inds[0]
            if len(barr_inds) > 0 and i > barr_inds[0]:
                inst_, qargs_, cargs_ = inst_tups_sub_new.pop(0)
                inst_tups.append((inst_, map_qubits(qargs_), cargs_))
                barr_inds = barr_inds[1:]

    # circ = create_circuit_from_inst_tups(inst_tups)
    return inst_tups


def simplify_xzpi2x(
    circ: Union[QuantumCircuit, Sequence[InstructionTuple]]
) -> Union[QuantumCircuit, Sequence[InstructionTuple]]:
    """Simplifies :math:`X_{\pi/2}Z_{\pi/2}X_{\pi/2}` to :math:`Z_{\pi/2}X_{\pi/2}Z_{\pi/2}`
    on a single-qubit circuit.
    
    Args:
        circ: A circuit on which :math:`X_{\pi/2}Z_{\pi/2}X_{\pi/2}` need to be simplified.
    
    Returns:
        circ_new: The new circuit after the simplifying :math:`X_{\pi/2}Z_{\pi/2}X_{\pi/2}`.
    """
    # assert len(circ.qregs[0]) == 1
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:
        inst_tups = circ.copy()
    inst_tups_new = []
    inst_tups_running = []

    called = False
    for inst, qargs, cargs in inst_tups:
        # Always append the instruction tuple to inst_tups_new.
        # If transpilation is carried out, inst_tups_new is modified at the end.
        inst_tups_new.append((inst, qargs, cargs))

        if inst == RXGate(np.pi / 2):
            # If the gate is X(pi/2), append to inst_tups_running if it contains 0 or 2 elements,
            # otherwise reset inst_tups_running to [X(pi/2)].
            if len(inst_tups_running) == 0 or len(inst_tups_running) == 2:
                inst_tups_running.append((inst, qargs, cargs))
            elif len(inst_tups_running) == 1:
                inst_tups_running = [(inst, qargs, cargs)]
        elif inst.name == "rz" and abs(inst.params[0]) == np.pi / 2:
            # If the gate is Z(pi/2), append to inst_tups_running if it contains 1 element,
            # otherwise reset inst_tups_running to [].
            if len(inst_tups_running) == 1:
                inst_tups_running.append((inst, qargs, cargs))
            elif len(inst_tups_running) == 0 or len(inst_tups_running) == 2:
                inst_tups_running = []
        else:
            # The gate is neither X(pi/2) nor Z(pi/2). Reset inst_tups_running to [].
            inst_tups_running = []

        # print('len(inst_tups_running) =', len(inst_tups_running))
        if len(inst_tups_running) == 3:
            # inst_tups_running is of the form [X(pi/2), Z(pi/2), X(pi/2)].
            # Remove the last 3 elements of inst_tups_new and append [Z(pi/2), X(pi/2), Z(pi/2)].
            inst_tups_new = inst_tups_new[:-3]
            z_angle = inst_tups_running[1][0].params[0]
            inst_tups_new += [
                (RZGate(z_angle), [0], []),
                (RXGate(np.pi / 2), [0], []),
                (RZGate(z_angle), [0], []),
            ]

            # Set called to True so that the function will be called again recursively.
            # Reset inst_tups_running to [] so that the chain of gates will be built up again.
            called = True
            inst_tups_running = []

    assert circuit_equal(inst_tups, inst_tups_new, init_state_0=False)
    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new

    # If called is True, call the function again recursively. Otherwise return the circuit.
    if called:
        circ_new = simplify_xzpi2x(circ_new)
    return circ_new


def simplify_xzpix(
    circ: Union[QuantumCircuit, Sequence[InstructionTuple]]
) -> Union[QuantumCircuit, Sequence[InstructionTuple]]:
    """Simplifies :math:`X_{\pi/2}Z_{\pi}X_{\pi/2}` to :math:`Z_{\pi}` recursively
    on a single-qubit subcircuit.
    
    Args:
        circ: A circuit on which :math:`X_{\pi/2}Z_{\pi}X_{\pi/2}` need to be simplified.
    
    Returns:
        circ_new: The new circuit after simplifying :math:`X_{\pi/2}Z_{\pi}X_{\pi/2}`.
    """
    # assert len(circ.qregs[0]) == 1
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:
        inst_tups = circ.copy()
    inst_tups_new = []
    inst_tups_running = []

    called = False
    for inst, qargs, cargs in inst_tups:
        # Always append the instruction tuple to inst_tups_new.
        # If transpilation is carried out, inst_tups_new is modified at the end.
        inst_tups_new.append((inst, qargs, cargs))

        if inst == RXGate(np.pi / 2):
            # If the gate is X(pi/2), append to inst_tups_running if it contains 0 or 2 elements,
            # otherwise reset inst_tups_running to [X(pi/2)].
            if len(inst_tups_running) == 0 or len(inst_tups_running) == 2:
                inst_tups_running.append((inst, qargs, cargs))
            elif len(inst_tups_running) == 1:
                inst_tups_running = [(inst, qargs, cargs)]
        elif inst == RZGate(np.pi):
            # If the gate is Z(pi), append to inst_tups_running if it contains 1 element,
            # otherwise reset inst_tups_running to [].
            if len(inst_tups_running) == 1:
                inst_tups_running.append((inst, qargs, cargs))
            elif len(inst_tups_running) == 0 or len(inst_tups_running) == 2:
                inst_tups_running = []
        else:
            # The gate is neither X(pi/2) nor Z(pi/2). Reset inst_tups_running to [].
            inst_tups_running = []

        # print('len(inst_tups_running) =', len(inst_tups_running))
        if len(inst_tups_running) == 3:
            # inst_tups_running is of the form [X(pi/2), Z(pi/2), X(pi/2)].
            # Remove the last 3 elements of inst_tups_new and append [Z(pi/2), X(pi/2), Z(pi/2)].
            inst_tups_new = inst_tups_new[:-3]
            # inst_tups_new += [(RZGate(np.pi/2), [0], []),
            #                   (RXGate(np.pi/2), [0], []),
            #                   (RZGate(np.pi/2), [0], [])]
            inst_tups_new.append((RZGate(np.pi), [0], []))

            # Set called to True so that the function will be called again recursively.
            # Reset inst_tups_running to [] so that the chain of gates will be built up again.
            called = True
            inst_tups_running = []

    # print('called =', called)

    assert circuit_equal(inst_tups, inst_tups_new, init_state_0=False)
    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new

    # If called is True, call the function again recursively. Otherwise return the circuit.
    if called:
        circ_new = simplify_xzpix(circ_new)
    return circ_new


def simplify_3xpi2(
    circ: Union[QuantumCircuit, Sequence[InstructionTuple]]
) -> Union[QuantumCircuit, Sequence[InstructionTuple]]:
    """Simplifies :math:`X_{\pi/2}X_{\pi/2}X_{\pi/2}` to :math:`Z_{\pi}X_{\pi/2}Z_{\pi}`
    on a single-qubit circuit.
    
    Args:
        circ: A circuit on which :math:`X_{\pi/2}X_{\pi/2}X_{\pi/2}` need to be simplified.
        
    Returns:
        circ_new: The new circuit after simplifying :math:`X_{\pi/2}X_{\pi/2}X_{\pi/2}`.
    """
    # assert len(circ.qregs[0]) == 1
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:
        inst_tups = circ.copy()
    inst_tups_new = []
    inst_tups_running = []

    called = False
    for inst, qargs, cargs in inst_tups:
        # Always append the instruction tuple to inst_tups_new.
        # If transpilation is carried out, inst_tups_new is modified at the end.
        inst_tups_new.append((inst, qargs, cargs))

        if inst == RXGate(np.pi / 2):
            # Append to inst_tups_running if the gate is X(pi/2).
            inst_tups_running.append((inst, qargs, cargs))
        else:
            # The gate is not X(pi/2). Reset inst_tups_running to [].
            inst_tups_running = []

        # print('len(inst_tups_running) =', len(inst_tups_running))
        if len(inst_tups_running) == 3:
            # inst_tups_running is of the form [X(pi/2), Z(pi/2), X(pi/2)].
            # Remove the last 3 elements of inst_tups_new and append [Z(pi/2), X(pi/2), Z(pi/2)].
            inst_tups_new = inst_tups_new[:-3]
            inst_tups_new += [
                (RZGate(np.pi), [0], []),
                (RXGate(np.pi / 2), [0], []),
                (RZGate(np.pi), [0], []),
            ]

            # Set called to True so that the function will be called again recursively.
            # Reset inst_tups_running to [] so that the chain of gates will be built up again.
            called = True
            inst_tups_running = []

    # print('called =', called)

    assert circuit_equal(inst_tups, inst_tups_new, init_state_0=False)
    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new

    # If called is True, call the function again recursively. Otherwise return the circuit.
    if called:
        circ_new = simplify_xzpix(circ_new)
    return circ_new


def simplify_z_gates(
    circ: Union[QuantumCircuit, Sequence[InstructionTuple]]
) -> Union[QuantumCircuit, Sequence[InstructionTuple]]:
    """Simplifes series of Z gates on a single-qubit circuit.
    
    Args:
        circ: A circuit on which series of Z gates need to be simplified.
        
    Returns:
        circ_new: The new circuit after simplifying series of Z gates.
    """
    # assert len(circ.qregs[0]) == 1
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:
        inst_tups = circ.copy()
    inst_tups_new = []
    z_angles = []

    # combined = False
    inst_tups += [(Barrier(1), [0], [])]  # Append sentinel
    for inst, qargs, cargs in inst_tups:
        if inst.name == "rz":
            z_angles.append(inst.params[0])
        else:
            if len(z_angles) > 0:
                angle = sum(z_angles) % (2 * np.pi)
                if angle > np.pi:
                    angle -= 2 * np.pi  # [-pi, pi)
                if abs(angle) > 1e-8:
                    inst_tups_new.append((RZGate(angle), [0], []))
                z_angles = []
            inst_tups_new.append((inst, qargs, cargs))
    inst_tups_new = inst_tups_new[:-1]  # Remove setinel

    # Create the new circuit and check if the two circuits are equivalent.
    # circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(inst_tups, inst_tups_new, init_state_0=False)

    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new
    return circ_new


def simplify_swap_gates(
    circ: Union[QuantumCircuit, Sequence[InstructionTuple]]
) -> Union[QuantumCircuit, Sequence[InstructionTuple]]:
    """Simplifies SWAP gates by exchanging the qubits between two SWAP gates and
    removing the SWAP gates.
    
    Args:
        circ: A circuit on which SWAP gates need to be simplified.
        
    Returns:
        circ_new: The new circuit after simplifying the SWAP gates.
    """
    # assert len(circ.qregs[0]) == 2
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:
        inst_tups = circ.copy()
    inst_tups_new = []
    inst_tups_running = []

    called = False
    status = 0
    for inst, qargs, cargs in inst_tups:
        inst_tups_new.append((inst, qargs, cargs))

        if inst.name == "swap":
            inst_tups_running.append((inst, qargs, cargs))
            qargs_2q = qargs.copy()
            status += 1  # 0 to 1 or 1 to 2
        else:
            if status == 1:
                if len(qargs) == 1:
                    inst_tups_running.append(
                        (inst, list(set(qargs_2q).difference(set(qargs))), cargs)
                    )
                if len(qargs) == 2:
                    if inst.name == "barrier":
                        status = 0
                        inst_tups_running = []
                    else:
                        inst_tups_running.append((inst, list(reversed(qargs)), cargs))

        # print('status =', status)
        if status == 2:
            # print(len(inst_tups_running))
            # print(inst_tups_running)
            inst_tups_new = inst_tups_new[: -len(inst_tups_running)]
            inst_tups_new += inst_tups_running[1:-1]
            called = True
            inst_tups_running = []
            status = 0

    # print(create_circuit_from_inst_tups(inst_tups))
    # print(create_circuit_from_inst_tups(inst_tups_new))

    assert circuit_equal(inst_tups, inst_tups_new, init_state_0=False)
    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new

    if called:
        circ_new = simplify_swap_gates(circ_new)
    return circ_new


def simplify_czxcz(
    circ: Union[QuantumCircuit, Sequence[InstructionTuple]]
) -> Union[QuantumCircuit, Sequence[InstructionTuple]]:
    """Simplifies CZ IX CZ into ZX on a two-qubit circuit.
    
    Args:
        circ: A circuit on which CZ IX CZ need to be simplified.
        
    Returns:
        circ_new: The new circuit after simplifying CZ IX CZ.
    """
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:  # instruction tuples
        inst_tups = circ.copy()
    inst_tups_new = []
    inst_tups_running = []

    called = False
    for inst, qargs, cargs in inst_tups:
        inst_tups_new.append((inst, qargs, cargs))
        if inst.name == "cz":
            qargs_2q = qargs.copy()
            if len(inst_tups_running) == 0 or len(inst_tups_running) == 2:
                inst_tups_running.append((inst, qargs, cargs))
            else:
                assert len(inst_tups_running) == 1
                inst_tups_running = [(inst, qargs, cargs)]

        elif inst.name == "x":
            qargs_1q = qargs.copy()
            if len(inst_tups_running) == 1:
                inst_tups_running.append((inst, qargs, cargs))
            else:
                assert len(inst_tups_running) == 0 or len(inst_tups_running) == 2
                inst_tups_running = []

        if len(inst_tups_running) == 3:
            inst_tups_new = inst_tups_new[:-3]
            inst_tups_new += [
                (RZGate(np.pi), list(set(qargs_2q).difference(set(qargs_1q))), []),
                (XGate(), qargs_1q, []),
            ]

            called = True
            inst_tups_running = []

    assert circuit_equal(inst_tups, inst_tups_new, False)
    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new
    return circ_new


transpilation_dict = {
    "xzpi2x": simplify_xzpi2x,
    "combz": simplify_z_gates,
    "xzpix": simplify_xzpix,
    "3xpi2": simplify_3xpi2,
    "swap": simplify_swap_gates,
    "czxcz": simplify_czxcz,
}
