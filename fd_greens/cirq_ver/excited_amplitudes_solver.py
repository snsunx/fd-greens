"""
====================================================================================
N-Electron Transition Amplitudes Solver (:mod:`fd_greens.excited_amplitudes_solver`)
====================================================================================
"""

from itertools import product
from typing import Sequence

import h5py
import json
import cirq

from .molecular_hamiltonian import MolecularHamiltonian
from .operators import ChargeOperators
from .circuit_constructor import CircuitConstructor
from .circuit_string_converter import CircuitStringConverter
from .transpilation import transpile_into_berkeley_gates
from .parameters import get_method_indices_pairs
from .general_utils import histogram_to_array


class ExcitedAmplitudesSolver:
    """Solver for transition amplitudes between ground state and N-electron excited states."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        qubits: Sequence[cirq.Qid],
        method: str = "exact",
        repetitions: int = 1000,
        fname: str = "lih",
        suffix: str = "",
    ) -> None:
        """Initializes an ``ExcitedAmplitudesSolver`` object.

        Args:
            hamiltonian: The molecular Hamiltonian.
            qubits: Qubits in the circuit.
            method: The method for extracting the transition amplitudes.
            repetitions: Number of repetitions used in the simulations.
            fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
        """
        assert method in ["exact", "tomo"]

        # Input attributes.
        self.hamiltonian = hamiltonian
        self.qubits = qubits
        self.method = method
        self.repetitions = repetitions
        self.h5fname = fname + '.h5'
        self.suffix = suffix

        # Derived attributes.
        self.n_orbitals = len(self.hamiltonian.active_indices)
        self.orbital_labels = list(product(range(self.n_orbitals), ['u', 'd']))

        self.circuits = dict()
        self.circuit_string_converter = CircuitStringConverter(self.qubits)
        with h5py.File(self.h5fname, 'r') as h5file:
            qtrl_strings = json.loads(h5file['gs/ansatz'][()])
            ansatz = self.circuit_string_converter.convert_strings_to_circuit(qtrl_strings)
        self.circuit_constructor = CircuitConstructor(ansatz, self.qubits)

        self.charge_operators = ChargeOperators(self.qubits)
        self.charge_operators.transform(get_method_indices_pairs('d'))

    def _run_diagonal_circuits(self) -> None:
        """Runs diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")
        for i in range(2 * self.n_orbitals):
            m, s = self.orbital_labels[i]
            circuit_label = f'circ{m}{s}'

            # Build the diagonal circuit and save to HDF5 file.
            circuit = self.circuit_constructor.build_diagonal_circuit(
                self.charge_operators[2 * i], self.charge_operators[2 * i + 1])

            # Transpile the circuit and save to HDF5 file.
            circuit = transpile_into_berkeley_gates(circuit)
            self.circuits[circuit_label] = circuit
            qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(circuit)
            dset_transpiled = h5file.create_dataset(f'{circuit_label}/transpiled', data=json.dumps(qtrl_strings))

            # Execute the circuit and store the statevector in the HDF5 file.
            state_vector = cirq.sim.final_state_vector(circuit)
            state_vector[abs(state_vector) < 1e-8] = 0.0
            dset_transpiled.attrs[f'psi{self.suffix}'] = state_vector

            if self.method == "tomo":
                tomography_circuits = self.circuit_constructor.build_tomography_circuits(
                    circuit, self.qubits[1:3], self.qubits[:3]) # XXX: 1:3 is hardcoded
                
                for tomography_label, tomography_circuit in tomography_circuits.items():                    
                    self.circuits[f'{circuit_label}{tomography_label}'] = tomography_circuit
                    qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(tomography_circuit)
                    dset_tomography = h5file.create_dataset(
                        f'{circuit_label}/{tomography_label}', data=json.dumps(qtrl_strings))
                    
                    result = cirq.Simulator().run(tomography_circuit, repetitions=self.repetitions)
                    histogram = result.multi_measurement_histogram(keys=[str(q) for q in self.qubits[:3]])
                    dset_tomography.attrs[f'counts{self.suffix}'] = histogram_to_array(histogram, n_qubits=3)

        h5file.close()

    def _run_off_diagonal_circuits(self) -> None:
        """Runs off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")
        for i in range(2 * self.n_orbitals):
            m, s = self.orbital_labels[i]
            for j in range(i + 1, 2 * self.n_orbitals):
                m_, s_ = self.orbital_labels[j]
                circuit_label = f'circ{m}{s}{m_}{s_}'

                # Build the off-diagonal circuit and save to HDF5 file.
                circuit = self.circuit_constructor.build_off_diagonal_circuit(
                    self.charge_operators[2 * i], self.charge_operators[2 * i + 1],
                    self.charge_operators[2 * j], self.charge_operators[2 * j + 1])

                # Transpile the circuit and save to HDF5 file.
                circuit = transpile_into_berkeley_gates(circuit)
                self.circuits[circuit_label] = circuit
                qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(circuit)
                dset_transpiled = h5file.create_dataset(f'{circuit_label}/transpiled', data=json.dumps(qtrl_strings))

                state_vector = cirq.sim.final_state_vector(circuit)
                state_vector[abs(state_vector) < 1e-8] = 0.0
                dset_transpiled.attrs[f'psi{self.suffix}'] = state_vector

                if self.method == "tomo":
                    tomography_circuits = self.circuit_constructor.build_tomography_circuits(
                        circuit, self.qubits[2:4], self.qubits[:4]) # XXX: 2:4 is hardcoded
                    for tomography_label, tomography_circuit in tomography_circuits.items():
                        self.circuits[f'{circuit_label}{tomography_label}'] = tomography_circuit
                        qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(tomography_circuit)
                        dset_tomography = h5file.create_dataset(
                            f'{circuit_label}/{tomography_label}', data=json.dumps(qtrl_strings))

                        result = cirq.Simulator().run(tomography_circuit, repetitions=self.repetitions)
                        histogram = result.multi_measurement_histogram(keys=[str(q) for q in self.qubits[:4]])
                        dset_tomography.attrs[f'counts{self.suffix}'] = histogram_to_array(histogram, n_qubits=4)

        h5file.close()

    def run(self) -> None:
        """Runs all the functions to compute transition amplitudes."""
        print(f"Start N-electron amplitudes solver.")
        self._run_diagonal_circuits()
        self._run_off_diagonal_circuits()
        print(f"N-electron amplitudes solver finished.")
