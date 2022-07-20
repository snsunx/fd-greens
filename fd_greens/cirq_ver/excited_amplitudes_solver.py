"""
====================================================================================
N-Electron Transition Amplitudes Solver (:mod:`fd_greens.excited_amplitudes_solver`)
====================================================================================
"""

from itertools import product
from typing import Sequence, Optional  

import h5py
import json
import cirq

from .molecular_hamiltonian import MolecularHamiltonian
from .operators import ChargeOperators
from .circuit_constructor import CircuitConstructor
from .circuit_string_converter import CircuitStringConverter
from .transpilation import transpile_into_berkeley_gates
from .parameters import CircuitConstructionParameters, MethodIndicesPairs
from .noise_parameters import NoiseParameters
from .general_utils import histogram_to_array
from .helpers import save_to_hdf5


class ExcitedAmplitudesSolver:
    """Solver for transition amplitudes between ground state and N-electron excited states."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        qubits: Sequence[cirq.Qid],
        method: str = "exact",
        noise_fname: Optional[str] = None,
        repetitions: int = 10000,
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
        assert method in ["exact", "tomo", "alltomo"]

        # Input attributes.
        self.hamiltonian = hamiltonian
        self.qubits = qubits
        self.method = method
        self.noise_fname = noise_fname
        self.repetitions = repetitions
        self.h5fname = fname + '.h5'
        self.suffix = suffix

        self.circuit_params = CircuitConstructionParameters()
        self.circuit_params.write(fname)

        if noise_fname is not None:
            self.simulator = cirq.DensityMatrixSimulator()
            self.noise_params = NoiseParameters.from_file(noise_fname)
        else:
            self.simulator = cirq.Simulator()
            self.noise_params = None

        # Derived attributes.
        self.n_spatial_orbitals = len(self.hamiltonian.active_indices)
        self.n_spin_orbitals = 2 * self.n_spatial_orbitals
        self.orbital_labels = list(product(range(self.n_spatial_orbitals), ['u', 'd']))

        self.circuits = dict()
        self.circuit_string_converter = CircuitStringConverter(self.qubits)
        with h5py.File(self.h5fname, 'r') as h5file:
            qtrl_strings = json.loads(h5file['gs/ansatz'][()])
            ansatz = self.circuit_string_converter.convert_strings_to_circuit(qtrl_strings)
        self.circuit_constructor = CircuitConstructor(ansatz, self.qubits)

        method_indices_pairs = MethodIndicesPairs.get_pairs('')
        self.charge_operators = ChargeOperators(self.qubits)
        self.charge_operators.transform(method_indices_pairs)

        self.n_system_qubits = self.n_spin_orbitals - method_indices_pairs.n_tapered

    def _run_diagonal_circuits(self) -> None:
        """Runs diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")
        for i in range(self.n_spin_orbitals):
            m, s = self.orbital_labels[i]
            circuit_label = f'circ{m}{s}'

            # Build the diagonal circuit and save to HDF5 file.
            circuit = self.circuit_constructor.build_diagonal_circuit(
                self.charge_operators[2 * i], self.charge_operators[2 * i + 1])

            # Transpile the circuit and save to HDF5 file.
            circuit = transpile_into_berkeley_gates(circuit, circuit_params=self.circuit_params)
            circuit = self.circuit_string_converter.adapt_to_hardware(circuit)
            self.circuits[circuit_label] = circuit
            qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(circuit)
            dset_transpiled = save_to_hdf5(
                h5file, f"{circuit_label}/transpiled",
                data=json.dumps(qtrl_strings), return_dataset=True)

            # Execute the circuit and store the statevector in the HDF5 file.
            state_vector = cirq.sim.final_state_vector(circuit)
            dset_transpiled.attrs[f'psi{self.suffix}'] = state_vector

            if self.method in ["tomo", "alltomo"]:                
                if self.method == "tomo":
                    tomographed_qubits = self.qubits[1:self.n_system_qubits + 1]
                else:
                    tomographed_qubits = self.qubits[:self.n_system_qubits + 1]
                measured_qubits = self.qubits[:self.n_system_qubits + 1]
                tomography_circuits = self.circuit_constructor.build_tomography_circuits(
                    circuit, tomographed_qubits=tomographed_qubits, measured_qubits=measured_qubits)
                
                for tomography_label, tomography_circuit in tomography_circuits.items():                    
                    self.circuits[f'{circuit_label}{tomography_label}'] = tomography_circuit
                    qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(tomography_circuit)
                    dset_tomography = save_to_hdf5(
                        h5file, f"{circuit_label}/{tomography_label}", 
                        json.dumps(qtrl_strings), return_dataset=True)
                    
                    if self.noise_params is not None:
                        tomography_circuit = self.noise_params.add_noise_to_circuit(tomography_circuit)
                    result = self.simulator.run(tomography_circuit, repetitions=self.repetitions)
                    histogram = result.multi_measurement_histogram(
                        keys=[str(q) for q in self.qubits[:self.n_system_qubits + 1]])
                    dset_tomography.attrs[f'counts{self.suffix}'] = histogram_to_array(
                        histogram, n_qubits=self.n_system_qubits + 1)

        h5file.close()

    def _run_off_diagonal_circuits(self) -> None:
        """Runs off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")
        for i in range(self.n_spin_orbitals):
            m, s = self.orbital_labels[i]
            for j in range(i + 1, self.n_spin_orbitals):
                m_, s_ = self.orbital_labels[j]
                circuit_label = f'circ{m}{s}{m_}{s_}'

                # Build the off-diagonal circuit and save to HDF5 file.
                circuit = self.circuit_constructor.build_off_diagonal_circuit(
                    self.charge_operators[2 * i], self.charge_operators[2 * i + 1],
                    self.charge_operators[2 * j], self.charge_operators[2 * j + 1])

                # Transpile the circuit and save to HDF5 file.
                circuit = transpile_into_berkeley_gates(circuit, circuit_params=self.circuit_params)
                circuit = self.circuit_string_converter.adapt_to_hardware(circuit)
                self.circuits[circuit_label] = circuit
                qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(circuit)
                dset_transpiled = save_to_hdf5(
                    h5file, f"{circuit_label}/transpiled",
                    data=json.dumps(qtrl_strings), return_dataset=True)

                state_vector = cirq.sim.final_state_vector(circuit)
                state_vector[abs(state_vector) < 1e-8] = 0.0
                dset_transpiled.attrs[f'psi{self.suffix}'] = state_vector

                if self.method in ["tomo", "alltomo"]:
                    if self.method == "tomo":
                        tomographed_qubits = self.qubits[2:self.n_system_qubits + 2]
                    else:
                        tomographed_qubits = self.qubits[:self.n_system_qubits + 2]
                    measured_qubits = self.qubits[:self.n_system_qubits + 2]
                    tomography_circuits = self.circuit_constructor.build_tomography_circuits(
                        circuit, tomographed_qubits=tomographed_qubits, measured_qubits=measured_qubits)

                    for tomography_label, tomography_circuit in tomography_circuits.items():
                        self.circuits[f'{circuit_label}{tomography_label}'] = tomography_circuit
                        qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(tomography_circuit)
                        dset_tomography = save_to_hdf5(
                            h5file, f"{circuit_label}/{tomography_label}",
                            data=json.dumps(qtrl_strings), return_dataset=True)

                        if self.noise_params is not None:
                            tomography_circuit = self.noise_params.add_noise_to_circuit(tomography_circuit)
                        result = self.simulator.run(tomography_circuit, repetitions=self.repetitions)
                        histogram = result.multi_measurement_histogram(
                            keys=[str(q) for q in self.qubits[:self.n_system_qubits + 2]])
                        dset_tomography.attrs[f'counts{self.suffix}'] = histogram_to_array(
                            histogram, n_qubits=self.n_system_qubits + 2)

        h5file.close()

    def run(self) -> None:
        """Runs all the functions to compute transition amplitudes."""
        print(f"Start N-electron amplitudes solver.")
        self._run_diagonal_circuits()
        self._run_off_diagonal_circuits()
        print(f"N-electron amplitudes solver finished.")
