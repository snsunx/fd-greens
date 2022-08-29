import sys
sys.path.append('../../..')
import argparse

import cirq
import h5py
import json

from fd_greens import CircuitStringConverter, print_circuit_statistics

qubits = cirq.LineQubit.range(4)
converter = CircuitStringConverter(qubits)


def print_parameters(h5fname: str) -> None:
    with h5py.File(h5fname + '.h5', 'r') as h5file:
        print("Circuit Parameters:")
        for key, val in h5file['params/circ'].attrs.items():
            print(f'{key} = {val}')
        print('')

        print("Mitigation Parameters:")
        for key, val in h5file['params/miti'].attrs.items():
            print(f'{key} = {val}')

def print_circuit_information(h5fname: str) -> None:
    with h5py.File(h5fname + '.h5', 'r') as h5file:
        for key in ['circ0u', 'circ1u', 'circ0d', 'circ1d', 'circ01u', 'circ01d']:
            if key in h5file:
                print('=' * 15 + ' ' + key + ' ' + '=' * 15) 
                qtrl_strings = json.loads(h5file[f'{key}/transpiled'][()])
                circuit = converter.convert_strings_to_circuit(qtrl_strings)
                print_circuit_statistics(circuit, True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", dest="parameters_h5fname")
    parser.add_argument("--circuit", dest="circuit_h5fname")
    args = parser.parse_args()

    if args.parameters_h5fname is not None:
        print_parameters(args.parameters_h5fname)

    if args.circuit_h5fname is not None:
        print_circuit_information(args.circuit_h5fname)
    
if __name__ == '__main__':
    main()
