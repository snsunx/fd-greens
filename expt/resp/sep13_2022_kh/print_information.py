import sys
sys.path.append('../../..')

import cirq
import h5py
import json

from fd_greens import CircuitStringConverter, print_circuit_statistics

qubits = cirq.LineQubit.range(4)
converter = CircuitStringConverter(qubits)

filename = "kh_resp_exact.h5"

def print_parameters():
    with h5py.File(filename, 'r') as h5file:
        print("Circuit Parameters:")
        for key, val in h5file['params/circ'].attrs.items():
            print(f'{key} = {val}')
        print('')

        print("Mitigation Parameters:")
        for key, val in h5file['params/miti'].attrs.items():
            print(f'{key} = {val}')

def print_circuit_information():
    with h5py.File(filename, 'r') as h5file:
        for key in h5file.keys():
            if key[:4] == 'circ':
                print('=' * 25 + ' ' + key + ' ' + '=' * 25) 
                qtrl_strings = json.loads(h5file[f'{key}/transpiled'][()])
                circuit = converter.convert_strings_to_circuit(qtrl_strings)
                print_circuit_statistics(circuit)

def main():
    # print_parameters()
    print_circuit_information()
    
if __name__ == '__main__':
    main()
