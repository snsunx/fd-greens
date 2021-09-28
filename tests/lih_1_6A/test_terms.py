"""Test the terms feature in circuits.py."""
import numpy as np
from openfermion.ops import PolynomialTensor
from openfermion.transforms import get_fermion_operator, jordan_wigner

n_qubits = 5
ind = 2
arr = np.zeros((n_qubits,))
arr[ind] = 1.
poly_tensor = PolynomialTensor({(0,): arr})
print(poly_tensor)
ferm_op = get_fermion_operator(poly_tensor)
print(ferm_op)
qubit_op = jordan_wigner(ferm_op)
terms = list(qubit_op.terms)

print(terms)
