import sys
sys.path.append('../src')
import unittest

from qiskit.quantum_info import SparsePauliOp, PauliTable
from z2_symmetries import apply_cnot_z2, taper, transform_4q_hamiltonian

class Z2SymmetryTest(unittest.TestCase):
	"""Tests functions operating on Z2 symmetries."""

	def test_apply_cnot_z2(self):
		"""Tests applying CNOT in Z2 representation."""
		table = PauliTable.from_labels(['XX', 'YY', 'ZZ'])
		op = SparsePauliOp(table)
		op_new = apply_cnot_z2(op, 0, 1)

		table = PauliTable.from_labels(['IX', 'ZX', 'ZI'])
		reference = SparsePauliOp(table, coeffs=[1., -1., 1.])
		
		self.assertEqual(op_new, reference)

	def test_taper(self):
		"""Tests tapering qubits off an operator."""
		table = PauliTable.from_labels(['XIZX', 'YZIY'])
		op = SparsePauliOp(table)
		op_new = taper(op, [0, 1], [1, 1])

		table = PauliTable.from_labels(['XI', 'YZ'])
		reference = SparsePauliOp(table, coeffs=[-1., -1j])

		self.assertEqual(op_new, reference)

	def test_transform_4q_hamiltonian(self):
		"""Tests transforming a 4-qubit Hamiltonian."""
		table = PauliTable.from_labels(['XXXX', 'ZZZZ'])
		op = SparsePauliOp(table)
		op_new = transform_4q_hamiltonian(op, [0, 1])

		table = PauliTable.from_labels(['XX', 'II'])
		reference = SparsePauliOp(table, coeffs=[1., -1.])

		self.assertEqual(op_new, reference)

if __name__ == '__main__':
	unittest.main()