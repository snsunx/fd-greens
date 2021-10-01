import sys
sys.path.append('../src')

import unittest
from qubit_indices import QubitIndices

class TestQubitIndices(unittest.TestCase):
    """Tests methods of QubitIndices."""

    def test_list_form_to_str_form(self):
        inds = QubitIndices([[0, 1, 1], [1, 0, 0]])
        inds_str_form = inds.str_form

        reference = ['110', '001']
        self.assertEqual(inds_str_form, reference)

    def test_int_form_to_str_form(self):
        inds = QubitIndices([6, 1], n_qubits=3)
        inds_str_form = inds.str_form

        reference = ['110', '001']
        self.assertEqual(inds_str_form, reference)

    def test_include_ancilla(self):
        inds = QubitIndices(['10', '00'], n_qubits=2)
        inds_new = inds.include_ancilla('1')

        reference = QubitIndices(['101', '001'])
        self.assertEqual(inds_new.str_form, reference.str_form)

if __name__ == '__main__':
    unittest.main()