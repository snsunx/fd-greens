import sys
sys.path.append('../src')

import unittest
from qubit_indices import QubitIndices

class TestQubitIndices(unittest.TestCase):
    """Tests methods of QubitIndices."""

    def test_include_ancilla(self):
        inds = QubitIndices(['10', '00'], n_qubits=2)
        inds_new = inds.include_ancilla('1')

        reference = QubitIndices(['101', '001'])
        self.assertEqual(inds_new.str_form, reference.str_form)

if __name__ == '__main__':
    unittest.main()
