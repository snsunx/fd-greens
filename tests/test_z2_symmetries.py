import sys
sys.path.append("..")

import unittest

from fd_greens.cirq_ver.main import QubitIndices, QubitIndicesTransformer


class TestQubitIndicesTransformer(unittest.TestCase):
    """Tests methods of QubitIndices."""

    def test_cnot(self):
        qubit_indices = QubitIndices([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
        transformer = QubitIndicesTransformer(qubit_indices)
        transformer.transform(cnot=[0, 2])

        reference = QubitIndices([[0, 1, 1], [1, 0, 1], [1, 0, 0]])
        self.assertEqual(qubit_indices, reference)

    def test_swap(self):
        qubit_indices = QubitIndices([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
        transformer = QubitIndicesTransformer(qubit_indices)
        transformer.transform(swap=[0, 1])

        reference = QubitIndices([[1, 0, 1], [0, 1, 0], [0, 1, 1]])
        self.assertEqual(qubit_indices, reference)

    def test_taper(self):
        qubit_indices = QubitIndices([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
        transformer = QubitIndicesTransformer(qubit_indices)
        transformer.transform(taper=[0, 2])

        reference = QubitIndices([[1], [0], [0]])
        self.assertEqual(qubit_indices, reference)

if __name__ == "__main__":
    unittest.main()
