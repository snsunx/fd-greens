import sys

sys.path.append("../src/")
import warnings

warnings.filterwarnings("ignore")
import unittest
import numpy as np

from ground_state_solvers import GroundStateSolver
from number_states_solvers import EHStatesSolver
from amplitudes_solvers_new import EHAmplitudesSolver
from utils import get_lih_hamiltonian, get_quantum_instance, initialize_hdf5


class TestLiH(unittest.TestCase):
    def test_lih(self):
        h = get_lih_hamiltonian(3.0)
        initialize_hdf5()

        gs_solver = GroundStateSolver(h)
        gs_solver.run()

        eh_solver = EHStatesSolver(h)
        eh_solver.run()

        amp_solver = EHAmplitudesSolver(h)
        amp_solver.run()

        B_e = amp_solver.B["e"]
        B_h = amp_solver.B["h"]
        np.testing.assert_array_almost_equal(
            B_e[0, 0], [0.017964 + 0.0j, 0.065905 + 0.0j]
        )
        np.testing.assert_array_almost_equal(
            B_e[1, 1], [0.909705 + 0.0j, 0.006425 + 0.0j]
        )
        np.testing.assert_array_almost_equal(
            B_e[0, 1], [-0.127837 + 0.0j, 0.020578 + 0.0j]
        )

        np.testing.assert_array_almost_equal(
            B_h[0, 0], [0.900782 + 0.0j, 0.015348 + 0.0j]
        )
        np.testing.assert_array_almost_equal(
            B_h[1, 1], [0.00586 + 0.0j, 0.078009 + 0.0j]
        )
        np.testing.assert_array_almost_equal(
            B_h[0, 1], [0.072657 + 0.0j, 0.034602 + 0.0j]
        )
        # print(B_e)
        # print(B_h)


if __name__ == "__main__":
    unittest.main()
