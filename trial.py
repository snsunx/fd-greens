from qiskit.algorithms.optimizers import Optimizer, L_BFGS_B


print(isinstance(L_BFGS_B, Optimizer))
print(isinstance(L_BFGS_B(), Optimizer))
