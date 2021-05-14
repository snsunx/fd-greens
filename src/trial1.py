import numpy as np
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import jordan_wigner

constant = 0
two_body_tensor = np.zeros((4, 4, 4, 4))

one_body_tensor = np.diag([0, 1, 0, 0])
interaction_operator = InteractionOperator(constant, one_body_tensor, two_body_tensor)
fermion_operator = jordan_wigner(interaction_operator)
print(interaction_operator)
print(fermion_operator)
