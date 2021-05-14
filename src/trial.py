from openfermion.chem import MolecularData
from openfermion.ops.representations import InteractionOperator
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner


molecule = MolecularData('H 0 0 0; H 0 0 0.74', 'sto3g', 1, 0)
run_pyscf(molecule)
int_op = molecule.get_molecular_hamiltonian()
ferm_op = get_fermion_operator(int_op)
qubit_op = jordan_wigner(ferm_op)
for term in qubit_op.terms:
    print(term)
"""
constant = interaction_operator.constant
one_body_tensor = interaction_operator.one_body_tensor
two_body_tensor = interaction_operator.two_body_tensor
print(two_body_tensor.shape)


interaction_operator1 = InteractionOperator(constant, one_body_tensor, two_body_tensor)
print(interaction_operator1)


print(interaction_operator == interaction_operator1)
"""
