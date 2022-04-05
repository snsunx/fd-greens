# Quantum Computation of Frequency-Domain Green's Functions

This project implements a quantum algorithm based on [linear combination of unitaries](https://arxiv.org/abs/1202.5822) to 
compute frequency-domain [Green's functions](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.012330)
and [response functions](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033043). The experiments are
performed on a quantum processor at UC Berkeley using a recently developed protocol for the 
[iToffoli gate](https://arxiv.org/abs/2108.10288).

## Description

This source code in the directory `fd_greens` contains two subpackages:

- `main`: Main modules, including ground state, excited state and transition amplitude solvers.
- `utils`: Utility modules, including functions for circuit manipulation, transpilation and data post-processing.

There are two other directories on the same level as the source code directory `fd_greens`:

- `sim`: Simulator runs of Green's functions and response functions.
- `expt`: Experimental runs of Green's functions and response functions.
