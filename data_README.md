# Quantum Computation of Frequency-Domain Molecular Response Properties Using a Three-Qubit iToffoli Gate
---

This dataset is for the Science Advances manuscript No. adg8040 titled "Quantum Computation of Frequency-Domain Molecular Response Properties Using a Three-Qubit iToffoli Gate." The data are collected from a quantum simulation experiment on a superconducting quantum device at University of California, Berkeley. The goal of the experiment is to compute spectral functions and density-density response functions of diatomic molecules through the transition amplitudes determined on the quantum device. This dataset includes both intermediate results, such as circuit fidelities, and final results, such as spectral functions and density-density response functions.

## Description of the data and file structure

Each folder in the dataset corresponds to a figure in the main text or supplementary. Below is a description of how the files in each folder are organized and what data each file contains.

- `fig3_spectral_function` contains the data for Fig. 3 Spectral function of diatomic molecules. Files with the `nah` prefix are for the NaH molecule; those with the `kh` prefix are for the KH molecule. Files with names containing `exact` are from classical exact simulation; those with names containing `tomo_expt` are from experiments after tomography. In each file, the first column is the frequency and the second column is the spectral function.
- `fig4_fidelity_vs_depth` contains the data for Fig. 4 Fidelity versus circuit depth of the $(0\uparrow, 0\downarrow)$-circuit for NaH. The two files with the `n0814` suffix are from simulation, and the two files without the suffix are from experiments. `circ0u1u` denotes the circuits decomposed with the iToffoli gates, and `circ0u1u2q` denotes the circuits decomposed with the CZ gates. Each file has three columns: the first column corresponds to the circuit depth; the second column corresponds to the type of gates in the cycle (1 for single-qubit gate, 2 for two-qubit gate, 3 for three-qubit iToffoli gate); the third column is the fidelity of the circuit at that depth.
- `fig5_fidelity_and_trace` contains the data for Fig. 5 System-qubit state fidelities in the response function calculation of NaH. Each file contains data in a 4 by 4 matrix form, which is consistent with the matrix-form data layout in the figure. The files with names containing `rc` are obtained from circuits constructed from randomized compiling (RC), while the files without `rc` are not. The files with the suffix `raw` are not processed with purification, while the files with the suffix `pur` are processed with purification.
- `fig6_response_function` contains the data for Fig. 6 Density-density response function of NaH. Among the 10 data  files in this folder, the two files with `exact` in the file names are from classical exact simulation. The files with the `chi00` suffix are for $\chi_{00}$, while the files with the `chi01` suffix are for $\chi_{01}$. The files that contain `rc` in the file names are obtained from circuits with randomized compiling (RC), while files that do not contain `rc` in the file names are not. Each file has three columns: the first column corresponds to the frequency; the second column corresponds to the real part of the response function; the third column corresponds to the imaginary part of the response function.
- `figs1_spectator_error` contains the data for Fig. S1 Cancellation of spectator error during the iToffoli gate. The file `CP67_no_cancel.csv` contains data for panel (B), where the spectator error is not canceled; the file `CP67_yes_cancel_2.csv`contains data for panel (D), where the spectator error is canceled with the additional $CZ_{\phi}$ drive. The "$Q_3$ population in $|0\rangle$" quantity plotted in the figure are from the last column in both files. The "sweep angle" quantity is determined from the column with the title "phase." Among the 42 data points in each column, the first 21 points correspond to the data obtained by preparing $Q_2$ in $|1\rangle$, and the last 21 points correspond to the data obtained by preparing $Q_2$ in $|0\rangle$. Note that this is consistent with the column titled "initial state," where the data obtained by preparing $Q_2$ in $|1\rangle$ are labeled "101+," and the data obtained by preparing $Q_2$ in $|0\rangle$ are labeled "100+."
- `figs2_fidelity_and_trace_kh` contains the data for Fig. S2 System-qubit state fidelities for the response function calculation of KH. This folder follows the same structure as `fig5_fidelity_and_trace`, with the only difference being the substitution of `nah` with `kh` in the file names.
- `figs3_response_function_kh` contains the data for Fig. S3 Density-density response function of KH. This folder follows the same structure as `fig6_response_function`, with the only difference being the substitution of `nah` with `kh` in the file names.


## Sharing/Access information

The data can also be accessed from the public Github repository at https://github.com/snsunx/fd-greens. The figures can be generated by running the Python scripts in the following five subdirectories contained in the `plots` directory:

- `fig3_spectral_function` for Fig. 3 Spectral function of diatomic molecules
- `fig4_fidelity_vs_depth` for Fig. 4 Fidelity versus circuit depth of the $(0\uparrow, 0\downarrow)$-circuit for NaH
- `fig5_fidelity_and_trace` for Fig. 5 System-qubit state fidelities in the response function calculation of NaH and Fig. S2 System-qubit state fidelities for the response function calculation of KH
- `fig6_response_function` for Fig. 6 Density-density response function of NaH and Fig. S3 Density-density response function of KH
- `figs1_spectator_error` for Fig. S1 Cancellation of spectator error during the iToffoli gate

Data collected in this dataset is original and not derived from any source.