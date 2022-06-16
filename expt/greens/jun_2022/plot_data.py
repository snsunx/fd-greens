import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import plot_spectral_function, plot_trace_self_energy
from fd_greens.cirq_ver.parameters import linestyles_A, linestyles_TrSigma

def main_spectral_function():
	plot_spectral_function(h5fnames, suffixes, linestyles=linestyles_A, labels=labels, text="legend")

def main_trace_self_energy():
	plot_trace_self_energy(h5fnames, suffixes, linestyles=linestyles_TrSigma, labels=labels, text="legend")

if __name__ == "__main__":
	h5fnames = ['lih_3A_sim', 'lih_3A_expt0']
	suffixes = ['', '']
	labels = ['Sim', 'Expt']
	
	main_spectral_function()
	main_trace_self_energy()
