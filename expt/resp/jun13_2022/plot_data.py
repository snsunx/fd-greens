import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import plot_response_function
from fd_greens.cirq_ver.parameters import linestyles_chi

def main_response_function():
	plot_response_function(h5fnames, suffixes, linestyles=linestyles_chi, labels=labels, text="legend", dirname=f"figs{h5fnames[-1][-1]}")

if __name__ == "__main__":
	h5fnames = ['lih_resp_sim', 'lih_3A_expt1']
	suffixes = ['', '']
	labels = ['Sim', 'Expt']
	
	main_response_function()
