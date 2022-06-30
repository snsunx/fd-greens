import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import display_fidelities

def main():
    display_fidelities('lih_resp_exact', 'lih_resp_expt1', 'n')

if __name__ == '__main__':
    main()
