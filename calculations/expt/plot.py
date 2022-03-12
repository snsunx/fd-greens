import sys
sys.path.append('../../src')
from helpers import plot_A, plot_TrS, plot_chi

import numpy as np
import matplotlib.pyplot as plt

def main_1p6A():
    h5fnames = ['lih_1p6A', 'lih_1p6A_run2']
    suffixes = ['_d', '_d_exp_proc']
    linestyles = [{}, {'ls': '--', 'marker': 'x', 'markevery': 30}]
    
    plot_A(h5fnames, suffixes, linestyles=linestyles, figname='A1p6')
    plot_TrS(h5fnames, suffixes, linestyles=linestyles, figname='TrS1p6')

def main_3A():
    h5fnames = ['lih_3A', 'lih_3A_run2']
    suffixes = ['_d', '_d_exp_proc']
    linestyles = [{}, {'ls': '--', 'marker': 'x', 'markevery': 30}]

    plot_A(h5fnames, suffixes, linestyles=linestyles, figname='A3')
    plot_TrS(h5fnames, suffixes, linestyles=linestyles, figname='TrS3')

def main_chi():
    h5fnames = ['lih', 'lih_run2']
    suffixes = ['_exact', '_noisy_exp_proc']
    linestyles = [{}, {'ls': '--', 'marker': 'x', 'markevery': 30}]

    plot_chi(h5fnames, suffixes, circ_label='00', linestyles=linestyles)
    plot_chi(h5fnames, suffixes, circ_label='01', linestyles=linestyles)

if __name__ == '__main__':
    main_1p6A()
    main_3A()
    main_chi()
