from math import log2
import numpy as np

def frequency_select_physical(freq):
    power = log2(len(freq))
    selA = np.logical_or(freq == 0, np.isin(freq, 1/(2**np.arange(1,power+1))))
    return selA


def frequency_selelect_positive(freq):
    selA = freq >= 0
    return selA