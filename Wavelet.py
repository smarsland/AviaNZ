# Just a copy of the pywt.Wavelet class for
# simple processing of homebrew WPs.

# Essentially just a named list.

import numpy as np


class Wavelet:
    def __init__(self, name):
        filename = 'Wavelets/' + name + '.txt'
        filter_bank = np.loadtxt(filename)

        if(len(filter_bank)) != 4:
            msg = "ERROR: wavelet expects four filter coefficients"
            raise ValueError(msg)
        else:
            self.dec_lo = np.asarray(filter_bank[0], dtype=np.float64)
            self.dec_hi = np.asarray(filter_bank[1], dtype=np.float64)
            self.rec_lo = np.asarray(filter_bank[2], dtype=np.float64)
            self.rec_hi = np.asarray(filter_bank[3], dtype=np.float64)

        if self.dec_lo.ndim!=1 or self.dec_hi.ndim!=1 or self.rec_lo.ndim!=1 or self.rec_hi.ndim!=1:
            msg = "ERROR: all filters must be 1D"
            raise ValueError(msg)
