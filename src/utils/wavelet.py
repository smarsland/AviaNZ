
# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Essentially just a named list.

import numpy as np
import os


class Wavelet:
    def __init__(self, name):
        # Get the absolute path to the wavelets directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        wavelets_dir = os.path.join(current_dir, '..', 'data', 'wavelets')
        filename = os.path.join(wavelets_dir, name + '.txt')
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
