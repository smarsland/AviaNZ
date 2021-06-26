# Shapes.py
# Shape container class and detection functions for the AviaNZ program

# Version 3.3 25/06/21
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2021

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

import math
import numpy as np
from ext import ce_denoise

class Shape():
    """ Container for storing a shape.
        Arguments:
        1. Start time, s. Represents the time of the first point
          in the shape.
        2. End time, s. To be used mainly to match shapes to
          segments and may not represent the last point's
          position.
        3. Time step size, s
        4. Y (shape) sequence, in pixel coordinates
        5. Y step size (Hz)
        6. Y offset (Hz)
    """
    def __init__(self, tstart, tend, tunit, y, yunit, ystart=0):
        self.tstart = tstart
        self.tend = tend
        self.tunit = tunit
        self.y = np.asarray(y)
        self.yunit = yunit
        self.ystart = ystart

    def __repr__(self):
        return "A shape over %f-%f s, step size %f s: "%(self.tstart, self.tend, self.tunit) + str(self.y)

    def toJSON(self):
        """ Returns a dictionary representation of the shape
            that can be used to export it as JSON (or otherwise pickle).
        """
        return {"t": (self.tstart, self.tend), "tunit": self.tunit, "yunit": self.yunit, "ystart": self.ystart, "y": self.y}


def stupidShaper(segment, specxunit, specyunit):
    """ Placeholder shape detector.
        Takes a segment and outputs a constant line in its middle.
        Specxunit: number of s in one spectrogram column
        Specyunit: number of Hz in one spectrogram row
    """
    midfreq = (segment[2]+segment[3])/2  # in Hz
    midy = midfreq / specyunit
    # repeat this value for each spec column in this segment
    midyrepeated = [midy]*math.floor((segment[1]-segment[0])/specxunit)

    newshape = Shape(segment[0], segment[1], specxunit, midyrepeated, specyunit)
    # print("Detected shape: ", newshape)
    return newshape

def fundFreqShaper(data, Wsamples, thr, fs):
    """ Compute squared diffs between signal and shifted signal
        (Yin fund freq estimator)
        over all tau<W, for each start position.
        Starts are shifted by half a window size, i.e. Wsamples//2.

        Will return pitch as self.fs/besttau for each window,
        and then convert to a shape.
    """
    if len(data) <= 2*Wsamples:
        print("Warning: data too short for F0: must be %d, is %d" % (2*Wsamples, len(data)))
        pitch = np.array([])
    else:
        # prep a copy of the input audio. not sure if needed
        data2 = np.zeros(len(data))
        data2[:] = data[:]
        # returns a series of freqs (in Hz) with -1 indicating "too weak"/error
        # (datatype: a float for each column in range(0, len(data)-2W, W//2))
        pitch = ce_denoise.FundFreqYin(data2, Wsamples, thr, fs)

    shape = Shape(y=pitch, tstart=0, tend=len(data)/fs, tunit=(Wsamples//2)/fs, yunit=1, ystart=0)
    return shape