
# Version 4.0 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

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

# Code to extract segments from sound files

from core import spectrogram
from core import config_loader
from core import audio_data
from utils import shapes

import numpy as np
import scipy.ndimage as spi
from scipy import signal
import librosa
import os
import copy
from scipy.interpolate import interp1d
from scipy.signal import medfilt
import skimage.measure as skm
import tensorflow as tf

DIAMOND_3X3 = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=int)

DIAMOND_5X5 = np.array([[0, 0, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 0, 0]], dtype=int)

FIR_COEFFICIENTS = [
    0.078573, 0.053921, 0.041608, 0.036006, 0.031521, 0.029435, 0.028122, 0.027287,
    0.026241, 0.025226, 0.024076, 0.022927, 0.021704, 0.020487, 0.019721, 0.019015,
    0.018564, 0.017953, 0.01753, 0.017077, 0.016544, 0.015762, 0.015056, 0.014457,
    0.013913, 0.013299, 0.012879, 0.012568, 0.012455, 0.012056, 0.011634, 0.011077,
    0.010707, 0.010217, 0.009884, 0.009596, 0.009361, 0.00902, 0.008691, 0.008384,
    0.008148, 0.007919, 0.007636, 0.007341, 0.007069, 0.006844, 0.006587, 0.006369,
    0.00617, 0.005974, 0.005756, 0.005535, 0.005363, 0.00518, 0.004974, 0.004743,
    0.004565, 0.004397, 0.004246, 0.004102, 0.00395, 0.003801, 0.003635, 0.003486,
    0.003327, 0.003207, 0.003057, 0.002921, 0.002776, 0.002656, 0.00253, 0.002419,
    0.002297, 0.002186, 0.00207, 0.001955, 0.001856, 0.001756, 0.001661, 0.001552,
    0.001448, 0.001349, 0.00126, 0.001179, 0.001091, 0.001005, 0.000915, 0.000821,
    0.000745, 0.000672, 0.000603, 0.000534, 0.000461, 0.000391, 0.000327, 0.000262,
    0.00019, 0.00012, 0.000061, 0.0
]

class Segmenter:
    """Segmentation algorithms for audio analysis.
    
    Implements six forms of segmentation:
    - Amplitude threshold (basic/deprecated)
    - Energy threshold
    - Harma
    - Median clipping of spectrogram
    - Fundamental frequency using yin
    - FIR
    
    And two forms of recognition:
    - Cross-correlation
    - DTW
    
    Important parameters:
        mingap: minimum gap between segments (otherwise merge them)
        minlength: minimum segment length (otherwise delete it)
        ignoreInsideEnvelope: whether to keep superset or individuals when merging
        maxlength: maximum segment size (currently unused)
        threshold: generally mean + threshold * std dev for filtering
    
    Segment Format Convention:
        - Methods without suffix: operate on 2-element segments [[start, end], ...]
        - Methods with '3' suffix: operate on 3-element segments [[[start, end], certainty], ...]
          Examples: deleteShort vs deleteShort3, checkSegmentOverlap vs checkSegmentOverlap3
    
    Each segmentation method returns start and stop times (in seconds) as a list of pairs.
    It is up to the caller to convert these to a true SegmentList.
    See also the species-specific segmentation in WaveletSegment.
    """

    def __init__(self, sp=None, fs=0, mingap=0.3, minlength=0.2):
        """Initialize segmenter.
        
        Args:
            sp: Spectrogram object with audio_data, sg, incr attributes
            fs: Sample rate (ignored if sp is provided, kept for backwards compatibility)
            mingap: Minimum gap between segments
            minlength: Minimum segment length
        """
        self.sp = sp
        self.mingap = mingap
        self.minlength = minlength

    def bestSegments(self,FIRthr=0.7, medianClipthr=3.0, yinthr=0.9):
        """ A reasonably good segmentaion - a merged version of FIR, median clipping, and fundamental frequency using yin"""
        segs1 = self.segmentByFIR(FIRthr)
        segs2 = self.medianClip(medianClipthr)
        segs3 = self.yinSegs(100, thr=yinthr)
        segs1 = self.mergeSegments(segs1, segs2)
        segs = self.mergeSegments(segs1, segs3)
        # mergeSegments also sorts, so not needed:
        # segs = sorted(segs, key=lambda x: x[0])
        return segs

    def mergeSegments(self, segs1, segs2=None):
        """ Given two segmentations of the same file, join them,
        and if one wasn't empty, merge any overlapping segments.
        format: [[1,3] [2,4] [5,7] [7,8]] -> [[1,4] [5,7] [7,8]]
        Can take in one or two lists. """
        if segs1 == [] and segs2 == []:
            return []
        elif segs1 == []:
            return segs2
        elif segs2 == []:
            return segs1

        if segs2 is not None:
            segs1.extend(segs2)
        out = self.checkSegmentOverlap(segs1)
        return out


    @staticmethod
    def convert01(presabs, window=1):
        """ Turns a list of presence/absence [0 1 1 1]
            into a list of start-end segments [[1,4]].
            Can use non-1 s units of pres/abs.
        """
        # squeeze any extra axes except axis 0 (don't make scalars)
        presabs = np.reshape(presabs, (np.shape(presabs)[0]))
        out = []
        t = 0
        while t < len(presabs):
            if presabs[t]==1:
                start = t
                while t<len(presabs) and presabs[t]!=0:
                    t += 1
                out.append([start*window, t*window])
            t += 1
        return out

    @staticmethod
    def deleteShort(segments, minlength=0.25):
        """Remove segments shorter than minlength.
        
        Operates on 2-element segments: [[start, end], [start, end], ...]
        Example: [[1,3], [4,5]] -> [[1,3]]
        """
        out = []
        if minlength == 0:
            minlength = 0.25
        for seg in segments:
            if seg[1]-seg[0] >= minlength:
                out.append(seg)
        return out

    @staticmethod
    def deleteShort3(segments, minlength=0.25):
        """Remove segments shorter than minlength (with certainty values).
        
        Operates on 3-element segments: [[[start, end], certainty], ...]
        Example: [[[1,3], 50], [[4,5], 90]] -> [[[1,3], 50]]
        
        Note: '3' suffix indicates this operates on segments WITH certainty values.
        """
        out = []
        if minlength == 0:
            minlength = 0.25
        for seg in segments:
            if seg[0][1]-seg[0][0] >= minlength:
                out.append(seg)
        return out

    @staticmethod
    def splitLong3(segments, maxlen=10):
        """Split long segments evenly (with certainty values).
        
        Operates on 3-element segments: [[[start, end], certainty], ...]
        Example: [[[1,5], 50]] -> [[[1,3], 50], [[3,5], 50]]
        
        Note: '3' suffix indicates this operates on segments WITH certainty values.
        """
        out = []
        for seg in segments:
            l = seg[0][1]-seg[0][0]
            if l > maxlen:
                n = int(np.ceil(l/maxlen))
                d = l/n
                for i in range(n):
                    end = min(l, d * (i+1))
                    out.append([[seg[0][0] + d*i, seg[0][0] + end], seg[1]])
            else:
                out.append(seg)
        return out

    @staticmethod
    def checkSegmentOverlap(segments):
        """Merge overlapping segments (2-element format).
        
        Does not merge if segments only touch.
        Operates on 2-element segments: [[start, end], [start, end], ...]
        Example: [[1,3], [2,4]] -> [[1,4]]
        """
        if isinstance(segments, np.ndarray):
            segments = segments.tolist()
        if len(segments) == 0:
            return []
        
        segments = sorted(segments, key=lambda x: x[0])
        out = []
        i = 0
        while i < len(segments):
            start = segments[i][0]
            end = segments[i][1]
            while i+1 < len(segments) and segments[i+1][0] < end:
                i += 1
                end = max(end, segments[i][1])
            out.append([start, end])
            i += 1
        return out

    @staticmethod
    def joinGaps(segments, maxgap=3):
        """Merge segments within maxgap units (2-element format).
        
        Merges segments if the gap between them is <= maxgap.
        Operates on 2-element segments: [[start, end], [start, end], ...]
        Example: [[1,3], [5,7]] with maxgap=3 -> [[1,7]]
        """
        if isinstance(segments, np.ndarray):
            segments = segments.tolist()
        if len(segments) == 0:
            return []
        
        segments = sorted(segments, key=lambda x: x[0])
        out = []
        i = 0
        while i < len(segments):
            start = segments[i][0]
            end = segments[i][1]
            while i+1 < len(segments) and segments[i+1][0] - end <= maxgap:
                i += 1
                end = max(end, segments[i][1])
            out.append([start, end])
            i += 1
        return out

    @staticmethod
    def checkSegmentOverlap3(segments):
        """Merge overlapping segments (with certainty values).
        
        Operates on 3-element segments: [[[start, end], certainty], ...]
        Certainties are averaged when segments overlap.
        
        Note: '3' suffix indicates this operates on segments WITH certainty values.
        """
        if isinstance(segments, np.ndarray):
            segments = segments.tolist()
        if len(segments) == 0:
            return []

        segments.sort(key=lambda seg: seg[0][0])
        out = []
        i = 0
        while i < len(segments):
            start = segments[i][0][0]
            end = segments[i][0][1]
            cert = [segments[i][1]]
            while i+1 < len(segments) and segments[i+1][0][0] < end:
                i += 1
                end = max(end, segments[i][0][1])
                cert.append(segments[i][1])
            out.append([[start, end], np.mean(cert)])
            i += 1
        return out

    @staticmethod
    def joinGaps3(segments, maxgap=3):
        """Merge segments within maxgap units (with certainty values).
        
        Operates on 3-element segments: [[[start, end], certainty], ...]
        
        Note: '3' suffix indicates this operates on segments WITH certainty values.
        """
        if isinstance(segments, np.ndarray):
            segments = segments.tolist()
        if len(segments) == 0:
            return []

        segments.sort(key=lambda seg: seg[0][0])
        out = []
        i = 0
        while i < len(segments):
            start = segments[i][0][0]
            end = segments[i][0][1]
            cert = [segments[i][1]]
            while i+1 < len(segments) and segments[i+1][0][0] - end <= maxgap:
                i += 1
                end = max(end, segments[i][0][1])
                cert.append(segments[i][1])
            out.append([[start, end], np.mean(cert)])
            i += 1
        return out

    def segmentByFIR(self, threshold):
        """ Segmentation using FIR envelope.
        """
        nsecs = len(self.sp.audio_data.data) / float(self.sp.audio_data.sample_rate)
        fftrate = int(np.shape(self.sp.sg)[0]) / nsecs
        upperlimit = 100
        f = interp1d(np.arange(0, len(FIR_COEFFICIENTS)), np.squeeze(FIR_COEFFICIENTS))
        samples = f(np.arange(1, upperlimit, float(upperlimit) / int(fftrate / 10.)))
        padded = np.concatenate((np.zeros(int(fftrate / 10.)), np.mean(self.sp.sg, axis=1), np.zeros(int(fftrate / 10.))))
        envelope = spi.filters.convolve(padded, samples, mode='constant')[:-int(fftrate / 10.)]
        ind = envelope > np.median(envelope) + threshold * np.std(envelope)
        segs = self.convert01(ind, self.sp.incr / self.sp.audio_data.sample_rate)
        return segs

    def segmentByAmplitude(self, threshold, usePercent=True):
        """ Bog standard amplitude segmentation.
        A straw man, do not use.
        """
        if usePercent:
            threshold = threshold*np.max(self.sp.audio_data.data)
        seg = np.abs(self.sp.audio_data.data)>threshold
        seg = self.convert01(seg, self.sp.audio_data.sample_rate)
        return seg

    def segmentByEnergy(self, thr, width, min_width=450):
        """ Segmentation based on energy curve using windowed sum of absolute amplitudes.
        Based on Jinnai et al. 2012.
        """
        data = np.abs(self.sp.audio_data.data)
        E = np.zeros(len(data))
        E[width] = np.sum(data[:2*width+1])
        for i in range(width+1, len(data)-width):
            E[i] = E[i-1] - data[i-width-1] + data[i+width]
        E = E / (2*width)

        Em = np.zeros(len(data))
        for i in range(width, len(data)-width):
            Em[i] = np.median(E[i-width:i+width])
        for i in range(width):
            Em[i] = np.median(E[0:2*i])
            Em[-i] = np.median(E[-2*i:])

        threshold = np.mean(Em) + thr * np.std(Em)

        starts = []
        ends = []
        insegment = False
        for i in range(len(data)-1):
            if not insegment:
                if Em[i] < threshold and Em[i+1] > threshold:
                    starts.append(i)
                    insegment = True
            if insegment:
                if Em[i] > threshold and Em[i+1] < threshold:
                    ends.append(i)
                    insegment = False
        if insegment:
            ends.append(len(data))
        
        maxpoints = []
        Emm = np.zeros(len(data))
        for i in range(len(starts)):
            if ends[i] - starts[i] > min_width:
                maxpoints.append(np.argmax(Em[starts[i]:ends[i]]))
                Emm[starts[i]:ends[i]] = Em[starts[i]:ends[i]]

        segs = []
        for i in range(len(starts)):
            segs.append([float(starts[i]) / self.sp.audio_data.sample_rate, float(ends[i]) / self.sp.audio_data.sample_rate])
        return segs

    def Harma(self, thr=10., stop_thr=0.8, minSegment=50):
        """ Harma's method, but with a different stopping criterion
        # Assumes that spectrogram is not normalised
        maxFreqs = 10. * np.log10(np.max(self.sp.sg, axis = 1))
        """
        maxFreqs = 10. * np.log10(np.max(self.sp.sg, axis=1))
        maxFreqs = medfilt(maxFreqs,21)
        biggest = np.max(maxFreqs)
        segs = []

        while np.max(maxFreqs)>stop_thr*biggest:
            t0 = np.argmax(maxFreqs)
            a_n = maxFreqs[t0]

            # Go backwards looking for where the syllable stops
            t = t0
            while maxFreqs[t] > a_n - thr and t>0:
                t -= 1
            t_start = t

            # And forwards
            t = t0
            while maxFreqs[t] > a_n - thr and t<len(maxFreqs)-1:
                t += 1
            t_end = t

            # Set the syllable just found to 0
            maxFreqs[t_start:t_end] = 0
            if float(t_end - t_start)*self.sp.incr/self.sp.audio_data.sample_rate*1000.0 > minSegment:
                segs.append([float(t_start)* self.sp.incr / self.sp.audio_data.sample_rate,float(t_end)* self.sp.incr / self.sp.audio_data.sample_rate])

        return segs

    def segmentByPower(self, thr=1.):
        """ Segmentation simply on the power"""
        maxFreqs = 10. * np.log10(np.max(self.sp.sg, axis=1))
        maxFreqs = medfilt(maxFreqs, 21)
        ind = maxFreqs > (np.mean(maxFreqs)+thr*np.std(maxFreqs))
        segs = self.convert01(ind, self.sp.incr / self.sp.audio_data.sample_rate)
        return segs

    def medianClip(self, thr=3.0, medfiltersize=5, minaxislength=5, minSegment=70):
        """ Median clipping for segmentation based on Lasseck's method.
        
        Args:
            thr: threshold multiplier for median clipping
            medfiltersize: size of median filter
            minaxislength: min length of minor axis for valid regions
            minSegment: min number of pixels for valid segment
        """
        sg = self.sp.sg / np.max(self.sp.sg)
        rowmedians = np.median(sg, axis=1)
        colmedians = np.median(sg, axis=0)

        clipped = np.zeros(np.shape(sg), dtype=int)
        for i in range(np.shape(sg)[0]):
            for j in range(np.shape(sg)[1]):
                if (sg[i, j] > thr * rowmedians[i]) and (sg[i, j] > thr * colmedians[j]):
                    clipped[i, j] = 1
        print("Found", np.sum(clipped), "pixels")

        clipped = spi.binary_closing(clipped, structure=DIAMOND_5X5).astype(int)
        clipped = spi.binary_dilation(clipped, structure=DIAMOND_5X5).astype(int)
        clipped = spi.median_filter(clipped, size=medfiltersize)
        clipped = spi.binary_fill_holes(clipped)

        blobs = skm.regionprops(skm.label(clipped.astype(int)))

        keep = []
        for i in range(len(blobs)):
            if blobs[i].filled_area > minSegment and blobs[i].minor_axis_length > minaxislength:
                keep.append(i)

        blobs = [blobs[i] for i in keep]
        out = []
        for l in blobs:
            out.append([float(l.bbox[0] * self.sp.incr / self.sp.audio_data.sample_rate), float(l.bbox[2] * self.sp.incr / self.sp.audio_data.sample_rate)])
        return out

    def checkSegmentOverlapCentroids(self, blobs, minSegment=50):
        centroids = np.array([(i[1] - i[0]) / 2 for i in blobs])
        ind = np.argsort(centroids)
        centroids = centroids[ind]
        blobs = np.array(blobs)[ind]

        current = 0
        centroid = centroids[0]
        count = 0
        merged = [blobs[0].tolist()]
        
        for i in centroids:
            if (i - centroid) * 1000 < minSegment / 2. * 10:
                if blobs[ind[count]][0] < merged[current][0]:
                    merged[current][0] = blobs[ind[count]][0]
                if blobs[ind[count]][1] > merged[current][1]:
                    merged[current][1] = blobs[ind[count]][1]
            else:
                current += 1
                centroid = centroids[count]
                merged.append([blobs[ind[count]][0], blobs[ind[count]][1]])
            count += 1

        segments = []
        for i in merged:
            if float(i[1] - i[0]) * 1000 > minSegment:
                segments.append([i[0], i[1]])
        return segments

    def onsets(self, thr=3.0):
        """ Segmentation using the onset times from librosa.
        There are no offset times -- compute an energy drop?
        A straw man really.
        """
        o_env = librosa.onset.onset_strength(self.sp.audio_data.data, sr=self.sp.audio_data.sample_rate, aggregate=np.median)
        cutoff = np.mean(o_env) + thr * np.std(o_env)
        o_env = np.where(o_env > cutoff, o_env, 0)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=self.sp.audio_data.sample_rate)
        times = librosa.frames_to_time(np.arange(len(o_env)), sr=self.sp.audio_data.sample_rate)

        segments = []
        for i in range(len(onsets)):
            segments.append([times[onsets[i]],times[onsets[i]]+0.2])
        return segments

    def yinSegs(self, minfreq=100, minperiods=3, thr=0.5, W=1000):
        """ Segmentation by computing the fundamental frequency.
            Uses the Yin algorithm of de Cheveigne and Kawahara (2002).
            Args:
            minfreq: lowest freq (Hz) to consider as plausible.
            thr: the threshold for accepting F0,
              necessarily higher than the 0.1 in the paper.
            W: the window in samples used.
        """
        # Window width W should be at least 3*period.
        # A sample rate of 16000 and a min fundamental frequency of 100Hz would then therefore suggest reasonably short windows
        minwin = float(self.sp.audio_data.sample_rate) / minfreq * minperiods
        if W < minwin:
            print("Extending window width to ", minwin)
            W = int(minwin)

        # returns pitch in Hz for each window of Wsamples/2.
        # As this uses the full audio data, it is up to caller to adjust times
        # to real seconds if the data only contained e.g. a segment
        shape = shapes.fundFreqShaper(self.sp.audio_data.data, W, thr, self.sp.audio_data.sample_rate)

        pitch = shape.y
        if len(pitch)==0:
            return np.array([])
        units = shape.tunit

        # drop any pitch under minfreq
        ind = pitch > minfreq
        segs = self.convert01(ind, units)
        return segs

    def findCCMatches(self, seg, sg, thr):
        """ Cross-correlation. Takes a segment and looks for others that match it to within thr.
        match_template computes fast normalised cross-correlation
        """
        from skimage.feature import match_template

        # seg and sg have the same $y$ size, so the result of match_template is 1D
        #m = match_template(sg,seg)
        matches = np.squeeze(match_template(sg, seg))

        import peakutils
        md = np.shape(seg)[0]/2
        threshold = thr*np.max(matches)
        indices = peakutils.indexes(matches, thres=threshold, min_dist=md)
        return indices

    def findDTWMatches(self, seg, data):
        # TODO: This is slow and crap. Note all the same length, for a start, and the fact that it takes forever!
        # Use MFCC first?
        d = np.zeros(len(data))
        for i in range(len(data)):
            d[i] = self.dtw(seg, data[i:i+len(seg)])
        return d

    def dtw(self, x, y, wantDistMatrix=False):
        # Compute the dynamic time warp between two 1D arrays
        dist = np.zeros((len(x)+1,len(y)+1))
        dist[1:, :] = np.inf
        dist[:, 1:] = np.inf
        for i in range(len(x)):
            for j in range(len(y)):
                dist[i+1, j+1] = np.abs(x[i]-y[j]) + min(dist[i, j+1], dist[i+1, j], dist[i, j])
        if wantDistMatrix:
            return dist
        else:
            return dist[-1, -1]

    def dtw_path(self, d):
        # Shortest path through DTW matrix
        i = np.shape(d)[0]-2
        j = np.shape(d)[1]-2
        xpath = [i]
        ypath = [j]
        while i>0 or j>0:
                next = np.argmin((d[i,j],d[i+1,j],d[i,j+1]))
                if next == 0:
                    i -= 1
                    j -= 1
                elif next == 1:
                    j -= 1
                else:
                    i -= 1
                xpath.insert(0,i)
                ypath.insert(0,j)
        return xpath, ypath

    # def testDTW(self):
    #     x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
    #     y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
    #
    #     d = self.dtw(x,y,wantDistMatrix=True)
    #     print self.dtw_path(d)


class PostProcess:
    """ This class implements few post processing methods basically to avoid false positives.
    Operates on detections from a single subfilter.

    segments:   wavelet filter detections in format [[s1,e1], [s2,e2],...]
        Will be converted to a list of [[s1, e1], prob] upon load,
        and subsequent functions deal with certainty values.
    subfilter:  AviaNZ format subfilter
    cert:       Default certainty to attach to the segments
    """

    def __init__(self, configdir, audioData=None, sampleRate=0, tgtsampleRate=0, segments=[], subfilter={}, NNmodel=None, cert=0):
        self.configdir = configdir
        # Store as AudioData object for consistency with Spectrogram API
        if audioData is not None:
            if isinstance(audioData, audio_data.AudioData):
                self.audioData = audioData
            else:
                # Convert raw numpy array to AudioData object
                self.audioData = audio_data.AudioData(
                    data=audioData, 
                    sample_rate=sampleRate,
                    sample_format='float32',
                    sample_size=32,
                    channels=1
                )
        else:
            self.audioData = None
        self.sampleRate = sampleRate
        self.subfilter = subfilter

        # Convert to [[s1, e1], cert]
        self.segments = []
        for seg in segments:
            if len(seg) != 2:
                continue
            if seg[0]<0 or seg[1]<0:
                continue
            self.segments.append([seg, cert])

        if NNmodel:
            # Configure TensorFlow GPU memory growth to prevent it from allocating all GPU memory at once
            try:
                physical_devices = tf.config.list_physical_devices('GPU')
                if physical_devices:
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except Exception as e:
                print(f"Warning: Could not configure GPU memory growth: {e}")

            cl = config_loader.ConfigLoader()
            self.LearningDict = cl.learningParams(os.path.join(configdir, "LearningParams.txt"))

            self.NNmodel = NNmodel[0]    # NNmodel is a list [model, win, inputdim, outputdict, windowInc, thrs]
            self.NNwindow = NNmodel[1][0]  # size of each frame
            # self.NNhop = NNmodel[1][1]
            self.NNhop = self.LearningDict['hopScaling']*self.NNwindow
            self.NNinputdim = NNmodel[2]
            self.NNoutputs = NNmodel[3]
            self.NNwindowInc = NNmodel[4]  # [window,incr] for making the spec
            self.NNthrs = NNmodel[5]
            if NNmodel[6]:
                self.NNfRange = NNmodel[7]
            else:
                self.NNfRange = None
            self.tgtsampleRate = tgtsampleRate
        else:
            self.NNmodel = None

        if subfilter != {}:
            self.minLen = subfilter['TimeRange'][0]
            self.maxLen = subfilter['TimeRange'][1]
            if 'F0Range' in subfilter:
                self.F0 = subfilter['F0Range']
            self.fLow = subfilter['FreqRange'][0]
            self.fHigh = subfilter['FreqRange'][1]
            self.minLen = subfilter['TimeRange'][0]
            self.calltype = subfilter['calltype']
            self.syllen = subfilter['TimeRange'][2]
        else:
            self.minLen = 0.25
            self.fLow = 0
            self.fHigh = 0

    def getCertainty(self, meanprob, ctkey):
        if meanprob[ctkey] >= self.NNthrs[ctkey][-1]:
            return 90
        elif meanprob[ctkey] >= self.NNthrs[ctkey][0]:
            return 50
        else:
            return 0

    def expand_short_segment(self, seg, min_length):
        """Expand segment in-place to meet minimum length requirement"""
        start, end = seg[0][0], seg[0][1]
        duration = end - start
        
        # Already long enough - no changes needed
        if duration >= min_length:
            return duration
        
        # Expand symmetrically, with small padding for safety
        PADDING = 0.005
        extend_by = (min_length - duration) / 2 + PADDING
        new_start = start - extend_by
        new_end = end + extend_by
        
        # Clip to audio boundaries
        audio_length_secs = len(self.audioData.data) / self.sampleRate
        
        if new_start < 0:
            # Hit start boundary - anchor at 0 and extend forward
            new_start = 0
            new_end = min_length + 0.01  # Small extra padding
        elif new_end > audio_length_secs:
            # Hit end boundary - anchor at end and extend backward
            new_end = audio_length_secs
            new_start = max(0, audio_length_secs - min_length - 0.01)
        
        # Modify segment in place
        seg[0][0] = new_start
        seg[0][1] = new_end
        
        return new_end - new_start

    def generate_nn_features(self, audio_data, duration):
        """Generate NN features from an AudioData object.
        
        Args:
            audio_data: AudioData object containing the audio segment
            duration: Duration of the segment in seconds
        """
        nn_window_width = self.NNinputdim[0]
        sp = spectrogram.Spectrogram(window_width=nn_window_width, incr=self.NNwindowInc[1])
        sp.audio_data = audio_data
        
        if self.sampleRate != self.tgtsampleRate:
            sp.resample(self.tgtsampleRate)
        
        featuress = sp.generateFeaturesNN(seglen=duration, real_spec_width=self.NNinputdim[1], 
                                         frame_size=self.NNwindow, frame_hop=self.NNhop, 
                                         NNfRange=self.NNfRange)
        return featuress.astype('float32')

    def predict_nn_batched(self, featuress, batchsize=5):
        numframes = featuress.shape[0]
        if numframes == 0:
            return None
        
        probs = np.empty((numframes, len(self.NNoutputs)))
        for start in range(0, numframes, batchsize):
            end = min(numframes, start + batchsize)
            p = self.NNmodel(tf.convert_to_tensor(featuress[start:end, :, :, :], dtype=tf.float32))
            probs[start:end, :] = p
        return probs

    def compute_certainty_from_probs(self, probs, ctkey):
        if probs is None:
            return 0
        
        if self.activelength(probs[:, ctkey], self.NNthrs[ctkey][-1]) >= self.subfilter['TimeRange'][0]:
            return 90
        elif self.activelength(probs[:, ctkey], self.NNthrs[ctkey][0]) >= self.subfilter['TimeRange'][0]:
            return 50
        else:
            return 0

    def NN(self):
        if not self.NNmodel:
            print("ERROR: no NN model specified")
            return
        if len(self.segments) == 0:
            print("No segments to classify by NN")
            return
        
        ctkey = int(list(self.NNoutputs.keys())[list(self.NNoutputs.values()).index(self.calltype)])
        print('call type: ', self.calltype)
        batchsize = 5

        for ix in reversed(range(len(self.segments))):
            seg = self.segments[ix]
            print('\n--- Segment', seg)
            
            duration = self.expand_short_segment(seg, self.NNwindow)
            
            # Extract audio segment as AudioData object
            start_sample = int(seg[0][0] * self.sampleRate)
            end_sample = int(seg[0][1] * self.sampleRate)
            segment_audio = audio_data.AudioData(
                data=self.audioData.data[start_sample:end_sample],
                sample_rate=self.sampleRate,
                sample_format='float32',
                sample_size=32,
                channels=1
            )
            
            featuress = self.generate_nn_features(segment_audio, duration)
            
            if featuress.shape != (featuress.shape[0], self.NNinputdim[0], self.NNinputdim[1], 1):
                print("ERROR: features shape incorrect", featuress.shape)
                raise AssertionError
            
            probs = self.predict_nn_batched(featuress, batchsize)
            certainty = self.compute_certainty_from_probs(probs, ctkey)
            
            print("probabilities: ", probs)
            if certainty == 0:
                print('Deleted by NN')
                del self.segments[ix]
            else:
                print('Not deleted by NN')
                self.segments[ix][-1] = certainty

        print("Segments remaining after NN: ", len(self.segments))

    def activelength(self, probs, thr):
        """
        Returns the max length (secs) above thr given the probabilities of the images (overlapped)
        """
        binaryout = np.asarray(probs>=thr, dtype=int)
        subsegs = Segmenter.convert01(binaryout)
        lengths = [seg[1]-seg[0] for seg in subsegs]
        if lengths:
            return max(lengths)*self.NNhop
        else:
            return 0

    def NNDiagnostic(self):
        if not self.NNmodel:
            print("ERROR: no NN model specified")
            return
        if len(self.segments) == 0:
            print("No segments to classify by NN")
            return

        self.NNhop = self.NNwindow
        for ix in reversed(range(len(self.segments))):
            seg = self.segments[ix]

            if self.NNwindow >= seg[0][1] - seg[0][0]:
                print('Current page is smaller than NN input (%f)' % (self.NNwindow))
                # Use full audio data
                segment_audio = self.audioData
            else:
                # Use full audio data (this seems like it should be a segment slice, but preserving original logic)
                segment_audio = self.audioData
            
            featuress = self.generate_nn_features(segment_audio, seg[0][1] - seg[0][0])
            
            if np.shape(featuress)[0] > 0:
                probs = self.NNmodel.predict(featuress)
            else:
                probs = 0
        return self.NNwindow, probs

    def wind_cal(self, data, sampleRate, fn_peak=0.35):
        """ Calculate wind characteristics from audio data.
        Adopted from Automatic Identification of Rainfall in Acoustic Recordings by Carol Bedoya et al.
        """
        wind_lower = 2.0 * 50 / sampleRate
        wind_upper = 2.0 * 500 / sampleRate
        f, p = signal.welch(data, fs=sampleRate, window='hamming', nperseg=512, detrend=False)
        p = np.log10(p)

        limite_inf = int(round(len(p) * wind_lower))
        limite_sup = int(round(len(p) * wind_upper))
        a_wind = p[limite_inf:limite_sup]

        fn = False
        if self.fLow > 500 or self.fLow == 0:
            ind = np.abs(f - 500).argmin()
            if self.fLow == 0:
                ind_fLow = ind
            else:
                ind_fLow = np.abs(f - self.fLow).argmin() - ind
            if self.fHigh == 0:
                ind_fHigh = len(f) - 1
            else:
                ind_fHigh = np.abs(f - self.fHigh).argmin() - ind
            p = p[ind:]

            peaks, _ = signal.find_peaks(p)
            peaks = [i for i in peaks if (ind_fLow <= i <= ind_fHigh)]
            prominences = signal.peak_prominences(p, peaks)[0]
            if len(prominences) > 0 and np.max(prominences) > fn_peak:
                fn = True

        return np.mean(a_wind), np.std(a_wind), fn

    def wind(self, windT=2.5, fn_peak=0.35):
        if len(self.segments) == 0:
            print("No segments to remove wind from")
            return

        newSegments = copy.deepcopy(self.segments)
        for seg in self.segments:
            data = self.audioData.data[int(seg[0][0] * self.sampleRate):int(seg[0][1] * self.sampleRate)]
            ind = np.flatnonzero(data).tolist()
            data = np.asarray(data)[ind].tolist()
            if len(data) == 0:
                continue
            m, _, fn = self.wind_cal(data=data, sampleRate=self.sampleRate, fn_peak=fn_peak)
            if m > windT and not fn:
                print(seg[0], m, 'windy, deleted')
                newSegments.remove(seg)
            elif m > windT and fn:
                print(seg[0], m, 'windy, but possible bird call')
            else:
                print(seg[0], m, 'not windy/possible bird call')
        self.segments = newSegments
        print("Segments remaining after wind: ", len(self.segments))



    def fundamentalFrq(self, fileName=None):
        Wsamples = 1024
        minfreq = 100
        thr = 0.5

        for segix in reversed(range(len(self.segments))):
            seg = self.segments[segix][0]
            secs = int(seg[1] - seg[0])
            
            sp = spectrogram.Spectrogram(256, 128)
            if fileName:
                sp.readSoundFile(fileName, secs, seg[0])
                self.sampleRate = sp.audio_data.sample_rate
                self.audioData = sp.audio_data
            else:
                sp.audio_data = self.audioData

            minwin = 3 * sp.audio_data.sample_rate / minfreq
            if Wsamples < minwin:
                print("Extending window width to ", minwin)
                Wsamples = int(minwin)

            pitch = shapes.fundFreqShaper(sp.audio_data.data, Wsamples, thr, sp.audio_data.sample_rate)
            pitch = pitch.y
            ind = np.squeeze(np.where(pitch > minfreq))
            pitch = pitch[ind]

            if pitch.size == 0:
                print('Segment ', seg, ' *++ no fundamental freq detected, could be faded call or noise')
                del self.segments[segix]
            else:
                meanF0 = np.mean(pitch)
                if (meanF0 < self.F0[0]) or (meanF0 > self.F0[1]):
                    print('segment* ', seg, meanF0, pitch, ' *-- fundamental freq is out of range, could be noise')
                    del self.segments[segix]
        print("Segments remaining after fundamental frequency: ", len(self.segments))

    # The following are just wrappers for easier parsing of 3-element segment lists:
    # Segmenter class has corresponding methods that operate on 2-element lists
    def joinGaps(self, maxgap):
        self.segments = Segmenter.joinGaps3(self.segments, maxgap=maxgap)
        print("Segments remaining after merge (<=%.2f secs): %d" % (maxgap, len(self.segments)))

    def deleteShort(self, minlength):
        self.segments = Segmenter.deleteShort3(self.segments, minlength=minlength)
        print("Segments remaining after deleting short (<%.2f secs): %d" % (minlength, len(self.segments)))

    def splitLong(self, maxlen):
        self.segments = Segmenter.splitLong3(self.segments, maxlen=maxlen)
        print('Segments after splitting long segments (>%.2f secs): %d' % (maxlen, len(self.segments)))

    def checkSegmentOverlap(self):
        # Used for merging call types or different segmenter outputs
        self.segments = Segmenter.checkSegmentOverlap3(self.segments)
        print("Segments produced after merging: %d" % len(self.segments))
