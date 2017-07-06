# Version 0.3 27/11/16
# Author: Stephen Marsland

import numpy as np
import scipy.ndimage as spi
import librosa

# TODO:
# Pure energy threshold-> think about threshold, add automatic gain
# Median clipping: play with the various parameters, get some that I'm happy with
#   -> size of diamond, med filter, min size of blob
#   -> think about how to remove duplicate segments
# Amplitude: threshold parameter ?!
# onset_detect from librosa
#   If we want to use it, it needs work -> remove overlaps, sort out length
#   -> doesn't do offset :(
# Add yaapt or bana to yin for fundamental frequency
# Frechet distance for DTW?

class Segment:
    # This class implements six forms of segmentation for the AviaNZ interface:
    # Amplitude threshold, energy threshold, Harma, median clipping of spectrogram, yin, FIR
    # And can merge them

    # It also implements two forms of recognition:
    # Cross-correlation and DTW

    # Each returns start and stop times for each segment (in seconds) as a Python list of pairs
    # See also the species-specific segmentation in WaveletSegment

    def __init__(self,data,sg,sp,fs,minSegment,window_width=256,incr=128):
        self.data = data
        self.fs = fs
        # This is the length of a window to average to get the power
        #self.length = 100
        # This is the (raw -- psd) spectrogram
        self.sg = sg
        # This is the reference to SignalProc
        self.sp = sp
        self.minSegment = minSegment
        # print "minSegment:", minSegment
        # These are the spectrogram params. Needed to compute times
        self.window_width = window_width
        self.incr = incr

    def setNewData(self, data, sg, fs, window_width, incr):
        # To be called when a new sound file is loaded
        self.data = data
        self.fs = fs
        self.sg = sg
        self.window_width = window_width
        self.incr = incr

    def bestSegments(self):
        segs1 = self.segmentByFIR(0.7)
        segs2 = self.medianClip()
        segs3, p, t = self.yin(100, thr=0.9, returnSegs=True)
        segs1 = self.mergeSegments(segs1, segs2)
        return self.mergeSegments(segs1,segs3)

    def mergeSegments(self,segs1,segs2):
        # Given two segmentations of the same file, return the merged set of them
        # Two similar segments should be replaced by their union
        # Those that are inside another should be removed (?) or the too-large one deleted?
        # This version is the first of those, next is the second

        from intervaltree import Interval, IntervalTree
        t = IntervalTree()

        # Put the first set into the tree
        for s in segs1:
            t[s[0]:s[1]] = s

        # Decide whether or not to put each segment in the second set in
        for s in segs2:
            overlaps = t.search(s[0],s[1])
            # If there are no overlaps, add it
            if len(overlaps)==0:
                t[s[0]:s[1]] = s
            else:
                # Search for any enveloped, if there are remove and add the new one
                envelops = t.search(s[0],s[1],strict=True)
                if len(envelops) > 0:
                    # Remove any inside the envelope of the test point
                    t.remove_envelop(s[0],s[1])
                    overlaps = t.search(s[0], s[1])
                    #print s[0], s[1], overlaps
                    # Open out the region, delete the other
                    for o in overlaps:
                        if o.begin < s[0]:
                            s[0] = o.begin
                            t.remove(o)
                        if o.end > s[1]:
                            s[1] = o.end
                            t.remove(o)
                    t[s[0]:s[1]] = s
                else:
                    # Check for those that intersect the ends, widen them out a bit
                    for o in overlaps:
                        if o.begin > s[0]:
                            t[s[0]:o[1]] = (s[0],o[1])
                            t.remove(o)
                        if o.end < s[1]:
                            t[o[0]:s[1]] = (o[0],s[1])
                            t.remove(o)

        segs = []
        for a in t:
            segs.append([a[0],a[1]])
        return segs

    def mergeSegments2(self,segs1,segs2):
        # Given two segmentations of the same file, return the merged set of them
        # Two similar segments should be replaced by their union
        # Those that are inside another should be removed (?) or the too-large one deleted?
        # This version is the second of those

        from intervaltree import Interval, IntervalTree
        t = IntervalTree()

        # Put the first set into the tree
        for s in segs1:
            t[s[0]:s[1]] = s

        # Decide whether or not to put each segment in the second set in
        for s in segs2:
            overlaps = t.search(s[0],s[1])
            # If there are no overlaps, add it
            if len(overlaps)==0:
                t[s[0]:s[1]] = s
            else:
                # Search for any enveloped, if there are remove and add the new one
                envelops = t.search(s[0],s[1],strict=True)
                if len(envelops) > 0:
                    # Don't include this one
                    pass
                else:
                    # Check for those that intersect the ends, widen them out a bit
                    for o in overlaps:
                        if o.begin > s[0]:
                            t[s[0]:o[1]] = (s[0],o[1])
                            t.remove(o)
                        if o.end < s[1]:
                            t[o[0]:s[1]] = (o[0],s[1])
                            t.remove(o)

        segs = []
        for a in t:
            segs.append([a[0],a[1]])
        return segs

    def checkSegmentLength(self,segs, mingap=0.3, minlength=0.2, maxlength=5.0):
        # Checks whether start/stop segments are long enough
        # These are species specific!
        # Doesn't currently use maxlength
        for i in range(len(segs))[-1::-1]:
            if i<len(segs)-1:
                if np.abs(segs[i][1] - segs[i+1][0]) < mingap:
                    segs[i][1] = segs[i+1][1]
                    del segs[i+1]
            if np.abs(segs[i][1] - segs[i][0]) < minlength:
                del segs[i]
        return segs

    def identifySegments(self, seg, maxgap=1, minlength=1):
        # Turns presence/absence segments into a list of start/stop times
        # Note the two parameters
        segments = []
        start = seg[0]
        for i in range(1, len(seg)):
            if seg[i] <= seg[i - 1] + maxgap:
                pass
            else:
                # See if segment is long enough to be worth bothering with
                if (seg[i - 1] - start) > minlength:
                    segments.append([float(start) * self.incr / self.fs, float(seg[i - 1]) * self.incr / self.fs])
                start = seg[i]
        if seg[-1] - start > minlength:
            segments.append([float(start) * self.incr / self.fs, float(seg[-1]) * self.incr / self.fs])

        return segments

    def segmentByFIR(self, threshold):
        # Create an FIR envelope (as a negative exponential -- params are hacks!)
        from scipy.interpolate import interp1d
        # print self.fs, self.window_width, self.incr
        nsecs = len(self.data) / float(self.fs)
        fftrate = int(np.shape(self.sg)[0]) / nsecs
        upperlimit = 100
        # a = np.arange(upperlimit)
        # FIR = 0.06 * np.exp(-a / 5.)
        FIR = [0.078573000000000004, 0.053921000000000004, 0.041607999999999999, 0.036006000000000003, 0.031521,
               0.029435000000000003, 0.028122000000000001, 0.027286999999999999, 0.026241000000000004,
               0.025225999999999998, 0.024076, 0.022926999999999999, 0.021703999999999998, 0.020487000000000002,
               0.019721000000000002, 0.019015000000000001, 0.018563999999999997, 0.017953, 0.01753,
               0.017077000000000002, 0.016544, 0.015762000000000002, 0.015056, 0.014456999999999999, 0.013913,
               0.013299, 0.012879, 0.012568000000000001, 0.012454999999999999, 0.012056000000000001, 0.011634,
               0.011077, 0.010707, 0.010217, 0.0098840000000000004, 0.0095959999999999986, 0.0093607000000000013,
               0.0090197999999999997, 0.0086908999999999997, 0.0083841000000000002, 0.0081481999999999995,
               0.0079185000000000002, 0.0076363000000000004, 0.0073406000000000009, 0.0070686999999999998,
               0.0068438999999999991, 0.0065873000000000008, 0.0063688999999999994, 0.0061700000000000001,
               0.0059743000000000001, 0.0057561999999999995, 0.0055351000000000003, 0.0053633999999999991,
               0.0051801, 0.0049743000000000001, 0.0047431000000000001, 0.0045648999999999993,
               0.0043972000000000004, 0.0042459999999999998, 0.0041016000000000004, 0.0039503000000000003,
               0.0038013000000000005, 0.0036351, 0.0034856000000000002, 0.0033270999999999999,
               0.0032066999999999998, 0.0030569999999999998, 0.0029206999999999996, 0.0027760000000000003,
               0.0026561999999999996, 0.0025301999999999998, 0.0024185000000000001, 0.0022967,
               0.0021860999999999998, 0.0020696999999999998, 0.0019551999999999998, 0.0018563,
               0.0017562000000000001, 0.0016605000000000001, 0.0015522000000000001, 0.0014482999999999998,
               0.0013492000000000001, 0.0012600000000000001, 0.0011788, 0.0010909000000000001, 0.0010049,
               0.00091527999999999998, 0.00082061999999999999, 0.00074465000000000002, 0.00067159000000000001,
               0.00060258999999999996, 0.00053370999999999996, 0.00046135000000000002, 0.00039071,
               0.00032736000000000001, 0.00026183000000000001, 0.00018987999999999999, 0.00011976000000000001,
               6.0781000000000006e-05, 0.0]
        f = interp1d(np.arange(0, len(FIR)), np.squeeze(FIR))
        samples = f(np.arange(1, upperlimit, float(upperlimit) / int(fftrate / 10.)))
        padded = np.concatenate((np.zeros(int(fftrate / 10.)), np.mean(self.sg, axis=1), np.zeros(int(fftrate / 10.))))
        envelope = spi.filters.convolve(padded, samples, mode='constant')[:-int(fftrate / 10.)]
        seg = np.squeeze(np.where(envelope > np.median(envelope) + threshold * np.std(envelope)))
        # print seg
        return self.identifySegments(seg, minlength=10) #, envelope

    def segmentByAmplitude(self,threshold):
        seg = np.where(np.abs(self.data)>threshold)
        #print seg
        # print self.incr
        # return self.identifySegments(np.squeeze(seg))
        return self.identifySegments(np.squeeze(seg)/float(self.incr))

    def segmentByAmplitude1(self,threshold): #replaced the amplitude value thr to a percentage. e.g. extract top 70%
        seg = np.where(np.abs(self.data)>threshold*np.max(self.data))
        #print seg
        #print np.shape(np.squeeze(seg))[0]
        if np.shape(np.squeeze(seg))[0]>0:
            return self.identifySegments(np.squeeze(seg)/float(self.incr))  #TODO: self.incr is the right value to devide with?
        else:
            return []

    def segmentByEnergy(self,thr,width,min_width=450):
        # Based on description in Jinnai et al. 2012 paper in Acoustics
        # Computes the 'energy curve' as windowed sum of absolute values of amplitude
        # I median filter it, 'cos it's very noisy
        # And then threshold it (no info on their threshold) and find max in each bit above threshold
        # I also check for width of those (they don't say anything)
        # They then return the max-width:max+width segments for each max
        data = np.abs(self.data)
        E = np.zeros(len(data))
        E[width] = np.sum(data[:2*width+1])
        for i in range(width+1,len(data)-width):
            E[i] = E[i-1] - data[i-width-1] + data[i+width]
        E = E/(2*width)

        # TODO: Automatic energy gain (normalisation method)

        # This thing is noisy, so I'm going to median filter it. SoundID doesn't seem to?
        Em = np.zeros(len(data))
        for i in range(width,len(data)-width):
            Em[i] = np.median(E[i-width:i+width])
        for i in range(width):
            Em[i] = np.median(E[0:2*i])
            Em[-i] = np.median(E[-2 * i:])

        # TODO: Better way to do this?
        threshold = np.mean(Em) + thr*np.std(Em)

        # Pick out the regions above threshold and the argmax of each, assuming they are wide enough
        starts = []
        ends = []
        insegment = False
        for i in range(len(data)-1):
            if not insegment:
                if Em[i]<threshold and Em[i+1]>threshold:
                    starts.append(i)
                    insegment = True
            if insegment:
                if Em[i]>threshold and Em[i+1]<threshold:
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

        # TODO: SoundID appears to now compute the 44 LPC coeffs for each [midpoint-width:midpoint+width]
        # TODO: And then compute the geometric distance to templates
        #return Emm, maxpoints
        segs = []
        for i in range(len(starts)):
            segs.append([float(starts[i])/self.fs,float(ends[i])/self.fs])
        return segs


    def Harma(self,thr=10.,stop_thr=0.8):
        # Harma's method, but with a different stopping criterion
        # Note that this will go wrong with librosa's load because the elements of the spectrogram lie in [0,1] and hence interesting things with the log
        #print np.shape(self.sg), np.min(self.sg), np.max(self.sg)
        maxFreqs = 10. * np.log10(np.max(self.sg, axis = 1))
        from scipy.signal import medfilt
        maxFreqs = medfilt(maxFreqs,21)
        #import pylab as pl
        #pl.savetxt('poo.txt',maxFreqs)
        biggest = np.max(maxFreqs)
        segs = []

        while np.max(maxFreqs)>stop_thr*biggest:
            t0 = np.argmax(maxFreqs)
            a_n = maxFreqs[t0]
            #print t0, a_n

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

            #print t_start, t_end
            # Set the syllable just found to 0
            maxFreqs[t_start:t_end] = 0
            if float(t_end - t_start)*self.incr/self.fs*1000.0 > self.minSegment:
                segs.append([float(t_start)* self.incr / self.fs,float(t_end)* self.incr / self.fs])

            #print t_start, t_end, stop_thr*biggest, np.max(maxFreqs)
        return segs

    def segmentByPower(self,thr=1.):
        # Harma's method, but with a different stopping criterion
        # Note that this will go wrong with librosa's load because the elements of the spectrogram lie in [0,1] and hence interesting things with the log
        #print np.shape(self.sg), np.min(self.sg), np.max(self.sg)
        maxFreqs = 10. * np.log10(np.max(self.sg, axis = 1))
        # print np.shape(maxFreqs)
        from scipy.signal import medfilt
        maxFreqs = medfilt(maxFreqs,21)
        seg = np.squeeze(np.where(maxFreqs > (np.mean(maxFreqs)+thr*np.std(maxFreqs))))
        #print seg
        return self.identifySegments(seg,minlength=10)

    def medianClip(self,thr=3.0,medfiltersize=5,minsize=80,minaxislength=5):
        # Median clipping for segmentation
        # Based on Lasseck's method (but with multi-tapered spectrogram)
        # This version only clips in time, ignoring frequency
        # And it opens up the segments to be maximal (so assumes no overlap)
        # TODO: Parameters!!
        # Use the multitaper spectrogram, it helps a lot
        #print np.max(self.sg)
        sg = self.sg.T/np.max(self.sg)

        # This next line gives an exact match to Lasseck, but screws up bitterns!
        #sg = sg[4:232, :]

        rowmedians = np.median(sg, axis=1)
        colmedians = np.median(sg, axis=0)

        clipped = np.zeros(np.shape(sg),dtype=int)
        for i in range(np.shape(sg)[0]):
            for j in range(np.shape(sg)[1]):
                if (sg[i, j] > thr * rowmedians[i]) and (sg[i, j] > thr * colmedians[j]):
                    clipped[i, j] = 1

        # This is the stencil for the closing and dilation. It's a 5x5 diamond. Can also use a 3x3 diamond
        diamond = np.zeros((5,5),dtype=int)
        diamond[2,:] = 1
        diamond[:,2] = 1
        diamond[1,1] = diamond[1,3] = diamond[3,1] = diamond[3,3] = 1
        #diamond[2, 1:4] = 1
        #diamond[1:4, 2] = 1

        import scipy.ndimage as spi
        clipped = spi.binary_closing(clipped,structure=diamond).astype(int)
        clipped = spi.binary_dilation(clipped,structure=diamond).astype(int)
        clipped = spi.median_filter(clipped,size=medfiltersize)
        clipped = spi.binary_fill_holes(clipped)

        import skimage.measure as skm
        blobs = skm.regionprops(skm.label(clipped.astype(int)))

        # Delete blobs that are too small
        todelete = []
        for i in blobs:
            if i.filled_area < minsize or i.minor_axis_length < minaxislength:
                todelete.append(i)

        for i in todelete:
            blobs.remove(i)

        # Delete overlapping boxes by computing the centroids and picking out overlaps
        # Could also look at width and so just merge boxes that are about the same size
        centroids = []
        for i in blobs:
            centroids.append(i.centroid[1])
        centroids = np.array(centroids)
        ind = np.argsort(centroids)
        centroids = centroids[ind]

        current = 0
        centroid = centroids[0]
        count = 0
        list = []
        list.append([blobs[ind[0]].bbox[1],blobs[ind[0]].bbox[1]])
        for i in centroids:
            if i - centroid < minsize / 2.:
                if blobs[ind[count]].bbox[1]<list[current][0]:
                    list[current][0] = blobs[ind[count]].bbox[1]
                if blobs[ind[count]].bbox[3] > list[current][1]:
                    list[current][1] = blobs[ind[count]].bbox[3]
            else:
                current += 1
                centroid = centroids[count]
                list.append([blobs[ind[count]].bbox[1], blobs[ind[count]].bbox[1]])
            count += 1

        segments = []
        for i in list:
            #print i
            if float(i[1] - i[0])*self.incr/self.fs*1000 > self.minSegment:
                segments.append([float(i[0]) * self.incr / self.fs, float(i[1]) * self.incr / self.fs])
            #else:
                #print i[1], i[0], float(i[1] - i[0])*self.incr/self.fs

        return segments

    def onsets(self,thr=3.0):
        # TODO: What about offsets, check threshold (and note it currently isn't passed in!)
        # This gets the onset times from librosa. But there are no offset times -- compute an energy drop?
        o_env = librosa.onset.onset_strength(self.data, sr=self.fs, aggregate=np.median)
        cutoff = np.mean(o_env) + thr * np.std(o_env)
        o_env = np.where(o_env > cutoff, o_env, 0)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=self.fs)
        times = librosa.frames_to_time(np.arange(len(o_env)), sr=self.fs)

        segments = []
        for i in range(len(onsets)):
            segments.append([times[onsets[i]],times[onsets[i]]+0.2])
        return segments

    def yin(self,minfreq=100, minperiods=3, thr=0.5, W=1000, returnSegs=False):
        # Compute fundamental frequency using Yin algorithm of de Cheveigne and Kawahara (2002)

        # fs, data = wavfile.read('../Sound Files/tril1.wav')
        if self.data.dtype == 'int16':
            data = self.data.astype(float)/32768.0
        else:
            data = self.data

        # The threshold is necessarily higher than the 0.1 in the paper

        # Window width W should be at least 3*period.
        # A sample rate of 16000 and a min fundamental frequency of 100Hz would then therefore suggest reasonably short windows
        minwin = float(self.fs) / minfreq * minperiods
        if minwin > W:
            print "Extending window width to ", minwin
            W = minwin
        # Make life easier, and make W be a function of the spectrogram window width
        W = int(round(W/self.window_width)*self.window_width)
        pitch = np.zeros((int((len(data) - 2 * W) * 2. / W) + 1))

        # Compute squared diff between signal and shift
        # sd = np.zeros(W)
        # for tau in range(1,W):
        #    sd[tau] = np.sum((data[:W] - data[tau:tau+W])**2)

        ints = np.arange(1, W)
        starts = range(0, len(data) - 2 * W, W / 2)

        for i in starts:
            # Compute squared diff between signal and shift
            sd = np.zeros(W)
            for tau in range(1, W):
                sd[tau] = np.sum((data[i:i + W] - data[i + tau:i + tau + W]) ** 2)

                # If not using window
                # if i>0:
                # for tau in range(1,W):
                # sd[tau] -= np.sum((data[i-1] - data[i-1+tau])**2)
                # sd[tau] += np.sum((data[i+W] - data[i+W+tau])**2)

            # Compute cumulative mean of normalised diff
            d = np.zeros(W)
            d[0] = 1
            d[1:] = sd[1:] * ints / np.cumsum(sd[1:])

            tau = 1
            notFound = True
            while tau < W-1 and notFound:
                tau += 1
                if d[tau] < thr:
                    notFound = False
                    while tau+1 < W and d[tau+1] < d[tau]:
                        tau = tau+1

            if tau == W-1 or d[tau] >= thr:
                #print "none found"
                pitch[int(i*2./W)] = -1
            else:
                # Parabolic interpolation to improve the estimate
                if tau==0:
                    s0 = d[tau]
                else:
                    s0 = d[tau-1]
                if tau==W-1:
                    s2 = d[tau]
                else:
                    s2 = d[tau+1]
                newtau = tau + (s2 - s0)/(2.0*(2.0*d[tau] - s0 - s2))

                # Compute the pitch
                pitch[int(i*2./W)] = float(self.fs)/newtau

        if returnSegs:
            ind = np.squeeze(np.where(pitch > minfreq))
            segs = self.identifySegments(ind)
            #ind = ind.astype('float')
            for s in segs:
                s[0] *= W / self.window_width
                s[1] *= W / self.window_width
            # segs = []
            # ind = np.squeeze(np.where(pitch > minfreq))
            # inseg = False
            # i = 0
            # while i < len(ind):
            #     if not inseg:
            #         start = ind[i]
            #         inseg = True
            #     elif ind[i] != ind[i - 1] + 1:
            #         # TODO: Consider allowing for a couple of points to be missed?
            #         inseg = False
            #         # TODO: Consider not including segments that are too short
            #         if float(starts[ind[i-1]] - starts[start]) / self.fs * 1000.0 > self.minSegment:
            #             segs.append([float(starts[start]) / self.fs, float(starts[ind[i-1]]) / self.fs])
            #     i += 1
            # # Add the last segment
            # if (starts[ind[i - 1]] - starts[start]) / self.fs  * 1000.0> self.minSegment:
            #     segs.append([float(starts[start]) / self.fs, float(starts[ind[i - 1]]) / self.fs])
            return segs, pitch, np.array(starts)
        else:
            return pitch, np.array(starts), minfreq, W

    # Function for cross-correlation to find segments that match the currently selected one.
    def findCCMatches(self,seg,sg,thr):
        # This takes a segment and looks for others that match it according to cross-correlation
        # match_template is fast normalised cross-correlation
        # TODO: There is a messy parameter (the threshold) unfortunately, ditto the min_dist
        # TODO: remove multiple boxes -- compute overlap
        from skimage.feature import match_template

        # At the moment seg and sg have the same $y$ size, so the result of match_template is 1D
        matches = np.squeeze(match_template(sg, seg))

        import peakutils
        md = np.shape(seg)[0]/2
        threshold = thr*np.max(matches)
        indices = peakutils.indexes(matches, thres=threshold, min_dist=md)
        #print indices
        return indices

    def findDTWMatches(self,seg,data,thr):
        # TODO: This is slow and crap. Note all the same length, for a start, and the fact that it takes forever!
        # Use MFCC first?
        d = np.zeros(len(data))
        for i in range(len(data)):
            d[i] = self.dtw(seg,data[i:i+len(seg)])
        #print d
        return d

    def dtw(self,x,y,wantDistMatrix=False):
        # Compute the dynamic time warp between two 1D arrays
        # I've taught it to second years, should be easy!
        dist = np.zeros((len(x)+1,len(y)+1))
        dist[1:,:] = np.inf
        dist[:,1:] = np.inf
        for i in range(len(x)):
            for j in range(len(y)):
                dist[i+1,j+1] = np.abs(x[i]-y[j]) + min(dist[i,j+1],dist[i+1,j],dist[i,j])
        if wantDistMatrix:
            return dist
        else:
            return dist[-1,-1]

    def dtw_path(self,d):
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

def testsegEnergy():
    data, sampleRate = librosa.load('Sound Files/tril1.wav',sr=None)
    import SignalProc
    sp = SignalProc.SignalProc(data,sampleRate,256,128)
    sg = sp.spectrogram(data,multitaper=True)
    s = Segment(data,sg,sp,fs)
    segments, midpoints = s.segmentByEnergy(0.005,200)
    #print midpoints
    pl.figure()
    pl.plot(data)
    pl.plot(segments)

def testsegMC():
    #data, fs = librosa.load('Sound Files/male1.wav')
    #data, fs = librosa.load('Sound Files/tril1.wav',sr=None)
    from scipy.io import wavfile
    fs, data = wavfile.read('Sound Files/tril1.wav')
    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128)
    # sg = sp.spectrogram(data)
    # segments, clipped, blobs = s.medianClip()
    # print segments
    #
    # pl.figure(), pl.imshow(clipped)
    # for i in blobs:
    #     pl.gca().add_patch(
    #         pl.Rectangle((i.bbox[1], i.bbox[0]), i.bbox[3] - i.bbox[1], i.bbox[2] - i.bbox[0], facecolor='w',
    #                      alpha=0.5))
    sg = sp.spectrogram(data,multitaper=True)
    s = Segment.Segment(data,sg,sp,fs)
    #segments, clipped, blobs = s.medianClip()
    segments = s.medianClip()
    #print segments
    return segments
    # pl.figure(), pl.imshow(clipped)
    # for i in blobs:
    #     pl.gca().add_patch(
    #         pl.Rectangle((i.bbox[1], i.bbox[0]), i.bbox[3] - i.bbox[1], i.bbox[2] - i.bbox[0], facecolor='w',
    #                      alpha=0.5))

def testsegHarma():
    from scipy.io import wavfile
    fs, data = wavfile.read('Sound Files/tril1.wav')
    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128)
    sg = sp.spectrogram(data,multitaper=True)
    s = Segment(data, sg, sp, fs)
    #segments, clipped, blobs = s.medianClip()
    segments = s.Harma()
    print segments
    onsets = s.onsets()
    print onsets
#testseg()

def testsegDTW():
    data, fs = librosa.load('Sound Files/tril1.wav',sr=None)
    segment = data[79:193]
    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128)
    sg = sp.spectrogram(data,fs,multitaper=False)
    s = Segment(data,sg,sp,fs)
    d = s.findDTWMatches(segment,data, 0.7)
    pl.figure, pl.plot(d)

def testYin():
    data, fs = librosa.load('OpenE.wav',sr=None)
    data = data[1000:2500]
    data = data.astype('float') / 32768.0
    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128)
    sg = sp.spectrogram(data,fs,multitaper=False)
    import yin
    reload(yin)
    pitches = yin.yin(data,fs)
    print pitches
    import pylab as pl
    pl.ion()
    pl.figure, pl.imshow(sg)
    pl.show()

def testsegFIR():
    #from scipy.io import wavfile
    import wavio
    #fs, data = wavfile.read('Sound Files/New folder/robin.wav')
    wavobj = wavio.read('Sound Files/20170515_223004 piping.wav')
    #wavobj = wavio.read('Sound Files/tril1.wav')
    #wavobj = wavio.read('Sound Files/kiwi_1min.wav')
    fs = wavobj.rate
    data = wavobj.data
    if data.dtype is not 'float':
        data = data.astype('float')  # / 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    #data = data[:fs*30]
    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128)
    sg = sp.spectrogram(data,multitaper=False,mean_normalise=False)
    s = Segment(data,sg,sp,fs,50) #(data,sg,sp,fs,minSegment,window_width=256,incr=128)
    segments, samp = s.segmentByFIR(1.0)

    import pylab as pl
    pl.ion()
    pl.plot(samp)
    pl.plot(np.arange(len(samp)),(np.median(samp)+np.std(samp))*np.ones(len(samp)))
    pl.show()

    print segments


def testMergeSeg():
    from scipy.io import wavfile
    fs, data = wavfile.read('Sound Files/tril1.wav')
    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128,)
    sg = sp.spectrogram(data,multitaper=False,mean_normalise=False)
    s = Segment(data,sg,sp,fs,50) #(data,sg,sp,fs,minSegment,window_width=256,incr=128)
    segs1 = s.segmentByFIR(1.0)
    segs2 = s.medianClip()
    print "1"
    print segs1
    print "2"
    print segs2
    segs = s.mergeSegments(segs1,segs2)
    print "==="
    print segs

def testMergeSeg2():
    segs1 = [[0.1, 0.3], [0.6,0.8],[1.1,1.3], [1.5, 1.7]]
    segs2 = [[0.05,0.2], [0.65,0.75], [0.7,0.9], [1.05,1.75]]
    print "1"
    print segs1
    print "2"
    print segs2
    s = Segment(None, None, None, None, 50)
    segs = s.mergeSegments(segs1,segs2)
    print "==="
    print segs
    print "[[0.05, 0.3], [0.6,0.9], [1.05, 1.75]]"

def convertAmpltoSpec(x,fs,inc):
    """ Unit conversion """
    return x*fs/inc

def showSegs():
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    import wavio
    import WaveletSegment

    wavobj = wavio.read('Sound Files/tril1.wav')
    #wavobj = wavio.read('Sound Files/010816_202935_p1.wav')
    #wavobj = wavio.read('Sound Files/20170515_223004 piping.wav')
    #wavobj = wavio.read('Sound Files/kiwi_1min.wav')
    fs = wavobj.rate
    data = wavobj.data[:20*fs]

    if data.dtype is not 'float':
        data = data.astype('float') # / 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    # Is this worth doing?
    # import sys
    # sys.path.append("test")
    # import audio_tools
    # newdata, newsg, E = audio_tools.time_attack_agc(data,fs)
    # newsg = np.real(newsg)

    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128)
    sg = sp.spectrogram(data,fs,multitaper=False)
    s = Segment(data,sg,sp,fs,50)

    # FIR: threshold doesn't matter much, but low is better (0.01).
    # Amplitude: not great, will have to work on width and abs if want to use it (threshold about 0.6)
    # Power: OK, but threshold matters (0.5)
    # Median clipping: OK, threshold of 3 fine.
    # Onsets: Threshold of 4.0 was fine, lower not. Still no offsets!
    # Yin: Threshold 0.9 is pretty good
    # Energy: Not great, but thr 1.0

    # Harma: some errors?

    s1, samp = s.segmentByFIR(0.1)
    s1 = s.checkSegmentLength(s1)
    s2, x = s.segmentByFIR(1.0)
    s2 = s.checkSegmentLength(s2)
    s3 = s.checkSegmentLength(s.medianClip(3.0))
    s4, p, t = s.yin(100, thr=0.9, returnSegs=True)
    s4 = s.checkSegmentLength(s4)
    s5 = s.mergeSegments(s1,s3)
    s6 = s.mergeSegments(s1,s4)
    s7 = s.mergeSegments2(s1,s3)
    #s7 = s.mergeSegments(s1,s.mergeSegments(s3,s4))

    #s4, samp = s.segmentByFIR(0.4)
    #s4 = s.checkSegmentLength(s4)
    #s2 = s.segmentByAmplitude1(0.6)
    #s5 = s.checkSegmentLength(s.segmentByPower(0.3))
    #s6, samp = s.segmentByFIR(0.6)
    #s6 = s.checkSegmentLength(s6)
    #s7 = []
    #s5 = s.onsets(3.0)
    #s6 = s.segmentByEnergy(1.0,500)

    import pylab as pl
    pl.ion()
    pl.plot(samp)
    pl.show()

    print np.mean(samp), np.std(samp)

    #s5 = s.Harma(5.0,0.8)
    #s4 = s.Harma(10.0,0.8)
    #s7 = s.Harma(15.0,0.8)

    #s2 = s.segmentByAmplitude1(0.7)
    #s3 = s.segmentByPower(1.)
    #s4 = s.medianClip(3.0)
    #s5 = s.onsets(3.0)
    #s6, p, t = s.yin(100,thr=0.5,returnSegs=True)
    #s7 = s.Harma(10.0,0.8)

    app = QtGui.QApplication([])

    mw = QtGui.QMainWindow()
    mw.show()
    mw.resize(800, 600)

    win = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(win)
    vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    im1 = pg.ImageItem(enableMouse=False)
    vb1.addItem(im1)
    im1.setImage(10.*np.log10(sg))

    vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=1, col=0)
    im2 = pg.ImageItem(enableMouse=False)
    vb2.addItem(im2)
    im2.setImage(10.*np.log10(sg))

    vb3 = win.addViewBox(enableMouse=False, enableMenu=False, row=2, col=0)
    im3 = pg.ImageItem(enableMouse=False)
    vb3.addItem(im3)
    im3.setImage(10.*np.log10(sg))

    vb4 = win.addViewBox(enableMouse=False, enableMenu=False, row=3, col=0)
    im4 = pg.ImageItem(enableMouse=False)
    vb4.addItem(im4)
    im4.setImage(10.*np.log10(sg))

    vb5 = win.addViewBox(enableMouse=False, enableMenu=False, row=4, col=0)
    im5 = pg.ImageItem(enableMouse=False)
    vb5.addItem(im5)
    im5.setImage(10.*np.log10(sg))

    vb6 = win.addViewBox(enableMouse=False, enableMenu=False, row=5, col=0)
    im6 = pg.ImageItem(enableMouse=False)
    vb6.addItem(im6)
    im6.setImage(10.*np.log10(sg))

    vb7 = win.addViewBox(enableMouse=False, enableMenu=False, row=6, col=0)
    im7 = pg.ImageItem(enableMouse=False)
    vb7.addItem(im7)
    im7.setImage(10.*np.log10(sg))

    print s1
    for seg in s1:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb1.addItem(a, ignoreBounds=True)

    print s2
    for seg in s2:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb2.addItem(a, ignoreBounds=True)

    print s3
    for seg in s3:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb3.addItem(a, ignoreBounds=True)

    print s4
    for seg in s4:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb4.addItem(a, ignoreBounds=True)

    print s5
    for seg in s5:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb5.addItem(a, ignoreBounds=True)

    print s6
    for seg in s6:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb6.addItem(a, ignoreBounds=True)

    print s7
    for seg in s7:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb7.addItem(a, ignoreBounds=True)

    QtGui.QApplication.instance().exec_()

#showSegs()
#testsegFIR()

#testMergeSeg2()
# testsegFIR()

# A=[51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68, 69,  70,  72,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
#   98,  99, 100, 101, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 137, 140, 141, 142, 143, 144, 145, 146, 147,
#  148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161]
# [[0.408, 0.56], [0.664, 0.808], [0.888, 1.064], [1.12, 1.288]]

# B=self.identifySegments(A, minlength=10)

def test_segMerge():
    s = Segment(None, None, None, None, 5)
    seg = [[0.808, 1.48], [1.49, 2.22], [3.0, 3.02], [3.2,3.4], [3.5,4.1], [4.408, 5.12], [7.72, 8.36]]
    segs = s.checkSegmentLength(seg)
    print segs

#test_segMerge()