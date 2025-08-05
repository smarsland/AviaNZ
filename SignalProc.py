
# Version 3.4 18/12/24
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

# SignalProc.py
# This file holds signal processing functions that don't use the full spectrogram or audio data
import scipy.signal as signal
import pyfftw as fft
#import scipy.fftpack as fft
import numpy as np
import copy
import Spectrogram

def ButterworthBandpass(data,sampleRate,low=0,high=None,band=0.005):
    """ Basic IIR bandpass filter.
        Identifies order of filter, max 10. If single-stage polynomial is unstable,
        switches to order 30, second-order filter.
        Args:
        1-2. data and sample rate.
        3-4. Low and high pass frequencies in Hz
        5. difference between stopband and passband, in fraction of Nyquist.
        Filter will lose no more than 3 dB in freqs [low,high], and attenuate
        at least 40 dB outside [low-band*Fn, high+band*Fn].

        Does double-pass filtering - slower, but keeps original phase.
    """

    if data is None:
        print("No data given")
        return
    if sampleRate is None:
        print("No sample rate given")
        return data
    nyquist = sampleRate/2

    if high is None:
        high = nyquist
    low = max(low,0)
    high = min(high,nyquist)

    # convert freqs to fractions of Nyquist:
    lowPass = low/nyquist
    highPass = high/nyquist
    lowStop = lowPass-band
    highStop = highPass+band
    # safety checks for values near edges
    if lowStop<=0:
        lowStop = lowPass/2
    if highStop>=1:
        highStop = (1+highPass)/2

    if lowPass == 0 and highPass == 1:
        print("No filter needed!")
        return data
    elif lowPass == 0:
        # Low pass
        # calculate the best order
        order,wN = signal.buttord(highPass, highStop, 3, 40)
        if order>10:
            order=10
        b, a = signal.butter(order,wN, btype='lowpass')
    elif highPass == 1:
        # High pass
        # calculate the best order
        order,wN = signal.buttord(lowPass, lowStop, 3, 40)
        if order>10:
            order=10
        b, a = signal.butter(order,wN, btype='highpass')
    else:
        # Band pass
        # calculate the best order
        order,wN = signal.buttord([lowPass, highPass], [lowStop, highStop], 3, 40)
        if order>10:
            order=10
        b, a = signal.butter(order,wN, btype='bandpass')

    # check if filter is stable
    filterUnstable = np.any(np.abs(np.roots(a))>1)
    if filterUnstable:
        # redesign to SOS and filter.
        # uses order=30 because why not
        print("single-stage filter unstable, switching to SOS filtering")
        if lowPass == 0:
            sos = signal.butter(30, wN, btype='lowpass', output='sos')
        elif highPass == 1:
            sos = signal.butter(30, wN, btype='highpass', output='sos')
        else:
            sos = signal.butter(30, wN, btype='bandpass', output='sos')

        # do the actual filtering
        data = signal.sosfiltfilt(sos, data)
    else:
        # do the actual filtering
        data = signal.filtfilt(b, a, data)

    return data

def FastButterworthBandpass(data,low=0,high=None):
    """ Basic IIR bandpass filter.
        Streamlined to be fast - for use in antialiasing etc.
        Tries to construct a filter of order 7, with critical bands at +-0.002 Fn.
        This corresponds to +- 16 Hz or so.
        If single-stage polynomial is unstable,
        switches to order 30, second-order filter.
        Args:
        1-2. data and sample rate.
        3-4. Low and high pass frequencies in fraction of Nyquist

        Does single-pass filtering, so does not retain phase.
    """

    if data is None:
        print("No data given")
        return

    # convert freqs to fractions of Nyquist:
    lowPass = max(low-0.002, 0)
    highPass = min(high+0.002, 1)

    if lowPass == 0 and highPass == 1:
        print("No filter needed!")
        return data
    elif lowPass == 0:
        # Low pass
        b, a = signal.butter(7, highPass, btype='lowpass')
    elif highPass == 1:
        # High pass
        b, a = signal.butter(7, lowPass, btype='highpass')
    else:
        # Band pass
        b, a = signal.butter(7, [lowPass, highPass], btype='bandpass')

    # check if filter is stable
    filterUnstable = True
    try:
        filterUnstable = np.any(np.abs(np.roots(a))>1)
    except Exception as e:
        print("Warning:", e)
        filterUnstable = True
    if filterUnstable:
        # redesign to SOS and filter.
        # uses order=30 because why not
        print("single-stage filter unstable, switching to SOS filtering")
        if lowPass == 0:
            sos = signal.butter(30, highPass, btype='lowpass', output='sos')
        elif highPass == 1:
            sos = signal.butter(30, lowPass, btype='highpass', output='sos')
        else:
            sos = signal.butter(30, [lowPass, highPass], btype='bandpass', output='sos')

        # do the actual filtering
        data = signal.sosfilt(sos, data)
    else:
        data = signal.lfilter(b, a, data)

    return data

def bandpassFilter(data,sampleRate,start=0,end=-1):
    """ FIR bandpass filter
    128 taps, Hamming window, very basic.
    """

    if data is None:
        print("No data given")
        return
    if sampleRate is None:
        print("No sample rate given")
        return data
    if end==-1 or end is None:
        end = sampleRate/2

    start = max(start,0)
    end = min(end,sampleRate/2)

    if start == 0 and end == sampleRate/2:
        print("No filter needed!")
        return data

    nyquist = sampleRate/2
    ntaps = 129

    if start == 0:
        # Low pass
        taps = signal.firwin(ntaps, cutoff=[end / nyquist], window=('hamming'), pass_zero=True)
    elif end == sampleRate/2:
        # High pass
        taps = signal.firwin(ntaps, cutoff=[start / nyquist], window=('hamming'), pass_zero=False)
    else:
        # Bandpass
        taps = signal.firwin(ntaps, cutoff=[start / nyquist, end / nyquist], window=('hamming'), pass_zero=False)
    #print("Taps:", taps)
    #ntaps, beta = signal.kaiserord(ripple_db, width)
    #taps = signal.firwin(ntaps,cutoff = [500/nyquist,8000/nyquist], window=('kaiser', beta),pass_zero=False)
    return signal.lfilter(taps, 1.0, data)

# TODO: Here or in spectrogram? Needs some work either way
# The next functions perform spectrogram inversion
def invertSpectrogram(sg,window_width=256,incr=64,nits=10, window='Hamming'):
    # Assumes that this is the plain (not power) spectrogram
    sp = Spectrogram.Spectrogram()
    # Assume the spectrogram is a bmp, so it is real (no phase) and one-sided, make it two-sided
    #sg = np.concatenate([sg, sg[:, ::-1]], axis=1)
    print(np.shape(sg))

    current_sg = copy.deepcopy(sg)
    for i in range(nits):
        new_wave = inversion_iteration(current_sg, incr, calculate_offset=True, iteration=i, window=window)
        sp.setData(new_wave)
        new_sg = sp.spectrogram(window_width, incr, onesided=False,need_even=False, complex_values=True, window=window)
        if new_sg.shape[1] != sg.shape[1]:
            new_sg = new_sg[:,sg.shape[1]]
        new_phase = new_sg / np.maximum(np.max(sg)/1E8, np.abs(new_sg))
        print(np.shape(new_phase),np.shape(new_sg),np.shape(sg))
        current_sg = sg * new_phase
        #print(np.max(current_sg))

    #new_wave = inversion_iteration(current_sg, incr, calculate_offset=True, iteration=nits, window=window)
    return new_wave

def inversion_iteration(sg, incr, calculate_offset=True, iteration = 0, window='Hamming'):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    windowSize = int(np.shape(sg)[1] // 2)
    wave = np.zeros(((np.shape(sg)[0]) * incr + windowSize - 1),dtype='float64')
    # Getting overflow warnings with 32 bit...
    #wave = wave.astype('float64')
    total_windowing_sum = np.zeros(((np.shape(sg)[0]) * incr + windowSize - 1))
    #Virginia: adding different windows

    
   # Set of window options
    if window=='Hann':
        # This is the Hann window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(windowSize) / (windowSize - 1)))
    elif window=='Parzen':
        # Parzen (window_width even)
        n = np.arange(windowSize) - 0.5*windowSize
        window = np.where(np.abs(n)<0.25*windowSize,1 - 6*(n/(0.5*windowSize))**2*(1-np.abs(n)/(0.5*windowSize)), 2*(1-np.abs(n)/(0.5*windowSize))**3)
    elif window=='Welch':
        # Welch
        window = 1.0 - ((np.arange(windowSize) - 0.5*(windowSize-1))/(0.5*(windowSize-1)))**2
    elif window=='Hamming':
        # Hamming
        alpha = 0.54
        beta = 1.-alpha
        window = alpha - beta*np.cos(2 * np.pi * np.arange(windowSize) / (windowSize - 1))
    elif window=='Blackman':
        # Blackman
        alpha = 0.16
        a0 = 0.5*(1-alpha)
        a1 = 0.5
        a2 = 0.5*alpha
        window = a0 - a1*np.cos(2 * np.pi * np.arange(windowSize) / (windowSize - 1)) + a2*np.cos(4 * np.pi * np.arange(windowSize) / (windowSize - 1))
    elif window=='BlackmanHarris':
        # Blackman-Harris
        a0 = 0.358375
        a1 = 0.48829
        a2 = 0.14128
        a3 = 0.01168
        window = a0 - a1*np.cos(2 * np.pi * np.arange(windowSize) / (windowSize - 1)) + a2*np.cos(4 * np.pi * np.arange(windowSize) / (windowSize - 1)) - a3*np.cos(6 * np.pi * np.arange(windowSize) / (windowSize - 1))
    elif window=='Ones':
        window = np.ones(windowSize)
    else:
        print("Unknown window, using Hann")
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(windowSize) / (windowSize - 1)))

    fft_start = int(windowSize // 2) -1
    fft_end = fft_start + windowSize 
    for i in range(sg.shape[0]):
        wave_start = incr * i
        wave_end = wave_start + windowSize 
        if np.iscomplex(sg).any():
            sg_col = sg[i,:]
        else:
            print("Treating as complex")
            sg_col = sg[i,:].real + 0j
        wave_est = np.real(fft.interfaces.scipy_fft.fftshift(fft.interfaces.scipy_fft.ifft(sg_col)))

        if calculate_offset and i > 0 and iteration==0:
            offset_size = windowSize - incr 
            cor = fast_xcorr(wave[wave_start:wave_start+offset_size],wave_est[fft_start:fft_start+offset_size])
            ind = np.argmax(cor[offset_size//2:-offset_size//2])+offset_size//2
            bestOffset = ind-offset_size
            print("Offset: ",bestOffset)
            #print(wave_start, wave_end, len(wave), est_start, offset_size, bestOffset)
            #offset = xcorr_offset(wave[wave_start:wave_start + offset_size], wave_est[est_start:est_start + offset_size])
            #print("New offset: ",offset)
        else:
            bestOffset = 0
        #if est_end-offset >= windowSize:
            #offset+=(est_end-offset-windowSize)
            #wave_end-=(est_end-offset-windowSize)
        wave[wave_start:wave_end] += wave_est[fft_start - bestOffset:fft_end - bestOffset]
        total_windowing_sum[wave_start:wave_end] += window #**2 #Virginia: needed square
    inds = np.where(total_windowing_sum!=0)
    wave = np.real(wave[inds]) / total_windowing_sum[inds]
    return wave

def xcorr_offset(x1, x2):
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2
    corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    return corrs.argmax() - len(x1)

def fast_xcorr(x1,x2):
    X1 = fft.interfaces.scipy_fft.fft(np.hstack((x1, 0*x1)))
    X2 = fft.interfaces.scipy_fft.fft(np.hstack((x2, 0*x2)))
    y = fft.interfaces.scipy_fft.fftshift(fft.interfaces.scipy_fft.ifft(X1*np.conj(X2)))
    return y[1:]

def medianFilter(data,width=11):
    # Median Filtering
    # Uses smaller width windows at edges to remove edge effects
    # TODO: Use abs rather than pure median?
    if data is None:
        print("No data")
        return
    mData = np.zeros(len(data))
    for i in range(width,len(data)-width):
        mData[i] = np.median(data[i-width:i+width])
    for i in range(len(data)):
        wid = min(i,len(data)-i,width)
        mData[i] = np.median(data[i - wid:i + wid])

    return mData

def wsola(x, s, win_type='hann', win_size=1024, syn_hop_size=512, tolerance=512):
    from scipy.interpolate import interp1d
    """Modify length of the audio sequence using WSOLA algorithm.
    This implementation is largely from pytsmod

    Parameters
    ----------

    start, stop : the part of the sound to play

    s : number > 0 [scalar] or numpy.ndarray [shape=(2, num_points)]
        the time stretching factor. Either a constant value (alpha)
        or an 2 x n array of anchor points which contains the sample points
        of the input signal in the first row
        and the sample points of the output signal in the second row.
    win_type : str
            type of the window function. hann and sin are available.
    win_size : int > 0 [scalar]
            size of the window function.
    syn_hop_size : int > 0 [scalar]
            hop size of the synthesis window.
            Usually half of the window size.
    tolerance : int >= 0 [scalar]
                number of samples the window positions
                in the input signal may be shifted
                to avoid phase discontinuities when overlap-adding them
                to form the output signal (given in samples).

    Returns
    -------

    y : numpy.ndarray [shape=(channel, num_samples) or (num_samples)]
        the modified output audio sequence.
    """

    x = np.expand_dims(x, 0)
    anc_points = np.array([[0, np.shape(x)[1] - 1], [0, np.ceil(s * np.shape(x)[1]) - 1]])
    #anc_points = np.array([[0, np.shape(x)[1] - 1], [0, np.ceil(s * np.shape(x)[1]) - 1]])
    n_chan = x.shape[0]
    output_length = int(anc_points[-1, -1]) + 1

    win = np.hanning(win_size)

    sw_pos = np.arange(0, output_length + win_size // 2, syn_hop_size)
    ana_interpolated = interp1d(anc_points[1, :], anc_points[0, :],
                                fill_value='extrapolate')
    aw_pos = np.round(ana_interpolated(sw_pos)).astype(int)
    ana_hop = np.insert(aw_pos[1:] - aw_pos[0: -1], 0, 0)
    
    y = np.zeros((n_chan, output_length))

    min_fac = np.min(syn_hop_size / ana_hop[1:])

    # padding the input audio sequence.
    left_pad = int(win_size // 2 + tolerance)
    right_pad = int(np.ceil(1 / min_fac) * win_size + tolerance)
    x_padded = np.pad(x, ((0, 0), (left_pad, right_pad)), 'constant')

    aw_pos = aw_pos + tolerance

    # Applying WSOLA to each channels
    for c, x_chan in enumerate(x_padded):
        y_chan = np.zeros(output_length + 2 * win_size)
        ow = np.zeros(output_length + 2 * win_size)

        delta = 0

        for i in range(len(aw_pos) - 1):
            x_adj = x_chan[aw_pos[i] + delta: aw_pos[i] + win_size + delta]
            y_chan[sw_pos[i]: sw_pos[i] + win_size] += x_adj * win
            ow[sw_pos[i]: sw_pos[i] + win_size] += win

            nat_prog = x_chan[aw_pos[i] + delta + syn_hop_size:
                            aw_pos[i] + delta + syn_hop_size + win_size]

            next_aw_range = np.arange(aw_pos[i+1] - tolerance,
                                    aw_pos[i+1] + win_size + tolerance)

            x_next = x_chan[next_aw_range]

            cross_corr = np.correlate(nat_prog, x_next)
            max_index = np.argmax(cross_corr)

            delta = tolerance - max_index

        # Calculate last frame
        x_adj = x_chan[aw_pos[-1] + delta: aw_pos[-1] + win_size + delta]
        y_chan[sw_pos[-1]: sw_pos[-1] + win_size] += x_adj * win
        ow[sw_pos[-1]: sw_pos[-1] + win_size] += + win

        ow[ow < 1e-3] = 1

        y_chan = y_chan / ow
        y_chan = y_chan[win_size // 2:]
        y_chan = y_chan[: output_length]

        y[c, :] = np.int_(y_chan)

    return y.squeeze()

def impMask(data,sampleRate,engp=90, fp=0.75):
    """
    Impulse mask
    :param engp: energy percentile (for rows of the spectrogram)
    :param fp: frequency proportion to consider it as an impulse (cols of the spectrogram)
    :return: audiodata
    """
    print('Impulse masking...')
    imps = impulse_cal(data,sampleRate, engp=engp, fp=fp)
    print('Samples to mask: ', len(data) - np.sum(imps))
    # Mask only the affected samples
    return np.multiply(data, imps)

def impulse_cal(data,sampleRate, engp=90, fp=0.75, blocksize=10):
    """
    Find sections where impulse sounds occur e.g. clicks
    window  -   window length (no overlap)
    engp    -   energy percentile (thr), the percentile of energy to inform that a section got high energy across
                frequency bands
    fp      -   frequency percentage (thr), the percentage of frequency bands to have high energy to mark a section
                as having impulse noise
    blocksize - max number of consecutive blocks, 10 consecutive blocks (~1/25 sec) is a good value, to not to mask
                very close-range calls
    :return: a binary list of length len(data) indicating presence of impulsive noise (0) otherwise (1)
    """
    # for impulse masking
    from itertools import chain, repeat

    # Calculate window length
    w1 = np.floor(sampleRate/250)      # Window length of 1/250 sec selected experimentally
    arr = [2 ** i for i in range(5, 11)]
    pos = np.abs(arr - w1).argmin()
    window = arr[pos]

    sp = Spectrogram.Spectrogram(window, window)     # No overlap
    sp.data = data
    sp.audioFormat.setSampleRate(sampleRate)
    sg = sp.spectrogram()

    # For each frq band get sections where energy exceeds some (90%) percentile, engp
    # and generate a binary spectrogram
    sgb = np.zeros((np.shape(sg)))
    ep = np.percentile(sg, engp, axis=0)    # note thr - 90% for energy percentile
    for y in range(np.shape(sg)[1]):
        ey = sg[:, y]
        sgb[np.where(ey > ep[y]), y] = 1

    # If lots of frq bands got 1 then predict a click
    # 1 - presence of impulse noise, 0 - otherwise here
    impulse = np.where(np.count_nonzero(sgb, axis=1) > np.shape(sgb)[1] * fp, 1, 0)     # Note thr fp

    # When an impulsive noise detected, it's better to check neighbours to make sure its not a bird call
    # very close to the microphone.
    imp_inds = np.where(impulse > 0)[0].tolist()
    imp = countConsecutive(imp_inds, len(impulse))

    impulse = []
    for item in imp:
        if item > blocksize or item == 0:        # Note threshold - blocksize, 10 consecutive blocks ~1/25 sec
            impulse.append(1)
        else:
            impulse.append(0)

    impulse = list(chain.from_iterable(repeat(e, window) for e in impulse))  # Make it same length as self.audioData

    if len(impulse) > len(data):      # Sanity check
        impulse = impulse[0:len(data)]
    elif len(impulse) < len(data):
        gap = len(data) - len(impulse)
        impulse = np.pad(impulse, (0, gap), 'constant')

    return impulse

def countConsecutive(nums, length):
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    edges = list(zip(edges, edges))
    edges_reps = [item[1] - item[0] + 1 for item in edges]
    res = np.zeros((length)).tolist()
    t = 0
    for item in edges:
        for i in range(item[0], item[1]+1):
            res[i] = edges_reps[t]
        t += 1
    return res

