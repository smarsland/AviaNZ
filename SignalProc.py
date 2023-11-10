
# SignalProc.py
# This file holds signal processing functions that don't use the full spectrogram or audio data

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
    print(start,end,sampleRate,len(data))

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

# The next functions perform spectrogram inversion
def invertSpectrogram(sg,window_width=256,incr=64,nits=10, window='Hann'):
    # Assumes that this is the plain (not power) spectrogram
    # Make the spectrogram two-sided and make the values small
    sg = np.concatenate([sg, sg[:, ::-1]], axis=1)

    sg_best = copy.deepcopy(sg)
    for i in range(nits):
        invertedSgram = self.inversion_iteration(sg_best, incr, calculate_offset=True,set_zero_phase=(i==0), window=window)
        self.setData(invertedSgram)
        est = self.spectrogram(window_width, incr, onesided=False,need_even=True, window=window)
        phase = est / np.maximum(np.max(sg)/1E8, np.abs(est))
        sg_best = sg * phase[:len(sg)]
    invertedSgram = self.inversion_iteration(sg_best, incr, calculate_offset=True,set_zero_phase=False, window=window)
    return np.real(invertedSgram)

def inversion_iteration(sg, incr, calculate_offset=True, set_zero_phase=True, window='Hann'):
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
    size = int(np.shape(sg)[1] // 2)
    wave = np.zeros((np.shape(sg)[0] * incr + size),dtype='float64')
    # Getting overflow warnings with 32 bit...
    #wave = wave.astype('float64')
    total_windowing_sum = np.zeros((np.shape(sg)[0] * incr + size))
    #Virginia: adding different windows

    
   # Set of window options
    if window=='Hann':
        # This is the Hann window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))
    elif window=='Parzen':
        # Parzen (window_width even)
        n = np.arange(size) - 0.5*size
        window = np.where(np.abs(n)<0.25*size,1 - 6*(n/(0.5*size))**2*(1-np.abs(n)/(0.5*size)), 2*(1-np.abs(n)/(0.5*size))**3)
    elif window=='Welch':
        # Welch
        window = 1.0 - ((np.arange(size) - 0.5*(size-1))/(0.5*(size-1)))**2
    elif window=='Hamming':
        # Hamming
        alpha = 0.54
        beta = 1.-alpha
        window = alpha - beta*np.cos(2 * np.pi * np.arange(size) / (size - 1))
    elif window=='Blackman':
        # Blackman
        alpha = 0.16
        a0 = 0.5*(1-alpha)
        a1 = 0.5
        a2 = 0.5*alpha
        window = a0 - a1*np.cos(2 * np.pi * np.arange(size) / (size - 1)) + a2*np.cos(4 * np.pi * np.arange(size) / (size - 1))
    elif window=='BlackmanHarris':
        # Blackman-Harris
        a0 = 0.358375
        a1 = 0.48829
        a2 = 0.14128
        a3 = 0.01168
        window = a0 - a1*np.cos(2 * np.pi * np.arange(size) / (size - 1)) + a2*np.cos(4 * np.pi * np.arange(size) / (size - 1)) - a3*np.cos(6 * np.pi * np.arange(size) / (size - 1))
    elif window=='Ones':
        window = np.ones(size)
    else:
        print("Unknown window, using Hann")
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(sg.shape[0]):
        wave_start = int(incr * i)
        wave_end = wave_start + size
        if set_zero_phase:
            spectral_slice = sg[i].real + 0j
        else:
            # already complex
            spectral_slice = sg[i]

        wave_est = np.real(fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - incr
            print("Offset: ",offset_size)
            if offset_size <= 0:
                print("WARNING: Large step size >50\% detected! " "This code works best with high overlap - try " "with 75% or greater")
                offset_size = incr
            print(wave_start, est_start, offset_size, len(wave), len(wave_est), est_start+offset_size<len(wave_est))
            offset = self.xcorr_offset(wave[wave_start:wave_start + offset_size], wave_est[est_start:est_start + offset_size])
            print("New offset: ",offset)
        else:
            offset = 0
        print(wave_end-wave_start, len(window), est_end-est_start,len(wave_est), len(wave),est_start, offset)
        if est_end-offset >= size:
            offset+=(est_end-offset-size)
            wave_end-=(est_end-offset-size)
        wave[wave_start:wave_end] += window * wave_est[est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += window**2 #Virginia: needed square
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
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

