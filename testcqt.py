
import wavio
import numpy as np
import pylab as pl

def nsgcwin(fmin, fmax, n_bins, fs, signal_len, gamma):
    """
    Nonstationary Gabor window calculation

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    # use a hanning window
    # no fractional shifts
    fftres = float(fs) / signal_len
    fmin = float(fmin)
    fmax = float(fmax)
    gamma = float(gamma)
    nyq = fs / 2.
    b = np.floor(n_bins * np.log2(fmax / fmin))
    fbas = fmin * 2 ** (np.arange(b + 1) / float(n_bins))
    Q = 2 ** (1. / n_bins) - 2 ** (-1. / n_bins)
    cqtbw = Q * fbas + gamma
    cqtbw = cqtbw.ravel()
    maxidx = np.where(fbas + cqtbw / 2. > nyq)[0]
    if len(maxidx) > 0:
        # replicate bug in MATLAB version...
        # or is it a feature
        if sum(maxidx) == 0:
            first = len(cqtbw) - 1
        else:
            first = maxidx[0]
        fbas = fbas[:first]
        cqtbw = cqtbw[:first]
    minidx = np.where(fbas - cqtbw / 2. < 0)[0]
    if len(minidx) > 0:
        fbas = fbas[minidx[-1]+1:]
        cqtbw = cqtbw[minidx[-1]+1:]

    fbas_len = len(fbas)
    fbas_new = np.zeros((2 * (len(fbas) + 1)))
    fbas_new[1:len(fbas) + 1] = fbas
    fbas = fbas_new
    fbas[fbas_len + 1] = nyq
    fbas[fbas_len + 2:] = fs - fbas[1:fbas_len + 1][::-1]
    bw = np.zeros_like(fbas)
    bw[0] = 2 * fmin
    bw[1:len(cqtbw) + 1] = cqtbw
    bw[len(cqtbw) + 1] = fbas[fbas_len + 2] - fbas[fbas_len]
    bw[-len(cqtbw):] = cqtbw[::-1]
    bw = bw / fftres
    fbas = fbas / fftres

    posit = np.zeros_like(fbas)
    posit[:fbas_len + 2] = np.floor(fbas[:fbas_len + 2])
    posit[fbas_len + 2:] = np.ceil(fbas[fbas_len + 2:])
    base_shift = -posit[-1] % signal_len
    shift = np.zeros_like(posit).astype("int32")
    shift[1:] = (posit[1:] - posit[:-1]).astype("int32")
    shift[0] = base_shift

    bw = np.round(bw)
    bwfac = 1
    M = bw
    M = np.int_(M)

    min_win = 4
    for ii in range(len(bw)):
        if bw[ii] < min_win:
            bw[ii] = min_win
            M[ii] = bw[ii]

    def _win(numel):
        if numel % 2 == 0:
            s1 = np.arange(0, .5, 1. / numel)
            if len(s1) != numel // 2:
                # edge case with small floating point numbers...
                s1 = s1[:-1]
            s2 = np.arange(-.5, 0, 1. / numel)
            if len(s2) != numel // 2:
                # edge case with small floating point numbers...
                s2 = s2[:-1]
            x = np.concatenate((s1, s2))
        else:
            s1 = np.arange(0, .5, 1. / numel)
            s2 = np.arange(-.5 + .5 / numel, 0, 1. / numel)
            if len(s2) != numel // 2:  # assume integer truncate 27 // 2 = 13
                s2 = s2[:-1]
            x = np.concatenate((s1, s2))
        assert len(x) == numel
        g = .5 + .5 * np.cos(2 * np.pi * x)
        return g

    multiscale = [_win(bi) for bi in bw]
    bw = bwfac * np.ceil(M / bwfac)

    for kk in [0, fbas_len + 1]:
        if M[kk] > M[kk + 1]:
            multiscale[kk] = np.ones(int(M[kk])).astype(multiscale[0].dtype)
            i1 = int(np.floor(M[kk] / 2) - np.floor(M[kk + 1] / 2))
            i2 = int(np.floor(M[kk] / 2) + np.ceil(M[kk + 1] / 2))
            # Very rarely, gets an off by 1 error? Seems to be at the end...
            # for now, slice
            multiscale[kk][i1:i2] = _win(M[kk + 1])
            multiscale[kk] = multiscale[kk] / np.sqrt(M[kk])
    return multiscale, shift, M


def nsgtf_real(X, multiscale, shift, window_lens):
    """
    Nonstationary Gabor Transform for real values

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    # This will break with multchannel input
    signal_len = len(X)
    N = len(shift)
    X_fft = np.fft.fft(X)

    fill = np.sum(shift) - signal_len
    if fill > 0:
        X_fft_tmp = np.zeros((signal_len + shift))
        X_fft_tmp[:len(X_fft)] = X_fft
        X_fft = X_fft_tmp
    posit = np.cumsum(shift) - shift[0]
    scale_lens = np.array([len(m) for m in multiscale])
    N = np.where(posit - np.floor(scale_lens) <= (signal_len + fill) / 2)[0][-1]
    c = []
    # c[0] is almost exact
    for ii in range(N):
        idx_l = np.arange(np.ceil(scale_lens[ii] / 2), scale_lens[ii])
        idx_r = np.arange(np.ceil(scale_lens[ii] / 2))
        idx = np.concatenate((idx_l, idx_r))
        idx = idx.astype("int32")
        subwin_range = posit[ii] + np.arange(-np.floor(scale_lens[ii] / 2),
                                             np.ceil(scale_lens[ii] / 2))
        if len(subwin_range) < np.shape(multiscale[ii][idx])[0]:
            subwin_range = posit[ii] + np.arange(-np.floor(scale_lens[ii] / 2),
                                             np.ceil(scale_lens[ii] / 2 )+1)
        elif len(subwin_range) > np.shape(multiscale[ii][idx])[0]:
            subwin_range = posit[ii] + np.arange(-np.floor(scale_lens[ii] / 2)+1,
                                             np.ceil(scale_lens[ii] / 2 ))

        win_range = subwin_range % (signal_len + fill)
        win_range = win_range.astype("int32")
        if window_lens[ii] < scale_lens[ii]:
            raise ValueError("Not handling 'not enough channels' case")
        else:
            temp = np.zeros((window_lens[ii],)).astype(X_fft.dtype)
            temp_idx_l = np.arange(len(temp) - np.floor(scale_lens[ii] / 2),
                                   len(temp))
            temp_idx_r = np.arange(np.ceil(scale_lens[ii] / 2))
            temp_idx = np.concatenate((temp_idx_l, temp_idx_r))
            temp_idx = temp_idx.astype("int32")
            temp[temp_idx] = X_fft[win_range] * multiscale[ii][idx]
            fs_new_bins = window_lens[ii]
            fk_bins = posit[ii]
            displace = fk_bins - np.floor(fk_bins / fs_new_bins) * fs_new_bins
            displace = displace.astype("int32")
            temp = np.roll(temp, displace)
        c.append(np.fft.ifft(temp))

    if 0:
        # cell2mat concatenation
        c = np.concatenate(c)
    return c


def nsdual(multiscale, shift, window_lens):
    """
    Calculation of nonstationary inverse gabor filters

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    N = len(shift)
    posit = np.cumsum(shift)
    seq_len = posit[-1]
    posit = posit - shift[0]

    diagonal = np.zeros((seq_len,))
    win_range = []

    for ii in range(N):
        filt_len = len(multiscale[ii])
        idx = np.arange(-np.floor(filt_len / 2), np.ceil(filt_len / 2))
        win_range.append((posit[ii] + idx) % seq_len)
        subdiag = window_lens[ii] * np.fft.fftshift(multiscale[ii]) ** 2
        ind = win_range[ii].astype(np.int)
        diagonal[ind] = diagonal[ind] + subdiag

    dual_multiscale = multiscale
    for ii in range(N):
        ind = win_range[ii].astype(np.int)
        dual_multiscale[ii] = np.fft.ifftshift(
            np.fft.fftshift(dual_multiscale[ii]) / diagonal[ind])
    return dual_multiscale


def nsgitf_real(c, c_dc, c_nyq, multiscale, shift):
    """
    Nonstationary Inverse Gabor Transform on real valued signal

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    c_l = []
    c_l.append(c_dc)
    c_l.extend([ci for ci in c])
    c_l.append(c_nyq)

    posit = np.cumsum(shift)
    seq_len = posit[-1]
    posit -= shift[0]
    out = np.zeros((seq_len,)).astype(c_l[1].dtype)

    for ii in range(len(c_l)):
        filt_len = len(multiscale[ii])
        win_range = posit[ii] + np.arange(-np.floor(filt_len / 2),
                                          np.ceil(filt_len / 2))
        win_range = (win_range % seq_len).astype(np.int)
        temp = np.fft.fft(c_l[ii]) * len(c_l[ii])

        fs_new_bins = len(c_l[ii])
        fk_bins = posit[ii]
        displace = int(fk_bins - np.floor(fk_bins / fs_new_bins) * fs_new_bins)
        temp = np.roll(temp, -displace)
        l = np.arange(len(temp) - np.floor(filt_len / 2), len(temp))
        r = np.arange(np.ceil(filt_len / 2))
        temp_idx = (np.concatenate((l, r)) % len(temp)).astype(np.int)
        temp = temp[temp_idx]
        lf = np.arange(filt_len - np.floor(filt_len / 2), filt_len)
        rf = np.arange(np.ceil(filt_len / 2))
        filt_idx = np.concatenate((lf, rf)).astype(np.int)
        m = multiscale[ii][filt_idx]
        out[win_range] = out[win_range] + m * temp

    nyq_bin = int(np.floor(seq_len // 2) + 1)
    out_idx = np.arange( nyq_bin - np.abs(1 - seq_len % 2) - 1, 0, -1).astype(np.int)
    out[nyq_bin:] = np.conj(out[out_idx])
    t_out = np.real(np.fft.ifft(out)).astype(np.float64)
    return t_out


def cqt(X, fs, n_bins=48, fmin=27.5, fmax="nyq", gamma=20):
    """
    Constant Q Transform

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    if fmax == "nyq":
        fmax = fs / 2.
    multiscale, shift, window_lens = nsgcwin(fmin, fmax, n_bins, fs,
                                             len(X), gamma)
    fbas = fs * np.cumsum(shift[1:]) / len(X)
    fbas = fbas[:len(window_lens) // 2 - 1]
    bins = window_lens.shape[0] // 2 - 1
    window_lens[1:bins + 1] = window_lens[bins + 2]
    window_lens[bins + 2:] = window_lens[1:bins + 1][::-1]
    norm = 2. * window_lens[:bins + 2] / float(len(X))
    norm = np.concatenate((norm, norm[1:-1][::-1]))
    multiscale = [norm[ii] * multiscale[ii] for ii in range(2 * (bins + 1))]

    c = nsgtf_real(X, multiscale, shift, window_lens)
    c_dc = c[0]
    c_nyq = c[-1]
    c_sub = c[1:-1]
    c = np.vstack(c_sub)
    return c, c_dc, c_nyq, multiscale, shift, window_lens


def icqt(X_cq, c_dc, c_nyq, multiscale, shift, window_lens):
    """
    Inverse constant Q Transform

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    new_multiscale = nsdual(multiscale, shift, window_lens)
    X = nsgitf_real(X_cq, c_dc, c_nyq, new_multiscale, shift)
    return X


def run_cqt_example():
    wavobj = wavio.read("/Users/marslast/Projects/AviaNZ/Sound Files/kiwi_1min.wav")
    d = np.squeeze(wavobj.data)

    # take only left channel
    #if np.shape(np.shape(d))[0] > 1:
        #d= d[:, 0]

    # force float type
    if d.dtype != 'float':
        d = d.astype('float')

    fs = wavobj.rate

    X = d[:,0]
    #X = d[:44100]
    X_cq, c_dc, c_nyq, multiscale, shift, window_lens = cqt(X, fs)
    X_r = icqt(X_cq, c_dc, c_nyq, multiscale, shift, window_lens)
    SNR = 20 * np.log10(np.linalg.norm(X - X_r) / np.linalg.norm(X))
    X_cq = X_cq[int(np.shape(X_cq)[0]/2.):,:]
    pl.ion()
    pl.figure()
    pl.imshow(np.flipud(10*np.log10(np.real(X_cq*np.conj(X_cq))))[:,10000:], cmap=pl.cm.gray_r)
    pl.figure()
    pl.plot(X_r)
    
run_cqt_example()
    #wavfile.write("cqt_original.wav", fs, soundsc(X))
    #wavfile.write("cqt_reconstruction.wav", fs, soundsc(X_r))

