import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal

sys.path.insert(0, os.path.dirname(__file__))
from src.data.spectrogram import Spectrogram

# ── Parameters ────────────────────────────────────────────────────────────────
AUDIO_FILE   = "Sound Files/kiwi_1min.wav"
WINDOW_WIDTH = 512
INCR         = 128

N_BANDS   = 100      # total number of bandpass bands
FREQ_LOW  = 200     # Hz — lower edge of lowest band
FREQ_HIGH = None    # Hz — upper edge of highest band; None = fs/2 - 50
# ─────────────────────────────────────────────────────────────────────────────


def make_bands(fs, n_bands, f_low, f_high=None):
    """Logarithmically spaced bands from f_low to f_high.

    Log spacing gives wide bands at low frequencies (high time resolution)
    and narrow bands at high frequencies (finer frequency detail), which
    matches both perceptual scales and kiwi's high-frequency specialisation.
    """
    if f_high is None:
        f_high = fs / 2 - 50
    centers   = np.logspace(np.log10(f_low), np.log10(f_high), n_bands)
    mid_edges = np.sqrt(centers[:-1] * centers[1:])   # geometric midpoints
    low_edges  = np.r_[f_low,      mid_edges]
    high_edges = np.r_[mid_edges,  f_high]
    return list(zip(low_edges, high_edges))


def bandpass_fir(sig, low_hz, high_hz, fs):
    """Windowed-sinc bandpass FIR; tap count auto-scaled to bandwidth."""
    bw      = high_hz - low_hz
    n_taps  = int(np.ceil(8.0 * fs / bw))
    n_taps += 1 - n_taps % 2       # ensure odd
    half    = (n_taps - 1) // 2
    n       = np.arange(n_taps) - half

    def lp_sinc(fc):
        return np.where(n == 0, 2 * fc,
                        np.sin(2 * np.pi * fc * n) / (np.pi * n))

    h   = (lp_sinc(high_hz / fs) - lp_sinc(low_hz / fs)) * np.hamming(n_taps)
    out = signal.fftconvolve(sig, h, mode='full')
    return out[half: half + len(sig)]


def build_bandpass_spectrogram(audio, fs, bands):
    """Filter into each band, compute envelope, resample to a shared time axis.

    Each band is smoothed and downsampled at a rate proportional to its
    bandwidth (wider band → smaller hop → finer time resolution), then
    interpolated to the common grid set by the widest band.
    Returns (arr, t_common) where arr has shape (n_bands, n_time).
    """
    max_bw   = max(h - l for l, h in bands)
    dt       = 1.0 / (4.0 * max_bw)
    n_common = int(len(audio) / fs / dt) + 1
    t_common = np.linspace(0, len(audio) / fs, n_common)

    rows = []
    for low, high in bands:
        bw      = high - low
        filt    = bandpass_fir(audio, low, high, fs)
        env     = np.abs(signal.hilbert(filt))
        win_len = max(3, (int(fs * 2.0 / bw) | 1))   # smooth ∝ 1/bw, kept odd
        w       = np.hamming(win_len)
        env     = np.convolve(env, w / w.sum(), mode='same')
        hop     = max(1, int(fs / (4.0 * bw)))
        env_ds  = env[::hop]
        t_band  = np.arange(len(env_ds)) * hop / fs
        rows.append(np.interp(t_common, t_band, env_ds))

    arr = np.array(rows, dtype=np.float32)
    return np.log1p(arr / (arr.max() + 1e-9)), t_common


def main():
    audio_path = os.path.join(os.path.dirname(__file__), AUDIO_FILE)
    if not os.path.isfile(audio_path):
        print(f"ERROR: file not found: {audio_path}")
        sys.exit(1)

    sp = Spectrogram(window_width=WINDOW_WIDTH, incr=INCR)
    sp.readSoundFile(audio_path)
    fs    = sp.audio_data.sample_rate
    audio = sp.audio_data.data.astype(np.float64)
    print(f"Loaded  {AUDIO_FILE}  |  {fs} Hz  |  {len(audio)/fs:.1f} s")

    sp.spectrogram(window_width=WINDOW_WIDTH, incr=INCR, window='Hann',
                   sgType='Standard', sgScale='Linear', mean_normalise=True)
    sg_log   = sp.normalisedSpec(tr="Log")
    sg_times = np.arange(sg_log.shape[0]) * INCR / fs
    sg_freqs = np.linspace(0, fs / 2, sg_log.shape[1])

    f_high      = FREQ_HIGH if FREQ_HIGH is not None else fs / 2 - 50
    valid_bands = make_bands(fs, N_BANDS, FREQ_LOW, f_high)
    print(f"Computing {len(valid_bands)}-band bandpass spectrogram  "
          f"({FREQ_LOW:.0f}–{f_high:.0f} Hz) …")
    bp_arr, t_bp = build_bandpass_spectrogram(audio, fs, valid_bands)

    freq_edges = np.array([valid_bands[0][0]] + [h for _, h in valid_bands])
    dt_bp      = t_bp[1] - t_bp[0]
    t_edges    = np.r_[t_bp - dt_bp / 2, t_bp[-1] + dt_bp / 2]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 1, hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(sg_log.T, origin='lower', aspect='auto', cmap='inferno',
               extent=[sg_times[0], sg_times[-1], sg_freqs[0], sg_freqs[-1]])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(f"STFT spectrogram  —  {os.path.basename(AUDIO_FILE)}")

    ax2 = fig.add_subplot(gs[1])
    ax2.pcolormesh(t_edges, freq_edges, bp_arr, cmap='inferno', shading='flat')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Multiresolution bandpass spectrogram")

    plt.savefig("bandpass_explore.png", dpi=150, bbox_inches='tight')
    print("Saved bandpass_explore.png")
    plt.show()


if __name__ == "__main__":
    main()
