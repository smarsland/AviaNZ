import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal

sys.path.insert(0, os.path.dirname(__file__))
from src.data.spectrogram import Spectrogram

# ═══════════════════════════════════════════════════════════════════════════════
# USER-TUNEABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
AUDIO_FILE = "Sound Files/kiwi_1min.wav"

CENTER_FREQ_HZ    = 700    # centre of the passband (Hz)
BANDWIDTH_HZ      = 400    # full width of the passband (Hz)
NUM_FILTER_TAPS   = 1001   # FIR filter length — must be odd; more taps = sharper roll-off

WINDOW_WIDTH      = 512    # STFT window width in samples
INCR              = 128    # STFT hop size in samples

ENVELOPE_SMOOTH_MS = 20    # envelope smoothing window (ms)
# ═══════════════════════════════════════════════════════════════════════════════


def bandpass_fir(signal_in, low_hz, high_hz, fs, num_taps):
    """
    Bandpass filter built from scratch using the windowed-sinc method.

    An ideal lowpass filter with cut-off frequency fc has the impulse response:

        h[n] = sin(2π fc/fs · n) / (π n)     (the sinc function, centred at n=0)

    This is the inverse Fourier transform of a rectangular window in the
    frequency domain that is 1 below fc and 0 above it.  The sinc is infinite
    in time, so we truncate it to `num_taps` samples centred on n=0.

    A bandpass filter is the DIFFERENCE of two such lowpass filters:
        h_bp[n] = h_highcut[n] - h_lowcut[n]
    Subtracting removes all frequencies below low_hz (already blocked by the
    highcut filter) and all frequencies above high_hz (blocked by both), leaving
    only the band in between.

    Truncating the sinc to a finite window introduces Gibbs ringing at the
    cut-off edges.  We suppress that by multiplying by a Hamming window, which
    tapers the coefficients smoothly to zero at both ends.

    Finally we convolve the filter coefficients with the input signal using
    numpy's overlap-add FFT convolution (mode='same' keeps the output the same
    length as the input).  Because we apply the filter only once (not
    forward-backward), we correct the resulting time delay by shifting the
    output back by (num_taps - 1) // 2 samples.
    """
    assert num_taps % 2 == 1, "num_taps must be odd"
    M = num_taps - 1          # filter order
    half = M // 2             # index of the centre tap (n = 0)
    n = np.arange(num_taps) - half   # n runs from -half to +half

    # ── Ideal sinc lowpass at high_hz ──────────────────────────────────────────
    fc_high = high_hz / fs    # normalised cut-off (cycles per sample)
    h_high = np.where(
        n == 0,
        2 * fc_high,                                  # limit of sinc at n=0
        np.sin(2 * np.pi * fc_high * n) / (np.pi * n)
    )

    # ── Ideal sinc lowpass at low_hz ───────────────────────────────────────────
    fc_low = low_hz / fs
    h_low = np.where(
        n == 0,
        2 * fc_low,
        np.sin(2 * np.pi * fc_low * n) / (np.pi * n)
    )

    # ── Bandpass = difference of the two lowpass impulse responses ─────────────
    h_bp = h_high - h_low

    # ── Hamming window: taper coefficients to zero at both ends ────────────────
    h_bp *= np.hamming(num_taps)

    # ── Convolve with the signal; correct for the half-tap time delay ──────────
    out = np.convolve(signal_in, h_bp, mode='full')
    # 'full' output has length len(signal) + num_taps - 1;
    # trim back to the original length, accounting for the delay
    delay = half
    out = out[delay: delay + len(signal_in)]
    return out


def compute_envelope(filtered, fs, smooth_ms):
    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)
    win_len = max(3, int(smooth_ms * fs / 1000))
    if win_len % 2 == 0:
        win_len += 1
    w = np.hamming(win_len)
    w /= w.sum()
    return np.convolve(envelope, w, mode='same')


def main():
    audio_path = os.path.join(os.path.dirname(__file__), AUDIO_FILE)
    if not os.path.isfile(audio_path):
        print(f"ERROR: file not found: {audio_path}")
        sys.exit(1)

    # ── 1. Load audio via existing Spectrogram class ───────────────────────────
    sp = Spectrogram(window_width=WINDOW_WIDTH, incr=INCR)
    sp.readSoundFile(audio_path)
    fs        = sp.audio_data.sample_rate
    raw_audio = sp.audio_data.data.astype(np.float64)
    duration  = len(raw_audio) / fs
    print(f"Loaded {AUDIO_FILE}  |  {fs} Hz  |  {duration:.1f} s")

    # ── 2. Spectrogram via existing code ───────────────────────────────────────
    sp.spectrogram(window_width=WINDOW_WIDTH, incr=INCR, window='Hann',
                   sgType='Standard', sgScale='Linear', mean_normalise=True)
    sg_log = sp.normalisedSpec(tr="Log")   # shape: (time_frames, freq_bins)

    n_frames = sg_log.shape[0]
    sg_times = np.arange(n_frames) * INCR / fs
    sg_freqs = np.linspace(0, fs / 2, sg_log.shape[1])

    # ── 3. Bandpass filter (from scratch) ─────────────────────────────────────
    low_cut  = max(CENTER_FREQ_HZ - BANDWIDTH_HZ / 2, 1.0)
    high_cut = min(CENTER_FREQ_HZ + BANDWIDTH_HZ / 2, fs / 2 - 1)
    print(f"Bandpass  {low_cut:.0f}–{high_cut:.0f} Hz  |  {NUM_FILTER_TAPS} taps")

    filtered = bandpass_fir(raw_audio, low_cut, high_cut, fs, NUM_FILTER_TAPS)

    # ── 4. Amplitude envelope ──────────────────────────────────────────────────
    envelope  = compute_envelope(filtered, fs, ENVELOPE_SMOOTH_MS)
    time_axis = np.arange(len(raw_audio)) / fs

    # ── 5. Plot ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.38)

    ax_sg = fig.add_subplot(gs[0])
    ax_sg.imshow(sg_log.T, origin='lower', aspect='auto', cmap='inferno',
                 extent=[sg_times[0], sg_times[-1], sg_freqs[0], sg_freqs[-1]])
    ax_sg.axhline(low_cut,  color='cyan', lw=1.0, ls='--', alpha=0.85,
                  label=f'Passband  {low_cut:.0f}–{high_cut:.0f} Hz')
    ax_sg.axhline(high_cut, color='cyan', lw=1.0, ls='--', alpha=0.85)
    ax_sg.set_xlabel("Time (s)")
    ax_sg.set_ylabel("Frequency (Hz)")
    ax_sg.set_title(f"Log-normalised spectrogram — {os.path.basename(AUDIO_FILE)}")
    ax_sg.legend(loc='upper right', fontsize=8)

    ax_bp = fig.add_subplot(gs[1])
    ax_bp.plot(time_axis, filtered,  color='steelblue', lw=0.4, alpha=0.4,
               label='Filtered signal')
    ax_bp.plot(time_axis,  envelope, color='orangered', lw=1.5,
               label=f'Envelope (smoothed {ENVELOPE_SMOOTH_MS} ms)')
    ax_bp.plot(time_axis, -envelope, color='orangered', lw=1.5)
    ax_bp.set_xlabel("Time (s)")
    ax_bp.set_ylabel("Amplitude")
    ax_bp.set_title(
        f"Windowed-sinc bandpass  |  centre {CENTER_FREQ_HZ} Hz, "
        f"width {BANDWIDTH_HZ} Hz  ({low_cut:.0f}–{high_cut:.0f} Hz)"
    )
    ax_bp.legend(loc='upper right', fontsize=8)
    ax_bp.set_xlim(sg_times[0], sg_times[-1])

    plt.savefig("bandpass_explore.png", dpi=150, bbox_inches='tight')
    print("Saved bandpass_explore.png")
    plt.show()


if __name__ == "__main__":
    main()
