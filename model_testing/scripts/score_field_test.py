#!/usr/bin/env python3
"""
Score a trained model against the hand-labelled field test set in
Sound Files/FinalTestData.

This mirrors the app's inference path exactly (src/core/nn_segmenter.py):
sliding non-overlapping time_bins windows, RMS audio normalisation, mel
spectrogram, optional fg/bg self-reconstruction, sigmoid + kbird max-k
normalisation, then per-class thresholds.

A model window scores a hit for a GT segment if it temporally overlaps the
segment and assigns that class a probability >= the operating threshold.

Usage:
    python3 scripts/score_field_test.py \
        model_testing/selected_models/regnet_combined_bgprob50

    # multiple models, flat 0.5 instead of tuned thresholds
    python3 scripts/score_field_test.py <model_dir> [<model_dir>...] --flat 0.5

    # restrict which GT species are scored
    python3 scripts/score_field_test.py <model_dir> --only chaffinch,blackbird
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.core.spectrogram import Spectrogram
from src.core.audio_data import AudioData
from src.models import inference, model_loader
from model_testing.src.data.normalizer import split_spectrogram_components
from model_testing.src.data.spectrogram_utils import highpass_audio

FIELD_DIR = REPO_ROOT / 'Sound Files' / 'FinalTestData'

# HandAnnotation label -> model class name(s). The hand labels use the app's
# display names; the model uses lowercase DOC-style names.
GT_TO_MODEL = {
    'blackbird': ['blackbird'],
    'Chaffinch': ['chaffinch'],
    'song thrush': ['song thrush'],
    'Bellbird/Tui': ['tui/bellbird'],
    'grey warbler': ['grey warbler'],
    'Silvereye': ['silvereye'],
    'Goldfinch': ['goldfinch'],
    'Fantail (spp)': ['fantail'],
    'Morepork': ['morepork'],
    'Cuckoo (Shining)': ['shining cuckoo'],
    'Kiwi (spp)': ['north island brown kiwi', 'southern brown kiwi',
                   'great spotted kiwi', 'little spotted kiwi', 'okarito brown kiwi'],
    'Wētā': ['tree weta'],
    'Kaka (spp)': ['kaka'],
    "Don't Know": [],
}


def load_field_annotations(field_dir):
    """Return {wav_basename: [(start, end, [gt_label, ...]), ...]}."""
    out = {}
    for wav in sorted(glob.glob(str(field_dir / '*.wav'))):
        name = os.path.basename(wav)
        data_file = field_dir / 'HandAnnotations' / f'{name}.data'
        if not data_file.exists():
            continue
        segs = json.load(open(data_file))[1:]
        out[name] = [(s[0], s[1], [lab['species'] for lab in s[4]]) for s in segs]
    return out


class FieldModel:
    """Load a model_testing model and run app-parity windowed inference."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        stem = self._find_stem()
        with open(self.model_dir / f'{stem}_config.json') as f:
            self.config = json.load(f)
        self.class_names = self.config['class_names']
        self.class_idx = {c: i for i, c in enumerate(self.class_names)}
        self.model = model_loader.loadModel(stem, str(self.model_dir)).eval()

        self.sample_rate = self.config.get('sample_rate', 32000)
        self.window_width = int(self.config.get('window_seconds', 0.064) * self.sample_rate)
        self.incr = int(self.config.get('hop_seconds', 0.01) * self.sample_rate)
        self.time_bins = self.config.get('time_bins', 1024)
        self.freq_bins = self.config.get('freq_bins', 224)
        self.spec_params = self.config.get('spectrogram_params', {})
        self.fg_bg_swap = self.config.get('fg_bg_swap', False)
        self.kbird_prior = self.config.get('kbird_prior', 0.0)

        thresh_path = self.model_dir / 'thresholds_combined.csv'
        self.thresholds = None
        if thresh_path.exists():
            df = pd.read_csv(thresh_path)
            self.thresholds = dict(zip(df['class'], df['threshold'].astype(float)))

    def _find_stem(self):
        for cfg in glob.glob(str(self.model_dir / '*_config.json')):
            stem = os.path.basename(cfg).replace('_config.json', '')
            for cand in (stem + '_best.pt', stem + '.pt', stem + '.pth'):
                if (self.model_dir / cand).exists():
                    return stem
        raise FileNotFoundError(f'No model weights found in {self.model_dir}')

    def window_probs(self, audio, sr):
        """Yield (start_s, prob_vector) per non-overlapping window."""
        audio = np.asarray(audio, dtype=np.float64)
        if 'highpass_hz' in self.spec_params:
            audio = highpass_audio(audio, sr, self.spec_params['highpass_hz'])
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-8:
            audio = audio / rms * 0.1

        clip_len = (self.time_bins - 1) * self.incr + self.window_width
        hop = self.time_bins * self.incr

        for s0 in range(0, len(audio) - self.window_width, hop):
            clip = audio[s0:s0 + clip_len]
            if len(clip) <= self.window_width:
                break
            sp = Spectrogram(self.window_width, self.incr)
            sp.audio_data = AudioData(data=clip, sample_rate=sr, sample_format='float32',
                                      sample_size=32, channels=1)
            sp.spectrogram(
                window_width=self.window_width, incr=self.incr,
                window=self.spec_params.get('windowType', 'Hamming'),
                sgType=self.spec_params.get('sgType', 'Standard'),
                sgScale=self.spec_params.get('sgScale', 'Mel Frequency'),
                nfilters=self.spec_params.get('nfilters', 224),
                mean_normalise=self.spec_params.get('mean_normalise', False),
                equal_loudness=self.spec_params.get('equal_loudness', False),
                onesided=True)

            raw = np.rot90(sp.sg).copy()
            fb = min(self.freq_bins, raw.shape[0])
            if self.fg_bg_swap:
                comp = split_spectrogram_components(raw[:fb], stochastic=False)
                combined = comp['norm_bg'].copy()
                combined[comp['fg_mask']] += comp['fg_diff'][comp['fg_mask']]
                raw[:fb] = np.clip(
                    combined * (comp['true_sigma'] + 1e-8) + comp['true_mu'], 0, None)

            proc = np.log(np.maximum(raw[:fb], 0.0) + 1e-7).astype(np.float32)
            x = np.zeros((fb, self.time_bins), dtype=np.float32)
            n = min(proc.shape[1], self.time_bins)
            x[:, :n] = proc[:, :n]
            x = x.reshape(1, fb, self.time_bins, 1)

            probs = torch.sigmoid(
                torch.from_numpy(inference.predict_batch(self.model, x)[0])).numpy()
            if self.kbird_prior > 0:
                total = max(float(probs.sum()), 1e-7)
                probs = np.clip(probs / max(total / self.kbird_prior, 1.0), 0.0, 1.0)
            yield s0 / sr, probs


def score_model(model_dir, gt_by_file, flat=None, only=None):
    """Return (per-GT recall stats, FP counts, window count) for one model."""
    fm = FieldModel(model_dir)
    cidx = fm.class_idx

    def threshold_for(cls):
        if flat is not None:
            return flat
        if fm.thresholds is None:
            return 0.5
        return fm.thresholds.get(cls, 1.0)

    stats = defaultdict(lambda: {'pos': 0, 'hit': 0, 'top1': 0})
    fp_counts = defaultdict(int)
    n_windows = 0

    only = set(only) if only else None

    for wav_name, gts in gt_by_file.items():
        audio, sr = sf.read(str(FIELD_DIR / wav_name))
        starts, prob_rows = [], []
        for st, probs in fm.window_probs(audio, sr):
            starts.append(st)
            prob_rows.append(probs)
        if not starts:
            continue
        prob_rows = np.stack(prob_rows)
        n_windows += len(starts)
        win_len = fm.time_bins * fm.incr / sr

        for si, st in enumerate(starts):
            en = st + win_len
            active = set()
            for g0, g1, names in gts:
                if g0 < en and g1 > st:
                    for nm in names:
                        active.update(GT_TO_MODEL.get(nm, []))
            for ci, cname in enumerate(fm.class_names):
                if prob_rows[si][ci] >= threshold_for(cname) and cname not in active:
                    fp_counts[cname] += 1

        for g0, g1, names in gts:
            mapped = [mc for nm in names for mc in GT_TO_MODEL.get(nm, [])]
            if not mapped:
                continue
            key = '|'.join(sorted(set(names)))
            if only and not any(mc in only for mc in mapped):
                continue
            stats[key]['pos'] += 1
            best = {mc: 0.0 for mc in mapped}
            top1_hit = False
            for si, st in enumerate(starts):
                en = st + win_len
                if not (g0 < en and g1 > st):
                    continue
                for mc in mapped:
                    best[mc] = max(best[mc], prob_rows[si][cidx[mc]])
                if fm.class_names[int(np.argmax(prob_rows[si]))] in mapped:
                    top1_hit = True
            if any(best[mc] >= threshold_for(mc) for mc in mapped):
                stats[key]['hit'] += 1
            stats[key]['top1'] += int(top1_hit)

    return fm, stats, fp_counts, n_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model_dirs', nargs='+', help='Model dir(s) with *_config.json + weights')
    parser.add_argument('--flat', type=float, default=None,
                        help='Use this flat threshold for all classes instead of thresholds_combined.csv')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated model class names to restrict scoring')
    args = parser.parse_args()

    gt_by_file = load_field_annotations(FIELD_DIR)
    n_segs = sum(len(v) for v in gt_by_file.values())
    print(f'Field test: {len(gt_by_file)} files, {n_segs} annotated segments '
          f'from {FIELD_DIR}')
    if args.flat is not None:
        print(f'Using FLAT threshold {args.flat} for all classes')

    for model_dir in args.model_dirs:
        fm, stats, fp_counts, n_windows = score_model(
            model_dir, gt_by_file, flat=args.flat,
            only=args.only.split(',') if args.only else None)

        mode = f'flat {args.flat}' if args.flat is not None else 'tuned thresholds'
        print(f'\n{"=" * 60}\n{fm.model_dir.name}  ({mode}, {n_windows} windows)\n{"=" * 60}')
        print(f'{"GT species":45s} {"n":>4s} {"recall":>8s} {"top1":>6s}')
        for key in sorted(stats, key=lambda k: -stats[k]['pos']):
            s = stats[key]
            print(f'{key:45s} {s["pos"]:>4d} {s["hit"] / s["pos"]:>8.2f} '
                  f'{s["top1"] / s["pos"]:>6.2f}')

        # overall recall over all scored GT segments
        tot = sum(s['pos'] for s in stats.values())
        hit = sum(s['hit'] for s in stats.values())
        t1 = sum(s['top1'] for s in stats.values())
        if tot:
            print(f'\n{"OVERALL":45s} {tot:>4d} {hit / tot:>8.2f} {t1 / tot:>6.2f}')

        print('\nTop false-positive classes (windows above threshold, class not annotated there):')
        for c, n in sorted(fp_counts.items(), key=lambda kv: -kv[1])[:10]:
            thr = args.flat if args.flat is not None else (fm.thresholds or {}).get(c, 0.5)
            print(f'  {c:30s} {n:>5d}  (thr={thr:.3f})')


if __name__ == '__main__':
    main()
