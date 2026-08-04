
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

# UI-independent neural-net segmentation.
#
# This is the single source of truth for running the model_testing RegNet (and
# similar torch models) over an audio buffer. Both the manual interface
# (manual_interface.segment, alg == 'NN_Model') and the batch processor use it,
# so the preprocessing MUST stay in exact parity with the model_testing training
# pipeline - RegNet has no internal input normalisation. See the
# nn-inference-pipeline memory for the non-obvious requirements (RMS-normalise
# audio, natural-log spec transform, per-clip bg_subtract, Standard sgType, ...).

import os
import json
import glob

import numpy as np

# Fallback gate used only when a model ships no per-class threshold table. For
# multilabel models with thresholds_combined.csv the per-class thresholds fully
# override this; it still applies to single-label models.
DEFAULT_MIN_CONFIDENCE = 0.5
LOG_OFFSET = 1e-7


def discover_models():
    """ Find available NN models on disk.

        Returns a list of (display_name, (dir_path, stem)) tuples, matching the
        segmentation dialog's populateNNModels() so the manual and batch UIs
        offer the same choices.
    """
    models = []  # list of (display_name, (dir_path, stem))

    # Legacy Models/ directory (.pth files)
    models_dir = 'Models'
    for json_file in glob.glob(os.path.join(models_dir, '*_config.json')):
        stem = os.path.basename(json_file).replace('_config.json', '')
        if os.path.exists(os.path.join(models_dir, stem + '.pth')):
            models.append((stem, (models_dir, stem)))

    # model_testing/ subdirectories (.pt or .pth files)
    for json_file in glob.glob(os.path.join('model_testing', '*', '*_config.json')):
        d = os.path.dirname(json_file)
        stem = os.path.basename(json_file).replace('_config.json', '')
        candidates = [stem + '.pth', stem + '.pt', stem + '_best.pt']
        for c in candidates:
            if os.path.exists(os.path.join(d, c)):
                display = f"{stem} ({os.path.basename(d)})"
                models.append((display, (d, stem)))
                break

    models.sort()
    return models


class NNSegmenter:
    """ Runs a trained torch model over an audio buffer and returns segments.

        Load the model once (constructor) and reuse it across many files/pages -
        loading is expensive, inference is not.
    """

    def __init__(self, model_dir, model_stem):
        """ Load the model, its config, and per-class thresholds.

            Args:
                model_dir: directory holding the model files
                model_stem: file stem (loadModel resolves _best.pt / .pt / .pth)

            Raises FileNotFoundError if the config is missing.
        """
        from src.models import model_loader

        self.model_dir = model_dir
        self.model_stem = model_stem

        config_path = os.path.join(model_dir, f'{model_stem}_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Per-class thresholds (thresholds_combined.csv) override the uniform gate
        # for multilabel models when present.
        self.per_class_thresholds = None
        thresh_path = os.path.join(model_dir, 'thresholds_combined.csv')
        if os.path.exists(thresh_path):
            import pandas as pd
            thresh_df = pd.read_csv(thresh_path)
            self.per_class_thresholds = dict(
                zip(thresh_df['class'], thresh_df['threshold'].astype(float)))
            print(f"Loaded per-class thresholds from {thresh_path}")

        from src.utils.device import get_device

        self.model = model_loader.loadModel(model_stem, model_dir)
        self.model = self.model.to(get_device())
        self.model.eval()

        # Cache the frequently-used config values.
        self.model_sample_rate = self.config.get('sample_rate', 32000)
        self.class_names = self.config.get('class_names', [])
        self.multilabel = self.config.get('multilabel', False)
        self.spec_params = self.config.get('spectrogram_params', {})
        self.window_seconds = self.config.get('window_seconds', 0.025)
        self.hop_seconds = self.config.get('hop_seconds', 0.01)
        self.time_bins = self.config.get('time_bins', 400)
        self.freq_bins = self.config.get('freq_bins', 128)
        self.spec_transform = self.config.get('spec_transform', 'Log')
        self.bg_subtract = self.config.get('bg_subtract', False)
        self.median_filter = self.config.get('median_filter', False)

    def _resample_if_needed(self, audio, sample_rate, progress_cb=None):
        """ Return (audio, sample_rate) resampled to the model's rate if needed.

            Does not modify the caller's array.
        """
        if sample_rate == self.model_sample_rate:
            return audio, sample_rate

        if progress_cb:
            progress_cb(f'Resampling audio from {sample_rate} Hz to {self.model_sample_rate} Hz...')

        import src.core.spectrogram as spectrogram
        import src.core.audio_data as audio_data

        temp_sp = spectrogram.Spectrogram(256, 128)
        temp_sp.audio_data = audio_data.AudioData(
            data=audio.copy(), sample_rate=sample_rate,
            sample_format='float32', sample_size=32, channels=1)
        temp_sp.resample(self.model_sample_rate)
        return temp_sp.audio_data.data, self.model_sample_rate

    def _raw_detections(self, audio, inference_sr, progress_cb=None):
        """ Run the model clip-by-clip and collect gated detections.

            Returns a list of [start_time, end_time, class_idx, certainty], with
            times relative to the start of `audio`.
        """
        import torch
        import torch.nn.functional as F
        import src.core.spectrogram as spectrogram
        import src.core.audio_data as audio_data
        from src.models import inference
        from model_testing.src.data.normalizer import normalize_spectrogram

        window_width = int(self.window_seconds * inference_sr)
        incr = int(self.hop_seconds * inference_sr)
        time_bins = self.time_bins
        freq_bins = self.freq_bins

        # Process the file one CLIP at a time (time_bins columns per clip). This
        # matches how training built each clip and bounds memory - a whole-file
        # spectrogram OOMs on multi-minute files.
        clip_hop_samples = time_bins * incr
        clip_len_samples = (time_bins - 1) * incr + window_width
        n_audio = len(audio)
        n_clips = max(1, int(np.ceil(max(1, n_audio - window_width + 1) / clip_hop_samples)))

        raw_detections = []

        for k in range(n_clips):
            s0 = k * clip_hop_samples
            clip = audio[s0:s0 + clip_len_samples]
            if len(clip) <= window_width:
                break

            clip_sp = spectrogram.Spectrogram(window_width, incr)
            clip_sp.audio_data = audio_data.AudioData(
                data=clip, sample_rate=inference_sr,
                sample_format='float32', sample_size=32, channels=1)
            clip_sp.spectrogram(
                window_width=window_width, incr=incr,
                window=self.spec_params.get('windowType', 'Hann'),
                sgType=self.spec_params.get('sgType', 'Standard'),
                sgScale=self.spec_params.get('sgScale', 'Linear'),
                nfilters=self.spec_params.get('nfilters', 128),
                mean_normalise=self.spec_params.get('mean_normalise', True),
                equal_loudness=self.spec_params.get('equal_loudness', False),
                onesided=True)

            # (time, freq) -> (freq, time), row 0 = highest frequency
            raw = np.rot90(clip_sp.sg).copy()
            fb = min(freq_bins, raw.shape[0])
            actual_frames = min(raw.shape[1], time_bins)

            # Match training preprocessing.
            if self.spec_transform == 'Log':
                proc = np.log(np.maximum(raw[:fb], 0.0) + LOG_OFFSET)
            elif self.spec_transform in ('None', None):
                proc = np.asarray(raw[:fb], dtype=np.float32)
            else:
                print(f"Warning: spec_transform '{self.spec_transform}' not implemented for inference, using Log")
                proc = np.log(np.maximum(raw[:fb], 0.0) + LOG_OFFSET)

            if self.bg_subtract or self.median_filter:
                proc = normalize_spectrogram(
                    proc, median_filter=self.median_filter, bg_subtract=self.bg_subtract)

            # Zero-pad the time axis to time_bins.
            segment = np.zeros((fb, time_bins), dtype=np.float32)
            segment[:, :actual_frames] = proc[:, :actual_frames]
            segment = segment.reshape(1, fb, time_bins, 1)

            logits = inference.predict_batch(self.model, segment)[0]
            pred_tensor = torch.from_numpy(logits)

            start_time = s0 / inference_sr
            end_time = (s0 + actual_frames * incr) / inference_sr

            if self.multilabel:
                probs = torch.sigmoid(pred_tensor).numpy()
                for class_idx, prob in enumerate(probs):
                    prob = float(np.clip(prob, 0.0, 1.0))
                    if self.per_class_thresholds is not None:
                        cname = self.class_names[class_idx] if class_idx < len(self.class_names) else None
                        thr = self.per_class_thresholds.get(cname, 1.0)
                    else:
                        thr = DEFAULT_MIN_CONFIDENCE
                    if prob >= thr:
                        raw_detections.append([start_time, end_time, class_idx, prob])
            else:
                probs = F.softmax(pred_tensor, dim=0).numpy()
                max_idx = int(np.argmax(probs))
                max_prob = float(np.clip(probs[max_idx], 0.0, 1.0))
                if max_prob >= DEFAULT_MIN_CONFIDENCE:
                    raw_detections.append([start_time, end_time, max_idx, max_prob])

            if progress_cb:
                progress_cb(f'Running NN inference... clip {k + 1}/{n_clips}')

        return raw_detections

    def run(self, audio, sample_rate, do_post_process=True, maxgap=1.0, minlen=0.5,
            progress_cb=None):
        """ Segment an audio buffer with the model.

            Args:
                audio: 1-D numpy array of samples
                sample_rate: sample rate of `audio` (resampled internally if needed)
                do_post_process: merge nearby windows and drop short segments
                maxgap: max gap (s) between windows to merge (post-processing)
                minlen: min segment length (s) to keep (post-processing)
                progress_cb: optional callable(str) for status updates

            Returns a list of segment dicts, times relative to the start of `audio`:
                {"start", "end", "freq_low", "freq_high",
                 "labels": [{"species", "certainty"}]}
            certainty is on AviaNZ's 0-100 scale.
        """
        audio, inference_sr = self._resample_if_needed(audio, sample_rate, progress_cb)

        # RMS-normalise the audio exactly as the training pipeline does.
        audio = np.asarray(audio, dtype=np.float64)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-8:
            audio = audio / rms * 0.1

        raw_detections = self._raw_detections(audio, inference_sr, progress_cb)

        # Build multilabel segments, one per detection window: detections that
        # share a (start, end) are collected into a single {class_idx: certainty}.
        window_labels = {}
        for start_time, end_time, class_idx, certainty in raw_detections:
            labels = window_labels.setdefault((start_time, end_time), {})
            if class_idx not in labels or certainty > labels[class_idx]:
                labels[class_idx] = certainty

        segments = [[list(win), labels] for win, labels in window_labels.items()]
        segments.sort(key=lambda s: s[0][0])

        # Post-processing: merge windows that overlap or sit within maxgap of each
        # other, unioning labels (max certainty per class), then drop short ones.
        if do_post_process and segments:
            merged = [segments[0]]
            for seg in segments[1:]:
                (cur_start, cur_end), cur_labels = merged[-1]
                (seg_start, seg_end), seg_labels = seg
                if seg_start <= cur_end + maxgap + 0.01:
                    merged[-1][0][1] = max(cur_end, seg_end)
                    for cls, cert in seg_labels.items():
                        if cls not in cur_labels or cert > cur_labels[cls]:
                            cur_labels[cls] = cert
                else:
                    merged.append(seg)
            segments = [s for s in merged if (s[0][1] - s[0][0]) >= minlen]

        print(f'NN produced {len(segments)} multilabel segments from '
              f'{len(raw_detections)} raw detections '
              f'(post-processing {"on" if do_post_process else "off"})')

        return self._to_label_segments(segments, inference_sr)

    def _to_label_segments(self, segments, inference_sr):
        """ Convert [[start, end], {class_idx: certainty}] pairs into segment dicts
            with species labels and 0-100 certainties. Windows with no labels are
            dropped.
        """
        freq_low = 0
        freq_high = inference_sr // 2

        out = []
        for (start_time, end_time), class_certainties in segments:
            label_list = []
            for class_idx, certainty in class_certainties.items():
                if self.class_names and 0 <= class_idx < len(self.class_names):
                    species_label = self.class_names[class_idx]
                else:
                    species_label = f"Class_{class_idx}"
                label_list.append({
                    "species": species_label,
                    "certainty": round(float(certainty) * 100, 1),
                })

            if label_list:
                out.append({
                    "start": start_time,
                    "end": end_time,
                    "freq_low": freq_low,
                    "freq_high": freq_high,
                    "labels": label_list,
                })
            else:
                print(f"Warning: segment at {start_time}-{end_time} has no labels, skipping")

        return out
