
# Version 4.0 9/10/25
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

# BatDetector.py
#
# Bat detection logic

import math
import numpy as np
from src.core import Spectrogram

class BatDetector:
    """
    Handles bat detection using click detection, NN classification, and labeling.
    """
    
    def __init__(self):
        """Initialize bat detector."""
        pass
    
    def findBatNNModel(self, filters, NNDicts):
        """Find and return the bat NN model from available models."""
        for filt in filters:
            if 'NN' in filt:
                nn_name = filt['NN']['NN_name']
                if nn_name in NNDicts:
                    return NNDicts[nn_name]
        return None
    
    def detectBatsInFile(self, sp, segments, currentFilename, filters, NNDicts, testmode=False, check_cancelled=None):
        """Main bat detection method - handles BMP spectrograms and audio files."""
        # Get NN model for bat classification
        NNmodel = self.findBatNNModel(filters, NNDicts)
        if NNmodel is None:
            print("Warning: No NN model found for bat detection")
            return
        
        if testmode:
            print("Warning: Bat detection not fully supported in test mode")
            return
        
        # Process entire file (no paging for now)
        batSegments = self.processBatFile(sp, currentFilename, NNmodel, check_cancelled)
        if len(batSegments) > 0:
            self.makeBatSegments(segments, batSegments[0])
    
    def processBatFile(self, sp, filename, NNmodel, check_cancelled=None):
        """Detect clicks, run NN classification, and generate labels for entire file."""
        print("Processing bat file...")
        
        # Get input dimensions and thresholds from model
        model = NNmodel[0]
        inputdim = NNmodel[2]
        thr1 = float(np.atleast_1d(NNmodel[5][0])[0])
        thr2 = float(np.atleast_1d(NNmodel[5][1])[0])
        
        # Step 1: Detect clicks in spectrogram
        click_label, data_test, count = self.clickSearch(sp, filename, inputdim=inputdim, virginia=True, check_cancelled=check_cancelled)
        print(f"Click detection: {click_label}, {count} clicks found")
        
        # Check for cancellation after click detection
        if check_cancelled and check_cancelled():
            from src.utils.exceptions import GentleExitException
            raise GentleExitException
        
        if click_label != 'Click' or count == 0:
            print("No clicks detected")
            return []
        
        # Step 2: Prepare data for NN classification
        num_spectrograms = len(data_test)
        if num_spectrograms == 0:
            print("No spectrograms to process")
            return []
        
        # Convert data_test elements back to numpy arrays
        first_sg = np.array(data_test[0][0])
        sg_test = np.ndarray(shape=(num_spectrograms, first_sg.shape[0], first_sg.shape[1]), dtype=float)
        print(f'Number of spectrograms: {num_spectrograms}')
        
        for j in range(num_spectrograms):
            sg_array = np.array(data_test[j][0])
            maxg = np.max(sg_array)
            sg_test[j][:] = sg_array / maxg
        
        # Step 3: Run NN classification
        print(f"Shape before reshape: {sg_test.shape}")
        test_images = sg_test.reshape(sg_test.shape[0], inputdim[0], inputdim[1], 1)
        print(f"Shape after reshape: {test_images.shape}")
        test_images = test_images.astype('float32')
        
        predictions = model.predict(test_images)
        
        # Step 4: Generate labels from predictions
        print('Assessing file label...')
        labels = self.labelBatFile(predictions, thr1=thr1, thr2=thr2)
        print('NN detected:', labels)
        
        if len(labels) == 0:
            return []
        
        # Step 5: Create segment spanning entire file
        file_duration = sp.get_duration()
        return [[0, file_duration, labels]]
    
    def updateDataset(self, file_name, featuress, count, spectrogram, click_start, click_end, dt=None, inputdim=None):
        """Extract and resize spectrogram window centered on click, append to featuress."""
        if inputdim is None:
            raise ValueError("inputdim is required - must provide target dimensions for model input")
        
        target_time_pixels = inputdim[0]
        target_freq_bins = inputdim[1]
        
        ls = np.shape(spectrogram)[1] - 1
        click_center = int((click_start + click_end) / 2)
        
        # Extract a window centered on the click
        # Use approximately target_time_pixels / 4 on each side of click center
        win_pixel = max(1, target_time_pixels // 4)
        
        start_pixel = max(0, click_center - win_pixel)
        end_pixel = min(ls, click_center + win_pixel)
        
        # Extract the window
        sgRaw = spectrogram[:, start_pixel:end_pixel + 1]
        
        # Flip and transpose to get [time, freq] format
        sgRaw = (np.flipud(sgRaw)).T
        
        # Resize to target dimensions using bilinear interpolation
        from scipy import ndimage
        zoom_factors = (target_time_pixels / sgRaw.shape[0], target_freq_bins / sgRaw.shape[1])
        sgRaw = ndimage.zoom(sgRaw, zoom_factors, order=1)
        
        featuress.append([sgRaw.tolist(), file_name, count])
        count += 1

        return featuress, count

    def clickSearch(self, sp, file, inputdim=None, virginia=True, check_cancelled=None):
        """Detect bat clicks in 24-54 kHz band; return extracted patches (virginia=True) or indices (virginia=False)."""
        imspec = sp.sg
        featuress = []
        count = 0

        df = sp.audio_data.sample_rate // 2 / (np.shape(imspec)[0] + 1)  # frequency increment
        dt = sp.incr / sp.audio_data.sample_rate  # sp.incr is set to 512 for bats
        up_len = 17
        
        # Frequency band
        f0 = 24000
        index_f0 = -1 + math.floor(f0 / df)  # lower bound needs to be rounded down
        f1 = 54000
        index_f1 = -1 + math.ceil(f1 / df)  # upper bound needs to be rounded up

        # Mean in the frequency band
        mean_spec = np.mean(imspec[index_f0:index_f1, :], axis=0)

        # Threshold
        mean_spec_all = np.mean(imspec, axis=0)[2:]
        thr_spec = (np.mean(mean_spec_all) + np.std(mean_spec_all)) * np.ones((np.shape(mean_spec)))

        ## clickfinder
        # check when the mean is bigger than the threshold
        # clicks is an array which elements are equal to 1 only where the sum is bigger
        # than the mean, otherwise are equal to 0
        clicks = mean_spec > thr_spec
        
        if virginia:
            clicks_indices = np.nonzero(clicks)
            # check: if I have found somenthing
            if np.shape(clicks_indices)[1] == 0:
                click_label = 'None'
                return click_label, featuress, count
                # not saving spectrograms

            # Discarding segments too long or too short and saving spectrogram images
            click_start = clicks_indices[0][0]
            click_end = clicks_indices[0][0]
            for i in range(1, np.shape(clicks_indices)[1]):
                # Check for cancellation periodically (every 100 clicks)
                if check_cancelled and i % 100 == 0 and check_cancelled():
                    from src.utils.exceptions import GentleExitException
                    raise GentleExitException
                
                if clicks_indices[0][i] == click_end + 1:
                    click_end = clicks_indices[0][i]
                else:
                    if click_end - click_start + 1 > up_len:
                        clicks[click_start:click_end + 1] = False
                    else:
                        # savedataset
                        featuress, count = self.updateDataset(file, featuress, count, imspec, click_start, click_end, dt, inputdim)
                    # update
                    click_start = clicks_indices[0][i]
                    click_end = clicks_indices[0][i]

            # checking last loop with end
            if click_end - click_start + 1 > up_len:
                clicks[click_start:click_end + 1] = False
            else:
                featuress, count = self.updateDataset(file, featuress, count, imspec, click_start, click_end, dt, inputdim)

            # Assigning: click label
            if np.any(clicks):
                click_label = 'Click'
            else:
                click_label = 'None'

            return click_label, featuress, count
        else:
            inds = np.where(clicks > 0)[0]
            if (len(inds)) > 0:
                # Have found something, now find first that isn't too long
                flag = False
                start = inds[0]
                while not flag:
                    i = 1
                    while i < len(inds) and inds[i] - inds[i - 1] == 1:
                        i += 1
                    end = i - 1
                    if inds[end] - inds[start] < up_len:
                        flag = True
                    else:
                        if i < len(inds):
                            start = inds[i]
                        else:
                            break

                first = inds[start] if flag else None

                # And last that isn't too long
                flag = False
                end = inds[-1]
                start_idx = len(inds) - 1
                while not flag and start_idx >= 0:
                    i = start_idx
                    while i > 0 and inds[i] - inds[i - 1] == 1:
                        i -= 1
                    start = i
                    if inds[start_idx] - inds[start] < up_len:
                        flag = True
                    else:
                        if start > 0:
                            start_idx = start - 1
                        else:
                            break

                last = inds[start_idx] if flag else None
                
                if first is not None and last is not None:
                    return [first, last]
                else:
                    return None
            else:
                return None

    def labelBatFile(self, predictions, thr1, thr2):
        """Generate Long-tailed/Short-tailed bat labels from NN predictions using mean and best3mean thresholds."""

        # Assessing file label
        # inizialization
        # vectors storing classes probabilities
        LT_prob = []  # class 0
        ST_prob = []  # class 1
        NT_prob = []  # class 2
        spec_num = 0   # counts number of spectrograms per file
        # flag: if no click detected no spectrograms
        click_detected_flag = False
        # looking for all the spectrogram related to this file

        for k in range(np.shape(predictions)[0]):
            click_detected_flag = True
            spec_num += 1
            LT_prob.append(predictions[k][0])
            ST_prob.append(predictions[k][1])
            NT_prob.append(predictions[k][2])

        # if no clicks => automatically Noise
        label = []

        if click_detected_flag:
            # mean
            LT_mean = np.mean(LT_prob) * 100
            ST_mean = np.mean(ST_prob) * 100

            # best3mean
            LT_best3mean = 0
            ST_best3mean = 0

            # LT
            ind = np.array(LT_prob).argsort()[-3:][::-1]
            # adding len ind in order to consider also the cases when we do not have 3 good examples
            if len(ind) == 1:
                # this means that there is only one prob!
                LT_best3mean += LT_prob[0]
            else:
                for j in range(len(ind)):
                    LT_best3mean += LT_prob[ind[j]]
            LT_best3mean /= 3
            LT_best3mean *= 100

            # ST
            ind = np.array(ST_prob).argsort()[-3:][::-1]
            # adding len ind in order to consider also the cases when we do not have 3 good examples
            if len(ind) == 1:
                # this means that there is only one prob!
                ST_best3mean += ST_prob[0]
            else:
                for j in range(len(ind)):
                    ST_best3mean += ST_prob[ind[j]]
            ST_best3mean /= 3
            ST_best3mean *= 100

            # ASSESSING FILE LABEL
            hasST = ST_mean >= thr1 or ST_best3mean >= thr2
            hasLT = LT_mean >= thr1 or LT_best3mean >= thr2
            hasSTlow = ST_mean < thr1 and ST_best3mean < thr2
            hasLTlow = LT_mean < thr1 and LT_best3mean < thr2
            reallyHasST = ST_mean >= thr1 and ST_best3mean >= thr2
            reallyHasLT = LT_mean >= thr1 and LT_best3mean >= thr2
            HasBat = LT_mean >= thr1 and ST_mean >= thr1

            if reallyHasLT and hasSTlow:
                label.append({"species": "Long-tailed bat", "certainty": 100})
            elif reallyHasLT and reallyHasST:
                label.append({"species": "Long-tailed bat", "certainty": 100})
            elif hasLT and ST_mean < thr1:
                label.append({"species": "Long-tailed bat", "certainty": 50})
            elif HasBat:
                label.append({"species": "Long-tailed bat", "certainty": 50})

            if reallyHasST and hasLTlow:
                label.append({"species": "Short-tailed bat", "certainty": 100})
            elif reallyHasLT and reallyHasST:
                label.append({"species": "Short-tailed bat", "certainty": 100})
            elif hasST and LT_mean < thr1:
                label.append({"species": "Short-tailed bat", "certainty": 50})
            elif HasBat:
                label.append({"species": "Short-tailed bat", "certainty": 50})

        return label
    
    def makeBatSegments(self, segmentsList, segmentsNew):
        """Add bat segment with labels to segment list."""
        from src.core import Segment
        
        # Batmode: segmentsNew should be already prepared as: [x1, x2, labels]
        y1 = 0
        y2 = 0
        if len(segmentsNew) != 3:
            print("Warning: segment format does not match bat mode")
        segment = Segment.Segment([segmentsNew[0], segmentsNew[1], y1, y2, segmentsNew[2]])
        segmentsList.addSegment(segment)