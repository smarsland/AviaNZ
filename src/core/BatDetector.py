# bat_detector.py
#
# Bat detection logic extracted from AviaNZ_batch.py

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

import math
import numpy as np


class BatDetector:
    """
    Handles bat detection using click detection, NN classification, and labeling.
    Extracted from AviaNZ_batchProcess to separate bat-specific logic.
    """
    
    def __init__(self):
        """Initialize bat detector."""
        pass
    
    def detectBatsInFile(self, sp, segments, currentFilename, filters, NNDicts, testmode=False, ui_callback=None):
        """
        Main bat detection method for a single file.
        
        Args:
            sp: Spectrogram object with loaded audio data and spectrogram
            segments: SegmentList to store results in
            currentFilename: Current filename being processed
            filters: List of filter dictionaries
            NNDicts: Neural network models dictionary
            testmode: Whether running in test mode
            ui_callback: Optional callback for UI updates (function that returns True if cancelled)
            
        Returns:
            None (modifies segments list in place)
        """
        # Check if this is a BMP file (spectrogram) or audio file
        if hasattr(sp, 'sg') and sp.sg is not None and (sp.audio_data is None or sp.audio_data.data is None):
            # Processing BMP spectrogram file
            print("Processing BMP spectrogram file...")
            
            # For BMP files, process the entire spectrogram as one page
            # Get NN model for bats
            NNmodel = None
            print(f"Available NN models: {list(NNDicts.keys())}")
            print(f"Available filters: {len(filters)}")
            for i, filt in enumerate(filters):
                print(f"Filter {i}: has NN = {'NN' in filt}")
                if 'NN' in filt:
                    print(f"  NN_name: {filt['NN']['NN_name']}")
                    if filt['NN']['NN_name'] in NNDicts.keys():
                        NNmodel = NNDicts[filt['NN']['NN_name']]
                        print(f"  Found matching model!")
                        break
                    else:
                        print(f"  No matching model for {filt['NN']['NN_name']}")
            
            if NNmodel is None:
                print("Warning: No NN model found for bat detection")
                return
            
            if not testmode:
                # Process BMP file directly
                batSegments = self.processBatFile(sp, currentFilename, 0, sp.sg.shape[1], 
                                                sp.sg.shape[1] * 0.002909090909090909, NNmodel)
                if len(batSegments) > 0:
                    self._makeBatSegments(segments, batSegments[0])
            else:
                print("Warning: Bat detection not fully supported in test mode")
        else:
            # Processing audio file - use page-based processing
            print("Processing audio file...")
            samplesInPage = 900 * 16000  # Standard 15 min pages
            numPages = (len(sp.audio_data.data) - 1) // samplesInPage + 1
            
            for page in range(numPages):
                print("Processing bat page %d / %d" % (page+1, numPages))
                start = page * samplesInPage
                end = min(start + samplesInPage, len(sp.audio_data.data))
                thisPageLen = (end - start) / sp.audio_data.sample_rate
                
                if thisPageLen < 2:
                    print("Warning: can't process short file ends (%.2f s)" % thisPageLen)
                    continue
                
                # Get NN model for bats
                NNmodel = None
                print(f"Available NN models: {list(NNDicts.keys())}")
                print(f"Available filters: {len(filters)}")
                for i, filt in enumerate(filters):
                    print(f"Filter {i}: has NN = {'NN' in filt}")
                    if 'NN' in filt:
                        print(f"  NN_name: {filt['NN']['NN_name']}")
                        if filt['NN']['NN_name'] in NNDicts.keys():
                            NNmodel = NNDicts[filt['NN']['NN_name']]
                            print(f"  Found matching model!")
                            break
                        else:
                            print(f"  No matching model for {filt['NN']['NN_name']}")
                
                if NNmodel is None:
                    print("Warning: No NN model found for bat detection")
                    continue
                
                if not testmode:
                    # Process bat file with bat detector
                    batSegments = self.processBatFile(sp, currentFilename, start, end, thisPageLen, NNmodel)
                    if len(batSegments) > 0:
                        self._makeBatSegments(segments, batSegments[0])
                else:
                    print("Warning: Bat detection not fully supported in test mode")
                
                # Check for UI cancellation
                if ui_callback and ui_callback():
                    from src.utils.exceptions import GentleExitException
                    raise GentleExitException
    
    def processBatFile(self, sp, filename, start, end, thisPageLen, NNmodel):
        """
        Unified bat processing method that handles click detection, NN classification, and labeling.
        
        Args:
            sp: Spectrogram object with loaded audio data and spectrogram
            filename: path to the audio file
            start: start sample of the current page
            end: end sample of the current page  
            thisPageLen: length of the page in seconds
            NNmodel: list containing [model, win, inputdim, output, ..., [thr1, thr2]]
            
        Returns:
            List of segments in format [start_time, end_time, labels] or empty list if no bats detected
        """
        print("Processing bat file...")
        
        # Get input dimensions from the model
        inputdim = NNmodel[2]
        
        # Step 1: Detect clicks in spectrogram
        click_label, data_test, count = self.clickSearch(sp, filename, inputdim=inputdim, virginia=True)
        print(f"Click detection: {click_label}, {count} clicks found")
        
        if click_label != 'Click' or count == 0:
            print("No clicks detected")
            return []
        
        # Step 2: Prepare data for NN classification
        model = NNmodel[0]
        inputdim = NNmodel[2]  # Get input dimensions from the model
        # Ensure thresholds are scalar values, not arrays
        thr1 = float(np.atleast_1d(NNmodel[5][0])[0])
        thr2 = float(np.atleast_1d(NNmodel[5][1])[0])
        
        # Convert data_test elements back to numpy arrays
        # data_test format: [[spectrogram_as_list, filename, count], ...]
        num_spectrograms = len(data_test)
        if num_spectrograms == 0:
            print("No spectrograms to process")
            return []
        
        # Get dimensions from first spectrogram (convert from list back to array)
        first_sg = np.array(data_test[0][0])
        sg_test = np.ndarray(shape=(num_spectrograms, first_sg.shape[0], first_sg.shape[1]), dtype=float)
        spec_id = []
        print(f'Number of file spectrograms: {num_spectrograms}')
        
        for j in range(num_spectrograms):
            # Convert spectrogram from list back to numpy array
            sg_array = np.array(data_test[j][0])
            maxg = np.max(sg_array)
            sg_test[j][:] = sg_array / maxg
            spec_id.append(data_test[j][1:3])
        
        # Step 3: Run NN classification
        x_test = sg_test
        print(f"Shape before reshape: {x_test.shape}")
        # Use the model's input dimensions instead of hardcoded values
        test_images = x_test.reshape(x_test.shape[0], inputdim[0], inputdim[1], 1)
        print(f"Shape after reshape: {test_images.shape}")
        test_images = test_images.astype('float32')
        
        predictions = model.predict(test_images)
        
        # Step 4: Generate labels from predictions
        print('Assessing file label...')
        labels = self.labelBatFile(predictions, thr1=thr1, thr2=thr2)
        print('NN detected:', labels)
        
        if len(labels) == 0:
            return []
        
        # Step 5: Create segment with labels
        thisPageStart = start / sp.audio_data.sample_rate
        return [[thisPageStart, thisPageLen, labels]]
    
    def updateDataset(self, file_name, featuress, count, spectrogram, click_start, click_end, dt=None, inputdim=None):
        """
        Update Dataset with current segment
        It take a piece of the spectrogram with fixed length centered in the click
        
        Args:
            file_name: Name of the file being processed
            featuress: List to append features to
            count: Current count of features
            spectrogram: Spectrogram data
            click_start: Start index of click
            click_end: End index of click
            dt: Time delta (unused)
            inputdim: Target dimensions [time, freq] for extracted spectrograms (e.g., [64, 343])
                      If None, uses legacy behavior of [6, 512]
            
        Returns:
            Tuple of (updated_featuress, updated_count)
        """
        # Use legacy dimensions if inputdim not provided
        if inputdim is None:
            inputdim = [6, 512]
        
        target_time_pixels = inputdim[0]
        target_freq_bins = inputdim[1]
        
        # Calculate window size in time (pixels) needed to extract
        # We want to extract a window centered on the click
        # For backward compatibility with old models, when target is [6, 512]:
        # - Extract 3 pixels (win_pixel=1 means center ±1)
        # - Double it to 6 with np.repeat
        # For new models like [64, 343]:
        # - We need to extract more pixels and resize appropriately
        
        ls = np.shape(spectrogram)[1] - 1
        click_center = int((click_start + click_end) / 2)
        
        # For backward compatibility: if target is [6, 512], use old method
        if target_time_pixels == 6 and target_freq_bins == 512:
            win_pixel = 1
            start_pixel = click_center - win_pixel
            if start_pixel < 0:
                win_pixel2 = win_pixel + np.abs(start_pixel)
                start_pixel = 0
            else:
                win_pixel2 = win_pixel

            end_pixel = click_center + win_pixel2
            if end_pixel > ls:
                start_pixel -= end_pixel - ls + 1
                end_pixel = ls - 1
            
            sgRaw = spectrogram[:, start_pixel:end_pixel + 1]
            sgRaw = np.repeat(sgRaw, 2, axis=1)
            sgRaw = (np.flipud(sgRaw)).T
        else:
            # For new models: extract a larger window and resize
            # Extract approximately target_time_pixels / 2 on each side of click
            win_pixel = target_time_pixels // 4  # Divide by 4 because we'll double it
            if win_pixel < 1:
                win_pixel = 1
                
            start_pixel = click_center - win_pixel
            if start_pixel < 0:
                start_pixel = 0
                
            end_pixel = click_center + win_pixel
            if end_pixel > ls:
                end_pixel = ls
            
            # Extract the window
            sgRaw = spectrogram[:, start_pixel:end_pixel + 1]
            
            # Flip and transpose to get [time, freq] format
            sgRaw = (np.flipud(sgRaw)).T
            
            # Resize to target dimensions using scipy or simple interpolation
            from scipy import ndimage
            zoom_factors = (target_time_pixels / sgRaw.shape[0], target_freq_bins / sgRaw.shape[1])
            sgRaw = ndimage.zoom(sgRaw, zoom_factors, order=1)  # bilinear interpolation
        
        featuress.append([sgRaw.tolist(), file_name, count])

        count += 1

        return featuress, count

    def clickSearch(self, sp, file, inputdim=None, virginia=True):
        """
        Searches for clicks in the provided spectrogram, saves dataset
        Returns click_label, dataset and count of detections

        The search is made on the spectrogram image that we know to be generated
        with parameters (1024,512)
        Click presence is assessed for each spectrogram column: if the mean in the
        frequency band [f0, f1] (*) is bigger than a treshold we have a click
        thr=mean(all_spec)+std(all_spec) (*)

        The clicks are discarded if longer than 0.05 sec

        Clicks are stored into featuress using updateDataset

        Args:
            sp: Spectrogram object with audio data and spectrogram (sp.sg)
            file: filename (NOTE originally was basename, now full filename)
            inputdim: Target dimensions [time, freq] for extracted spectrograms (e.g., [64, 343])
            virginia: Boolean flag for processing mode

        Returns:
            If virginia=True: (click_label, featuress, count)
            If virginia=False: [first, last] indices or None
        """
        imspec = sp.sg
        featuress = []
        count = 0

        df = sp.audio_data.sample_rate // 2 / (np.shape(imspec)[0] + 1)  # frequency increment
        dt = sp.incr / sp.audio_data.sample_rate  # sp.incr is set to 512 for bats
        # dt=0.002909090909090909
        # up_len=math.ceil(0.05/dt) #0.5 second lenth in indices divided by 11
        up_len = 17
        # up_len=math.ceil((0.5/11)/dt)

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
        """
        Uses the predictions made by the NN to update the filewise annotations
        when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)

        METHOD: evaluation of probability over files combining mean of probability
            + best3mean of probability against thr1 and thr2, respectively

        Args:
            predictions: NN predictions array
            thr1: Primary threshold
            thr2: Secondary threshold

        Returns: 
            species labels (list of dicts), compatible w/ the label format on Segments
        """

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
    
    def _makeBatSegments(self, segmentsList, segmentsNew):
        """Add bat-specific segments with proper metadata."""
        from src.core import Segment
        
        # Batmode: segmentsNew should be already prepared as: [x1, x2, labels]
        y1 = 0
        y2 = 0
        if len(segmentsNew) != 3:
            print("Warning: segment format does not match bat mode")
        segment = Segment.Segment([segmentsNew[0], segmentsNew[1], y1, y2, segmentsNew[2]])
        segmentsList.addSegment(segment)