# review_workflow_manager.py
# Part of AviaNZ refactoring - handles complex review workflows coordination

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox
import os
import copy
import numpy as np
from ..core import Segment
from ..ui import SupportClasses_GUI
from ..ui import Dialogs
import pyqtgraph as pg


class ReviewWorkflowManager(QObject):
    """Manages complex review workflows and coordination.
    
    Handles:
    - Cross-file review coordination 
    - Review mode selection (quick vs one-by-one)
    - Review state tracking and persistence
    - Integration with existing managers for actual operations
    """
    
    # Signals for review workflow coordination
    review_started = pyqtSignal()
    segments_collected = pyqtSignal(list)  # collected segments
    review_completed = pyqtSignal()
    review_progress_updated = pyqtSignal(int, int, str)  # current, total, message
    
    def __init__(self, parent, config_manager, audio_file_manager, species_manager):
        super().__init__(parent)
        self.parent_window = parent
        self.config_manager = config_manager
        self.audio_file_manager = audio_file_manager
        self.species_manager = species_manager
        
        # Review state
        self.all_segments = []
        self.all_file_data = {}
        self.current_species = None
        self.current_dir = None
        
    def coordinate_review_workflow(self, dir_name, species=None, quick=False):
        """Main review coordination method.
        
        Args:
            dir_name: Directory to review
            species: Species to review (None for all)
            quick: If True, use quick review mode
            
        Returns:
            bool: True if review started successfully
        """
        if not self.setup_review(dir_name, species):
            return False
            
        self.review_started.emit()
        
        # For the legacy integration, we need to collect segments in the expected format
        # and delegate back to the GUI methods that know how to handle the complex dialogs
        try:
            # Collect all segments by processing each sound file
            sound_files = self.audio_file_manager.scan_directory_recursive(self.current_dir)
            
            all_segments = []
            self.all_file_data = {}
            
            for filename in sound_files:
                # Use the workflow manager's own method to get properly formatted data
                file_data = self._process_file_for_review_internal(filename, species)
                if file_data:
                    # Store the full file data for later use
                    self.all_file_data[filename] = file_data
                    
                    # Create segment entries in the format expected by review methods
                    segments = file_data['segments']
                    for i, segment in enumerate(segments):
                        all_segments.append({
                            'segment': segment,
                            'filename': filename,
                            'index': i
                        })
            
            self.segments_collected.emit(all_segments)
            
            # Set up the parent window's allFileData for compatibility
            self.parent_window.allFileData = self.all_file_data
            
            # Set up GUI instance variables that are expected by review methods
            # These would normally be set by individual processFileForReview calls
            if hasattr(self.parent_window, 'batmode'):
                # Use existing value if already set
                pass
            else:
                # Set a default based on the files processed
                self.parent_window.batmode = any(
                    data.get('batmode', False) for data in self.all_file_data.values()
                )
            
            # Delegate to the appropriate GUI method based on review mode
            if quick:
                return self.parent_window.reviewAllSegmentsQuick(all_segments)
            else:
                return self.parent_window.reviewAllSegmentsOneByOne(all_segments)
                
        except Exception as e:
            print(f"Error in coordinate_review_workflow: {e}")
            return False
            
    def setup_review(self, dir_name, species):
        """Setup review session parameters.
        
        Args:
            dir_name: Directory to review
            species: Species to review
            
        Returns:
            bool: True if setup successful
        """
        if not dir_name or dir_name == '':
            msg = SupportClasses_GUI.MessagePopup("w", "Select Folder", "Please select a folder first!")
            msg.exec()
            return False
            
        self.current_dir = dir_name
        self.current_species = species
        
        # Validate species selection if provided
        if species and not self.species_manager.validate_species_name(species):
            msg = SupportClasses_GUI.MessagePopup("w", "Select Species", "Please select a species to review!")
            msg.exec()
            return False
            
        return True
        
    def _process_file_for_review_internal(self, filename, species):
        """Internal method to process a file for review using existing logic.
        
        Args:
            filename: Path to audio file
            species: Species to review
            
        Returns:
            dict: File data with segments and metadata, or None if file should be skipped
        """
        # Use the workflow manager's own process_file_for_review method
        try:
            # Get parameters from parent window if available, otherwise use defaults
            cert_threshold = 50  # Default threshold - lower to include more segments for review
            if hasattr(self.parent_window, 'certBox'):
                cert_threshold = self.parent_window.certBox.value()
                
            chunk_settings = {
                'manual': False,
                'size': 10
            }
            
            if hasattr(self.parent_window, 'chunksizeManual'):
                chunk_settings['manual'] = self.parent_window.chunksizeManual.isChecked()
            if hasattr(self.parent_window, 'chunksizeBox'):
                chunk_settings['size'] = self.parent_window.chunksizeBox.value()
                
            # Use the workflow manager's own method
            return self.process_file_for_review(filename, species, cert_threshold, chunk_settings)
            
        except Exception as e:
            print(f"Error in _process_file_for_review_internal: {e}")
            return self._simple_file_processing(filename, species)
            
    def _simple_file_processing(self, filename, species):
        """Simple fallback file processing when GUI methods aren't available."""
        try:
            import os
            
            # Check if data file exists
            data_file = filename + '.data'
            if not os.path.exists(data_file):
                return None
                
            # Load segments
            segments = Segment.SegmentList()
            segments.parseJSON(data_file)
            
            if len(segments) == 0:
                return None
                
            return {
                'segments': segments,
                'goodsegments': [],
                'allsegments': segments,
                'chunksize': 0,
                'batmode': filename.lower().endswith('.bmp')
            }
            
        except Exception as e:
            print(f"Error in simple file processing for {filename}: {e}")
            return None
        
    def collect_segments_for_review(self):
        """Collect all segments from all files for review.
        
        Returns:
            bool: True if collection successful
        """
        sound_files = self.audio_file_manager.scan_directory_recursive(self.current_dir)
        total = len(sound_files)
        
        if total == 0:
            msg = SupportClasses_GUI.MessagePopup("w", "No Files", "No processable files found in directory!")
            msg.exec()
            return False
            
        # Initialize collection state
        self.all_segments = []
        self.all_file_data = {}
        
        # Collect segments from each file
        for cnt, filename in enumerate(sound_files):
            self.review_progress_updated.emit(cnt + 1, total, f"Collecting from file {cnt + 1}/{total}: {os.path.basename(filename)}")
            
            if not self._collect_segments_from_file(filename):
                continue
                
        self.segments_collected.emit(self.all_segments)
        return True
        
    def _collect_segments_from_file(self, filename):
        """Collect segments from a single file.
        
        Args:
            filename: Path to audio file
            
        Returns:
            bool: True if segments collected successfully
        """
        try:
            # Load segments from .data file
            data_file = filename + '.data'
            if not os.path.exists(data_file):
                return False
                
            segments = Segment.SegmentList()
            segments.parseJSON(data_file)
            
            # Store file data for later use
            self.all_file_data[filename] = {
                'segments': segments,
                'data_file': data_file
            }
            
            # Filter segments by species if specified
            file_segments = []
            for segment in segments:
                if self._should_include_segment(segment):
                    # Add file context to segment
                    segment_with_context = {
                        'segment': segment,
                        'filename': filename,
                        'file_segments': segments
                    }
                    file_segments.append(segment_with_context)
                    
            self.all_segments.extend(file_segments)
            return True
            
        except Exception as e:
            print(f"Error collecting segments from {filename}: {e}")
            return False
            
    def _should_include_segment(self, segment):
        """Check if segment should be included in review.
        
        Args:
            segment: Segment object to check
            
        Returns:
            bool: True if segment should be included
        """
        if not self.current_species:
            # Include all segments if no species filter
            return True
            
        # Check if segment contains the target species
        for label in segment.keys():
            species_info = self.species_manager.parse_species_name(label)
            if species_info and species_info[0] == self.current_species:
                return True
                
        return False
        
    def review_all_segments_quick(self):
        """Review all segments in quick mode.
        
        Returns:
            bool: True if review completed successfully
        """
        if not self.all_segments:
            msg = SupportClasses_GUI.MessagePopup("i", "No Segments", "No segments found for review!")
            msg.exec()
            return False
            
        # Quick review logic - show all segments for bulk operations
        # This would integrate with the existing quick review UI
        total_segments = len(self.all_segments)
        
        self.review_progress_updated.emit(0, total_segments, f"Quick review mode: {total_segments} segments")
        
        # Process segments in batches for performance
        batch_size = 100
        for i in range(0, total_segments, batch_size):
            batch = self.all_segments[i:i + batch_size]
            self._process_segment_batch_quick(batch)
            self.review_progress_updated.emit(min(i + batch_size, total_segments), total_segments, 
                                            f"Processed {min(i + batch_size, total_segments)}/{total_segments} segments")
            
        self.review_completed.emit()
        return True
        
    def review_all_segments_one_by_one(self):
        """Review all segments one by one.
        
        Returns:
            bool: True if review completed successfully
        """
        if not self.all_segments:
            msg = SupportClasses_GUI.MessagePopup("i", "No Segments", "No segments found for review!")
            msg.exec()
            return False
            
        total_segments = len(self.all_segments)
        
        # One-by-one review logic
        for cnt, segment_with_context in enumerate(self.all_segments):
            self.review_progress_updated.emit(cnt + 1, total_segments, 
                                            f"Reviewing segment {cnt + 1}/{total_segments}")
            
            if not self._review_single_segment(segment_with_context):
                # User cancelled review
                break
                
        self.review_completed.emit()
        return True
        
    def _process_segment_batch_quick(self, segment_batch):
        """Process a batch of segments in quick mode.
        
        Args:
            segment_batch: List of segments to process
        """
        # Quick processing logic - could involve:
        # - Batch validation
        # - Bulk label operations
        # - Statistical analysis
        for segment_context in segment_batch:
            self._validate_segment_quick(segment_context)
            
    def _validate_segment_quick(self, segment_context):
        """Quick validation of a segment.
        
        Args:
            segment_context: Segment with file context
        """
        segment = segment_context['segment']
        
        # Quick validation logic
        # - Check segment bounds
        # - Validate labels
        # - Check for overlaps
        
        # Use species_manager for label validation
        for label in segment.keys():
            species_info = self.species_manager.parse_species_name(label)
            if not species_info:
                print(f"Invalid species label: {label}")
                
    def _review_single_segment(self, segment_context):
        """Review a single segment in detail.
        
        Args:
            segment_context: Segment with file context
            
        Returns:
            bool: True to continue, False to cancel
        """
        segment = segment_context['segment']
        filename = segment_context['filename']
        
        # Single segment review logic
        # This would integrate with the detailed review UI
        # - Load spectrogram
        # - Show segment details
        # - Allow user modifications
        
        return True  # Continue by default
        
    def save_review_results(self):
        """Save all review results back to files."""
        saved_files = []
        
        for filename, file_data in self.all_file_data.items():
            try:
                # Save segments back to .data file
                segments = file_data['segments']
                data_file = file_data['data_file']
                
                segments.saveJSON(data_file)
                saved_files.append(filename)
                
            except Exception as e:
                print(f"Error saving {filename}: {e}")
                
        return saved_files
        
    def _delegate_to_parent_review(self):
        """Delegate complex review dialog management back to parent window."""
        # This handles the case where the one-by-one review needs to interface
        # with the existing GUI dialog system that's complex to refactor
        if hasattr(self.parent_window, 'reviewAllSegmentsOneByOne'):
            return self.parent_window.reviewAllSegmentsOneByOne(self.all_segments)
        else:
            print("Warning: Parent window doesn't support one-by-one review delegation")
            return 1
        
    def get_review_statistics(self):
        """Get statistics about current review session.
        
        Returns:
            dict: Review statistics
        """
        stats = {
            'total_files': len(self.all_file_data),
            'total_segments': len(self.all_segments),
            'species_filter': self.current_species,
            'directory': self.current_dir
        }
        
        if self.current_species:
            # Count segments with target species
            species_count = 0
            for segment_context in self.all_segments:
                segment = segment_context['segment']
                for label in segment.keys():
                    species_info = self.species_manager.parse_species_name(label)
                    if species_info and species_info[0] == self.current_species:
                        species_count += 1
                        break
            stats['species_segments'] = species_count
            
        return stats
    
    def review_all_segments_quick(self, all_segments, gui_params):
        """ Reviews all segments in quick mode using a single dialog with pagination.
            all_segments: list of dicts with 'segment', 'filename', 'index' keys
            gui_params: dict with GUI settings like frequency ranges, dialog preferences
            Returns 1 for clean completion, 0 for Esc press.
        """
        from collections import defaultdict
        from ..core import Spectrogram
        from ..core import SignalProc
        from ..ui import colourMaps
        from ..ui import Dialogs
        import pyqtgraph as pg
        import copy
        
        # Group segments by file for efficient loading
        segments_by_file = defaultdict(list)
        for segData in all_segments:
            filename = segData['filename']
            segments_by_file[filename].append(segData)
        
        # Create segment list and spectrogram list for all segments
        all_segment_list = Segment.SegmentList()
        all_sps = []
        all_indices = []
        
        idx_counter = 0
        for filename, file_segments in segments_by_file.items():
            file_data = self.all_file_data[filename]
            chunksize = file_data['chunksize']
            
            # Determine file format
            if filename.lower().endswith('.bmp'):
                batmode = True
            else:
                batmode = False
            
            # Load file metadata
            if batmode:
                # For bat files, get duration from existing segments
                duration = max(segData['segment'][1] for segData in file_segments) + 1.0
                samplerate = 1  # Placeholder for bmp files
            else:
                import soundfile as sf
                info = sf.info(filename)
                samplerate = info.samplerate
                duration = info.frames / samplerate
            
            # Get frequency range from GUI params
            minFreq = max(gui_params.get('fLow', 0), 0)
            maxFreq = min(gui_params.get('fHigh', 32000), samplerate//2) if not batmode else 200000
            
            # Process each segment from this file
            for segData in file_segments:
                seg = segData['segment']
                orig_idx = segData['index']
                
                # Create spectrogram for this segment
                config = self.config_manager.config
                sp = Spectrogram.Spectrogram(config['window_width'], config['incr'], minFreq, maxFreq)
                
                if chunksize > 0:
                    halfChunk = 1.1/2 * chunksize
                    mid = (seg[0]+seg[1])/2
                    x1 = max(0, mid-halfChunk)
                    x2 = min(duration, mid+halfChunk)
                    x1nob = max(seg[0], x1)
                    x2nob = min(seg[1], x2)
                else:
                    x1nob = seg[0]
                    x2nob = seg[1]
                    x1 = max(x1nob - config['reviewSpecBuffer'], 0)
                    x2 = min(x2nob + config['reviewSpecBuffer'], duration)
                
                # Load spectrogram data
                try:
                    if batmode:
                        sp.readBmp(filename, off=x1, duration=x2-x1, silent=True)
                        sp.sg = sp.normalisedSpec("Batmode")
                    else:
                        sp.readSoundFile(filename, off=x1, duration=x2-x1, silent=True)
                        sp.data = SignalProc.bandpassFilter(sp.data, sp.audioFormat.sampleRate(), minFreq, maxFreq)
                        sp.sg = sp.spectrogram(window_width=config['window_width'], 
                                             incr=config['incr'],
                                             window=config['windowType'],
                                             sgType=config['sgType'],
                                             sgScale=config['sgScale'],
                                             nfilters=config['nfilters'],
                                             mean_normalise=config['sgMeanNormalise'],
                                             equal_loudness=config['sgEqualLoudness'],
                                             onesided=config['sgOneSided'])
                        
                        # Trim spectrogram to frequency range
                        height = sp.audioFormat.sampleRate()//2 / np.shape(sp.sg)[1]
                        pixelstart = int(minFreq/height)
                        pixelend = int(maxFreq/height)
                        sp.sg = sp.sg[:,pixelstart:pixelend]
                    
                    # Store unbuffered limits
                    sp.x1nobspec = sp.convertAmpltoSpec(x1nob-x1)
                    sp.x2nobspec = sp.convertAmpltoSpec(x2nob-x1)
                    
                    # Add to collections
                    all_segment_list.addSegment(seg)
                    all_sps.append(sp)
                    all_indices.append(idx_counter)
                    
                    # Store mapping back to original file/index
                    if not hasattr(self, '_quick_mapping'):
                        self._quick_mapping = {}
                    self._quick_mapping[idx_counter] = (filename, orig_idx)
                    
                    idx_counter += 1
                    
                except Exception as e:
                    print(f"Error loading segment from {filename}: {e}")
                    # Add placeholder
                    all_segment_list.addSegment(seg)
                    all_sps.append(None)
                    all_indices.append(idx_counter)
                    if not hasattr(self, '_quick_mapping'):
                        self._quick_mapping = {}
                    self._quick_mapping[idx_counter] = (filename, orig_idx)
                    idx_counter += 1
        
        if len(all_segment_list) == 0:
            return 1
        
        # Set up color map
        config = self.config_manager.config
        cmap = config['cmap']
        pos, colour, mode = colourMaps.colourMaps(cmap)
        cmap = pg.ColorMap(pos, colour)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        
        # Create normalized spectrograms
        sgs = []
        for sp in all_sps:
            if sp is not None:
                if batmode:
                    sgs.append(sp.sg)
                else:
                    sgs.append(sp.normalisedSpec(config['sgNormMode']))
            else:
                sgs.append(None)
        
        # Set up frequency guides
        if config['guidelinesOn']=='always' or (config['guidelinesOn']=='bat' and batmode):
            guides = config['guidepos']
        else:
            guides = None
        
        # Create and show dialog with ALL segments - let HumanClassify2 handle pagination
        species = gui_params.get('species', 'Unknown')
        dialog_title = f"Quick Review - {species} ({len(all_segment_list)} segments)"
        dialog = Dialogs.HumanClassify2(all_sps, sgs, all_segment_list, all_indices,
                                       species, lut, config['invertColourMap'],
                                       config['brightness'], config['contrast'],
                                       guidefreq=guides, guidecol=config['guidecol'],
                                       loop=gui_params.get('loop', False), filename=dialog_title)
        
        # Restore dialog position if available
        if 'dialogSize' in gui_params and 'dialogPos' in gui_params:
            dialog.resize(gui_params['dialogSize'])
            dialog.move(gui_params['dialogPos'])
        
        # Connect close handler
        dialog.finish.clicked.connect(lambda: self._handle_quick_dialog_close(dialog, gui_params))
        dialog.setModal(True)
        success = dialog.exec()
        
        return success
    
    def _handle_quick_dialog_close(self, dialog, gui_params):
        """ Handles the close event for quick review dialog """
        # Store dialog properties back to GUI params
        gui_params['dialogSize'] = dialog.size()
        gui_params['dialogPos'] = dialog.pos()
        
        # Save display preferences
        brightness = dialog.specControls.brightSlider.value()
        contrast = dialog.specControls.contrSlider.value()
        if not self.config_manager.config['invertColourMap']:
            brightness = 100-brightness
        
        self.config_manager.save_review_preferences({
            'brightness': brightness,
            'contrast': contrast
        })
        
        # Process button states and update segments
        if not hasattr(self, '_quick_changes'):
            self._quick_changes = {}
            
        for btn in dialog.buttons:
            btn.stopPlayback()
            quick_idx = btn.index
            
            if quick_idx in self._quick_mapping:
                filename, orig_idx = self._quick_mapping[quick_idx]
                
                if filename not in self._quick_changes:
                    self._quick_changes[filename] = {}
                
                # Store the button state for later processing
                self._quick_changes[filename][orig_idx] = btn.mark
        
        dialog.done(1)

    def save_quick_results(self, reviewer_name, species):
        """ Saves all changes from quick review back to the original files """
        if not hasattr(self, '_quick_changes'):
            return
            
        for filename, changes in self._quick_changes.items():
            if filename not in self.all_file_data:
                continue
                
            file_data = self.all_file_data[filename]
            segments = file_data['segments']
            goodsegments = file_data['goodsegments']
            
            todelete = []
            toadd = []
            
            # Process changes for this file
            for orig_idx, mark in changes.items():
                if orig_idx >= len(segments):
                    continue
                    
                seg = segments[orig_idx]
                
                if mark == "red":
                    # Remove all labels for the current species
                    wipedAll = seg.wipeSpecies(species)
                    if wipedAll:
                        todelete.append(orig_idx)
                elif mark == "yellow":
                    # Set uncertainty
                    seg.questionLabels(species)
                elif mark == "green":
                    # Confirm labels
                    seg.confirmLabels(species)
                elif mark == "blue":
                    # Duplicate segment
                    seg.confirmLabels(species)
                    new_seg = copy.deepcopy(seg)
                    new_seg[0] += 0.1
                    new_seg[1] += 0.1
                    new_seg[2] += 50
                    new_seg[3] += 50
                    toadd.append(new_seg)
            
            # Apply deletions (reverse order to preserve indices)
            for idx in reversed(sorted(todelete)):
                del segments[idx]
            
            # Add new segments
            segments.extend(toadd)
            
            # Re-add good segments
            segments.extend(goodsegments)
            
            # Save the file
            cleanexit = segments.saveJSON(filename + '.data', reviewer_name)
            if cleanexit != 1:
                print(f"Warning: could not save segments for {filename}!")
        
        # Clean up
        delattr(self, '_quick_changes')
        if hasattr(self, '_quick_mapping'):
            delattr(self, '_quick_mapping')
    
    def review_all_segments_one_by_one(self, all_segments, gui_params):
        """ Reviews segments one by one across all files.
            all_segments: list of dicts with 'segment', 'filename', 'index' keys
            gui_params: dict with GUI settings and callbacks
            Returns 1 for clean completion, 0 for Esc press or other dirty exit.
        """
        # Initialize tracking variables
        self.allSegmentsToReview = all_segments
        self.currentSegmentIndex = 0
        self.segsAccepted = 0
        self.segsDeleted = 0
        self.segsQuestioned = 0
        self.nsegments = len(all_segments)
        self.returned = False
        self.toadd = {}  # filename -> list of new segments to add
        
        # Track state changes for action buttons (certainty changes and deletion)
        self.segmentChanges = {}  # segment index -> 'accepted', 'deleted', 'questioned'
        
        # Initialize storage for corrections tracking
        if self.config_manager.config['saveCorrections']:
            self.allOriginalSegments = {}
            for filename, filedata in self.all_file_data.items():
                self.allOriginalSegments[filename] = copy.deepcopy(filedata['segments'])
        
        # Load bird lists and known calls
        self._load_bird_lists(all_segments)
        
        if len(all_segments) == 0:
            return 1
            
        # Load first segment and create dialog
        self._load_current_segment()
        success = self._create_one_by_one_dialog(gui_params)
        
        if success == 0:
            # On Esc press, only apply tracked action button changes (species changes already saved)
            self._save_one_by_one_changes(confirmed_only=True, reviewer=gui_params.get('reviewer', 'Unknown'))
        else:
            # On normal completion, apply all tracked changes
            self._save_one_by_one_changes(confirmed_only=False, reviewer=gui_params.get('reviewer', 'Unknown'))

        return success
    
    def _load_bird_lists(self, all_segments):
        """Load bird lists and update with species from segments"""
        try:
            species_data = self.species_manager.load_species_lists(self.config_manager.configdir)
        except ValueError as e:
            print(f"Error loading species lists: {e}")
            return
        
        # Scan segments for species and call types
        segment_species = self.species_manager.scan_segments_for_species(all_segments)
        
        # Update short list with new species from segments
        species_data['short_list'] = self.species_manager.update_short_list_from_segments(
            species_data['short_list'], segment_species['new_species']
        )
        
        # Merge known calls with segment data
        species_data['known_calls'] = self.species_manager.merge_known_calls(
            species_data['known_calls'], segment_species['known_calls']
        )
        
        # Store for use in dialogs
        self.shortBirdList = species_data['short_list']
        self.longBirdList = species_data['long_list']  
        self.knownCalls = species_data['known_calls']
        self.batList = species_data['bat_list']
    
    def _load_current_segment(self):
        """Load the current segment for review"""
        # This will be called by the GUI to load individual segments
        # Implementation depends on integration with existing segment loading
        pass
    
    def _create_one_by_one_dialog(self, gui_params):
        """Create and configure the one-by-one review dialog"""
        # This method would create the HumanClassify1 dialog
        # and handle the review loop, but needs integration with GUI callbacks
        # For now, delegate back to GUI for complex dialog management
        return gui_params.get('dialog_handler', lambda: 1)()
    
    def _save_one_by_one_changes(self, confirmed_only=False, reviewer='Unknown'):
        """Save tracked changes to files
        
        Args:
            confirmed_only (bool): If True, only apply changes from action button presses.
            reviewer (str): Name of the reviewer
        """
        # Apply tracked changes to original data
        if not confirmed_only:
            self._apply_tracked_changes()
        else:
            self._apply_tracked_changes()
        
        # Save each modified file
        for filename, filedata in self.all_file_data.items():
            all_segments = filedata['segments'] + filedata['goodsegments']
            
            # Remove duplicates
            seen_segments = set()
            unique_segments = []
            for seg in all_segments:
                seg_id = id(seg)
                if seg_id not in seen_segments:
                    seen_segments.add(seg_id)
                    unique_segments.append(seg)
            
            # Handle deletions using segmentChanges
            if hasattr(self, 'segmentChanges') and hasattr(self, 'allSegmentsToReview'):
                segments_to_remove = []
                for segIndex, state in self.segmentChanges.items():
                    if state == 'deleted' and segIndex < len(self.allSegmentsToReview):
                        segData = self.allSegmentsToReview[segIndex]
                        if segData['filename'] == filename:
                            segments_to_remove.append(segData['segment'])
                
                for seg_to_remove in segments_to_remove:
                    if seg_to_remove in unique_segments:
                        unique_segments.remove(seg_to_remove)
                        
                print(f"Deleted {len(segments_to_remove)} segments from {filename}")
            
            # Add any new segments
            if hasattr(self, 'toadd') and filename in self.toadd:
                unique_segments.extend(self.toadd[filename])
            
            # Save the file
            segments = filedata['segments']
            segments.clear()
            segments.extend(unique_segments)
            
            cleanexit = segments.saveJSON(filename + '.data', reviewer)
            if cleanexit != 1:
                print(f"Warning: could not save segments for {filename}!")
    
    def _apply_tracked_changes(self):
        """Apply all tracked changes from segmentChanges to segment data"""
        if not hasattr(self, 'segmentChanges'):
            return
            
        if hasattr(self, 'allSegmentsToReview'):
            # Cross-file mode: apply changes to segments in allSegmentsToReview
            for segmentIndex, state in self.segmentChanges.items():
                if segmentIndex >= len(self.allSegmentsToReview):
                    continue
                    
                segData = self.allSegmentsToReview[segmentIndex]
                segment = segData['segment']
                
                # Apply certainty changes (deletion is handled separately)
                if state == 'questioned':
                    if len(segment[4]) > 0:
                        for label in segment[4]:
                            label["certainty"] = 50
                elif state == 'accepted':
                    if len(segment[4]) > 0:
                        for label in segment[4]:
                            label["certainty"] = 100
    
    def process_file_for_review(self, filename, species, cert_threshold, chunk_settings):
        """Process a single file for review. Returns segments data or None if file should be skipped."""
        import os
        import pyqtgraph as pg
        
        if os.stat(filename).st_size < 1000:
            print("Warning: file %s empty, skipping" % filename)
            return None

        # check if file is formatted correctly
        batmode = False
        if filename.lower().endswith('.wav'):
            with open(filename, 'br') as f:
                if f.read(4) != b'RIFF':
                    print("Warning: WAV file %s not formatted correctly, skipping" % filename)
                    return None
            batmode = False
        elif filename.lower().endswith('.flac'):
            with open(filename, 'br') as f:
                if f.read(4) != b'fLaC':
                    print("Warning: FLAC file %s not formatted correctly, skipping" % filename)
                    return None
            batmode = False
        elif filename.lower().endswith('.bmp'):
            with open(filename, 'br') as f:
                if f.read(2) != b'BM':
                    print("Warning: BMP file %s not formatted correctly" % filename)
                    return None
            batmode = True
        else:
            print("Warning: file %s format not recognised " % filename)
            return None

        # Load segments
        with pg.BusyCursor():
            allsegments = Segment.SegmentList()
            allsegments.parseJSON(filename+'.data')

            segments = Segment.SegmentList()
            segments.parseJSON(filename+'.data')

            print(f"Processing {filename}: loaded {len(segments)} segments, species={species}, cert_threshold={cert_threshold}")

            # Separate out segments which do not need review
            goodsegments = []
            for seg in reversed(segments):
                goodenough = True
                if species is None or species == 'All species':
                    # For "All species" mode, check all labels
                    for lab in seg[4]:
                        if lab["certainty"] <= cert_threshold:
                            goodenough = False
                            break
                else:
                    # For specific species mode, only check labels for that species
                    species_labels = [lab for lab in seg[4] if lab["species"] == species]
                    for lab in species_labels:
                        if lab["certainty"] <= cert_threshold:
                            goodenough = False
                            break
                    # If no labels for this species exist, keep the segment for review
                    if not species_labels:
                        goodenough = False
                        
                if goodenough:
                    goodsegments.append(seg)
                    segments.remove(seg)
                    
            print(f"After filtering: {len(segments)} segments for review, {len(goodsegments)} good segments")

        # Skip review dialog if there's no segments passing relevant criteria
        if len(segments)==0 or (species is not None and species != 'All species' and len(segments.getSpecies(species))==0):
            print("No segments found in file %s" % filename)
            return None

        # Split segments into chunks if requested
        if chunk_settings.get('manual', False):
            chunksize = chunk_settings.get('size', 10)
            segments.splitLongSeg(species=species, maxlen=chunksize)
        else:
            # Leave all (chunksize = max segment length)
            chunksize = 0
            if species is None or species == 'All species':
                # For all species, check all segments
                for seg in segments:
                    chunksize = max(chunksize, seg[1]-seg[0])
            else:
                # For specific species, check segments with that species
                thisspsegs = segments.getSpecies(species)
                for si in thisspsegs:
                    seg = segments[si]
                    chunksize = max(chunksize, seg[1]-seg[0])
            print("Auto-setting view size to:", chunksize)

        _ = segments.orderTime()

        return {
            'chunksize': chunksize,
            'segments': segments,
            'goodsegments': goodsegments,
            'allsegments': allsegments,
            'batmode': batmode
        }