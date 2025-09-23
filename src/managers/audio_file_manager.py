# audio_file_manager.py
# Part of AviaNZ refactoring - handles file I/O, loading, and navigation

import os
import re
import time
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from ..ui import SupportClasses_GUI
from ..core import Segment


class AudioFileManager(QObject):
    """Manages audio file operations, loading, saving, and navigation.
    
    Handles all file I/O operations including:
    - Opening and loading audio files
    - Saving annotation data
    - File navigation and list management
    - Recent files tracking
    - File validation and format checking
    """
    
    # Signals emitted when file operations occur
    file_loaded = pyqtSignal(str)  # filename
    file_saved = pyqtSignal(str)   # filename
    file_list_updated = pyqtSignal()
    file_navigation_changed = pyqtSignal(int)  # current file index
    
    def __init__(self, config_manager, audio_processor=None):
        super().__init__()
        self.config_manager = config_manager
        self.audio_processor = audio_processor
        self.config = config_manager.config
        
        # File state
        self.current_filename = None
        self.current_file_section = 0
        self.sound_file_dir = None
        self.start_time = 0
        
        # Audio data will be set by main window
        self.segments = None
        self.sp = None
        self.sg = None
        
    def set_audio_context(self, segments, sp, sg, sound_file_dir):
        """Set audio context from main window"""
        self.segments = segments
        self.sp = sp
        self.sg = sg
        self.sound_file_dir = sound_file_dir
        
    def open_file(self, filename=None, parent_widget=None):
        """Handle file opening with dialog if no filename provided.
        
        Args:
            filename: Path to file to open, or None to show dialog
            parent_widget: Parent widget for file dialog
            
        Returns:
            bool: True if file opened successfully
        """
        if filename is None:
            # Show file selection dialog
            filename, _ = QFileDialog.getOpenFileName(
                parent_widget, 
                'Choose File', 
                self.sound_file_dir or '', 
                "WAV or BMP files (*.wav *.bmp);; Only WAV files (*.wav);; Only BMP files (*.bmp);; FLAC files (*.flac)"
            )
        
        if not filename:
            return False
            
        print(f"Opening file {filename}")
        
        # Update working directory
        old_dir = self.sound_file_dir
        old_filename = self.current_filename
        
        self.sound_file_dir = os.path.dirname(filename)
        
        # Try to load the file
        success = self.load_from_file_list(os.path.basename(filename))
        
        if not success:
            print("Warning: could not load file, reverting to previous file")
            self.sound_file_dir = old_dir
            if old_filename:
                self.load_from_file_list(os.path.basename(old_filename))
            return False
            
        return True
        
    def load_from_file_list(self, filename):
        """Load file from file list with safety checks.
        
        Args:
            filename: Base filename to load
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        # Extract filename if it's a list item
        if hasattr(filename, 'text'):
            filename = filename.text()
            filename = re.sub(r'\/.*', '', filename)
            
        full_path = os.path.join(self.sound_file_dir, filename)
        
        # Safety checks
        if os.path.isdir(full_path):
            return False
            
        if not os.path.isfile(full_path):
            print(f"File {full_path} does not exist!")
            return False
            
        # Check file size
        if os.stat(full_path).st_size == 0:
            print(f"Cannot open file {full_path} of size 0!")
            return False
            
        if os.stat(full_path).st_size < 1000:
            print(f"File {full_path} appears to have only header")
            
        # Load the file
        return self.load_audio_file(full_path)
        
    def load_audio_file(self, filepath):
        """Load audio file and prepare basic file information.
        
        Args:
            filepath: Full path to audio file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(filepath):
                print(f"ERROR: tried to open non-existing file {filepath}")
                return False
                
            self.current_filename = filepath
            self.current_file_section = 0
            
            # Parse DOC recording format for time information
            doc_match = re.search(r'(\d{6})_(\d{6})', filepath[-17:-4])
            if doc_match:
                time_str = doc_match.group(2)
                hour = int(time_str[:2])
                
                if hour > 17 or hour < 7:  # 6pm to 6am
                    print("Night time DOC recording")
                else:
                    print("Day time DOC recording")
                    
                self.start_time = hour * 3600 + int(time_str[2:4]) * 60 + int(time_str[4:6])
            else:
                self.start_time = 0
                
            # Note: file_loaded signal will be emitted by main window after segments are set up
            return True
            
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
            
    def navigate_to_next_file(self, file_list_widget, main_window=None):
        """Navigate to next file in the list.
        
        Args:
            file_list_widget: QListWidget containing file list
            main_window: Reference to main window for listLoadFile call
            
        Returns:
            bool: True if navigation successful
        """
        # Check if we're in the right directory
        if hasattr(file_list_widget, 'soundDir'):
            if file_list_widget.soundDir != os.path.dirname(self.current_filename):
                return False
                
        current_row = file_list_widget.currentRow()
        if current_row + 1 < len(file_list_widget):
            file_list_widget.setCurrentRow(current_row + 1)
            self.file_navigation_changed.emit(current_row + 1)
            
            # Let main window handle the actual loading via listLoadFile
            if main_window and hasattr(main_window, 'listLoadFile'):
                return main_window.listLoadFile(file_list_widget.currentItem()) == 0
            else:
                return self.load_from_file_list(file_list_widget.currentItem())
        else:
            # Show completion message
            return False
            
    def save_annotations(self, segments, operator=None, reviewer=None):
        """Save annotation data to .data file.
        
        Args:
            segments: Current segments object from main window
            operator: Name of operator who made annotations
            reviewer: Name of reviewer who checked annotations
        """
        if segments is None:
            return False
            
        # Ensure we have a proper SegmentList with metadata
        if not hasattr(segments, 'metadata'):
            # Convert plain list to SegmentList if needed
            if isinstance(segments, list):
                new_segments = Segment.SegmentList()
                new_segments.extend(segments)
                segments = new_segments
            else:
                # Still can't add metadata - this shouldn't happen
                print("Error: segments object doesn't support metadata")
                return False
            
        # Ensure metadata is a dictionary
        if not hasattr(segments, 'metadata') or segments.metadata is None:
            segments.metadata = {}
            
        # Update metadata (even for empty segments list)
        if operator:
            segments.metadata["Operator"] = operator
        if reviewer:
            segments.metadata["Reviewer"] = reviewer
            
        # Save to JSON file (even if empty - this clears the annotations)
        if not self.current_filename:
            print("Error: No current filename set - cannot save")
            return False
            
        data_filename = str(self.current_filename) + '.data'
        
        try:
            segments.saveJSON(data_filename)
            
            # Emit signal
            self.file_saved.emit(self.current_filename)
            return True
        except Exception as e:
            print(f"Error saving segments: {e}")
            return False
        
    def close_current_file(self):
        """Close current file and clean up."""
        # Note: Saving is handled by main window before this method is called
        
        # Update recent files
        if self.current_filename and self.current_filename not in self.config['RecentFiles']:
            self.config['RecentFiles'].append(self.current_filename)
            if len(self.config['RecentFiles']) > 4:
                self.config['RecentFiles'] = self.config['RecentFiles'][-4:]
                
        # Clear current state
        self.current_filename = None
        self.current_file_section = 0
        
    def populate_file_list(self, directory, current_filename, file_list_widget):
        """Populate file list widget with files from directory.
        
        Args:
            directory: Directory to scan
            current_filename: Currently opened file to highlight
            file_list_widget: QListWidget to populate
        """
        if not os.path.isdir(directory):
            print(f"ERROR: directory {directory} doesn't exist")
            return
            
        if hasattr(file_list_widget, 'fill'):
            file_list_widget.fill(directory, current_filename)
            self.file_list_updated.emit()
            
    def validate_file(self, filepath):
        """Validate that file exists and has reasonable size.
        
        Args:
            filepath: Path to file to validate
            
        Returns:
            bool: True if file is valid
        """
        if not os.path.isfile(filepath):
            return False
            
        # Check file size
        size = os.stat(filepath).st_size
        if size == 0:
            return False
            
        if size < 1000:
            print(f"Warning: File {filepath} appears to have only header")
            
        return True
        
    def get_file_info(self):
        """Get information about current file.
        
        Returns:
            dict: File information including path, size, format
        """
        if not self.current_filename:
            return {}
            
        info = {
            'filename': self.current_filename,
            'basename': os.path.basename(self.current_filename),
            'directory': os.path.dirname(self.current_filename),
            'start_time': self.start_time,
            'section': self.current_file_section
        }
        
        if os.path.exists(self.current_filename):
            info['size'] = os.stat(self.current_filename).st_size
            info['exists'] = True
        else:
            info['exists'] = False
            
        return info
        
    def update_recent_files(self, filename):
        """Update recent files list with new file.
        
        Args:
            filename: File to add to recent list
        """
        if filename not in self.config['RecentFiles']:
            self.config['RecentFiles'].append(filename)
            if len(self.config['RecentFiles']) > 4:
                self.config['RecentFiles'] = self.config['RecentFiles'][-4:]
                
    # ========== BATCH OPERATIONS EXTENSION ==========
    
    def scan_directory_recursive(self, directory):
        """Recursively scan directory for processable audio files.
        
        Args:
            directory: Root directory to scan
            
        Returns:
            list: List of processable audio files with corresponding .data files
        """
        processable_files = []
        for root, dirs, files in os.walk(str(directory)):
            for filename in files:
                filepath = os.path.join(root, filename)
                if self._is_processable_audio_file(filepath):
                    processable_files.append(filepath)
        return processable_files
        
    def _is_processable_audio_file(self, filepath):
        """Check if audio file is processable (has corresponding .data file).
        
        Args:
            filepath: Path to audio file
            
        Returns:
            bool: True if file is processable
        """
        # Check if it's an audio file
        if not (filepath.lower().endswith('.wav') or 
                filepath.lower().endswith('.flac') or 
                filepath.lower().endswith('.bmp')):
            return False
            
        # Check if corresponding .data file exists
        data_file = filepath + '.data'
        return os.path.isfile(data_file)
        
    def validate_batch_files(self, file_list):
        """Validate files have required .data annotations.
        
        Args:
            file_list: List of file paths to validate
            
        Returns:
            dict: {'valid': [list], 'invalid': [list]}
        """
        valid_files = []
        invalid_files = []
        
        for filepath in file_list:
            if self._is_processable_audio_file(filepath):
                valid_files.append(filepath)
            else:
                invalid_files.append(filepath)
                
        return {'valid': valid_files, 'invalid': invalid_files}
        
    def get_file_processing_order(self, directory):
        """Return optimized file processing order.
        
        Args:
            directory: Directory to process
            
        Returns:
            list: Files in optimized processing order
        """
        files = self.scan_directory_recursive(directory)
        # Sort by file size (smaller files first for faster feedback)
        return sorted(files, key=lambda f: os.path.getsize(f) if os.path.exists(f) else 0)
        
    def find_files_by_pattern(self, directory, pattern):
        """Find files matching a pattern in directory tree.
        
        Args:
            directory: Root directory to search
            pattern: File pattern to match (e.g., '*.xlsx', '*.data')
            
        Returns:
            list: List of matching file paths
        """
        import fnmatch
        matching_files = []
        for root, dirs, files in os.walk(str(directory)):
            for filename in files:
                filepath = os.path.join(root, filename)
                if fnmatch.fnmatch(filepath, pattern) or filename.endswith(pattern.replace('*', '')):
                    matching_files.append(filepath)
        return matching_files
        
    def get_all_data_files(self, directory):
        """Get all .data annotation files in directory tree.
        
        Args:
            directory: Root directory to search
            
        Returns:
            list: List of .data file paths
        """
        return self.find_files_by_pattern(directory, '*.data')
