# Playback Management Module for AviaNZ
# Handles media playback control, volume, speed, and position tracking

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QStyle
from PyQt6.QtGui import QIcon

class PlaybackManager(QObject):
    """
    Handles media playback operations including:
    - Start/stop/pause playback of visible area or segments
    - Volume control
    - Playback speed adjustment
    - Position tracking and seeking
    - Button state management
    """
    
    # Signals
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()
    playback_paused = pyqtSignal()
    playback_position_changed = pyqtSignal(int)  # Position in spectrogram coordinates
    volume_changed = pyqtSignal(int)  # Volume level
    playback_finished = pyqtSignal()  # Playback reached the end
    button_state_changed = pyqtSignal(bool)  # True for pause state, False for play state
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.config = config_manager.config
        
        # Will be set by main window
        self.media_obj = None
        self.audio_processor = None
        
        # Playback state
        self.play_speed = 1.0
        self.segment_start = 0
        self.segment_stop = 0
        
    def set_media_object(self, media_obj):
        """Set the media object for playback control"""
        self.media_obj = media_obj
        
    def set_audio_processor(self, audio_processor):
        """Set the audio processor for coordinate conversions"""
        self.audio_processor = audio_processor
        
    def set_playback_speed(self, speed):
        """Set playback speed - handles both numeric and Unicode fraction strings"""
        if type(speed) is str:
            # convert Unicode fractions to floats
            speedchar = ord(speed)
            if speedchar == 188:
                speed = 0.25
            elif speedchar == 189:
                speed = 0.5
            elif speedchar == 190:
                speed = 0.75
        
        self.play_speed = 1/float(speed)
        if self.media_obj:
            self.media_obj.setSpeed(self.play_speed)
    
    def start_playback_visible(self, view_range, bar_position=None):
        """Start playback of the visible area
        
        Args:
            view_range: Tuple of (start_time, end_time) in seconds
            bar_position: Optional position from playback bar in spectrogram coordinates
        """
        if not self.media_obj:
            return
            
        if self.media_obj.isPlaying():
            self.pause_playback()
        else:
            self.segment_start = view_range[0] * 1000  # Convert to milliseconds
            self.segment_stop = view_range[1] * 1000
            
            if self.media_obj.isPlayingorPaused():
                self.media_obj.pressedPlay()
            else:
                # if bar was moved under pause, update the playback start position based on the bar:
                if bar_position is not None and bar_position > 0:
                    if self.audio_processor:
                        start = self.audio_processor.convertSpectoAmpl(bar_position) * 1000
                        print("found bar at %d ms" % start)
                    else:
                        start = self.segment_start
                else:
                    start = self.segment_start
                
                self.media_obj.pressedPlay(start=start, stop=self.segment_stop)
            
            self.playback_started.emit()
            self.button_state_changed.emit(True)  # Switch to pause state
    
    def start_playback_segment(self, segment_start_sec, segment_stop_sec, low_freq=None, high_freq=None):
        """Start playback of a specific segment
        
        Args:
            segment_start_sec: Start time in seconds
            segment_stop_sec: End time in seconds  
            low_freq: Optional low frequency limit for band-limited playback
            high_freq: Optional high frequency limit for band-limited playback
        """
        if not self.media_obj:
            return
            
        if self.media_obj.isPlayingorPaused():
            self.stop_playback()
        else:
            self.segment_start = segment_start_sec * 1000  # Convert to milliseconds
            self.segment_stop = segment_stop_sec * 1000
            
            self.media_obj.playSeg(self.segment_start, self.segment_stop, 
                                 speed=self.play_speed, low=low_freq, high=high_freq)
            
            self.playback_started.emit()
            self.button_state_changed.emit(True)  # Switch to pause state
    
    def start_playback_band_limited(self, segment_start_sec, segment_stop_sec, low_freq, high_freq):
        """Start band-limited playback of a segment
        
        Args:
            segment_start_sec: Start time in seconds
            segment_stop_sec: End time in seconds
            low_freq: Low frequency limit
            high_freq: High frequency limit
        """
        self.start_playback_segment(segment_start_sec, segment_stop_sec, low_freq, high_freq)
    
    def pause_playback(self):
        """Pause current playback"""
        if self.media_obj:
            self.media_obj.pressedPause()
            self.playback_paused.emit()
    
    def stop_playback(self):
        """Stop current playback"""
        if self.media_obj:
            self.media_obj.pressedStop()
            
        if not hasattr(self, 'segment_start') or self.segment_start is None:
            self.segment_start = 0
            
        self.playback_stopped.emit()
        self.button_state_changed.emit(False)  # Switch to play state
    
    def update_playback_position(self):
        """Update playback position - called by timer every 30ms
        
        Returns:
            int: Current position in spectrogram coordinates, or None if finished
        """
        if not self.media_obj:
            return None
            
        eltime = self.media_obj.processedUSecs() // 1000 // self.play_speed + self.media_obj.timeoffset
        
        # Check for playback finish (with small buffer for catching up)
        if eltime > (self.segment_stop - 10):
            print("Stopped at %d ms" % eltime)
            self.stop_playback()
            self.playback_finished.emit()
            return None
        else:
            # Convert to spectrogram coordinates (with small buffer)
            if self.audio_processor:
                spec_pos = int(self.audio_processor.convertAmpltoSpec(eltime / 1000.0 - 0.02))
                self.playback_position_changed.emit(spec_pos)
                return spec_pos
            return None
    
    def set_volume(self, value):
        """Set playback volume
        
        Args:
            value: Volume level (0-100)
        """
        if self.media_obj:
            self.media_obj.applyVolSlider(value)
            self.volume_changed.emit(value)
    
    def seek_to_position(self):
        """Handle seeking when playback bar is moved - resets player"""
        print("Resetting playback")
        if self.media_obj:
            self.media_obj.pressedStop()
    
    def is_playing(self):
        """Check if media is currently playing"""
        return self.media_obj.isPlaying() if self.media_obj else False
    
    def is_playing_or_paused(self):
        """Check if media is playing or paused"""
        return self.media_obj.isPlayingorPaused() if self.media_obj else False
