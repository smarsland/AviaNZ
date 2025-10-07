# coding=latin-1

# audio_player.py
# Audio playback functionality for the AviaNZ program


# Version 3.4 18/12/24
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
import threading
import numpy as np
from time import sleep
from PyQt6.QtCore import QTimer, QIODevice, QBuffer, QByteArray, pyqtSlot
from PyQt6.QtMultimedia import QAudio, QAudioSink, QAudioFormat, QMediaDevices

from src.core import SignalProc

try:
    import librosa
except ImportError:
    librosa = None


class ControllableAudio(QAudioSink):
    # This carries out the audio playback 
    # Pass in either an audioFormat or a ref to Spectrogram
    # If called by main interface, starts a timer for the moving bar

    def __init__(self, sp=None, loop=False, audioFormat=None,useBar=False):
        # Note the order here is audioFormat passed, otherwise sp.audioFormat
        #print(self.audioFormat.sample_format, self.audioFormat.sample_rate, self.audioFormat.bytesPerSample(), self.audioFormat.channels)

        if audioFormat is not None:
            self.audioFormat = audioFormat
        else:
            if sp is None:
                print("ERROR: need either audioFormat or SignalProc")
                return None
            # sp is a Spectrogram, extract the AudioData from it
            # sp.audio_data points to the AudioData object
            self.audioFormat = sp.audio_data
        
        # Convert AudioData format to QAudioFormat for PyQt6
        qtAudioFormat = self._convertToQtFormat(self.audioFormat)
        
        # Determine sample width from format
        sfmt = self.audioFormat.sample_format
        if 'Int16' in sfmt:
            self.sampwidth = 2
        elif 'Int32' in sfmt:
            self.sampwidth = 4
        elif 'UInt8' in sfmt or 'UInt8' == sfmt:
            self.sampwidth = 1
        else:
            print(f"ERROR: sample format {sfmt} not supported (size {self.audioFormat.sample_size})")
            self.sampwidth = 2  # default
        
        super(ControllableAudio, self).__init__(QMediaDevices.defaultAudioOutput(), format=qtAudioFormat)
        self.bytesPerSecond = int(self.sampwidth * self.audioFormat.sample_rate * self.audioFormat.channels)
        # TODO: or the size of the data if < 4 secs
        self.setBufferSize(int(self.bytesPerSecond/0.25)) # 4 s buffer

        # This is a timer for the moving bar. 
        # On this notify, move slider (connected where called)
        self.useBar = useBar
        if self.useBar:
            self.NotifyTimer = QTimer(self)

        self.timeoffset = 0  # start time of the played audio, in ms, relative to page start
        self.sp = sp
        self.audioThread = None
        self.audioThreadLoading = False
        self.audioThreadPaused = False
        self.playbackSpeed = 1.0
        self.bytesWritten = 0
    
    def _convertToQtFormat(self, audio_data):
        """Convert AudioData format to PyQt6 QAudioFormat."""
        qtFormat = QAudioFormat()
        qtFormat.setSampleRate(audio_data.sample_rate)
        qtFormat.setChannelCount(audio_data.channels)
        
        # Map string format to QAudioFormat.SampleFormat enum
        fmt_str = audio_data.sample_format
        if 'Int16' in fmt_str:
            qtFormat.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        elif 'Int32' in fmt_str:
            qtFormat.setSampleFormat(QAudioFormat.SampleFormat.Int32)
        elif 'UInt8' in fmt_str:
            qtFormat.setSampleFormat(QAudioFormat.SampleFormat.UInt8)
        elif 'Int8' in fmt_str:
            qtFormat.setSampleFormat(QAudioFormat.SampleFormat.Int8)
        else:
            # Default to Int16 if unknown
            qtFormat.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        
        return qtFormat
    
    def setSpeed(self, speed):
        self.playbackSpeed = speed

    @pyqtSlot()
    def isPlaying(self):
        return(self.state() == QAudio.State.ActiveState)

    @pyqtSlot()
    def isPlayingorPaused(self):
        return(self.state() == QAudio.State.ActiveState or self.state() == QAudio.State.SuspendedState)

    def endListener(self):
        # this should only be called if there's some misalignment between GUI and Audio
        # Should deal with underrun errors somehow
        print(self.bufferSize(),self.bytesFree(), self.elapsedUSecs())
        print("endlistener",self.state(),self.error())
        return

    @pyqtSlot()
    def pressedPlay(self, start=0, stop=0):
        # If playback bar has not moved, this can use resume() to continue from the same spot.
        # Otherwise assumes that the QAudioOutput was stopped/reset. In that case the updated 
        # position is passed as start, and playing starts anew from there.
        print("---", self.state(),start)
        if self.state() == QAudio.State.SuspendedState:
            print("resuming")
            self.resume()
            if self.useBar:
                self.NotifyTimer.start(30)
        else:
            self.playSeg(start,stop)

    @pyqtSlot()
    def pressedPause(self):
        self.audioThreadPaused = True
        self.suspend()
        if self.useBar:
            self.NotifyTimer.stop()

    @pyqtSlot()
    def pressedStop(self):
        # stop and reset to window/segment start

        # finish the threads
        self.audioThreadLoading = False
        if not self.audioThread is None:
            self.audioThread.join()
        
        # note if the audio was paused
        audio_was_paused = True if self.state() == QAudio.State.SuspendedState else False

        # do the reset
        self.reset()
        
        # Now if we were paused we resume. We couldn't do this before the reset, or it would play a short sound.
        if audio_was_paused:
            self.suspend()

        if self.useBar:
            self.NotifyTimer.stop()

    @pyqtSlot()
    def playSeg(self, start, stop, speed=1.0, audiodata=None, low=None, high=None):
        # Selects the data between start-stop ms, relative to file start
        # and plays it, optionally at a different speed and after bandpassing

        self.timeoffset = max(0, start)
        start = max(0, int(start * self.audioFormat.sample_rate // 1000))

        if audiodata is None:
            segment = self.sp.data[start:int(stop * self.audioFormat.sample_rate // 1000)]
        else:
            segment = audiodata

        if low is not None:
            segment = SignalProc.bandpassFilter(segment, self.audioFormat.sample_rate, start=low, end=high)

        if self.playbackSpeed != 1.0 and librosa is not None:
            segment = librosa.effects.time_stretch(segment.astype('float'), rate=self.playbackSpeed)

        print("Play starting ",start)
        self.loadArray(segment)

    def loadArray(self, audiodata):
        # Plays the entire audiodata 
        # Gets the format, then puts the data in a buffer
        # and then starts the QAudioOutput from that buffer
        fmt_str = self.audioFormat.sample_format
        if 'Int16' in fmt_str:
            audiodata = audiodata.astype('int16')  
        elif 'Int32' in fmt_str:
            audiodata = audiodata.astype('int32')  
        elif 'UInt8' in fmt_str or 'UInt8' == fmt_str:
            audiodata = audiodata.astype('uint8')  
        else:
            print("ERROR: sampleFormat %s not supported" % fmt_str)

        # double mono sound to get two channels -- simplifies reading
        chcount = self.audioFormat.channels

        if chcount == 2:
            audiodata = np.column_stack((audiodata, audiodata))

        self.audioByteArray = QByteArray(audiodata.tobytes())
        self.InBuffer = QBuffer(self.audioByteArray)
        self.InBuffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.bytesWritten = 0
        sleep(0.2)
        self.audioThreadLoading = True
        self.audioBuffer = self.start()
        self.audioThread = threading.Thread(target=self.fillBuffer)
        self.audioThread.start()

        if self.useBar:
            self.NotifyTimer.start(30)

    def fillBuffer(self):
        while self.InBuffer.bytesAvailable() > 0 and self.audioThreadLoading:
            if self.bytesFree() > 0 and not self.audioThreadPaused:
                data = self.InBuffer.read(min(self.bytesFree(), self.InBuffer.bytesAvailable()))
                self.audioBuffer.write(data)
                self.bytesWritten += len(data)
            sleep(0.01)

    @pyqtSlot()
    def applyVolSlider(self, value):
        # passes UI volume nonlinearly
        value = QAudio.convertVolume(value / 100, QAudio.VolumeScale.LogarithmicVolumeScale, QAudio.VolumeScale.LinearVolumeScale)
        self.setVolume(value)