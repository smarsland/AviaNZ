# AudioData.py
#
# Holds array and formats

# Version 4.0 09/10/25
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

class AudioData:
    """Container for loaded audio data with metadata."""
    def __init__(self, data, sample_rate, sample_format, sample_size, channels=1):
        self.data = data  # numpy array of audio samples
        self.sample_rate = sample_rate
        self.sample_format = sample_format  # e.g. 'Int16', 'Int32', 'UInt8'
        self.sample_size = sample_size  # in bits: 8, 16, 32
        self.channels = channels

    def extract_time_slice(self, start_sec, end_sec):
        """Extract a time slice of the audio data"""
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)
        sliced_data = self.data[start_sample:end_sample]        
        return AudioData(
            data=sliced_data,
            sample_rate=self.sample_rate,
            sample_format=self.sample_format,
            sample_size=self.sample_size,
            channels=self.channels
        )
