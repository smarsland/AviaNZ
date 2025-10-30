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

# TensorFlow inference utilities

import tensorflow as tf


def configure_gpu_memory():
    """Configure GPU memory settings for TensorFlow."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def predict_batch(model, features):
    """Run batch prediction on features using the given model.
    
    Args:
        model: TensorFlow/Keras model
        features: NumPy array with shape (batch, height, width, channels) or (batch, height, width)
        
    Returns:
        NumPy array of predictions
    """
    # TensorFlow expects (batch, height, width, channels)
    # If features don't have channel dimension, add it
    if len(features.shape) == 3:
        import numpy as np
        features = np.expand_dims(features, axis=-1)
    
    predictions = model.predict(features)
    return predictions
