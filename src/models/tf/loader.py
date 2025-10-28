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

# TensorFlow model loading utilities

import tensorflow as tf
from packaging import version
import json


customObjectScopes = {}
if version.parse(tf.__version__) >= version.parse("2.13.0"):
    from keras.saving import register_keras_serializable
    
    @register_keras_serializable()
    class SequentialWrapper(tf.keras.models.Sequential):
        pass

    customObjectScopes['Sequential'] = SequentialWrapper


def loadModelFromJson(jsonPath):
    with open(jsonPath, "r") as f:
        config = json.load(f)

    hasInputLayer = any(layer['class_name'] == 'InputLayer' for layer in config['config']['layers'])
    if not hasInputLayer and 'batch_input_shape' in config['config']['layers'][0]['config']:
        print("Likely missing InputLayer; older Keras may fail to load this.")
        input_shape = config["config"]["layers"][0]["config"].pop("batch_input_shape", None)
        if input_shape:
            config["config"]["layers"].insert(0, {
                "class_name": "InputLayer",
                "config": {
                    "batch_input_shape": input_shape,
                    "dtype": "float32",
                    "name": "input"
                }
            })
    model_json = json.dumps(config)
    return tf.keras.models.model_from_json(model_json, custom_objects=customObjectScopes)


def loadModelFromH5(h5Path):
    """Load a model directly from an H5 file with compatibility handling."""
    try:
        return tf.keras.models.load_model(h5Path, custom_objects=customObjectScopes)
    except (TypeError, ValueError) as e:
        if 'dtype' in str(e) or 'GlorotUniform' in str(e):
            print(f"Compatibility issue loading H5 model, trying alternative approach: {e}")
            try:
                model = tf.keras.models.load_model(h5Path, compile=False, custom_objects=customObjectScopes)
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                return model
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                raise ValueError(f"Unable to load H5 model {h5Path}. Consider converting to JSON format. Original error: {e}")
        else:
            raise


def loadWeights(model, weightsPath):
    """Load weights into an existing model.
    
    Args:
        model: TensorFlow/Keras model instance
        weightsPath: Path to .h5 or .weights.h5 file
        
    Returns:
        model with loaded weights
    """
    model.load_weights(weightsPath)
    return model
