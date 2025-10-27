
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

# Holds the models for NN training, does some saving / loading stuff

import tensorflow as tf
from packaging import version
import json
import numpy as np

def CNNModel(imageHeight,imageWidth,outputDim):
    apply_same_padding =  imageHeight < 120 or imageWidth < 120
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=[imageHeight, imageWidth, 1], padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (7, 7), activation='relu', padding="same" if apply_same_padding else "Valid"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding="same" if apply_same_padding else "Valid"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding="same" if apply_same_padding else "Valid"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same" if apply_same_padding else "Valid"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    # Flatten the results to one dimension for passing into our final layer
    model.add(tf.keras.layers.Flatten())
    # A hidden layer to learn with
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    # Another dropout
    model.add(tf.keras.layers.Dropout(0.5))
    # Final categorization from 0-ct+1 with softmax
    model.add(tf.keras.layers.Dense(outputDim, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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
        # Manually remove 'batch_input_shape' and insert InputLayer
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
            # Try loading with compile=False and then recompiling
            try:
                model = tf.keras.models.load_model(h5Path, compile=False, custom_objects=customObjectScopes)
                # Recompile with default settings
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                return model
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                # If both fail, suggest converting to JSON format
                raise ValueError(f"Unable to load H5 model {h5Path}. Consider converting to JSON format. Original error: {e}")
        else:
            raise

