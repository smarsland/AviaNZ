import tensorflow as tf
import os
import shutil
from packaging import version
import h5py
import json
from tensorflow.keras.models import model_from_json

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

class PatchLayer(tf.keras.layers.Layer):
    def __init__(self, patchSize, patchOverlap):
        super(PatchLayer, self).__init__()
        self.patchSize = patchSize
        self.patchOverlap = patchOverlap

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patchSize, self.patchSize, 1],
            strides=[1, self.patchSize - self.patchOverlap, self.patchSize - self.patchOverlap, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [tf.shape(patches)[0], -1, self.patchSize, self.patchSize, 3])
        return patches

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, numPatches, embeddingDim):
        super().__init__()
        self.pos_emb = self.add_weight("pos_emb", shape=[1, numPatches, embeddingDim], initializer="random_normal")

    def call(self, x):
        return x + self.pos_emb

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, numHeads, inputDim, ffDim, dropoutRate=0.1):
        super(TransformerBlock, self).__init__()
        self.layerNormBefore = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=numHeads, key_dim=inputDim//numHeads, dropout=dropoutRate)
        self.layerNormAfter = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ffDim, activation=tf.nn.gelu),
            tf.keras.layers.Dense(inputDim)
        ])

    def call(self, inputs, training=False):
        inputsNorm = self.layerNormBefore(inputs)
        attnOutput = self.attn(inputsNorm, inputsNorm, inputsNorm, training=training)
        attnOutputWithResidual = inputs + attnOutput
        attnOutputWithResidualNorm = self.layerNormAfter(attnOutputWithResidual)
        ffOutput = self.ff(attnOutputWithResidualNorm)
        ffOutputWithResidual = attnOutputWithResidual + ffOutput
        return ffOutputWithResidual

class ClsTokenLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim):
        super(ClsTokenLayer, self).__init__()
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        # Initialize clsToken as a trainable variable (weight)
        self.clsToken = self.add_weight(
            shape=(1, 1, self.embedding_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token"
        )
        super(ClsTokenLayer, self).build(input_shape)

    def call(self, inputs):
        # Concatenate clsToken with inputs (i.e., after patch extraction)
        # repeat clsToken for each input
        clsTokenRepeated = tf.repeat(self.clsToken, repeats=tf.shape(inputs)[0], axis=0) 
        return tf.concat([clsTokenRepeated, inputs], axis=1)

def AudioSpectogramTransformer(imageHeight, imageWidth, outputDim):
    patchSize = 16
    patchOverlap = 0
    transformerHeads = 12
    embeddingDim = 768
    transformerLayers = 12
    transformerNNDim = 3072
    dropoutRate = 0.1

    numPatches = ((imageHeight - patchSize) // (patchSize - patchOverlap) + 1) * ((imageWidth - patchSize) // (patchSize - patchOverlap) + 1)

    inputs = tf.keras.layers.Input(shape=(imageHeight, imageWidth, 1))
    # repeat to get 3 dimensions
    x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
    
    # Extract patches
    x = PatchLayer(patchSize, patchOverlap)(x)

    # embedding network
    x = tf.keras.layers.Reshape((-1, patchSize * patchSize * 3))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(embeddingDim, activation='linear'))(x)
    
    # Add the ClsTokenLayer after patch extraction
    x = ClsTokenLayer(embeddingDim)(x)

    # Add positional embedding
    positionalEmbedding = PositionalEmbedding(numPatches=numPatches+1, embeddingDim=embeddingDim)
    x = positionalEmbedding(x)
    
    for _ in range(transformerLayers):
        x = TransformerBlock(transformerHeads, embeddingDim, transformerNNDim, dropoutRate)(x)
    
    if outputDim==0: # no head
        outputs = x
    else:
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(outputDim, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def PretrainedAudioSpectogramTransformer(imageHeight, imageWidth, outputDim):
    if not imageHeight==224 or not imageWidth==224:
        print("Error: pretrained model requires imageHeight and imageWidth are set to 224. Change this in the learning parameters file.")
        return
    model = AudioSpectogramTransformer(224,224,outputDim)
    print("Loading weights...")
    loadWeights(model, "pre-trained_ViT.weights.h5")
    return model


customObjectScopes = {
    'PatchLayer': PatchLayer, 
    'PositionalEmbedding': PositionalEmbedding, 
    'TransformerBlock': TransformerBlock, 
    'ClsTokenLayer': ClsTokenLayer,
}

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
        if config["config"]["layers"][0]["class_name"] == "Conv2D":
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
    return model_from_json(model_json, custom_objects=customObjectScopes)

def loadWeights(model, path):
    try:
        print("Attemping to load NN model weights from", path)
        model.load_weights(path)
    except Exception as e:
        print(e)
        print("Warning: Model weights could not be loaded. This is likely because you are using an old model which was built in a previous version of tensorflow. Trying to load layer by layer...")

        name_map = {
            'conv2d': 'conv2d',
            'conv2d_2': 'conv2d_1',
            'conv2d_4': 'conv2d_2',
            'conv2d_6': 'conv2d_3',
            'conv2d_8': 'conv2d_4',
            'dense': 'dense',
            'dense_2': 'dense_1',
        }

        # print("\n--- Model layers ---")
        # for l in model.layers:
        #     print(l.name, [w.shape.as_list() for w in l.weights])
        # print("\n--- Checkpoint layers ---")
        # with h5py.File(path, 'r') as f:
        #     for layer_name in f.keys():
        #         print(f"Layer group: {layer_name}")
        #         for key in f[layer_name].keys():
        #             print(f"  Key: {key}, type: {type(f[layer_name][key])}")
        #             if isinstance(f[layer_name][key], h5py.Group):
        #                 for subkey in f[layer_name][key].keys():
        #                     print(f"    Subkey: {subkey}, type: {type(f[layer_name][key][subkey])}")
            
        with h5py.File(path, 'r') as f:
            layer_group = f['_layer_checkpoint_dependencies']
            for layer_name in layer_group.keys():
                vars_group = layer_group[layer_name]['vars']
                
                # Find the corresponding model layer by name (you must have layer names)
                model_layer_name = name_map.get(layer_name)
                if model_layer_name is None:
                    print(f"Skipping {layer_name}")
                    continue
                model_layer = next((l for l in model.layers if l.name == model_layer_name), None)

                if model_layer is None:
                    print(f"Layer {layer_name} not found in model.")
                    continue

                model_vars = model_layer.weights
                if len(model_vars) != len(vars_group):
                    print(f"Warning: Variable count mismatch for layer {layer_name}: model {len(model_vars)} vs checkpoint {len(vars_group)}")

                for var, (var_name, dataset) in zip(model_vars, vars_group.items()):
                    print(f"Assigning {var.name} {var.shape} <- {var_name} {dataset.shape}")
                    if var.shape == dataset.shape:
                        var.assign(dataset[()])
                    else:
                        print(f"Shape mismatch for {var.name}: model {var.shape}, checkpoint {dataset.shape}")