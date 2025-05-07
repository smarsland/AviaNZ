import tensorflow as tf
import numpy as np

def CNNModel(imageHeight,imageWidth,outputDim):
    apply_same_padding =  imageHeight < 120 or imageWidth < 120
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=[imageHeight, imageWidth, 1], padding='Same'))
    model.add(tf.keras.layers.Conv2D(64, (7, 7), activation='relu', padding="Same" if apply_same_padding else "Valid"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding="Same" if apply_same_padding else "Valid"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding="Same" if apply_same_padding else "Valid"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="Same" if apply_same_padding else "Valid"))
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class PatchLayer(tf.keras.layers.Layer):
    def __init__(self, patchSize, patchOverlap, channels, name=None):
        super(PatchLayer, self).__init__(name=name)
        self.patchSize = patchSize
        self.patchOverlap = patchOverlap
        self.channels = channels

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patchSize, self.patchSize, 1],
            strides=[1, self.patchSize - self.patchOverlap, self.patchSize - self.patchOverlap, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [tf.shape(patches)[0], -1, self.patchSize*self.patchSize*self.channels])
        return patches

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, numPatches, embeddingDim, name=None):
        super().__init__(name=name)
        self.pos_emb = self.add_weight("pos_emb", shape=[1, numPatches, embeddingDim], initializer="random_normal")

    def call(self, x):
        return x + self.pos_emb

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, numHeads, inputDim, ffDim, dropoutRate=0.1,name=None):
        super(TransformerBlock, self).__init__(name=name)
        self.layerNormBefore = tf.keras.layers.LayerNormalization(epsilon=1e-12,name="pre-norm")
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=numHeads, key_dim=inputDim//numHeads, dropout=dropoutRate, name="attn")
        self.layerNormAfter = tf.keras.layers.LayerNormalization(epsilon=1e-12,name="post-norm")
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ffDim, activation=tf.nn.gelu,name="ff-intermediate"),
            tf.keras.layers.Dense(inputDim,name="ff-output")
        ])
        self.finalDropout = tf.keras.layers.Dropout(dropoutRate)

    def call(self, inputs, training=False):
        inputsNorm = self.layerNormBefore(inputs)
        attnOutput = self.attn(inputsNorm, inputsNorm, inputsNorm, training=training)
        attnOutputWithResidual = inputs + attnOutput
        attnOutputWithResidualNorm = self.layerNormAfter(attnOutputWithResidual)
        ffOutput = self.ff(attnOutputWithResidualNorm)
        ffOutputWithResidual = attnOutputWithResidual + ffOutput
        finalOut = self.finalDropout(ffOutputWithResidual, training=training)
        return finalOut

class ClsTokenLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, name=None):
        super(ClsTokenLayer, self).__init__(name=name)
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

def AudioSpectogramTransformer(imageHeight, imageWidth, outputDim, patchSize, patchOverlap, transformerHeads, embeddingDim, transformerLayers, transformerNNDim, dropoutRate=0.1, silent=False):
    if not embeddingDim % transformerHeads == 0:
        print("Warning: your embedding is not a multiple of the number of heads. Model may fail to load...")

    numPatches = ((imageHeight - patchSize) // (patchSize - patchOverlap) + 1) * ((imageWidth - patchSize) // (patchSize - patchOverlap) + 1)

    inputs = tf.keras.layers.Input(shape=(imageHeight, imageWidth, 1),name="inputs")
    # repeat to get 3 dimensions
    x = tf.keras.layers.Concatenate(name="concatenate")([inputs, inputs, inputs])
    
    # Extract patches
    x = PatchLayer(patchSize, patchOverlap, 3, name="patch_layer")(x)

    # embedding network
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(embeddingDim, activation='linear',name="embedding_weights"), name="linear_embedding_layer")(x)
    
    # Add the ClsTokenLayer after patch extraction
    x = ClsTokenLayer(embeddingDim, name="cls_token_layer")(x)

    # Add positional embedding
    x = PositionalEmbedding(numPatches=numPatches+1, embeddingDim=embeddingDim, name="positional_embedding")(x)
    
    for l in range(transformerLayers):
        x = TransformerBlock(transformerHeads, embeddingDim, transformerNNDim, dropoutRate, name="transformer_block_"+str(l+1))(x)
    
    x = tf.keras.layers.LayerNormalization(epsilon=1e-12,name="final_normalization")(x)
    
    x = x[:, 0, :] # we only use the output at the CLS token normally.

    if outputDim>=0:
        outputs = tf.keras.layers.Dense(outputDim, activation='softmax', name="output_layer")(x)
    else:
        outputs = x

    model = tf.keras.models.Model(inputs, outputs)
    if not silent:
        model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

customObjectScopes = {
    'PatchLayer': PatchLayer, 
    'PositionalEmbedding': PositionalEmbedding, 
    'TransformerBlock': TransformerBlock, 
    'ClsTokenLayer': ClsTokenLayer
}