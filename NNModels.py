import tensorflow as tf

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
        patches = tf.reshape(patches, [tf.shape(patches)[0], -1, self.patchSize * self.patchSize])
        return patches

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, numPatches, embeddingDim):
        super().__init__()
        self.pos_emb = self.add_weight("pos_emb", shape=[1, numPatches, embeddingDim], initializer="random_normal")

    def call(self, x):
        return x + self.pos_emb

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, keyDim, numHeads, inputDim, ffDim, dropoutRate=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=numHeads, key_dim=keyDim, dropout=dropoutRate)
        self.dropout1 = tf.keras.layers.Dropout(dropoutRate)
        self.layerNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ffDim, activation='relu'),
            tf.keras.layers.Dense(inputDim)
        ])
        self.layerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = tf.keras.layers.Dropout(dropoutRate)

    def call(self, inputs, training=False):
        attnOutput = self.attn(inputs, inputs, inputs, training=training)
        attnOutput = self.dropout1(attnOutput, training=training)
        out = self.layerNorm1(inputs + attnOutput)
        ffOutput = self.ff(out)
        ffOutput = self.dropout2(ffOutput, training=training)
        out = self.layerNorm2(out + ffOutput)
        return out

def AudioSpectogramTransformer(imageHeight, imageWidth, outputDim):
    patchSize = 16
    patchOverlap = 6
    transformerHeads = 12
    transformerKeyDim = 12 # 64
    transformerLayers = 4
    transformerNNDim = 4*transformerKeyDim # have heard that this is a good value
    dropoutRate = 0.01

    embeddingDim = transformerHeads*transformerKeyDim
    numPatches = ((imageHeight - patchSize) // (patchSize - patchOverlap) + 1) * ((imageWidth - patchSize) // (patchSize - patchOverlap) + 1)

    inputs = tf.keras.layers.Input(shape=(imageHeight, imageWidth, 1))
    x = PatchLayer(patchSize, patchOverlap)(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(embeddingDim, activation='linear'))(x)
    x = PositionalEmbedding(numPatches=numPatches, embeddingDim=embeddingDim)(x)
    cls_token = tf.zeros((tf.shape(x)[0], 1, embeddingDim))
    x = tf.keras.layers.Concatenate(axis=1)([cls_token, x])
    for _ in range(transformerLayers):
        x = TransformerBlock(transformerKeyDim, transformerHeads, embeddingDim, transformerNNDim, dropoutRate)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(outputDim, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

customObjectScopes = {'PatchLayer': PatchLayer, 'PositionalEmbedding': PositionalEmbedding, 'TransformerBlock': TransformerBlock}
