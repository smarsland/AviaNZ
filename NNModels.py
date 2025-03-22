import tensorflow as tf

def CNNModel(imageheight,imagewidth,outputdim):
    apply_same_padding =  imageheight < 120 or imagewidth < 120
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=[imageheight, imagewidth, 1], padding='Same'))
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
    model.add(tf.keras.layers.Dense(outputdim, activation='softmax'))
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
        patches = tf.reshape(patches, [tf.shape(patches)[0], tf.shape(patches)[1] * tf.shape(patches)[2], self.patchSize * self.patchSize])
        return patches

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_emb = self.add_weight("pos_emb", shape=[1, num_patches, embed_dim], initializer="random_normal")

    def call(self, x):
        return x + self.pos_emb

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def AudioSpectogramTransformer(imageheight, imagewidth, outputdim):
    patchsize = 16
    patchoverlap = 6
    embeddingdim = 65
    numtransformerheads = 4
    numtransformerlayers = 4
    num_patches = ((imageheight - patchsize) // (patchsize - patchoverlap) + 1) * ((imagewidth - patchsize) // (patchsize - patchoverlap) + 1)
    inputs = tf.keras.layers.Input(shape=(imageheight, imagewidth, 1))
    x = PatchLayer(patchsize, patchoverlap)(inputs)
    x = tf.keras.layers.Dense(embeddingdim, activation='linear')(x)
    x = PositionalEmbedding(num_patches=num_patches, embed_dim=embeddingdim)(x)
    for _ in range(numtransformerlayers):
        x = TransformerBlock(embeddingdim, numtransformerheads, embeddingdim)(x)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=numtransformerheads, key_dim=embeddingdim)(x, x)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.Dense(embeddingdim, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(outputdim, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

customObjectScopes = {'PatchLayer': PatchLayer, 'PositionalEmbedding': PositionalEmbedding, 'TransformerBlock': TransformerBlock}
