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

def SingleLayerNetwork(imageheight,imagewidth,outputdim):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[imageheight, imagewidth, 1]))
    model.add(tf.keras.layers.Dense(outputdim, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model