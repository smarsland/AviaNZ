import tensorflow as tf
import numpy as np

def get_mnist():
    from tensorflow import keras
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, train_labels, test_images, test_labels

def basic_net():

    train_images, train_labels, test_images, test_labels = get_mnist()
    model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax) ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    predictions = model.predict(test_images)

    # Save the weights
    model.save_weights('./checkpoints/my_checkpoint')
    
    # Restore the weights
    model = create_model()
    model.load_weights('./checkpoints/my_checkpoint')
    
    loss,acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

def cnn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d( inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d( inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def useCNN():
    train_images, train_labels, test_images, test_labels = get_mnist()

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn( x={"x": train_images}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)

    # train one step and display the probabilties
    mnist_classifier.train( input_fn=train_input_fn, steps=1, hooks=[logging_hook])
    mnist_classifier.train(input_fn=train_input_fn, steps=1000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn( x={"x": test_images}, y=test_labels, num_epochs=1, shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

