Examples Organized by Pipeline Stage
====================================

Think of an idealized ML process as a pipeline; starting with inputing data, through defining the model, training and then using the output. Here are some examples highlighting choices at these different stages.


Input
-----

The input stage preprocesses the data (cleaning, normalizing, etc.) and then feeds into the model during training. Small datasets can be held in memory, while larger datasets need to be streamed from disk. Things get more complex when trying to avoid training stalls (pauses in training) with large datasets or augmented datasets.


In-memory arrays
^^^^^^^^^^^^^^^^

If your data fits entirely into memory, you can just load it up into an array and pass that directly to model.fit(). For example:

.. code-block:: python

    import tensorflow as tf

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [2.*x+1.2 for x in x]

    model = tf.keras.Sequential(
        [
         tf.keras.layers.Dense(1, activation='linear', input_shape=[1])
         ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01), loss='mse')
    model.fit(x, y, epochs=500)


`Linear Regression <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Example_Trivial_Linear_Regression_from_In_memory_Array.ipynb>`_


`Fashion MNIST <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Fashion_MNIST_Classifier.ipynb>`_


Data generators
^^^^^^^^^^^^^^^

Using Keras' Data Generators lets you efficiently stream batches of samples from disk. They will also handle automatically labelling samples based on the directory the sample comes from. 

`Binary Classifier <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Binary_Image_Classifier.ipynb>`_

Augment image data
^^^^^^^^^^^^^^^^^^

Specifically for image inputs, ImageDataGenerator gives several ways to transform your image data to help avoid overfitting. 

.. code-block:: python

  train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

  train_generator = train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',  # Source directory for training images
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary') # Since we use binary_crossentropy loss, we need binary labels

`Binary Classifier <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Binary_Image_Classifier.ipynb>`_



Model Definition
----------------

Treat image data as just a bunch of pixels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ignore the 2D structure of the image and treat it as a single array of unrelated inputs.

.. code-block:: python

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

`Fashion MNIST <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Fashion_MNIST_Classifier.ipynb>`_


Image classifier from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple, but quite effective, image classifier can be made up of just 2D convolutions and max pools, with a fully connected final stage.

.. code-block:: python

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='relu'),

        # One sigmoid output: zero means 'horses' and one means 'humans'
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


`Binary Classifier <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Binary_Image_Classifier.ipynb>`_


Using a pretrained model
^^^^^^^^^^^^^^^^^^^^^^^^

In this code, we get Inception with a 75x75 image input:

.. code-block:: python

    # Download the model and the weights
    inception = tf.keras.applications.inception_v3.InceptionV3(input_shape = (75, 75, 3),
                            include_top = False)

    # Lock it so it doesn't update during backpropagation
    for layer in inception.layers:
      layer.trainable = False

Next, we add an additional dense layer on top:

.. code-block:: python

    # Final layers to learn the classifier

    # 'mixed7' is the final CNN layer that we'll build on-top of.
    x = tf.keras.layers.Flatten()(inception.get_layer('mixed7').output)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)                  
    x = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)

    finetuned_inception = tf.keras.Model(inception.input, x)

This ``finetuned_inception`` model is just a Keras model, so it can be ``.compile()``d and ``.fit()`` as usual.

For a full example, see `Fine Tuning Inception <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Fine_Tuning.ipynb>`_


Modify the input of an existing model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's common to use a pretrained model for image tasks with a fixed input size. Moving the resizing into the graph has the advantage of ensuring the same resizing algorithm is used for both training and serving (avoiding any training-serving skew). Adding a resize layer to the front of a frozen model is easy, if a little finicky syntax-wise.

First, start with a pretrained model as in the previous section. 

Next, add the resize layer as the input of the the finetuned_inception model. Note the ``preprocess = finetuned_inception(preprocess)`` line is where we create a new layer that passes into the Inception model.

.. code-block:: python

    # Add a new layer at the beginning to resize the image up to Inception size.
    input = tf.keras.Input(shape=(32, 32, 3))
    preprocess = tf.keras.layers.experimental.preprocessing.Resizing(75, 75)(input)
    preprocess = finetuned_inception(preprocess)

    model = tf.keras.Model(input, preprocess)

For a full example, see `Fine Tuning Inception <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Fine_Tuning.ipynb>`_


Loss Function
-------------

Classification with sparse categorical labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training with a sparse categorical class inputs (e.g., integers from 0 to 9 for a ten-class problem) that are automatically converted to a size-ten one-hot vector by the loss function.

`Fashion MNIST <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Fashion_MNIST_Classifier.ipynb>`_


Binary classification
^^^^^^^^^^^^^^^^^^^^^

Binary crossentropy loss for use with a binary classification problem and a single sigmoidal output: `Binary Classifier <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Binary_Image_Classifier.ipynb>`_


Training
--------

Optimizers
^^^^^^^^^^

* Simple use of Adam optimizer:  `Fashion MNIST <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Fashion_MNIST_Classifier.ipynb>`_

* Simple use of RMSProp: `Binary Classifier <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Binary_Image_Classifier.ipynb>`_


Schedule of learning rates using callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Changing the learning rate while training can speed-up early training with a big step, then dial down the learning rate to avoid overshooting when close. These can be hard to tune, and often better handled in the optimizer, but worth trying if needed.

.. code-block:: python

    def lr_schedule(epoch, lr):
        if epoch > 50:
            return 0.001
        if epoch > 25:
            return 0.01
        # Use the starting rate until epoch 25
        return lr
    callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

    model.fit(image_train, label_train, epochs=15, callbacks=[callback])

`Learning Rate Callback <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Training_Callbacks.ipynb#scrollTo=0ByKI22dcEmg>`_

Evaluation
----------

* Record the training accuracy of a sparse categorical problem with the string keyword `accuracy`:  `Fashion MNIST <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Fashion_MNIST_Classifier.ipynb>`_

* Use a separate validation generator with notes on getting the number of validation steps right: `Binary Classifier <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Binary_Image_Classifier.ipynb>`_

* Early stopping at a target accuracy using a callback: `Early Stopping Callback <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Training_Callbacks.ipynb#scrollTo=-t4_wC_mGO4m>`_


Visualization
-------------

* Textual output of model.fit():  `Fashion MNIST <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Fashion_MNIST_Classifier.ipynb>`_

Matplotlib
^^^^^^^^^^

Record history and compare train and validation accuracy in a matplotlib plot: `Binary Classifier <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Binary_Image_Classifier.ipynb>`_

TensorBoard
^^^^^^^^^^^

Use TensorBoard to see progress during traning, directly in colab:

.. code-block:: python

    %load_ext tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    %tensorboard --logdir logs

    model.fit(image_train, label_train, epochs=5, callbacks=[tensorboard_callback])

`Visualizing with TensorBoard <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Example_TensorBoard_Example.ipynb>`_


Add profiling data to TensorBoard:

.. code-block:: python

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 profile_batch=(1, 10))
    # Train the model and use the TensorBoard Keras callback to collect
    # performance profiling data.
    model.fit(dataset,
              epochs=1,
              callbacks=[tb_callback])

