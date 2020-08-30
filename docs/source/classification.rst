Classification
==============

Keras provides some magic for handling common classification problems, but you need to be sure that the type of classiciation (sparse categorical, binary or dense categorical) matches up with your (a) data generator, (b) loss function, (c) metrics, and (d) model outputs.

Sparse Categorical
------------------

Input labels are expected to be one integer per sample, with an integer value for the class, e.g., for a four-class problem, the label can be one of 0, 1, 2, or 3.

Loss function
^^^^^^^^^^^^^

The `tf.keras.metrics.SparseCategoricalAccuracy()` loss function handles sparse categorical input, converting from the integer representation to the one-hot to compute the loss. Specify the magic string `accuracy` for the metric which model.compile() will intelligently convert to the proper sparse categorical accuracy metric.

.. code-block:: python

	model.compile(optimizer=tf.optimizers.Adam(),
	              loss=tf.keras.losses.sparse_categorical_crossentropy,
	              metrics=['accuracy'])


Model definition
^^^^^^^^^^^^^^^^

Sparse categorical models use a one-hot vector as output, with a whole-layer softmax activation function. For example, a ten-class output layer:

.. code-block:: python

	model = tf.keras.models.Sequential([
	  tf.keras.layers.Dense(512, activation=tf.nn.relu),
	  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
	])

`Fashion MNIST <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Fashion_MNIST_Classifier.ipynb>`_


Binary Classification
---------------------

When the model is distinguishing between only two possibilities, there's an advantage to using only one output neuron with a sigmoidal activation. Although this is functionally equivalent to a two neuron softmax layer, the choice of using one sigmoidal output neuron ripples through the code.

Input labels are expected to be either 0 or 1.

.. code-block:: python

	model = tf.keras.models.Sequential([
	  tf.keras.layers.Dense(512, activation=tf.nn.relu),
	  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
	])


`Binary Classifier <https://colab.research.google.com/github/slightperturbation/ml_examples/blob/master/ML_Examples_Binary_Image_Classifier.ipynb>`_

Dense Categorical
-----------------

Input labels are a one-hot vector per sample, e.g., for a four-class problem, one of [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], or [0, 0, 0, 1].
