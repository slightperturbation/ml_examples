Text
====

Text processing begins with converting the text into input symbols. There are three possibilities: convert each character into an input, convert each word into an input or convert each sub-word parts into an input.

Character-level input has the advantage of being strictly limited in number of inputs-- e.g., a mere 128 ASCII characters vs. the hundreds of thousands of different English words. Word-level inputs 


Tokenizing
----------

Input labels are expected to be one integer per sample, with an integer value for the class, e.g., for a four-class problem, the label can be one of 0,
1, 2, or 3.

Character-level inputs
^^^^^^^^^^^^^^^^^^^^^^

Word-level inputs
^^^^^^^^^^^^^^^^^


The `tf.keras.metrics.SparseCategoricalAccuracy()` loss function handles sparse categorical input, converting from the integer representation to the one
-hot to compute the loss. Specify the magic string `accuracy` for the metric which model.compile() will intelligently convert to the proper sparse categ
orical accuracy metric.

.. code-block:: python

	model.compile(optimizer=tf.optimizers.Adam(),
	              loss=tf.keras.losses.sparse_categorical_crossentropy,
	              metrics=['accuracy'])

