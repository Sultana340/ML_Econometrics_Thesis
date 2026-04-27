"""
Neural network architecture utilities for DeepIV.

This file defines a feed-forward network builder used by the DeepIV
treatment and response models.
"""

# --------------------------------------------------
# Keras / TensorFlow imports
# --------------------------------------------------
try:
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras import regularizers
except ImportError:
    from keras.layers import Dense, Dropout
    from keras import regularizers


def _get_activation(activations, layer_index):
    """
    Return the activation function for a given hidden layer.

    Parameters
    ----------
    activations : str, list, tuple, or None
        If a string is provided, the same activation is used for all layers.
        If a list/tuple is provided, the activation is selected by layer index.
    layer_index : int
        Index of the hidden layer.

    Returns
    -------
    activation : str or None
    """

    if activations is None:
        return None

    if isinstance(activations, (list, tuple)):
        if layer_index < len(activations):
            return activations[layer_index]
        return activations[-1]

    return activations


def feed_forward_net(
    inputs,
    output,
    hidden_layers=(128, 64, 32),
    dropout_rate=0.0,
    l2=0.0,
    activations="relu",
):
    """
    Construct a feed-forward neural network.

    Parameters
    ----------
    inputs : keras tensor
        Input layer or intermediate tensor.

    output : callable
        Output layer or output function. For example:
            Dense(1)
        or:
            lambda h: densities.mixture_of_gaussian_output(h, n_components)

    hidden_layers : tuple or list
        Number of hidden units in each layer.

    dropout_rate : float
        Dropout probability applied after each hidden layer.

    l2 : float
        L2 regularization strength.

    activations : str, list, tuple, or None
        Activation function for hidden layers.

    Returns
    -------
    keras tensor
        Output tensor of the feed-forward network.
    """

    h = inputs

    # L2 regularizer
    kernel_regularizer = regularizers.l2(l2) if l2 and l2 > 0 else None

    # Hidden layers
    for i, units in enumerate(hidden_layers):
        h = Dense(
            units,
            activation=_get_activation(activations, i),
            kernel_regularizer=kernel_regularizer,
        )(h)

        if dropout_rate is not None and dropout_rate > 0:
            h = Dropout(dropout_rate)(h)

    # Output layer or output function
    h = output(h)

    return h