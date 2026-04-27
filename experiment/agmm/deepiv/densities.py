"""
Density utilities for DeepIV.

This file defines the mixture-of-Gaussians output used in the DeepIV
first-stage treatment model p(T | Z).
"""

# --------------------------------------------------
# TensorFlow / Keras imports
# --------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Concatenate
except ImportError:
    import tensorflow as tf
    from keras.layers import Dense, Concatenate


def mixture_of_gaussian_output(h, n_components):
    """
    Create mixture-of-Gaussians output layer.

    Parameters
    ----------
    h : keras tensor
        Hidden representation from the treatment network.

    n_components : int
        Number of Gaussian mixture components.

    Returns
    -------
    keras tensor
        Concatenated tensor containing:
        - mixture logits
        - component means
        - component standard deviations
    """

    # Mixture weights before softmax
    logits = Dense(n_components, name="mixture_logits")(h)

    # Component means
    means = Dense(n_components, name="mixture_means")(h)

    # Positive component standard deviations
    stds = Dense(
        n_components,
        activation="softplus",
        name="mixture_stds",
    )(h)

    # Concatenate into one output tensor
    output = Concatenate(axis=1, name="mixture_output")([logits, means, stds])

    return output


def split_mixture_params(y_pred, n_components):
    """
    Split mixture output into logits, means, and standard deviations.

    Parameters
    ----------
    y_pred : tensor
        Output tensor from mixture_of_gaussian_output.

    n_components : int
        Number of mixture components.

    Returns
    -------
    logits, means, stds : tensors
        Mixture logits, component means, and positive standard deviations.
    """

    logits = y_pred[:, :n_components]
    means = y_pred[:, n_components : 2 * n_components]
    stds = y_pred[:, 2 * n_components : 3 * n_components]

    # Numerical safety
    stds = tf.maximum(stds, 1e-6)

    return logits, means, stds


def mixture_of_gaussians_loss(n_components):
    """
    Negative log-likelihood loss for scalar treatment under a Gaussian mixture.

    Parameters
    ----------
    n_components : int
        Number of mixture components.

    Returns
    -------
    loss_fn : callable
        Keras-compatible loss function.
    """

    def loss_fn(y_true, y_pred):
        """
        Compute negative log likelihood.

        y_true has shape (batch, 1).
        y_pred has shape (batch, 3 * n_components).
        """

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        logits, means, stds = split_mixture_params(y_pred, n_components)

        # Ensure y_true broadcasts against mixture components
        if len(y_true.shape) == 1:
            y_true_expanded = tf.expand_dims(y_true, axis=1)
        else:
            y_true_expanded = y_true

        # Gaussian log density for each component
        log_norm_const = -0.5 * tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=tf.float32))
        log_std = tf.math.log(stds)

        squared_term = -0.5 * tf.square((y_true_expanded - means) / stds)

        log_prob_components = log_norm_const - log_std + squared_term

        # Mixture weights
        log_weights = tf.nn.log_softmax(logits, axis=1)

        # Log mixture density
        log_prob = tf.reduce_logsumexp(log_weights + log_prob_components, axis=1)

        # Negative log likelihood
        return -log_prob

    return loss_fn


# --------------------------------------------------
# Aliases for compatibility with different model files
# --------------------------------------------------
def mixture_of_gaussians(y_true, y_pred, n_components=10):
    """
    Default mixture-of-Gaussians loss with n_components=10.

    This is useful if the model file calls the loss directly by function name.
    """

    return mixture_of_gaussians_loss(n_components)(y_true, y_pred)


def mixture_of_gaussian_loss(n_components):
    """
    Alias for mixture_of_gaussians_loss.
    """

    return mixture_of_gaussians_loss(n_components)


def get_loss(loss, n_components=10):
    """
    Return a Keras-compatible loss function.

    This is useful if models.py calls densities.get_loss(...)
    when compiling the Treatment model.
    """

    if loss in ["mixture_of_gaussians", "mixture_of_gaussian"]:
        return mixture_of_gaussians_loss(n_components)

    raise ValueError(f"Unknown density loss: {loss}")